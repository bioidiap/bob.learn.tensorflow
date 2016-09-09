#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# @date: Tue 09 Aug 2016 15:25:22 CEST

import logging
logger = logging.getLogger("bob.learn.tensorflow")
import tensorflow as tf
import threading
from ..analyzers import Analizer
from ..network import SequenceNetwork
import bob.io.base
import os


class SiameseTrainer(object):

    def __init__(self,

                 architecture=None,
                 use_gpu=False,
                 loss=None,
                 temp_dir="",
                 save_intermediate=False,

                 ###### training options ##########
                 convergence_threshold = 0.01,
                 iterations=5000,
                 base_lr=0.001,
                 momentum=0.9,
                 weight_decay=0.95,

                 # The learning rate policy
                 snapshot=100):

        self.loss = loss
        self.loss_instance = None
        self.optimizer = None
        self.temp_dir = temp_dir
        self.save_intermediate = save_intermediate

        self.architecture = architecture
        self.use_gpu = use_gpu

        self.iterations = iterations
        self.snapshot = snapshot
        self.base_lr = base_lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.convergence_threshold = convergence_threshold

    def train(self, train_data_shuffler, validation_data_shuffler=None):
        """
        Do the loop forward --> backward --|
                      ^--------------------|
        """

        def start_thread():
            threads = []
            for n in range(1):
                t = threading.Thread(target=load_and_enqueue)
                t.daemon = True  # thread will close when parent quits
                t.start()
                threads.append(t)
            return threads

        def load_and_enqueue():
            """
            Injecting data in the place holder queue
            """
            for i in range(self.iterations):
                batch_left, batch_right, labels = train_data_shuffler.get_pair()

                feed_dict = {train_placeholder_left_data: batch_left,
                             train_placeholder_right_data: batch_right,
                             train_placeholder_labels: labels}

                session.run(enqueue_op, feed_dict=feed_dict)

        bob.io.base.create_directories_safe(os.path.join(self.temp_dir, 'OUTPUT'))

        # Creating two graphs
        train_placeholder_left_data, train_placeholder_labels = train_data_shuffler.\
            get_placeholders_forprefetch(name="train_left")
        train_placeholder_right_data, _ = train_data_shuffler.get_placeholders(name="train_right")

        # Defining a placeholder queue for prefetching
        queue = tf.FIFOQueue(capacity=100,
                             dtypes=[tf.float32, tf.float32, tf.int64],
                             shapes=[train_placeholder_left_data.get_shape().as_list()[1:],
                                     train_placeholder_right_data.get_shape().as_list()[1:],
                                     []])
        # Fetching the place holders from the queue
        enqueue_op = queue.enqueue_many([train_placeholder_left_data,
                                         train_placeholder_right_data,
                                         train_placeholder_labels])
        train_left_feature_batch, train_right_label_batch, train_labels_batch = \
            queue.dequeue_many(train_data_shuffler.batch_size)

        # Creating the architecture for train and validation
        if not isinstance(self.architecture, SequenceNetwork):
            raise ValueError("The variable `architecture` must be an instance of "
                             "`bob.learn.tensorflow.network.SequenceNetwork`")

        # Creating the siamese graph
        train_left_graph = self.architecture.compute_graph(train_left_feature_batch)
        train_right_graph = self.architecture.compute_graph(train_right_label_batch)

        loss_train, within_class, between_class = self.loss(train_labels_batch,
                                                            train_left_graph,
                                                            train_right_graph)

        batch = tf.Variable(0)

        learning_rate = tf.train.exponential_decay(
            self.base_lr,  # Learning rate
            batch * train_data_shuffler.batch_size,
            train_data_shuffler.n_samples,
            self.weight_decay  # Decay step
        )
        #optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_train,
        #                                                                      global_step=batch)
        optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=0.99, use_locking=False,
                                               name='Momentum').minimize(loss_train, global_step=batch)

        print("Initializing !!")
        # Training
        hdf5 = bob.io.base.HDF5File(os.path.join(self.temp_dir, 'model.hdf5'), 'w')

        with tf.Session() as session:
            if validation_data_shuffler is not None:
                analizer = Analizer(validation_data_shuffler, self.architecture, session)

            tf.initialize_all_variables().run()

            # Start a thread to enqueue data asynchronously, and hide I/O latency.
            thread_pool = tf.train.Coordinator()
            tf.train.start_queue_runners(coord=thread_pool)
            threads = start_thread()

            train_writer = tf.train.SummaryWriter(os.path.join(self.temp_dir, 'LOGS'), session.graph)

            # Tensorboard data
            tf.scalar_summary('loss', loss_train)
            tf.scalar_summary('between_class', between_class)
            tf.scalar_summary('within_class', within_class)
            tf.scalar_summary('lr', learning_rate)
            merged = tf.merge_all_summaries()

            for step in range(self.iterations):

                _, l, lr, summary = session.run([optimizer, loss_train, learning_rate, merged])
                train_writer.add_summary(summary, step)

                if validation_data_shuffler is not None and step % self.snapshot == 0:
                    analizer()
                    if self.save_intermediate:
                        self.architecture.save(hdf5, step)
                    print str(step) + " - " + str(analizer.eer[-1])

            self.architecture.save(hdf5)
            del hdf5
            train_writer.close()

            thread_pool.request_stop()
            thread_pool.join(threads)

