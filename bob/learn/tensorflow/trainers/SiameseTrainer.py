#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# @date: Tue 09 Aug 2016 15:25:22 CEST

import logging
logger = logging.getLogger("bob.learn.tensorflow")
import tensorflow as tf
import threading
from ..analyzers import ExperimentAnalizer
from ..network import SequenceNetwork
import bob.io.base
from .Trainer import Trainer
import os
import sys

class SiameseTrainer(Trainer):

    def __init__(self,
                 architecture,
                 optimizer=tf.train.AdamOptimizer(),
                 use_gpu=False,
                 loss=None,
                 temp_dir="cnn",

                 # Learning rate
                 base_learning_rate=0.001,
                 weight_decay=0.9,

                 ###### training options ##########
                 convergence_threshold=0.01,
                 iterations=5000,
                 snapshot=100):

        super(SiameseTrainer, self).__init__(
            architecture=architecture,
            optimizer=optimizer,
            use_gpu=use_gpu,
            loss=loss,
            temp_dir=temp_dir,
            base_learning_rate=base_learning_rate,
            weight_decay=weight_decay,
            convergence_threshold=convergence_threshold,
            iterations=iterations,
            snapshot=snapshot
        )

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
            # for i in range(self.iterations+5):
            while not thread_pool.should_stop():
                batch_left, batch_right, labels = train_data_shuffler.get_pair()

                feed_dict = {train_placeholder_left_data: batch_left,
                             train_placeholder_right_data: batch_right,
                             train_placeholder_labels: labels}

                session.run(enqueue_op, feed_dict=feed_dict)

        # TODO: find an elegant way to provide this as a parameter of the trainer
        learning_rate = tf.train.exponential_decay(
            self.base_learning_rate,  # Learning rate
            train_data_shuffler.batch_size,
            train_data_shuffler.n_samples,
            self.weight_decay  # Decay step
        )

        # Creating directory
        bob.io.base.create_directories_safe(self.temp_dir)

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

        loss_train, between_class, within_class = self.loss(train_labels_batch,
                                                            train_left_graph,
                                                            train_right_graph)

        # Preparing the optimizer
        step = tf.Variable(0)
        self.optimizer._learning_rate = learning_rate
        optimizer = self.optimizer.minimize(loss_train, global_step=step)
        #optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=0.99, use_locking=False,
        #                                       name='Momentum').minimize(loss_train, global_step=step)

        print("Initializing !!")
        # Training
        hdf5 = bob.io.base.HDF5File(os.path.join(self.temp_dir, 'model.hdf5'), 'w')

        with tf.Session() as session:
            if validation_data_shuffler is not None:
                analizer = ExperimentAnalizer(validation_data_shuffler, self.architecture, session)

            tf.initialize_all_variables().run()

            # Start a thread to enqueue data asynchronously, and hide I/O latency.
            thread_pool = tf.train.Coordinator()
            tf.train.start_queue_runners(coord=thread_pool)
            threads = start_thread()

            # TENSOR BOARD SUMMARY
            train_writer = tf.train.SummaryWriter(os.path.join(self.temp_dir, 'LOGS'), session.graph)

            # Siamese specific summary
            tf.scalar_summary('loss', loss_train)
            tf.scalar_summary('between_class', between_class)
            tf.scalar_summary('within_class', within_class)
            tf.scalar_summary('lr', learning_rate)
            merged = tf.merge_all_summaries()

            # Architecture summary
            self.architecture.generate_summaries()
            merged_validation = tf.merge_all_summaries()

            for step in range(self.iterations):

                _, l, lr, summary = session.run(
                    [optimizer, loss_train, learning_rate, merged])
                #_, l, lr,b,w, summary = session.run([optimizer, loss_train, learning_rate,between_class,within_class, merged])
                #_, l, lr= session.run([optimizer, loss_train, learning_rate])
                train_writer.add_summary(summary, step)
                #print str(step) + " loss: {0}, bc: {1}, wc: {2}".format(l, b, w)
                #print str(step) + " loss: {0}".format(l)
                sys.stdout.flush()
                #import ipdb; ipdb.set_trace();

                if validation_data_shuffler is not None and step % self.snapshot == 0:
                    print str(step)
                    sys.stdout.flush()

                    summary = session.run(merged_validation)
                    train_writer.add_summary(summary, step)

                    summary = analizer()
                    train_writer.add_summary(summary, step)

            print("#######DONE##########")
            self.architecture.save(hdf5)
            del hdf5
            train_writer.close()

            thread_pool.request_stop()
            thread_pool.join(threads)
