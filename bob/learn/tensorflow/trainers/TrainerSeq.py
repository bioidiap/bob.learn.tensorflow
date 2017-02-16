#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# @date: Tue 09 Aug 2016 15:25:22 CEST

import tensorflow as tf
from ..network import SequenceNetwork
import threading
import os
import bob.io.base
import bob.core
from ..analyzers import SoftmaxAnalizer
from tensorflow.core.framework import summary_pb2
import time
from bob.learn.tensorflow.datashuffler.OnlineSampling import OnLineSampling
from bob.learn.tensorflow.utils.session import Session
from .learning_rate import constant

import numpy
logger = bob.core.log.setup("bob.learn.tensorflow")


class TrainerSeq(object):
    """
    One graph trainer.
    Use this trainer when your CNN is composed by one graph

    **Parameters**
    architecture:
      The architecture that you want to run. Should be a :py:class`bob.learn.tensorflow.network.SequenceNetwork`

    optimizer:
      One of the tensorflow optimizers https://www.tensorflow.org/versions/r0.10/api_docs/python/train.html

    use_gpu: bool
      Use GPUs in the training

    loss: :py:class:`bob.learn.tensorflow.loss.BaseLoss`
      Loss function

    temp_dir: str
      The output directory

    learning_rate: :py:class:`bob.learn.tensorflow.trainers.learningrate`
      Initial learning rate

    convergence_threshold:

    iterations: int
      Maximum number of iterations

    snapshot: int
      Will take a snapshot of the network at every `n` iterations

    prefetch: bool
      Use extra Threads to deal with the I/O

    model_from_file: str
      If you want to use a pretrained model

    analizer:
      Neural network analizer :py:mod:`bob.learn.tensorflow.analyzers`

    verbosity_level:

    """

    def __init__(self,
                 architecture,
                 optimizer=tf.train.AdamOptimizer(),
                 use_gpu=False,
                 loss=None,
                 temp_dir="cnn",

                 # Learning rate
                 learning_rate=constant(),

                 ###### training options ##########
                 convergence_threshold=0.01,
                 iterations=5000,
                 snapshot=500,
                 validation_snapshot=100,
                 prefetch=False,
                 epochs=10,

                 ## Analizer
                 analizer=SoftmaxAnalizer(),

                 ### Pretrained model
                 model_from_file="",

                 verbosity_level=2):

        if not isinstance(architecture, SequenceNetwork):
            raise ValueError("`architecture` should be instance of `SequenceNetwork`")

        self.architecture = architecture
        self.optimizer_class = optimizer
        self.use_gpu = use_gpu
        self.loss = loss
        self.temp_dir = temp_dir

        self.learning_rate = learning_rate

        self.iterations = iterations
        self.snapshot = snapshot
        self.validation_snapshot = validation_snapshot
        self.convergence_threshold = convergence_threshold
        self.prefetch = prefetch

        self.epochs = epochs  # how many epochs to run

        # Training variables used in the fit
        self.optimizer = None
        self.training_graph = None
        self.train_data_shuffler = None
        self.summaries_train = None
        self.train_summary_writter = None
        self.train_thread_pool = None
        self.train_threads = None

        # Validation data
        self.validation_graph = None
        self.validation_data_shuffler = None
        self.validation_summary_writter = None
        self.valid_thread_pool = None
        self.valid_threads = None

        # Analizer
        self.analizer = analizer

        self.enqueue_op_train = None
        self.enqueue_op_valid = None
        self.global_epoch = None
        self.threads_lock_train = threading.RLock()
        self.threads_lock_valid = threading.RLock()


        self.model_from_file = model_from_file
        self.session = None

        bob.core.log.set_verbosity_level(logger, verbosity_level)

    def __del__(self):
        tf.reset_default_graph()

    def compute_graph(self, data_shuffler, prefetch=False, name="", training=True):
        """
        Computes the graph for the trainer.

        ** Parameters **

            data_shuffler: Data shuffler
            prefetch: Uses prefetch
            name: Name of the graph
            training: Is it a training graph?
        """

        # Defining place holders
        if prefetch:
            [placeholder_data, placeholder_labels] = data_shuffler.get_placeholders_forprefetch(name=name)

            # Defining a placeholder queue for prefetching
            queue = tf.FIFOQueue(capacity=100000,
                                 dtypes=[tf.float32, tf.int64],
                                 shapes=[placeholder_data.get_shape().as_list()[1:], []])

            # Fetching the place holders from the queue
            if training:
                self.enqueue_op_train = queue.enqueue_many([placeholder_data, placeholder_labels])
            else:
                self.enqueue_op_valid = queue.enqueue_many([placeholder_data, placeholder_labels])
            feature_batch, label_batch = queue.dequeue_many(data_shuffler.batch_size)

            # Creating the architecture for train and validation
            if not isinstance(self.architecture, SequenceNetwork):
                raise ValueError("The variable `architecture` must be an instance of "
                                 "`bob.learn.tensorflow.network.SequenceNetwork`")
        else:
            [feature_batch, label_batch] = data_shuffler.get_placeholders(name=name)

        # Creating graphs and defining the loss
        network_graph = self.architecture.compute_graph(feature_batch, training=training)
        graph = self.loss(network_graph, label_batch)
        if not training:
            return [network_graph, graph, label_batch]

        return graph

    def get_feed_dict(self, data_shuffler):
        """
        Given a data shuffler prepared the dictionary to be injected in the graph

        ** Parameters **
            data_shuffler:

        """
        [data, labels] = data_shuffler.get_batch()
        # when we run out of data
        if data is None:
            return None

        [data_placeholder, label_placeholder] = data_shuffler.get_placeholders()

        feed_dict = {data_placeholder: data,
                     label_placeholder: labels}
        return feed_dict

    def fit(self):
        """
        Run one iteration (`forward` and `backward`)

        ** Parameters **
            session: Tensorflow session
            step: Iteration number

        """

        if self.prefetch:
            _, l, lr, summary = self.session.run([self.optimizer, self.training_graph,
                                                  self.learning_rate, self.summaries_train])
        else:
            feed_dict = self.get_feed_dict(self.train_data_shuffler)
            # if we run out of data
            if feed_dict is None:
                return None, None
            _, l, lr, summary = self.session.run([self.optimizer, self.training_graph,
                                                  self.learning_rate, self.summaries_train], feed_dict=feed_dict)
        return l, summary

    def compute_validation(self, data_shuffler):
        """
        Computes the loss in the validation set

        ** Parameters **
            session: Tensorflow session
            data_shuffler: The data shuffler to be used
            step: Iteration number

        """
        if self.prefetch:
            prediction, l, labels = self.session.run(self.validation_graph)
        else:
            # Opening a new session for validation
            [data, labels] = data_shuffler.get_batch()
            # when we run out of data
            if data is None:
                return None, None

            [data_placeholder, label_placeholder] = data_shuffler.get_placeholders()

            feed_dict = {data_placeholder: data,
                         label_placeholder: labels}

            prediction, l, labels = self.session.run(self.validation_graph, feed_dict=feed_dict)

        prediction = numpy.argmax(prediction, 1)
        prediction_err = numpy.asarray([prediction != labels], dtype=numpy.int16)

        return prediction_err, l

    def create_general_summary(self):
        """
        Creates a simple tensorboard summary with the value of the loss and learning rate
        """
        # Train summary
        tf.scalar_summary('loss', self.training_graph, name="train")
        tf.scalar_summary('lr', self.learning_rate, name="train")
        return tf.merge_all_summaries()

    def start_thread(self, training=True):
        """
        Start pool of train_threads for pre-fetching

        **Parameters**
          session: Tensorflow session
        """

        threads = []
        for n in range(10):
            if training:
                t = threading.Thread(target=self.load_and_enqueue, args=())
            else:
                t = threading.Thread(target=self.load_and_enqueue_valid, args=())
            t.daemon = True  # thread will close when parent quits
            t.start()
            threads.append(t)
        return threads

    def load_and_enqueue(self):
        """
        Injecting data in the place holder queue

        **Parameters**
          session: Tensorflow session
        """

        while not self.train_thread_pool.should_stop():
            with self.threads_lock_train:
                [train_data, train_labels] = self.train_data_shuffler.get_batch()

            # if we run out of data, stop
            if train_data is None or self.train_data_shuffler.data_finished:
#                print("None data, exiting the thread")
                self.train_thread_pool.request_stop()
#                self.train_thread_pool.join(self.train_threads)
                return

            [train_placeholder_data, train_placeholder_labels] = self.train_data_shuffler.get_placeholders()

            feed_dict = {train_placeholder_data: train_data,
                         train_placeholder_labels: train_labels}

            self.session.run(self.enqueue_op_train, feed_dict=feed_dict)

    def load_and_enqueue_valid(self):
        """
        Injecting data in the place holder queue

        **Parameters**
          session: Tensorflow session
        """
        if self.validation_data_shuffler is None:
            return

        while not self.valid_thread_pool.should_stop():
            with self.threads_lock_valid:
                [valid_data, valid_labels] = self.validation_data_shuffler.get_batch()

            # if we run out of data, stop
            if valid_data is None or self.validation_data_shuffler.data_finished:
#                print("None validation data, exiting the thread")
                self.valid_thread_pool.request_stop()
#                self.valid_thread_pool.join(self.valid_threads)
                return

            [valid_placeholder_data, valid_placeholder_labels] = self.validation_data_shuffler.get_placeholders()

            feed_dict = {valid_placeholder_data: valid_data,
                         valid_placeholder_labels: valid_labels}

            self.session.run(self.enqueue_op_valid, feed_dict=feed_dict)

    def bootstrap_graphs(self, train_data_shuffler, validation_data_shuffler):
        """
        Create all the necessary graphs for training, validation and inference graphs
        """

        # Creating train graph
        self.training_graph = self.compute_graph(train_data_shuffler, prefetch=self.prefetch, name="train")
        tf.add_to_collection("training_graph", self.training_graph)

        # Creating inference graph
        self.architecture.compute_inference_placeholder(train_data_shuffler.deployment_shape)
        self.architecture.compute_inference_graph()
        tf.add_to_collection("inference_placeholder", self.architecture.inference_placeholder)
        tf.add_to_collection("inference_graph", self.architecture.inference_graph)

        # Creating validation graph
        if validation_data_shuffler is not None:
            self.validation_graph = self.compute_graph(validation_data_shuffler, prefetch=self.prefetch,
                                                       name="validation", training=False)
            tf.add_to_collection("validation_graph", self.validation_graph)

        self.bootstrap_placeholders(train_data_shuffler, validation_data_shuffler)

    def bootstrap_placeholders(self, train_data_shuffler, validation_data_shuffler):
        """
        Persist the placeholders

         ** Parameters **
           train_data_shuffler: Data shuffler for training
           validation_data_shuffler: Data shuffler for validation

        """

        # Persisting the placeholders
        if self.prefetch:
            batch, label = train_data_shuffler.get_placeholders_forprefetch("train")
        else:
            batch, label = train_data_shuffler.get_placeholders()

        tf.add_to_collection("train_placeholder_data", batch)
        tf.add_to_collection("train_placeholder_label", label)

        # Creating validation graph
        if validation_data_shuffler is not None:
            if self.prefetch:
                batch, label = validation_data_shuffler.get_placeholders_forprefetch("validation")
            else:
                batch, label = validation_data_shuffler.get_placeholders()
            tf.add_to_collection("validation_placeholder_data", batch)
            tf.add_to_collection("validation_placeholder_label", label)

    def bootstrap_graphs_fromhdf5file(self, train_data_shuffler, validation_data_shuffler):

        self.bootstrap_graphs(train_data_shuffler, validation_data_shuffler)

        # TODO: find an elegant way to provide this as a parameter of the trainer
        self.global_epoch = tf.Variable(0, trainable=False, name="global_epoch")
        tf.add_to_collection("global_epoch", self.global_epoch)

        # Preparing the optimizer
        self.optimizer_class._learning_rate = self.learning_rate
        self.optimizer = self.optimizer_class.minimize(self.training_graph, global_step=self.global_epoch)
        tf.add_to_collection("optimizer", self.optimizer)
        tf.add_to_collection("learning_rate", self.learning_rate)

        # Train summary
        self.summaries_train = self.create_general_summary()
        tf.add_to_collection("summaries_train", self.summaries_train)

        tf.add_to_collection("summaries_train", self.summaries_train)

        tf.initialize_all_variables().run(session=self.session)

        # Original tensorflow saver object
        saver = tf.train.Saver(var_list=tf.all_variables())

        self.architecture.load_hdf5(self.model_from_file, shape=[1, 6560, 1])
        # fname, _ = os.path.splitext(self.model_from_file)
        # self.model_from_file = fname + '.ckp'
        # self.architecture.save(saver, self.model_from_file)
        return saver

    def bootstrap_graphs_fromfile(self, train_data_shuffler, validation_data_shuffler):
        """
        Bootstrap all the necessary data from file

         ** Parameters **
           session: Tensorflow session
           train_data_shuffler: Data shuffler for training
           validation_data_shuffler: Data shuffler for validation


        """
        saver = self.architecture.load(self.model_from_file, clear_devices=False)

        # Loading training graph
        self.training_graph = tf.get_collection("training_graph")[0]

        # Loding other elements
        self.optimizer = tf.get_collection("optimizer")[0]
        self.learning_rate = tf.get_collection("learning_rate")[0]
        self.summaries_train = tf.get_collection("summaries_train")[0]
        self.global_epoch = tf.get_collection("global_epoch")[0]

        if validation_data_shuffler is not None:
            self.validation_graph = tf.get_collection("validation_graph")[0]

        self.bootstrap_placeholders_fromfile(train_data_shuffler, validation_data_shuffler)

        return saver

    def bootstrap_placeholders_fromfile(self, train_data_shuffler, validation_data_shuffler):
        """
        Load placeholders from file

         ** Parameters **

           train_data_shuffler: Data shuffler for training
           validation_data_shuffler: Data shuffler for validation

        """

        train_data_shuffler.set_placeholders(tf.get_collection("train_placeholder_data")[0],
                                             tf.get_collection("train_placeholder_label")[0])

        if validation_data_shuffler is not None:
            train_data_shuffler.set_placeholders(tf.get_collection("validation_placeholder_data")[0],
                                                 tf.get_collection("validation_placeholder_label")[0])

    def launch_train_threads(self):
        self.train_thread_pool = tf.train.Coordinator()
        tf.train.start_queue_runners(coord=self.train_thread_pool, sess=self.session)
        self.train_threads = self.start_thread()

    def launch_valid_threads(self):
        self.valid_thread_pool = tf.train.Coordinator()
        tf.train.start_queue_runners(coord=self.valid_thread_pool, sess=self.session)
        self.valid_threads = self.start_thread(training=False)

    def train(self, train_data_shuffler, validation_data_shuffler=None):
        """
        Train the network:

         ** Parameters **

           train_data_shuffler: Data shuffler for training
           validation_data_shuffler: Data shuffler for validation
        """

        # Creating directory
        bob.io.base.create_directories_safe(self.temp_dir)
        self.train_data_shuffler = train_data_shuffler
        self.validation_data_shuffler = validation_data_shuffler

        logger.info("Initializing !!")

        # Pickle the architecture to save
        self.architecture.pickle_net(train_data_shuffler.deployment_shape)

        Session.create()
        self.session = Session.instance().session

        # Loading a pretrained model
        if self.model_from_file != "":
            logger.info("Loading pretrained model from {0}".format(self.model_from_file))
            if self.model_from_file.lower().endswith('.hdf5'):
                saver = self.bootstrap_graphs_fromhdf5file(train_data_shuffler, validation_data_shuffler)
            elif self.model_from_file.lower().endswith('.ckp'):
                saver = self.bootstrap_graphs_fromfile(train_data_shuffler, validation_data_shuffler)
            else:
                raise ValueError("Unknown format of the model %s. Only HDF5 or pickled formats are supported"
                                 % self.model_from_file)

            epoch = self.global_epoch.eval(session=self.session)

        else:
            epoch = 0
            # Bootstraping all the graphs
            self.bootstrap_graphs(train_data_shuffler, validation_data_shuffler)

            # TODO: find an elegant way to provide this as a parameter of the trainer
            self.global_epoch = tf.Variable(0, trainable=False, name="global_epoch")
            tf.add_to_collection("global_epoch", self.global_epoch)

            # Preparing the optimizer
            self.optimizer_class._learning_rate = self.learning_rate
            self.optimizer = self.optimizer_class.minimize(self.training_graph, global_step=self.global_epoch)
            tf.add_to_collection("optimizer", self.optimizer)
            tf.add_to_collection("learning_rate", self.learning_rate)

            # Train summary
            self.summaries_train = self.create_general_summary()
            tf.add_to_collection("summaries_train", self.summaries_train)

            tf.add_to_collection("summaries_train", self.summaries_train)

            tf.initialize_all_variables().run(session=self.session)

            # Original tensorflow saver object
            saver = tf.train.Saver(var_list=tf.all_variables())

        if isinstance(train_data_shuffler, OnLineSampling):
            train_data_shuffler.set_feature_extractor(self.architecture, session=self.session)

        self.architecture.save(saver, os.path.join(self.temp_dir, 'model_initial.ckp'))
        with self.session.as_default():
            path = os.path.join(self.temp_dir, 'model_initial.hdf5')
            self.architecture.save_hdf5(bob.io.base.HDF5File(path, 'w'))

        # TENSOR BOARD SUMMARY
        self.train_summary_writter = tf.train.SummaryWriter(os.path.join(self.temp_dir, 'train'), self.session.graph)
        start = time.time()
        total_train_data = 0
        total_valid_data = 0
        for epoch in range(epoch, self.epochs):

            batch_num = 0
            total_train_loss = 0
            logger.info("\nTRAINING EPOCH {0}".format(epoch))
            self.train_data_shuffler.data_finished = False

            # Start a thread to enqueue data asynchronously, and hide I/O latency.
            if self.prefetch:
                self.launch_train_threads()

            while True:
                # start = time.time()
                cur_loss, summary = self.fit()
                # end = time.time()
                # logger.info("Fit time = {0}".format(float(end - start)))
                # we are done when we went through the whole data
                if cur_loss is None or self.train_data_shuffler.data_finished:
                    break

                batch_num += 1
                total_train_loss += cur_loss

                # Reporting loss for each snapshot
                if batch_num % self.snapshot == 0:
                    logger.info("Loss training set, epoch={0}, batch_num={1} = {2}".format(
                        epoch, batch_num, total_train_loss/batch_num))
                    self.train_summary_writter.add_summary(summary, epoch*total_train_data+batch_num)
                    end = time.time()
                    logger.info("Training Batch = {0}, time = {1}".format(batch_num, float(end - start)))
                    summary = summary_pb2.Summary.Value(tag="elapsed_time", simple_value=float(end - start))
                    self.train_summary_writter.add_summary(
                        summary_pb2.Summary(value=[summary]), epoch*total_train_data+batch_num)
                    path = os.path.join(self.temp_dir, 'model_epoch{0}_batch{1}.ckp'.format(epoch, batch_num))
                    self.architecture.save(saver, path)
                    with self.session.as_default():
                        path = os.path.join(self.temp_dir, 'model_epoch{0}_batch{1}.hdf5'.format(epoch, batch_num))
                        self.architecture.save_hdf5(bob.io.base.HDF5File(path, 'w'))
                    start = time.time()

            total_train_data = batch_num
            logger.info("Number of training batches={0}".format(total_train_data))
            logger.info("Taking snapshot for epoch %d", epoch)
            if total_train_data:
                logger.info("Loss total TRAINING for epoch={0} = {1}".format(
                    epoch, total_train_loss / total_train_data))
            path = os.path.join(self.temp_dir, 'model_epoch{0}.ckp'.format(epoch))
            self.architecture.save(saver, path)
            with self.session.as_default():
                path = os.path.join(self.temp_dir, 'model_epoch{0}.hdf5'.format(epoch))
                self.architecture.save_hdf5(bob.io.base.HDF5File(path, 'w'))


            # Running validation for the current epoch
            if self.validation_data_shuffler is not None:
                batch_num = 0
                total_valid_loss = 0
                total_prediction_err = 0
                start = time.time()
                logger.info("\nVALIDATION EPOCH {0}".format(epoch))
                self.validation_data_shuffler.data_finished = False

                # Start a thread to enqueue data asynchronously, and hide I/O latency.
                if self.prefetch:
                    self.launch_valid_threads()

                while True:
                    prediction_err, cur_loss = self.compute_validation(self.validation_data_shuffler)
                    # we are done when we went through the whole data
                    if cur_loss is None or self.validation_data_shuffler.data_finished:
                        break

                    batch_num += 1
                    total_valid_loss += cur_loss
                    total_prediction_err += numpy.mean(numpy.array(prediction_err))

                    if self.validation_summary_writter is None:
                        self.validation_summary_writter = tf.train.SummaryWriter(
                            os.path.join(self.temp_dir, 'validation'), self.session.graph)
                    if batch_num % self.validation_snapshot == 0:
                        summaries = [summary_pb2.Summary.Value(tag="loss", simple_value=float(total_valid_loss/batch_num))]
                        self.validation_summary_writter.add_summary(
                            summary_pb2.Summary(value=summaries), epoch*total_valid_data+batch_num)
                        logger.info("Loss validation batch={0} = {1}".format(
                            batch_num, total_valid_loss/batch_num))
                        end = time.time()
                        logger.info("Validation Batch = {0}, time = {1}".format(batch_num, float(end - start)))

                        summaries = [summary_pb2.Summary.Value(tag="Error", simple_value=float(total_prediction_err / batch_num))]
                        self.validation_summary_writter.add_summary(
                            summary_pb2.Summary(value=summaries), epoch * total_valid_data + batch_num)
                        logger.info("Error validation batch={0} = {1}".format(
                            batch_num, total_prediction_err / batch_num))
                        start = time.time()

                total_valid_data = batch_num
                logger.info("Total number of validation batches={0}".format(total_valid_data))
                if total_valid_data:
                    logger.info("Loss total VALIDATION for epoch={0} = {1}".format(
                        epoch, total_valid_loss / total_valid_data))
                    logger.info("Error total VALIDATION for epoch={0} = {1}".format(
                        epoch, total_prediction_err / total_valid_data))

        logger.info("Training finally finished")

        self.train_summary_writter.close()
        if self.validation_data_shuffler is not None:
            self.validation_summary_writter.close()

        # Saving the final network
        path = os.path.join(self.temp_dir, 'model.ckp')
        self.architecture.save(saver, path)
        with self.session.as_default():
            path = os.path.join(self.temp_dir, 'model.hdf5')
            self.architecture.save_hdf5(bob.io.base.HDF5File(path, 'w'))

        if self.prefetch:
            # now they should definetely stop
            self.train_thread_pool.request_stop()
            self.valid_thread_pool.request_stop()
