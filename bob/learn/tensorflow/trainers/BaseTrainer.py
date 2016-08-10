#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# @date: Tue 09 Aug 2016 15:25:22 CEST

import logging
logger = logging.getLogger("bob.learn.tensorflow")
from ..DataShuffler import DataShuffler


class BaseTrainer(object):

    def __init__(self,
                 data=None,
                 labels=None,
                 network=None,
                 width=28,
                 height=28,
                 channels=1,
                 use_gpu=False,

                 data_shuffler=DataShuffler(),
                 optimization=tf.train.MomentumOptimizer(0.001, momentum=0.99, use_locking=False, name='Momentum'),
                 loss=None,

                 ###### training options ##########
                 convergence_threshold = 0.01,
                 percentage_train=0.5,
                 iterations=5000,
                 train_batch_size=1,
                 validation_batch_size=1000,
                 base_lr=0.00001,
                 momentum=0.9,
                 weight_decay=0.0005,

                 # The learning rate policy
                 lr_policy="inv",
                 gamma=0.0001,
                 power=0.75,
                 snapshot=5000):

        self.data = data
        self.labels = labels
        self.data_shuffler = DataShuffler(self.data, self.labels)
        self.width = width
        self.height = height
        self.channels = channels

        self.network = network
        self.use_gpu = use_gpu


        #TODO: PREPARE THE CONSTRUCTOR FOR THAT
        self.caffe_model_path = None
        self.deploy_architecture_path = None
        self.net = None

        # TODO: Parametrize this variable
        self.single_batch = False

        self.n_labels = 0

        self.train_batch_size = train_batch_size
        self.validation_batch_size = validation_batch_size
        self.iterations = iterations
        self.snapshot = snapshot
        self.solver_class = solver
        self.solver = None
        self.base_lr = base_lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.percentage_train = percentage_train
        self.convergence_threshold = convergence_threshold

        # The learning rate policy
        self.lr_policy = lr_policy
        self.gamma = gamma
        self.power = power


        # Shape of the data in the format [c, w, h] --> Channels, Width, Height
        self.data_shape = None

    def __setup_train(self):
        # Loading data

        # Defining place holders for train and validation
        train_data_node = tf.placeholder(tf.float32, shape=(self.train_batch_size, self.width, self.height, self.channels))
        train_labels_node = tf.placeholder(tf.int64, shape=self.train_batch_size)

        # Creating the architecture for train and validation
        architecture = self.network(use_gpu=self.use_gpu)

        train_graph = architecture.create_lenet(train_data_node)



    def __call__(self, image):
        """
        Forward the CNN uising the image.

        **Parameters:**

          image:

        """

        if self.use_gpu:
            caffe.set_device(0)
            caffe.set_mode_gpu()

        # Dealing with RGB or gray images
        if len(image.shape) == 2:
            image = numpy.reshape(image, (1, 1, image.shape[0], image.shape[1]))
        else:
            image = numpy.reshape(image, (1, image.shape[0], image.shape[1], image.shape[2]))

        # Computes features
        norm_image = bob.bio.caffe.backend.utils.scale_dummy(image, self.backend_type.mu)

        feature = self.net.forward(data=norm_image, end=self.network.end_cnn)[self.network.end_cnn]
        feature = feature.reshape((feature.shape[1])).astype("float64")
        return feature

    def load(self, extractor_file):
        """
        Load the neural net
        """

        if self.caffe_model_path is None:
            backend_path = BaseTrainer.get_backend_path(extractor_file)
            self.caffe_model_path = BaseTrainer.get_default_model_path(backend_path)
            self.deploy_architecture_path = BaseTrainer.get_default_deploy_architecture_path(backend_path)

        self.net = caffe.Net(self.deploy_architecture_path, self.caffe_model_path, caffe.TEST)

        # Loading the scale factor
        hdf5_file = bob.io.base.HDF5File(extractor_file)
        if type(hdf5_file.read("mu")) is numpy.float64:
            self.backend_type.mu = numpy.array([hdf5_file.read("mu")])
        else:
            self.backend_type.mu = hdf5_file.read("mu")

    def create_architecture(self, backend_file, n_labels, batch_size, data_shape):
        """
        Create the base architecture for the trainer

         **Parameters**
           n_labels: Number of labels

         **Returns**
           Will return the network specs

        """

        return self.network.create_architecture(backend_file, self.backend_type, n_labels,
                                               batch_size=batch_size,
                                               data_shape=data_shape)

    def define_single_train_validation(self, train_data):
        """
        Split the input data, organized by client, into train and validation set

         **Parameters**

           train_data: The data for training, here we expect the data organized by client

           percentage_train:

        """

        training_samples = 0
        test_samples = 0

        # Computing the number of clients for pre allocation
        for d in train_data:
            n_samples = len(d)
            client_training_samples = int(numpy.floor(n_samples * self.percentage_train))
            training_samples += client_training_samples
            test_samples += n_samples - client_training_samples

        feature_dimension = train_data[0][0].shape
        feature_dimension = list(feature_dimension)
        # One channel
        if len(feature_dimension) == 2:
            feature_dimension.insert(0, 1)

        train_feature_dimension = feature_dimension[:]
        train_feature_dimension.insert(0, training_samples)
        train_feature_dimension = tuple(train_feature_dimension)
        test_feature_dimension = feature_dimension[:]
        test_feature_dimension.insert(0, test_samples)
        test_feature_dimension = tuple(test_feature_dimension)

        # Preallocations
        train_data_caffe = numpy.zeros(shape=train_feature_dimension)
        train_label = numpy.ones((training_samples, 1))

        test_data_caffe = numpy.zeros(shape=test_feature_dimension)
        test_label = numpy.ones((test_samples, 1))

        # Filling up the data
        train_index = 0
        test_index = 0

        for i in range(len(train_data)):
            client_data = numpy.array(train_data[i])
            n_samples = len(client_data)
            train_offset = int(numpy.floor(n_samples * self.percentage_train))
            test_offset = n_samples - train_offset

            # Splinting train and test
            if len(client_data.shape) == 3:
                client_data = numpy.expand_dims(client_data, axis=1)

            train_data_temp = client_data[0:train_offset, :, :, :]
            test_data_temp = client_data[train_offset:train_offset + test_offset:, :, :]

            train_data_caffe[train_index: train_index + train_offset] = train_data_temp
            train_label[train_index: train_index + train_offset] = numpy.ones(shape=(train_offset, 1)) * i

            test_data_caffe[test_index: test_index + test_offset] = test_data_temp
            test_label[test_index: test_index + test_offset] = numpy.ones(shape=(test_offset, 1)) * i

            train_index += train_offset
            test_index += test_offset

        return train_data_caffe, train_label, test_data_caffe, test_label

    def create_backend(self, train_data, extractor_file):
        backend_path = BaseTrainer.get_backend_path(extractor_file)
        bob.io.base.create_directories_safe(backend_path)

        # Spliting between in train and validation
        train_data_caffe, train_label, validation_data_caffe, validation_label = self.define_single_train_validation\
            (train_data)

        # Persisting using some backend
        train_backend_file, validation_backend_file = self.backend_type(backend_path, train_data_caffe, train_label,
                                                                        validation_data_caffe, validation_label)

        # Defining the shape of the data
        self.data_shape = train_data_caffe.shape[1:]
        return train_backend_file, validation_backend_file

    def save_stats(self, extractor_file, it, snapshot, loss, eer):
        # Saving statistics
        hdf5 = bob.io.base.HDF5File(extractor_file, "w")
        hdf5.set("iterations", it)
        hdf5.set("snapshot", snapshot)
        #hdf5.set("validationloss", loss)
        hdf5.set("eer", eer)
        hdf5.set("mu", self.backend_type.mu)

        del hdf5

    def train_loop(self, extractor_file, analizer_architecture_file):
        """
        Do the loop forward --> backward --|
                      ^--------------------|
        """
        previous_eer = 1.
        for i in range(self.iterations):

            if isinstance(self.backend_type, bob.bio.caffe.backend.MemoryBackend):
                data_train, labels_train = self.backend_type.get_batch(self.train_batch_size)
                self.solver.net.set_input_arrays(data_train, labels_train, "data")

                #data_validation, labels_validation = self.backend_type.get_batch(self.validation_batch_size,
                #                                                                 is_target_set_train=False)
                #self.solver.test_nets[0].set_input_arrays(data_validation, labels_validation, "data")

            self.solver.step(1)

            if i % self.snapshot == 0:

                temp_model = os.path.join(BaseTrainer.get_backend_path(extractor_file), "model-it_{0}.model".format(i))
                self.solver.net.save(temp_model)

                # Setting up the analiser
                analizer = Analizer(temp_model, analizer_architecture_file, self.n_labels)

                # Loading data
                data_train, labels_train = self.backend_type.get_batch(self.validation_batch_size)
                data_validation, labels_validation = self.backend_type.get_batch(self.validation_batch_size,
                                                                                 is_target_set_train=True)
                # Analizing
                self.eer.append(analizer(data_train, labels_train, data_validation, labels_validation))
                self.validation_loss.append(self.solver.test_nets[0].blobs['loss'].data)
                if abs(self.eer[-1] - previous_eer) < self.convergence_threshold:
                    # Saving statistics
                    self.save_stats(extractor_file, i, self.snapshot,
                                    self.validation_loss, self.eer)

                    temp_model = BaseTrainer.get_default_model_path(extractor_file)
                    self.solver.net.save(temp_model)

                    break
                else:
                    previous_eer = self.eer[-1]

                # Saving statistics
                self.save_stats(extractor_file, i, self.snapshot, self.validation_loss, self.eer)

    def __setup_train(self, train_data, extractor_file):

        # Preparing the backend
        backend_path = BaseTrainer.get_backend_path(extractor_file)
        self.n_labels = len(train_data)
        logger.info(
            "Generating the Caffe backend in '{0}' for a {1} classes classification".format(backend_path, self.n_labels))
        train_backend_file, validation_backend_file = self.create_backend(train_data, extractor_file)

        # Setting up the architecture locations
        train_architecture_file = os.path.join(backend_path, "train_architecture.prototxt")
        #validation_architecture_file = os.path.join(backend_path, "validation_architecture.prototxt")
        deploy_architecture_file = os.path.join(backend_path, "deploy_architecture.prototxt")
        analizer_architecture_file = os.path.join(backend_path, "analizer_architecture.prototxt")

        #Creating the NET specs
        """
        train_net = self.create_architecture(train_backend_file, self.n_labels, self.train_batch_size, self.data_shape)

        #validation_net = self.create_architecture(validation_backend_file, self.n_labels, self.train_batch_size, self.data_shape)

        deploy_net = self.network.create_architecture(validation_backend_file, self.backend_type, self.n_labels,
                                                      batch_size=1,
                                                      data_shape=self.data_shape, deploy=True)
        # Spec used for the analized (n, c, w, h)
        if self.single_batch:
            analizer_net = self.network.create_architecture(validation_backend_file, self.backend_type, self.n_labels,
                                                            batch_size=self.validation_batch_size,
                                                            data_shape=self.data_shape, deploy=True)
        else:
            analizer_net = self.network.create_architecture(validation_backend_file, self.backend_type, self.n_labels,
                                                            batch_size=1, data_shape=self.data_shape, deploy=True)
        # Saving the specs
        open(train_architecture_file, 'w').write(str(train_net))
        #open(validation_architecture_file, 'w').write(str(validation_net))
        open(deploy_architecture_file, 'w').write(str(deploy_net))
        open(analizer_architecture_file, 'w').write(str(analizer_net))
        """
        if self.use_gpu:
            caffe.set_device(0)
            caffe.set_mode_gpu()

        # Defining the solver file
        from caffe.proto import caffe_pb2
        from google.protobuf import text_format

        solver_config = caffe_pb2.SolverParameter()
        solver_config.train_net = train_architecture_file
        #solver_config.test_net.append(validation_architecture_file)

        ##################################
        #TODO: Parametrize this
        ##################################
        #solver_config.test_iter.append(1)
        #solver_config.test_interval = self.snapshot

        # Training options
        solver_config.base_lr = self.base_lr
        solver_config.momentum = self.momentum
        solver_config.weight_decay = self.weight_decay

        solver_config.lr_policy = self.lr_policy
        solver_config.gamma = self.gamma
        solver_config.power = self.power

        solver_config.snapshot = 10000000
        solver_config.snapshot_prefix = backend_path

        solver_path = os.path.join(backend_path, "solver.prototxt")
        open(solver_path, 'w').write(text_format.MessageToString(solver_config))

        self.solver = self.solver_class(solver_path)

        return analizer_architecture_file

    def train(self, train_data, extractor_file):
        """
        Train the NN using the solver set in self.solver

        **Parameters**

           train_architecture_file:
           validation_architecture_file:
        """

        # Prepare the CNN
        analizer_architecture_file = self.__setup_train(train_data, extractor_file)

        # Do the loop for
        self.train_loop(extractor_file, analizer_architecture_file)

