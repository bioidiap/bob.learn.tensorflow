#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# @date: Thu 11 Aug 2016 09:39:36 CEST


import tensorflow as tf
import abc
import six
import numpy
import pickle

from collections import OrderedDict
from bob.learn.tensorflow.layers import Layer, MaxPooling, Dropout, Conv2D, FullyConnected
from bob.learn.tensorflow.utils.session import Session


class SequenceNetwork(six.with_metaclass(abc.ABCMeta, object)):
    """
    Sequential model is a linear stack of :py:mod:`bob.learn.tensorflow.layers`.

    **Parameters**

        default_feature_layer: Default layer name (:py:obj:`str`) used as a feature layer.

        use_gpu: If ``True`` uses the GPU in the computation.
    """

    def __init__(self,
                 default_feature_layer=None,
                 use_gpu=False):

        self.sequence_net = OrderedDict()
        self.default_feature_layer = default_feature_layer
        self.input_divide = 1.
        self.input_subtract = 0.
        self.use_gpu = use_gpu

        self.pickle_architecture = None# The trainer triggers this
        self.deployment_shape = None# The trainer triggers this

        # Inference graph
        self.inference_graph = None
        self.inference_placeholder = None

    def __del__(self):
        tf.reset_default_graph()

    def add(self, layer):
        """
        Add a :py:class:`bob.learn.tensorflow.layers.Layer` in the sequence network

        """
        if not isinstance(layer, Layer):
            raise ValueError("Input `layer` must be an instance of `bob.learn.tensorflow.layers.Layer`")
        self.sequence_net[layer.name] = layer

    def pickle_net(self, shape):
        """
        Pickle the class

         ** Parameters **
          shape: Shape of the input data for deployment
        """
        self.pickle_architecture = pickle.dumps(self.sequence_net)
        self.deployment_shape = shape

    def compute_graph(self, input_data, feature_layer=None, training=True):
        """Given the current network, return the Tensorflow graph

         **Parameter**

          input_data: tensorflow placeholder as input data

          feature_layer: Name of the :py:class:`bob.learn.tensorflow.layers.Layer` that you want to "cut".
                         If `None` will run the graph until the end.

          training: If `True` will generating the graph for training
        """

        input_offset = input_data
        for k in self.sequence_net.keys():
            current_layer = self.sequence_net[k]

            if training or not isinstance(current_layer, Dropout):
                current_layer.create_variables(input_offset)
                input_offset = current_layer.get_graph(training_phase=training)

                if feature_layer is not None and k == feature_layer:
                    return input_offset

        return input_offset

    def compute_inference_graph(self, feature_layer=None):
        """Generate a graph for feature extraction

        **Parameters**

        placeholder: tensorflow placeholder as input data
        """
        if feature_layer is None:
            feature_layer = self.default_feature_layer

        self.inference_graph = self.compute_graph(self.inference_placeholder, feature_layer, training=False)

    def compute_inference_placeholder(self, data_shape):
        self.inference_placeholder = tf.placeholder(tf.float32, shape=data_shape, name="feature")

    def __call__(self, data, feature_layer=None):
        """Run a graph and compute the embeddings

        **Parameters**

        data: tensorflow placeholder as input data

        session: tensorflow `session <https://www.tensorflow.org/versions/r0.11/api_docs/python/client.html#Session>`_

        feature_layer: Name of the :py:class:`bob.learn.tensorflow.layer.Layer` that you want to "cut".
                       If `None` will run the graph until the end.
        """

        session = Session.instance().session

        # Feeding the placeholder
        if self.inference_placeholder is None:
            self.compute_inference_placeholder(data.shape[1:])
        feed_dict = {self.inference_placeholder: data}

        if self.inference_graph is None:
            self.compute_inference_graph(self.inference_placeholder, feature_layer)

        embedding = session.run([self.inference_graph], feed_dict=feed_dict)[0]

        return embedding

    def predict(self, data):
        return numpy.argmax(self(data), 1)

    def dump_variables(self):
        """
        Return all the tensorflow `variables <https://www.tensorflow.org/versions/r0.11/api_docs/python/state_ops.html#Variable>`_ used in the graph
        """
        variables = {}
        for k in self.sequence_net:
            # TODO: IT IS NOT SMART TESTING ALONG THIS PAGE
            if not isinstance(self.sequence_net[k], MaxPooling) and not isinstance(self.sequence_net[k], Dropout):
                variables[self.sequence_net[k].W.name] = self.sequence_net[k].W
                variables[self.sequence_net[k].b.name] = self.sequence_net[k].b

                # Dumping batch norm variables
                if self.sequence_net[k].batch_norm:
                    variables[self.sequence_net[k].beta.name] = self.sequence_net[k].beta
                    variables[self.sequence_net[k].gamma.name] = self.sequence_net[k].gamma
                    #variables[self.sequence_net[k].mean.name] = self.sequence_net[k].mean
                    #variables[self.sequence_net[k].var.name] = self.sequence_net[k].var

        return variables

    def variable_summaries(self, var, name):
        """Attach a lot of summaries to a Tensor."""
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean/' + name, mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
            tf.summary.scalar('sttdev/' + name, stddev)
            tf.summary.scalar('max/' + name, tf.reduce_max(var))
            tf.summary.scalar('min/' + name, tf.reduce_min(var))
            tf.summary.histogram(name, var)

    def generate_summaries(self):
        for k in self.sequence_net.keys():
            current_layer = self.sequence_net[k]

            if not isinstance(self.sequence_net[k], MaxPooling) and not isinstance(self.sequence_net[k], Dropout):
                self.variable_summaries(current_layer.W, current_layer.name + '/weights')
                self.variable_summaries(current_layer.b, current_layer.name + '/bias')

    def compute_magic_number(self, hypothetic_image_dimensions=(28, 28, 1)):
        """

        Here it is done an estimative of the capacity of DNN.

        **Parameters**

            hypothetic_image_dimensions: Possible image dimentions `w, h, c` (width x height x channels)
        """

        stride = 1# ALWAYS EQUALS TO ONE
        current_image_dimensions = list(hypothetic_image_dimensions)

        samples_per_sample = 0
        flatten_dimension = numpy.prod(current_image_dimensions)
        for k in self.sequence_net.keys():
            current_layer = self.sequence_net[k]

            if isinstance(current_layer, Conv2D):
                #samples_per_sample += current_layer.filters * current_layer.kernel_size * current_image_dimensions[0] + current_layer.filters
                #samples_per_sample += current_layer.filters * current_layer.kernel_size * current_image_dimensions[1] + current_layer.filters
                samples_per_sample += current_layer.filters * current_image_dimensions[0] * current_image_dimensions[1] + current_layer.filters

                current_image_dimensions[2] = current_layer.filters
                flatten_dimension = numpy.prod(current_image_dimensions)

            if isinstance(current_layer, MaxPooling):
                current_image_dimensions[0] /= 2
                current_image_dimensions[1] /= 2
                flatten_dimension = current_image_dimensions[0] * current_image_dimensions[1] * current_image_dimensions[2]

            if isinstance(current_layer, FullyConnected):
                samples_per_sample += flatten_dimension * current_layer.output_dim + current_layer.output_dim
                flatten_dimension = current_layer.output_dim

        return samples_per_sample

    def save_hdf5(self, hdf5):
        """
        Save the state of the network in HDF5 format

        **Parameters**

            hdf5: An opened :py:class:`bob.io.base.HDF5File`

            step: The current training step. If not `None`, will create a HDF5 group with the current step.

        """

        def create_groups(path):
            split_path = path.split("/")
            for i in range(0, len(split_path)-1):
                p = split_path[i]
                if not hdf5.has_group(p):
                    hdf5.create_group(p)

        # Saving the architecture
        if self.pickle_architecture is not None:
            hdf5.set('architecture', self.pickle_architecture)
            hdf5.set('deployment_shape', self.deployment_shape)

        # Directory that stores the tensorflow variables
        hdf5.create_group('/tensor_flow')
        hdf5.cd('/tensor_flow')

        # Iterating the variables of the model
        for v in self.dump_variables().keys():
            create_groups(v)
            hdf5.set(v, self.dump_variables()[v].eval())

        hdf5.cd('..')
        hdf5.set('input_divide', self.input_divide)
        hdf5.set('input_subtract', self.input_subtract)

    def turn_gpu_onoff(self, state=True):
        for k in self.sequence_net:
            self.sequence_net[k].weights_initialization.use_gpu = state
            self.sequence_net[k].bias_initialization.use_gpu = state

    def load_variables_only(self, hdf5):
        """
        Load the variables of the model
        """

        session = Session.instance().session

        hdf5.cd('/tensor_flow')
        for k in self.sequence_net:
            # TODO: IT IS NOT SMART TESTING ALONG THIS PAGE
            if not isinstance(self.sequence_net[k], MaxPooling):
                self.sequence_net[k].W.assign(hdf5.read(self.sequence_net[k].W.name)).eval(session=session)
                session.run(self.sequence_net[k].W)
                self.sequence_net[k].b.assign(hdf5.read(self.sequence_net[k].b.name)).eval(session=session)
                session.run(self.sequence_net[k].b)

                if self.sequence_net[k].batch_norm:
                    self.sequence_net[k].beta.assign(hdf5.read(self.sequence_net[k].beta.name)).eval(session=session)
                    self.sequence_net[k].gamma.assign(hdf5.read(self.sequence_net[k].gamma.name)).eval(session=session)

        hdf5.cd("..")

    def load_hdf5(self, hdf5, shape=None, batch=1, use_gpu=False):
        """
        Load the network from scratch.
        This will build the graphs

        **Parameters**

            hdf5: The saved network in the :py:class:`bob.io.base.HDF5File` format
            shape: Input shape of the network. If `None`, the internal shape will be assumed
            session: An opened tensorflow `session <https://www.tensorflow.org/versions/r0.11/api_docs/python/client.html#Session>`_. If `None`, a new one will be opened
            batch: The size of the batch
            use_gpu: Load all the variables in the GPU?
        """

        session = Session.instance().session

        # Loading the normalization parameters
        self.input_divide = hdf5.read('input_divide')
        self.input_subtract = hdf5.read('input_subtract')

        # Loading architecture
        self.sequence_net = pickle.loads(hdf5.read('architecture'))
        self.deployment_shape = hdf5.read('deployment_shape')

        self.turn_gpu_onoff(use_gpu)

        if shape is None:
            shape = self.deployment_shape
            shape[0] = batch

        # Loading variables
        place_holder = tf.placeholder(tf.float32, shape=shape, name="load")
        self.compute_graph(place_holder)
        tf.global_variables_initializer().run(session=session)
        self.load_variables_only(hdf5, session)

    def save(self, saver, path):

        session = Session.instance().session

        open(path+"_sequence_net.pickle", 'wb').write(self.pickle_architecture)
        return saver.save(session, path)

    def load(self, path, clear_devices=False, session_from_scratch=False):

        session = Session.instance(new=session_from_scratch).session

        self.sequence_net = pickle.loads(open(path+"_sequence_net.pickle", 'rb').read())
        if clear_devices:
            saver = tf.train.import_meta_graph(path + ".meta", clear_devices=clear_devices)
        else:
            saver = tf.train.import_meta_graph(path + ".meta")

        saver.restore(session, path)
        self.inference_graph = tf.get_collection("inference_graph")[0]
        self.inference_placeholder = tf.get_collection("inference_placeholder")[0]

        return saver
