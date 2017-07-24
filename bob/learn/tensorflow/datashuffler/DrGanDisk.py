#!/usr/bin/env python
# encoding: utf-8

import numpy
import bob.io.base
import bob.io.image

from .Base import Base

import tensorflow as tf

logger = bob.core.log.setup("bob.learn.tensorflow")


from bob.learn.tensorflow.datashuffler.Normalizer import MinusOneOne


class DrGanDisk(Base):

  """
  This class implements a datashuffler to deal with DR-GAN,
  
  **Parameters**

  data:
    Input data

  id_labels:
    id labels of the retrieved faces. 

  pose_labels:
    pose labels of the retrieved faces. 

  input_shape:
    The shape of the inputs

  input_dtype:
    Type of the input

  batch_size:
    Batch size

  seed:
    The seed of the random generator

  normalizer:
    Algorithm used for feature scaling.

  """

  def __init__(self, data, id_labels, pose_labels,
               input_shape,
               input_dtype = "float64",
               batch_size = 128,
               seed = 1523,
               data_augmentation=None,
               normalizer=MinusOneOne(),
               ):

    if isinstance(data, list):
      data = numpy.array(data)

    if isinstance(id_labels, list):
      id_labels = numpy.array(id_labels)

    if isinstance(pose_labels, list):
      pose_labels = numpy.array(pose_labels)
   
    super(DrGanDisk, self).__init__(
            data=data,
            labels=id_labels,
            input_shape=input_shape,
            input_dtype=input_dtype,
            batch_size=batch_size,
            seed=seed,
            data_augmentation=data_augmentation,
            normalizer=normalizer
        )


    numpy.random.seed(seed)


    self.id_labels = id_labels
    self.pose_labels = pose_labels

    self.bob_shape = tuple([input_shape[2]] + list(input_shape[0:2]))

    self.data_placeholder = None
    self.id_label_placeholder = None
    self.pose_label_placeholder = None

    # number of training examples as a 'list'
    self.indexes = numpy.array(range(self.data.shape[0]))
    # shuffle the indexes to get randomized mini-batches
    numpy.random.shuffle(self.indexes)
    
  def get_placeholders(self, name=""):
    """
    Returns placeholders with the size of your batch
    """

    if self.data_placeholder is None:
      self.data_placeholder = tf.placeholder(tf.float32, shape=self.shape)

    if self.id_label_placeholder is None:
      self.id_label_placeholder = tf.placeholder(tf.int64, shape=self.shape[0])
    
    if self.pose_label_placeholder is None:
      self.pose_label_placeholder = tf.placeholder(tf.int64, shape=self.shape[0])
    
    return self.data_placeholder, self.id_label_placeholder, self.pose_label_placeholder

  def load_from_file(self, file_name):
    """load_from_file(file_name) -> data
    
    Load an image from a file, and rescale it if it does not fit the input data format
    Optionnally, data augmentation is performed.

    **Parameters**
      file_name: path
        The name of the (image) file to load.

    **Returns**
      data: numpy array
        The image data
    """
    d = bob.io.base.load(file_name)

    # Applying the data augmentation
    if self.data_augmentation is not None:
        d = self.data_augmentation(d)

    if d.shape[0] != 3 and self.input_shape[2] != 3: # GRAY SCALE IMAGE
        data = numpy.zeros(shape=(d.shape[0], d.shape[1], 1))
        data[:, :, 0] = d
        data = self.rescale(data)
    else:
        d = self.rescale(d)
        data = self.bob2skimage(d)

    # Checking NaN
    if numpy.sum(numpy.isnan(data)) > 0:
        logger.warning("######### Sample {0} has noise #########".format(file_name))

    return data


  def get_batch(self):
    """get_batch() -> selected_data, selected_pose_labels, selected_id_labels

     This function selects and returns data to be used in a minibatch iteration.
     Note that returned data is randomly selected in the training set
    
    **Returns**

    selected_data:
      The face images.

    selected_pose_labels:
      The pose labels

    selected_id_labels:
      The id labels
    """

    # Shuffling samples
    indexes = numpy.array(range(self.data.shape[0]))
    numpy.random.shuffle(indexes)

    selected_data = numpy.zeros(shape=self.shape)
    for i in range(self.batch_size):

      file_name = self.data[indexes[i]]
      data = self.load_from_file(file_name)

      selected_data[i, ...] = data

      # Scaling
      selected_data[i, ...] = self.normalize_sample(selected_data[i, ...])

      selected_id_labels = self.id_labels[indexes[0:self.batch_size]]
      selected_pose_labels = self.pose_labels[indexes[0:self.batch_size]]

    return [selected_data.astype("float32"), selected_id_labels.astype("int64"), selected_pose_labels.astype("int64")]


  def get_batch_epoch(self):
    """get_batch_epoch() -> selected_data, selected_pose_labels, selected_id_labels

    This function selects and returns data to be used in a minibatch iterations.
    Note that it works in epochs, i.e. all the training data should be seen
    during one epoch, which consists in several minibatch iterations.

    **Returns**

    selected_data:
      The face images.

    selected_pose_labels:
      The pose labels

    selected_id_labels:
      The id labels
    """
    # this is done to rebuild the whole list (i.e. at the end of one epoch)
    epoch_done = False

    # returned mini-batch
    selected_data = numpy.zeros(shape=self.shape)
    selected_id_labels = [] 
    selected_pose_labels = [] 

    # if there is not enough available data to fill the current mini-batch
    # add randomly some examples THAT ARE NOT STILL PRESENT in the dataset !
    if len(self.indexes) < self.batch_size:

      # since we reached the end of an epoch, we'll have to reconsider all the data
      epoch_done = True
      number_of_examples_to_add = self.batch_size - len(self.indexes) 
      added_examples = 0
        
      # generate a list of potential examples to add to this mini-batch
      potential_indexes = numpy.array(range(self.data.shape[0]))
      numpy.random.shuffle(potential_indexes)
        
      # add indexes that are not still present in the training data
      for pot_index in potential_indexes:
        if pot_index not in self.indexes:
          self.indexes = numpy.append(self.indexes, [pot_index])
          added_examples += 1
            
          # stop if we have enough examples
          if added_examples == number_of_examples_to_add:
            break
      
    # populate mini-batch
    for i in range(self.batch_size):

      current_index = self.batch_size - i - 1
       
      # TODO: try/catch for file loading

      # get the data example
      file_name = self.data[self.indexes[current_index]]
      data = self.load_from_file(file_name)
      selected_data[i, ...] = data
        
      # normalization
      selected_data[i, ...] = self.normalizer(selected_data[i, ...])
        
      # label
      selected_id_labels.append(self.id_labels[self.indexes[current_index]])
      selected_pose_labels.append(self.pose_labels[self.indexes[current_index]])

      # remove this example from the training set - used once in the epoch
      new_indexes = numpy.delete(self.indexes, current_index)
      self.indexes = new_indexes

    if isinstance(selected_id_labels, list):
      selected_id_labels = numpy.array(selected_id_labels)

    if isinstance(selected_pose_labels, list):
      selected_pose_labels = numpy.array(selected_pose_labels)

    # rebuild whole randomly shuffled training dataset
    if epoch_done:
      self.indexes = numpy.array(range(self.data.shape[0]))
      numpy.random.shuffle(self.indexes)

    return [selected_data.astype("float32"), selected_id_labels.astype("int64"), selected_pose_labels.astype("int64"), epoch_done]
