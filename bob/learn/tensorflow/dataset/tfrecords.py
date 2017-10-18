from functools import partial
import tensorflow as tf
from . import append_image_augmentation, DEFAULT_FEATURE


def example_parser(serialized_example, feature, data_shape, data_type):
    """
    Parses a single tf.Example into image and label tensors.
    
    """
    # Decode the record read by the reader
    features = tf.parse_single_example(serialized_example, features=feature)
    # Convert the image data from string back to the numbers
    image = tf.decode_raw(features['train/data'], data_type)
    # Cast label data into int64
    label = tf.cast(features['train/label'], tf.int64)
    # Reshape image data into the original shape
    image = tf.reshape(image, data_shape)
    return image, label


def image_augmentation_parser(serialized_example, feature, data_shape, data_type,
                              gray_scale=False, 
                              output_shape=None,
                              random_flip=False,
                              random_brightness=False,
                              random_contrast=False,
                              random_saturation=False,
                              per_image_normalization=True):

    """
    Parses a single tf.Example into image and label tensors.
    
    """
    # Decode the record read by the reader
    features = tf.parse_single_example(serialized_example, features=feature)
    # Convert the image data from string back to the numbers
    image = tf.decode_raw(features['train/data'], data_type)

    # Reshape image data into the original shape
    image = tf.reshape(image, data_shape)
    
    #Applying image augmentation
    image = append_image_augmentation(image, gray_scale=gray_scale,
                                      output_shape=output_shape,
                                      random_flip=random_flip,
                                      random_brightness=random_brightness,
                                      random_contrast=random_contrast,
                                      random_saturation=random_saturation,
                                      per_image_normalization=per_image_normalization)
    
    # Cast label data into int64
    label = tf.cast(features['train/label'], tf.int64)
    return image, label


def read_and_decode(filename_queue, data_shape, data_type=tf.float32,
                    feature=None):
                    
    """
    Simples parse possible for a tfrecord.
    It assumes that you have the pair **train/data** and **train/label**
    """
                    
    if feature is None:
        feature = DEFAULT_FEATURE
    # Define a reader and read the next record
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    return example_parser(serialized_example, feature, data_shape, data_type)


def create_dataset_from_records(tfrecord_filenames, data_shape, data_type,
                                feature=None):
    """
    Create dataset from a list of tf-record files
    
    **Parameters**
    
       tfrecord_filenames: 
          List containing the tf-record paths

       data_shape:
          Samples shape saved in the tf-record
          
       data_type:
          tf data type(https://www.tensorflow.org/versions/r0.12/resources/dims_types#data_types)
          
       feature:
    
    """
                                
    if feature is None:
        feature = DEFAULT_FEATURE
    dataset = tf.contrib.data.TFRecordDataset(tfrecord_filenames)
    parser = partial(example_parser, feature=feature, data_shape=data_shape,
                     data_type=data_type)
    dataset = dataset.map(parser)
    return dataset


def create_dataset_from_records_with_augmentation(tfrecord_filenames, data_shape, data_type,
                                feature=None,
                                gray_scale=False, 
                                output_shape=None,
                                random_flip=False,
                                random_brightness=False,
                                random_contrast=False,
                                random_saturation=False,
                                per_image_normalization=True):
    """
    Create dataset from a list of tf-record files
    
    **Parameters**
    
       tfrecord_filenames: 
          List containing the tf-record paths

       data_shape:
          Samples shape saved in the tf-record
          
       data_type:
          tf data type(https://www.tensorflow.org/versions/r0.12/resources/dims_types#data_types)
          
       feature:
    
    """
                                
                                
    if feature is None:
        feature = DEFAULT_FEATURE
    dataset = tf.contrib.data.TFRecordDataset(tfrecord_filenames)
    parser = partial(image_augmentation_parser, feature=feature, data_shape=data_shape,
                     data_type=data_type,
                     gray_scale=gray_scale, 
                     output_shape=output_shape,
                     random_flip=random_flip,
                     random_brightness=random_brightness,
                     random_contrast=random_contrast,
                     random_saturation=random_saturation,
                     per_image_normalization=per_image_normalization)
    dataset = dataset.map(parser)
    return dataset


def shuffle_data_and_labels_image_augmentation(tfrecord_filenames, data_shape, data_type,
                                              batch_size, epochs=None, buffer_size=10**3,
                                              gray_scale=False, 
                                              output_shape=None,
                                              random_flip=False,
                                              random_brightness=False,
                                              random_contrast=False,
                                              random_saturation=False,
                                              per_image_normalization=True):
    """
    Dump random batches from a list of tf-record files and applies some image augmentation

    **Parameters**

       tfrecord_filenames: 
          List containing the tf-record paths

       data_shape:
          Samples shape saved in the tf-record
          
       data_type:
          tf data type(https://www.tensorflow.org/versions/r0.12/resources/dims_types#data_types)
     
       batch_size:
          Size of the batch
          
       epochs:
           Number of epochs to be batched
       
       buffer_size:
            Size of the shuffle bucket

       gray_scale:
          Convert to gray scale?
          
       output_shape:
          If set, will randomly crop the image given the output shape

       random_flip:
          Randomly flip an image horizontally  (https://www.tensorflow.org/api_docs/python/tf/image/random_flip_left_right)

       random_brightness:
           Adjust the brightness of an RGB image by a random factor (https://www.tensorflow.org/api_docs/python/tf/image/random_brightness)

       random_contrast:
           Adjust the contrast of an RGB image by a random factor (https://www.tensorflow.org/api_docs/python/tf/image/random_contrast)

       random_saturation:
           Adjust the saturation of an RGB image by a random factor (https://www.tensorflow.org/api_docs/python/tf/image/random_saturation)

       per_image_normalization:
           Linearly scales image to have zero mean and unit norm.            
     
    """                            

    dataset = create_dataset_from_records_with_augmentation(tfrecord_filenames, data_shape,
                                          data_type,
                                          gray_scale=gray_scale, 
                                          output_shape=output_shape,
                                          random_flip=random_flip,
                                          random_brightness=random_brightness,
                                          random_contrast=random_contrast,
                                          random_saturation=random_saturation,
                                          per_image_normalization=per_image_normalization)
                                          
    dataset = dataset.shuffle(buffer_size).batch(batch_size).repeat(epochs)

    data, labels = dataset.make_one_shot_iterator().get_next()
    return data, labels


def shuffle_data_and_labels(tfrecord_filenames, data_shape, data_type,
                            batch_size, epochs=None, buffer_size=10**3):
    """
    Dump random batches from a list of tf-record files

    **Parameters**

       tfrecord_filenames: 
          List containing the tf-record paths

       data_shape:
          Samples shape saved in the tf-record
          
       data_type:
          tf data type(https://www.tensorflow.org/versions/r0.12/resources/dims_types#data_types)
     
       batch_size:
          Size of the batch
          
       epochs:
           Number of epochs to be batched
       
       buffer_size:
            Size of the shuffle bucket
     
    """                            

    dataset = create_dataset_from_records(tfrecord_filenames, data_shape,
                                          data_type)
    dataset = dataset.shuffle(buffer_size).batch(batch_size).repeat(epochs)

    data, labels = dataset.make_one_shot_iterator().get_next()
    return data, labels


def batch_data_and_labels(tfrecord_filenames, data_shape, data_type,
                          batch_size, epochs=1):
    """
    Dump in order batches from a list of tf-record files

    **Parameters**

       tfrecord_filenames: 
          List containing the tf-record paths

       data_shape:
          Samples shape saved in the tf-record
          
       data_type:
          tf data type(https://www.tensorflow.org/versions/r0.12/resources/dims_types#data_types)
     
       batch_size:
          Size of the batch
          
       epochs:
           Number of epochs to be batched
     
    """                             
    dataset = create_dataset_from_records(tfrecord_filenames, data_shape,
                                          data_type)
    dataset = dataset.batch(batch_size).repeat(epochs)

    data, labels = dataset.make_one_shot_iterator().get_next()
    return data, labels

