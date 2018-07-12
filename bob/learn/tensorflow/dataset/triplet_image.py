#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>

import tensorflow as tf
from functools import partial
from . import append_image_augmentation, triplets_random_generator, from_filename_to_tensor


def shuffle_data_and_labels_image_augmentation(filenames,
                                               labels,
                                               data_shape,
                                               data_type,
                                               batch_size,
                                               epochs=None,
                                               buffer_size=10**3,
                                               gray_scale=False,
                                               output_shape=None,
                                               random_flip=False,
                                               random_brightness=False,
                                               random_contrast=False,
                                               random_saturation=False,
                                               random_rotate=False,
                                               per_image_normalization=True,
                                               extension=None):
    """
    Dump random batches for triplee networks from a list of image paths and labels:

    The list of files and labels should be in the same order e.g.
    filenames = ['class_1_img1', 'class_1_img2', 'class_2_img1']
    labels = [0, 0, 1]

    The batches returned with tf.Session.run() with be in the following format:
    **data** a dictionary containing the keys ['anchor', 'positive', 'negative'].


    **Parameters**

       filenames:
          List containing the path of the images

       labels:
          List containing the labels (needs to be in EXACT same order as filenames)

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

       random_rotate:
           Randomly rotate face images between -5 and 5 degrees

       per_image_normalization:
           Linearly scales image to have zero mean and unit norm.

       extension:
           If None, will load files using `tf.image.decode..` if set to `hdf5`, will load with `bob.io.base.load`

    """

    dataset = create_dataset_from_path_augmentation(
        filenames,
        labels,
        data_shape,
        data_type,
        gray_scale=gray_scale,
        output_shape=output_shape,
        random_flip=random_flip,
        random_brightness=random_brightness,
        random_contrast=random_contrast,
        random_saturation=random_saturation,
        random_rotate=random_rotate,
        per_image_normalization=per_image_normalization,
        extension=extension)

    dataset = dataset.shuffle(buffer_size).batch(batch_size).repeat(epochs)
    #dataset = dataset.batch(buffer_size).batch(batch_size).repeat(epochs)

    data = dataset.make_one_shot_iterator().get_next()
    return data


def create_dataset_from_path_augmentation(filenames,
                                          labels,
                                          data_shape,
                                          data_type=tf.float32,
                                          gray_scale=False,
                                          output_shape=None,
                                          random_flip=False,
                                          random_brightness=False,
                                          random_contrast=False,
                                          random_saturation=False,
                                          random_rotate=False,
                                          per_image_normalization=True,
                                          extension=None):
    """
    Create dataset from a list of tf-record files

    **Parameters**

       filenames:
          List containing the path of the images

       labels:
          List containing the labels (needs to be in EXACT same order as filenames)

       data_shape:
          Samples shape saved in the tf-record

       data_type:
          tf data type(https://www.tensorflow.org/versions/r0.12/resources/dims_types#data_types)

       feature:

    """

    parser = partial(
        image_augmentation_parser,
        data_shape=data_shape,
        data_type=data_type,
        gray_scale=gray_scale,
        output_shape=output_shape,
        random_flip=random_flip,
        random_brightness=random_brightness,
        random_contrast=random_contrast,
        random_saturation=random_saturation,
        random_rotate=random_rotate,
        per_image_normalization=per_image_normalization,
        extension=extension)

    anchor_data, positive_data, negative_data = triplets_random_generator(
        filenames, labels)

    dataset = tf.data.Dataset.from_tensor_slices((anchor_data, positive_data,
                                                  negative_data))
    dataset = dataset.map(parser)
    return dataset


def image_augmentation_parser(anchor,
                              positive,
                              negative,
                              data_shape,
                              data_type=tf.float32,
                              gray_scale=False,
                              output_shape=None,
                              random_flip=False,
                              random_brightness=False,
                              random_contrast=False,
                              random_saturation=False,
                              random_rotate=False,
                              per_image_normalization=True,
                              extension=None):
    """
    Parses a single tf.Example into image and label tensors.
    """

    triplet = dict()
    for n, v in zip(['anchor', 'positive', 'negative'],
                    [anchor, positive, negative]):

        # Convert the image data from string back to the numbers
        image = from_filename_to_tensor(v, extension=extension)

        # Reshape image data into the original shape
        image = tf.reshape(image, data_shape)

        # Applying image augmentation
        image = append_image_augmentation(
            image,
            gray_scale=gray_scale,
            output_shape=output_shape,
            random_flip=random_flip,
            random_brightness=random_brightness,
            random_contrast=random_contrast,
            random_saturation=random_saturation,
            random_rotate=random_rotate,
            per_image_normalization=per_image_normalization)

        triplet[n] = image

    return triplet
