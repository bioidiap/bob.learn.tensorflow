#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# @date: Wed 11 May 2016 09:39:36 CEST 

import numpy
import bob.io.base
import bob.io.image
import bob.ip.base
import tensorflow as tf

from .BaseDataShuffler import BaseDataShuffler

#def scale_mean_norm(data, scale=0.00390625):
#    mean = numpy.mean(data)
#    data = (data - mean) * scale

#    return data, mean


class TextDataShuffler(BaseDataShuffler):
    def __init__(self, data, labels,
                 input_shape,
                 input_dtype="float64",
                 scale=True,
                 batch_size=1):
        """
         Shuffler that deal with file list

         **Parameters**
         data:
         labels:
         input_shape: Shape of the input. `input_shape != data.shape`, the data will be reshaped
         input_dtype="float64":
         scale=True:
         batch_size=1:
        """

        if isinstance(data, list):
            data = numpy.array(data)

        if isinstance(labels, list):
            labels = numpy.array(labels)

        super(TextDataShuffler, self).__init__(
            data=data,
            labels=labels,
            input_shape=input_shape,
            input_dtype=input_dtype,
            scale=scale,
            batch_size=batch_size
        )

        # TODO: very bad solution to deal with bob.shape images an tf shape images
        self.bob_shape = tuple([input_shape[2]] + list(input_shape[0:2]))

    def load_from_file(self, file_name, shape):
        d = bob.io.base.load(file_name)
        if d.shape[0] != 3 and self.input_shape[2] != 3: # GRAY SCALE IMAGE
            data = numpy.zeros(shape=(d.shape[0], d.shape[1], 1))
            data[:, :, 0] = d
            data = self.rescale(data)
        else:
            d = self.rescale(d)
            data = self.bob2skimage(d)

        return data

    def bob2skimage(self, bob_image):
        """
        Convert bob color image to the skcit image
        """

        skimage = numpy.zeros(shape=(bob_image.shape[1], bob_image.shape[2], 3))

        skimage[:, :, 0] = bob_image[0, :, :] #Copying red
        skimage[:, :, 1] = bob_image[1, :, :] #Copying green
        skimage[:, :, 2] = bob_image[2, :, :] #Copying blue

        return skimage

    def get_batch(self):

        # Shuffling samples
        indexes = numpy.array(range(self.data.shape[0]))
        numpy.random.shuffle(indexes)

        selected_data = numpy.zeros(shape=self.shape)
        for i in range(self.batch_size):

            file_name = self.data[indexes[i]]
            data = self.load_from_file(file_name, self.shape)

            selected_data[i, ...] = data
            if self.scale:
                selected_data[i, ...] *= self.scale_value

        selected_labels = self.labels[indexes[0:self.batch_size]]

        return selected_data.astype("float32"), selected_labels

    def rescale(self, data):
        """
        Reescale a single sample with input_shape

        """
        #if self.input_shape != data.shape:
        if self.bob_shape != data.shape:

            # TODO: Implement a better way to do this reescaling
            # If it is gray scale
            if self.input_shape[2] == 1:
                copy = data[:, :, 0].copy()
                dst = numpy.zeros(shape=self.input_shape[0:2])
                bob.ip.base.scale(copy, dst)
                dst = numpy.reshape(dst, self.input_shape)
            else:
                #dst = numpy.resize(data, self.bob_shape) # Scaling with numpy, because bob is c,w,d instead of w,h,c
                dst = numpy.zeros(shape=self.bob_shape)

                # TODO: LAME SOLUTION
                if data.shape[0] != 3: # GRAY SCALE IMAGES IN A RGB DATABASE
                    step_data = numpy.zeros(shape=(3, data.shape[0], data.shape[1]))
                    step_data[0, ...] = data[:, :]
                    step_data[1, ...] = data[:, :]
                    step_data[2, ...] = data[:, :]
                    data = step_data

                bob.ip.base.scale(data, dst)

            return dst
        else:
            return data

    def get_pair(self, zero_one_labels=True):
        """
        Get a random pair of samples

        **Parameters**
            is_target_set_train: Defining the target set to get the batch

        **Return**
        """

        data = numpy.zeros(shape=self.shape, dtype='float32')
        data_p = numpy.zeros(shape=self.shape, dtype='float32')
        labels_siamese = numpy.zeros(shape=self.shape[0], dtype='float32')

        genuine = True
        for i in range(self.shape[0]):
            file_name, file_name_p = self.get_genuine_or_not(self.data, self.labels, genuine=genuine)
            data[i, ...] = self.load_from_file(str(file_name), self.shape)
            data_p[i, ...] = self.load_from_file(str(file_name_p), self.shape)

            if zero_one_labels:
                labels_siamese[i] = not genuine
            else:
                labels_siamese[i] = -1 if genuine else +1
            genuine = not genuine

        if self.scale:
            data *= self.scale_value
            data_p *= self.scale_value

        return data, data_p, labels_siamese

    def get_random_triplet(self):
        """
        Get a random pair of samples

        **Parameters**
            is_target_set_train: Defining the target set to get the batch

        **Return**
        """

        data_a = numpy.zeros(shape=self.shape, dtype='float32')
        data_p = numpy.zeros(shape=self.shape, dtype='float32')
        data_n = numpy.zeros(shape=self.shape, dtype='float32')

        for i in range(self.shape[0]):
            file_name_a, file_name_p, file_name_n = self.get_one_triplet(self.data, self.labels)
            data_a[i, ...] = self.load_from_file(str(file_name_a), self.shape)
            data_p[i, ...] = self.load_from_file(str(file_name_p), self.shape)
            data_n[i, ...] = self.load_from_file(str(file_name_n), self.shape)

        if self.scale:
            data_a *= self.scale_value
            data_p *= self.scale_value
            data_n *= self.scale_value

        return data_a, data_p, data_n
