import tensorflow as tf
import numpy

DEFAULT_FEATURE = {'data': tf.FixedLenFeature([], tf.string),
                   'label': tf.FixedLenFeature([], tf.int64),
                   'key': tf.FixedLenFeature([], tf.string)}


def append_image_augmentation(image, gray_scale=False,
                              output_shape=None,
                              random_flip=False,
                              random_brightness=False,
                              random_contrast=False,
                              random_saturation=False,
                              per_image_normalization=True):
    """
    Append to the current tensor some random image augmentation operation

    **Parameters**
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

    # Casting to float32
    image = tf.cast(image, tf.float32)

    if output_shape is not None:
        assert len(output_shape) == 2
        image = tf.image.resize_image_with_crop_or_pad(image, output_shape[0], output_shape[1])

    if random_flip:
        image = tf.image.random_flip_left_right(image)

    if random_brightness:
        image = tf.image.random_brightness(image, max_delta=0.5)

    if random_contrast:
        image = tf.image.random_contrast(image, lower=0, upper=0.5)

    if random_saturation:
        image = tf.image.random_saturation(image, lower=0, upper=0.5)

    if gray_scale:
        image = tf.image.rgb_to_grayscale(image, name="rgb_to_gray")
        #self.output_shape[3] = 1

    # normalizing data
    if per_image_normalization:
        image = tf.image.per_image_standardization(image)

    return image


def arrange_indexes_by_label(input_labels, possible_labels):

    # Shuffling all the indexes
    indexes_per_labels = dict()
    for l in possible_labels:
        indexes_per_labels[l] = numpy.where(input_labels == l)[0]
        numpy.random.shuffle(indexes_per_labels[l])
    return indexes_per_labels


def triplets_random_generator(input_data, input_labels):
    """
    Giving a list of samples and a list of labels, it dumps a series of
    triplets for triple nets.

    **Parameters**

      input_data: List of whatever representing the data samples

      input_labels: List of the labels (needs to be in EXACT same order as input_data)
    """
    anchor = []
    positive = []
    negative = []

    def append(anchor_sample, positive_sample, negative_sample):
        """
        Just appending one element in each list
        """
        anchor.append(anchor_sample)
        positive.append(positive_sample)
        negative.append(negative_sample)

    possible_labels = list(set(input_labels))
    input_data = numpy.array(input_data)
    input_labels = numpy.array(input_labels)
    total_samples = input_data.shape[0]

    indexes_per_labels = arrange_indexes_by_label(input_labels, possible_labels)

    # searching for random triplets
    offset_class = 0
    for i in range(total_samples):

        anchor_sample = input_data[indexes_per_labels[possible_labels[offset_class]][numpy.random.randint(len(indexes_per_labels[possible_labels[offset_class]]))], ...]

        positive_sample = input_data[indexes_per_labels[possible_labels[offset_class]][numpy.random.randint(len(indexes_per_labels[possible_labels[offset_class]]))], ...]

        # Changing the class
        offset_class += 1

        if offset_class == len(possible_labels):
            offset_class = 0

        negative_sample = input_data[indexes_per_labels[possible_labels[offset_class]][numpy.random.randint(len(indexes_per_labels[possible_labels[offset_class]]))], ...]

        append(str(anchor_sample), str(positive_sample), str(negative_sample))
        #yield anchor, positive, negative
    return anchor, positive, negative


def siamease_pairs_generator(input_data, input_labels):
    """
    Giving a list of samples and a list of labels, it dumps a series of
    pairs for siamese nets.

    **Parameters**

      input_data: List of whatever representing the data samples

      input_labels: List of the labels (needs to be in EXACT same order as input_data)
    """

    # Lists that will be returned
    left_data = []
    right_data = []
    labels = []

    def append(left, right, label):
        """
        Just appending one element in each list
        """
        left_data.append(left)
        right_data.append(right)
        labels.append(label)

    possible_labels = list(set(input_labels))
    input_data = numpy.array(input_data)
    input_labels = numpy.array(input_labels)
    total_samples = input_data.shape[0]

    # Filtering the samples by label and shuffling all the indexes
    #indexes_per_labels = dict()
    #for l in possible_labels:
    #    indexes_per_labels[l] = numpy.where(input_labels == l)[0]
    #    numpy.random.shuffle(indexes_per_labels[l])
    indexes_per_labels = arrange_indexes_by_label(input_labels, possible_labels)

    left_possible_indexes = numpy.random.choice(possible_labels, total_samples, replace=True)
    right_possible_indexes = numpy.random.choice(possible_labels, total_samples, replace=True)

    genuine = True
    for i in range(total_samples):

        if genuine:
            # Selecting the class
            class_index = left_possible_indexes[i]

            # Now selecting the samples for the pair
            left = input_data[indexes_per_labels[class_index][numpy.random.randint(len(indexes_per_labels[class_index]))]]
            right = input_data[indexes_per_labels[class_index][numpy.random.randint(len(indexes_per_labels[class_index]))]]
            append(left, right, 0)
            #yield left, right, 0
        else:
            # Selecting the 2 classes
            class_index = list()
            class_index.append(left_possible_indexes[i])

            # Finding the right pair
            j = i
            # TODO: Lame solution. Fix this
            while j < total_samples: # Here is an unidiretinal search for the negative pair
                if left_possible_indexes[i] != right_possible_indexes[j]:
                    class_index.append(right_possible_indexes[j])
                    break
                j += 1

            if j < total_samples:
                # Now selecting the samples for the pair
                left = input_data[indexes_per_labels[class_index[0]][numpy.random.randint(len(indexes_per_labels[class_index[0]]))]]
                right = input_data[indexes_per_labels[class_index[1]][numpy.random.randint(len(indexes_per_labels[class_index[1]]))]]
                append(left, right, 1)


        genuine = not genuine
    return left_data, right_data, labels


def blocks_tensorflow(images, block_size):
    """Return all non-overlapping blocks of an image using tensorflow
    operations.

    Parameters
    ----------
    images : :any:`tf.Tensor`
        The input color images. It is assumed that the image has a shape of
        [?, H, W, C].
    block_size : (int, int)
        A tuple of two integers indicating the block size.

    Returns
    -------
    blocks : :any:`tf.Tensor`
        All the blocks in the batch dimension. The output will be of
        size [?, block_size[0], block_size[1], C].
    n_blocks : int
        The number of blocks that was obtained per image.
    """
    # normalize block_size
    block_size = [1] + list(block_size) + [1]
    output_size = list(block_size)
    output_size[0] = -1
    # extract image patches for each color space:
    output = []
    for i in range(3):
        blocks = tf.extract_image_patches(
            images[:, :, :, i:i + 1], block_size, block_size, [1, 1, 1, 1],
            "VALID")
        if i == 0:
            n_blocks = int(numpy.prod(blocks.shape[1:3]))
        blocks = tf.reshape(blocks, output_size)
        output.append(blocks)
    # concatenate the colors back
    output = tf.concat(output, axis=3)
    return output, n_blocks


def tf_repeat(tensor, repeats):
    """
    Parameters
    ----------
    tensor
        A Tensor. 1-D or higher.
    repeats
        A list. Number of repeat for each dimension, length must be the same as
        the number of dimensions in input

    Returns
    -------
    A Tensor. Has the same type as input. Has the shape of tensor.shape *
    repeats
    """
    with tf.variable_scope("repeat"):
        expanded_tensor = tf.expand_dims(tensor, -1)
        multiples = [1] + repeats
        tiled_tensor = tf.tile(expanded_tensor, multiples=multiples)
        repeated_tesnor = tf.reshape(tiled_tensor, tf.shape(tensor) * repeats)
    return repeated_tesnor