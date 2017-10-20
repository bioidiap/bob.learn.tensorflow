import tensorflow as tf
import numpy

DEFAULT_FEATURE = {'train/data': tf.FixedLenFeature([], tf.string),
                   'train/label': tf.FixedLenFeature([], tf.int64)}



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
    indexes_per_labels = dict()
    for l in possible_labels:
        indexes_per_labels[l] = numpy.where(input_labels == l)[0]
        numpy.random.shuffle(indexes_per_labels[l])

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

