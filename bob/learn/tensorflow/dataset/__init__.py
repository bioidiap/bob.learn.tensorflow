import tensorflow as tf


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
        assert output_shape.ndim == 2        
        image = tf.image.resize_image_with_crop_or_pad(image, output_shape[0], output_shape[1])
        
    if random_flip:
        image = tf.image.random_flip_left_right(image)

    if random_brightness:
        image = tf.image.random_brightness(image)

    if random_contrast:
        image = tf.image.random_contrast(image)

    if random_saturation:
        image = tf.image.random_saturation(image)

    if gray_scale:
        image = tf.image.rgb_to_grayscale(image, name="rgb_to_gray")
        #self.output_shape[3] = 1

    # normalizing data
    if per_image_normalization:
        image = tf.image.per_image_standardization(image)

    return image

