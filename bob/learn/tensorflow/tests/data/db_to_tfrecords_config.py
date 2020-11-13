import tensorflow as tf

from bob.learn.tensorflow.data import dataset_using_generator

mnist = tf.keras.datasets.mnist

(x_train, y_train), (_, _) = mnist.load_data()
x_train, y_train = x_train[:10], y_train[:10]
samples = zip(tf.keras.backend.arange(len(x_train)), x_train, y_train)


def reader(sample):
    data = sample[1]
    label = sample[2]
    key = str(sample[0]).encode("utf-8")
    return ({"data": data, "key": key}, label)


dataset = dataset_using_generator(samples, reader)
