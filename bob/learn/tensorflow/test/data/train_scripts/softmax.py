from bob.learn.tensorflow.datashuffler import Memory, ScaleFactor
from bob.learn.tensorflow.network import chopra
from bob.learn.tensorflow.trainers import Trainer, constant
from bob.learn.tensorflow.loss import MeanSoftMaxLoss
from bob.learn.tensorflow.utils import load_mnist
import tensorflow as tf
import numpy

BATCH_SIZE = 32
INPUT_SHAPE = [None, 28, 28, 1]
SEED = 10
USE_GPU = False

### PREPARING DATASHUFFLER ###
train_data, train_labels, validation_data, validation_labels = load_mnist()
train_data = numpy.reshape(train_data, (train_data.shape[0], 28, 28, 1))

train_data_shuffler = Memory(
    train_data,
    train_labels,
    input_shape=INPUT_SHAPE,
    batch_size=BATCH_SIZE,
    normalizer=ScaleFactor())

### ARCHITECTURE ###
architecture = chopra(seed=SEED, n_classes=10)

### LOSS ###
loss = MeanSoftMaxLoss()

### LEARNING RATE ###
learning_rate = constant(base_learning_rate=0.01)

### SOLVER ###
optimizer = tf.train.GradientDescentOptimizer(learning_rate)

### Trainer ###
trainer = Trainer
