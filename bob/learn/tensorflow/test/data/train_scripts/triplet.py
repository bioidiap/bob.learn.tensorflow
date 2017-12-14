from bob.learn.tensorflow.datashuffler import TripletMemory
from bob.learn.tensorflow.network import Chopra
from bob.learn.tensorflow.trainers import TripletTrainer as Trainer
from bob.learn.tensorflow.trainers import constant
from bob.learn.tensorflow.loss import TripletLoss
from bob.learn.tensorflow.utils import load_mnist
import tensorflow as tf
import numpy

BATCH_SIZE = 32
INPUT_SHAPE = [None, 28, 28, 1]
SEED = 10

### PREPARING DATASHUFFLER ###
train_data, train_labels, validation_data, validation_labels = \
    load_mnist()
train_data = numpy.reshape(train_data, (train_data.shape[0], 28, 28, 1))

train_data_shuffler = TripletMemory(
    train_data, train_labels, input_shape=INPUT_SHAPE, batch_size=BATCH_SIZE)

### ARCHITECTURE ###
architecture = Chopra(seed=SEED, n_classes=10)

### LOSS ###
loss = TripletLoss(margin=4.)

### LEARNING RATE ###
learning_rate = constant(base_learning_rate=0.01)

### SOLVER ###
optimizer = tf.train.GradientDescentOptimizer(learning_rate)

### Trainer ###
trainer = Trainer
