from bob.learn.tensorflow.datashuffler import Memory
from bob.learn.tensorflow.network import Chopra
from bob.learn.tensorflow.trainers import Trainer, constant
from bob.learn.tensorflow.loss import BaseLoss
from bob.learn.tensorflow.utils import load_mnist
import tensorflow as tf
import numpy

BATCH_SIZE = 32
INPUT_SHAPE = [28, 28, 1]
SEED = 10
USE_GPU = False


### PREPARING DATASHUFFLER ###
train_data, train_labels, validation_data, validation_labels = \
    load_mnist(data_dir="./src/bob.db.mnist/bob/db/mnist/")
train_data = numpy.reshape(train_data, (train_data.shape[0], 28, 28, 1))

train_data_shuffler = Memory(train_data, train_labels,
                             input_shape=INPUT_SHAPE,
                             batch_size=BATCH_SIZE)

### ARCHITECTURE ###
architecture = Chopra(seed=SEED, fc1_output=10, batch_norm=False, use_gpu=USE_GPU)

### LOSS ###
loss = BaseLoss(tf.nn.sparse_softmax_cross_entropy_with_logits, tf.reduce_mean)

### SOLVER ###
optimizer = tf.train.GradientDescentOptimizer(0.001)

### LEARNING RATE ###
learning_rate = constant(base_learning_rate=0.001)

### Trainer ###
trainer = Trainer
