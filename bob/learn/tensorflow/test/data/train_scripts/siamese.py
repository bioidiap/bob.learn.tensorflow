from bob.learn.tensorflow.datashuffler import SiameseMemory
from bob.learn.tensorflow.network import Chopra
from bob.learn.tensorflow.trainers import SiameseTrainer as Trainer
from bob.learn.tensorflow.trainers import constant
from bob.learn.tensorflow.loss import ContrastiveLoss
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

train_data_shuffler = SiameseMemory(train_data, train_labels,
                                    input_shape=INPUT_SHAPE,
                                    batch_size=BATCH_SIZE)

### ARCHITECTURE ###
architecture = Chopra(seed=SEED, fc1_output=10, batch_norm=False)

### LOSS ###
loss = ContrastiveLoss(contrastive_margin=4.)

### SOLVER ###
optimizer = tf.train.GradientDescentOptimizer(0.001)

### LEARNING RATE ###
learning_rate = constant(base_learning_rate=0.001)

### Trainer ###
trainer = Trainer
