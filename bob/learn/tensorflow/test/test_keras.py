#!/usr/bin/env python

import pprint

from keras.models import Sequential
from keras.engine import InputLayer
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Lambda
from keras.layers import Flatten
from keras.layers import Reshape

from keras.datasets import mnist
from keras.utils import np_utils

from keras.utils.layer_utils import print_summary

method = "drop-first" # See if/elif block below for explanation

n_epochs = 2
n_hidden = 32       # Inside the LSTM cell
n_drop_first = 2    # Number of first output to drop after LSTM

# Load you training data.
(X_train, y_train), (X_test, y_test) = mnist.load_data()

print("Train data {}".format(X_train.shape))
print("Train labels {}".format(y_train.shape))

if method == "squares":
    # Example where MNIST images are squares (28,28)
    X_train = X_train.reshape(X_train.shape[0], 28, 28).astype("float32")
    X_test  = X_test.reshape(X_test.shape[0],  28, 28).astype("float32")

    X_train = X_train / 255
    X_test = X_test / 255

    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)

    print("Train data for training {}".format(X_train.shape))

    # LSTM
    model = Sequential()
    model.add(LSTM(n_hidden, input_shape=(28,28)))
    model.add(Dense(10, activation="softmax"))

elif method == "lines":
    # Example where MNIST images are lines (1, 784)
    X_train = X_train.reshape(X_train.shape[0], 1, 784).astype("float32")
    X_test  = X_test.reshape(X_test.shape[0],  1, 784).astype("float32")

    X_train = X_train / 255
    X_test = X_test / 255

    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)

    print("Train data {}".format(X_train.shape))

    # LSTM
    model = Sequential()
    model.add(LSTM(n_hidden, input_shape=(1, 784)))
    model.add(Dense(10, activation="softmax"))

elif method == "drop-first":
    # Example where we drop first sequences, keep only the last ones
    # and plug a fully connected layer after. Input images are
    # squares, a 28-long sequence of 28 features
    X_train = X_train.reshape(X_train.shape[0], 28, 28).astype("float32")
    X_test  = X_test.reshape(X_test.shape[0],  28, 28).astype("float32")

    X_train = X_train / 255
    X_test = X_test / 255

    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)

    n_steps = X_train.shape[1]
    n_features = X_train.shape[2]
    n_classes = y_test.shape[1]
    n_new_steps = n_steps - n_drop_first

    print("n_steps {}".format(n_steps))
    print("n_features {}".format(n_features))
    print("n_classes {}".format(n_classes))

    # LSTM
    model = Sequential()
    model.add(LSTM(n_hidden, input_shape=(28, 28), return_sequences=True))
    model.add(Lambda(lambda x: x[:,n_drop_first:,:]))
    model.add(Reshape((n_hidden*n_new_steps,), input_shape=(n_new_steps, n_hidden)))
    model.add(Dense(n_classes, activation="softmax"))

######################################################################

print_summary(model)

model.compile(optimizer="adam",
              loss="categorical_crossentropy",
              metrics=["accuracy"])

model.fit(X_train, y_train,
          epochs=n_epochs,
          batch_size=32,
          validation_data=(X_test, y_test))

out = model.predict_on_batch(X_test[0:7, :])
print(out.shape)

# for layer in model.layers:
#     print("{} {}".format(layer.name, model.get_layer(layer.name).output.shape))
