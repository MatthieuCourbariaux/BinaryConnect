from __future__ import absolute_import
from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils

batch_size = 100
nb_classes = 10
nb_epoch = 100

# the data, shuffled and split between tran and test sets
(X_train, y_train), (X_valid, y_valid), (X_test, y_test) = mnist.load_data(path="/data/lisa/data/mnist/mnist.pkl")

X_train = X_train.reshape(50000, 784)
X_valid = X_valid.reshape(10000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype("float32")
X_valid = X_valid.astype("float32")
X_test = X_test.astype("float32")
print(X_train.shape[0], 'train samples')
print(X_valid.shape[0], 'valid samples')
print(X_test.shape[0], 'test samples')

# TODO preprocessing = center

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_valid = np_utils.to_categorical(y_valid, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

# for hinge loss
Y_train = 2* Y_train - 1.
Y_valid = 2* Y_valid - 1.
Y_test = 2* Y_test - 1.

model = Sequential()
# model.add(Dropout(0.2))
model.add(Dense(784, 1024,init='glorot_uniform'))
model.add(BatchNormalization(input_shape=(1024,),epsilon=1e-4, momentum=0.))
model.add(Activation('relu'))
# model.add(Dropout(0.5))
model.add(Dense(1024, 1024,init='glorot_uniform'))
model.add(BatchNormalization(input_shape=(1024,),epsilon=1e-4, momentum=0.))
model.add(Activation('relu'))
# model.add(Dropout(0.5))
model.add(Dense(1024, 1024,init='glorot_uniform'))
model.add(BatchNormalization(input_shape=(1024,),epsilon=1e-4, momentum=0.))
model.add(Activation('relu'))
# model.add(Dropout(0.5))
model.add(Dense(1024, 10))
model.add(BatchNormalization(input_shape=(10,),epsilon=1e-4, momentum=0.))
model.add(Activation('linear'))

# sgd = SGD(lr=0.003)
# model.compile(loss='squared_hinge', optimizer=sgd)
adam = Adam()
model.compile(loss='squared_hinge', optimizer=adam)
# rms = RMSprop()
# model.compile(loss='squared_hinge', optimizer=rms)

model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=True, verbose=2, validation_data=(X_valid, Y_valid))
score = model.evaluate(X_test, Y_test, show_accuracy=True, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])