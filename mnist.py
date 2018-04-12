'''Train a simple deep CNN on the CIFAR10 small images dataset.
GPU run command with Theano backend (with TensorFlow, the GPU is automatically used):
    THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatx=float32 python cifar10_cnn.py
It gets down to 0.65 test logloss in 25 epochs, and down to 0.55 after 50 epochs.
(it's still underfitting at that point, though).
'''

from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD
from keras.utils import np_utils
from custom import AdaptativeBiHyperbolic
import numpy as np
import os

nb_classes = 10
batch_size = 128
nb_epoch = 1000

# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = np_utils.to_categorical(y_train, nb_classes)
y_test = np_utils.to_categorical(y_test, nb_classes)

# create model
model = Sequential()

model.add(Dense(800, input_shape=(784,)))
model.add(AdaptativeBiHyperbolic())
for l in range(5):
    model.add(Dense(800))
    model.add(AdaptativeBiHyperbolic())
model.add(Dense(nb_classes))
model.add(Activation('softmax'))
model.summary()

# initiate RMSprop optimizer
#opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

opt = SGD(lr=0.01, momentum=0.9, nesterov=True)

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

X_train = X_train.astype('float32')

X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

history = model.fit(X_train, y_train,
              batch_size=batch_size,
              epochs=nb_epoch,
              verbose=1,
              validation_split=1/6)

# Score trained model.
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])  # -*- coding: utf-8 -*-
