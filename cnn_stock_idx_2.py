'''This example demonstrates the use of Convolution1D for text classification.
Gets to 0.89 test accuracy after 2 epochs.
90s/epoch on Intel i5 2.4Ghz CPU.
10s/epoch on Tesla K40 GPU.
'''

from __future__ import print_function

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.datasets import imdb
from load_data import load_data, evaluate, load_idx, load_pkl
import cPickle as pkl
import numpy as np
from keras import optimizers


# set parameters:
max_features = 5000
maxlen = 5
batch_size = 64 
embedding_dims = 250
filters = 600 
kernel_size = 2 
hidden_dims = 600
epochs = 22 

print('Loading data...')
(x_train, y_train), (x_dev, y_dev),(x_test, y_test) = load_pkl(path="sh_5day_feature_close.pkl")

print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')
'''
print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

x_train = np.reshape(x_train, (x_train.shape[0],1,x_train.shape[1]))
x_test = np.reshape(x_test, (x_test.shape[0],10,x_test.shape[1]))
'''

print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)
input_shape=x_train[0].shape
print('Build model...')
model = Sequential()

# we add a Convolution1D, which will learn filters
# word group filters of size filter_length:
model.add(Conv1D(filters,
                 kernel_size,
                 activation='relu',
                 strides=2,
                 input_shape=input_shape))
# we use max pooling:
model.add(GlobalMaxPooling1D())

# We add a vanilla hidden layer:
model.add(Dense(hidden_dims))
model.add(Dropout(0.5))
model.add(Activation('relu'))

# We project onto a single unit output layer, and squash it with a sigmoid:
model.add(Dense(1))
model.add(Activation('sigmoid'))
adam = optimizers.Adam(lr=0.001,decay=0.1)
model.compile(loss='binary_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test))
#score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
evaluate(model,x_test,y_test)
