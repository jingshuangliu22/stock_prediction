from __future__ import print_function

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.layers import Merge
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from load_data import load_data, evaluate_merge, load_pkl, load_idx
import cPickle as pkl
import numpy as np

from keras import optimizers


# set parameters:
filters = 300 
kernel_size = 4

max_features = 10000
maxlen = 12
embedding_dims = 300
filters2 = 300
kernel_size2 = 7

batch_size = 16
hidden_dims = 300
epochs = 25

print('Loading data...')
(x_train_text, y_train_text), (x_dev_text, y_dev_text), (x_test_text, y_test_text) = load_pkl(path="stock_week_vecs.pkl")
(x_train_idx, y_train_idx), (x_dev_idx, y_dev_idx),(x_test_idx, y_test_idx) = load_idx(path="stock_feature_close.pkl",num_words=max_features,maxlen=maxlen)

print(len(x_train_text), 'train_text sequences')
print(len(x_dev_text), 'dev_text sequences')
print(len(x_test_text), 'test_text sequences')

print('Pad sequences (samples x time)')
x_train_idx = sequence.pad_sequences(x_train_idx, maxlen=maxlen)
x_dev_idx = sequence.pad_sequences(x_dev_idx, maxlen=maxlen)
x_test_idx = sequence.pad_sequences(x_test_idx, maxlen=maxlen)
print(len(x_train_idx), 'train_idx sequences')
print(len(x_dev_idx), 'dev_idx sequences')
print(len(x_test_idx), 'test_idx sequences')


print('x_train_text shape:', x_train_text.shape)
print('x_test_text shape:', x_test_text.shape)

print('x_train_idx shape:', x_train_idx.shape)
print('x_test_idx shape:', x_test_idx.shape)


input_shape=x_train_text[0].shape
print('Build model...')
#text model
model1 = Sequential()
model1.add(Conv1D(filters,
                 kernel_size,
                 activation='relu',
                 strides=1,
                 input_shape=input_shape))
model1.add(GlobalMaxPooling1D())

#index model
model2 = Sequential()
model2.add(Embedding(max_features,
                    embedding_dims,
                    input_length=maxlen))
model2.add(Dropout(0.3))
model2.add(Conv1D(filters2,
                 kernel_size2,
                 padding='valid',
                 activation='relu',
                 strides=1))
model2.add(GlobalMaxPooling1D())


merged_model = Sequential()
merged_model.add(Merge([model1, model2], mode='concat'))
merged_model.add(BatchNormalization())


merged_model.add(Dense(hidden_dims))
merged_model.add(Dropout(0.3))
merged_model.add(Activation('relu'))

'''
merged_model.add(Dense(200))
merged_model.add(PReLU())
merged_model.add(Dropout(0.2))
merged_model.add(BatchNormalization())

merged_model.add(Dense(200))
merged_model.add(PReLU())
merged_model.add(Dropout(0.2))
merged_model.add(BatchNormalization())
'''

merged_model.add(Dense(1))
merged_model.add(Activation('sigmoid'))


adam = optimizers.Adam(lr=0.001,decay=0.4)
merged_model.compile(loss='binary_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])
merged_model.fit([x_train_text, x_train_idx], y_train_idx,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=([x_dev_text, x_dev_idx], y_dev_idx))

#score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
evaluate_merge(merged_model,[x_test_text, x_test_idx], y_test_text)
