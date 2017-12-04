import logging
import os
import numpy as np
import keras
import pandas as pd
from PIL import Image
import cv2
import pickle

from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers import Dense, Dropout, Activation, MaxPooling2D, Flatten
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator

## pickeファイル読み込み
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

## 訓練データ読み込み
data = np.empty((0,3072), int)
labels = []
nsamples = 0
label_file = unpickle("cifar10/batches.meta")
label_names = label_file[b'label_names']

for i in range(1, 6):
    file_name = "cifar10/data_batch_" + str(i)
    tmp_data_batch = unpickle(file_name)
    
    tmp_data = tmp_data_batch[b"data"]
    tmp_labels = np.array(tmp_data_batch[b"labels"])
    tmp_nsamples = len(tmp_data)
    
    nsamples += tmp_nsamples
    labels = np.append(labels, tmp_labels)
    data = np.append(data, tmp_data, axis = 0)

## テストデータ読み込み
test_data_batch = unpickle("cifar10/test_batch")
test_data = test_data_batch[b"data"]
test_labels = np.array(test_data_batch[b"labels"])
test_nsamples = len(tmp_data)

## データセット
X_train = data.reshape(len(data), 3, 32, 32).transpose(0, 2, 3, 1).astype('float32') / 255
Y_train = to_categorical(labels, 10)
X_test = test_data.reshape(len(test_data), 3, 32, 32).transpose(0, 2, 3, 1).astype('float32')/255
Y_test = to_categorical(test_labels, 10)

##################
### kerasでCNN ###
##################
model = Sequential()

model.add(Convolution2D(32,3,input_shape=(32,32,3)))
model.add(Activation('relu'))
model.add(Convolution2D(32,3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Convolution2D(64,3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(10, activation='softmax'))

adam = Adam(lr=1e-4)
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=["accuracy"])

## 画像の水増し作業
datagen = ImageDataGenerator(
    width_shift_range  = 1./8.,
    height_shift_range = 1./8.,
    rotation_range     = 0.,
    shear_range        = 0.,
    zoom_range         = 0.,
)

## 学習
history = model.fit_generator(datagen.flow(X_train, Y_train, batch_size = 125),
                    epochs = 40,
                    verbose = 1,
                    steps_per_epoch = len(X_train)/125,
                    validation_data = (X_test, Y_test)
                    )

## 予測の確認                    
loss, acc = model.evaluate(X_test, Y_test, verbose=0)
print('Test loss:', loss)
print('Test acc:', acc)
