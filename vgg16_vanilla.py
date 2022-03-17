# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 10:19:09 2020

@author: tonim
"""
import keras,os
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D , Flatten
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
##image_p=ImageDataGenerator()
##Image=image_p.flow_from_directory(directory="image_path", target_size=(224,224))
##path="image_path/"
trdata = ImageDataGenerator()
traindata = trdata.flow_from_directory(directory="train",target_size=(224,224))
tsdata = ImageDataGenerator()
testdata = tsdata.flow_from_directory(directory="test", target_size=(224,224))

model=Sequential()
#layer 1
model.add(Conv2D(input_shape=(224,224,3),filters=64, kernel_size=(3,3), padding="same", activation="sigmoid"))
model.add(Conv2D(filters=64,kernel_size=(3,3),activation="sigmoid"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
##input size needs not to be mentioned as the maxpool causes output to be 112*112 automatically this is the inpu for next layer
#layer 2
model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="sigmoid"))
model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="sigmoid"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
#layer 3
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="sigmoid"))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="sigmoid"))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="sigmoid"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
#layer 4
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="sigmoid"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="sigmoid"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="sigmoid"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
#layer 5
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="sigmoid"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="sigmoid"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="sigmoid"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
##output is 7*7

#layer 6
model.add(Flatten())
model.add(Dense(units=4096,activation="sigmoid"))
model.add(Dense(units=4096, activation="sigmoid"))
model.add(Dense(units=2,activation="softmax"))

from keras.optimizers import Adam
opt=Adam(learning_rate=0.001)
model.compile(optimizer=opt, loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])

model.summary()

#model checkpoint and early stopping imported in case of saturation
from keras.callbacks import ModelCheckpoint, EarlyStopping
checkpoint=ModelCheckpoint("vgg16_1.h5", monitor='val_acc',verbose=1,save_best_only=(True),save_weights_only=(False),mode='auto',save_freq=1)
early=EarlyStopping(monitor='val_acc',min_delta=0, patience=5, verbose=1, mode='auto')
hist=model.fit_generator(steps_per_epoch=(100),generator=traindata,validation_data=(testdata), validation_steps=(10),epochs=10, callbacks=[checkpoint,early])
#hist = Model.fit_generator(generator=traindata, steps_per_epoch=100, validation_data=testdata, validation_steps=10, epochs=100, callbacks=[checkpoint,early])
#plot
from matplotlib import pyplot as plt
plt.plot(hist.history["acc"])
plt.plot(hist.history['val_acc'])
plt.title("model data")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(["Accuracy","Validation Accuracy"])
plt.show()

from tensorflow.keras import backend as K

import tensorflow as tf

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

config = tf.Session()
