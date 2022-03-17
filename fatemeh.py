# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 16:08:33 2020

@author: PC
"""
import tensorflow
from keras.models import Sequential
from tensorflow.keras.layers import Input, ReLU, BatchNormalization, Conv2D, Dense, MaxPool2D, AvgPool2D, GlobalAvgPool2D,Concatenate

from keras import backend as k
import tensorflow.keras.backend as K


config = tensorflow.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
tensorflow.compat.v1.keras.backend.set_session(tensorflow.compat.v1.Session(config=config));
#k.set_session(tensorflow..Session(config=config))

#%%

def bn_relu_conv(x,filter,kerne_size):
  x = BatchNormalization()(x)
  x = ReLU()(x)
  x = Conv2D(filters=filter,kernel_size=kerne_size,padding='same')(x)
  return x

def dense_block(tensor,k,reps):
  for _ in range(reps):
    x = bn_relu_conv(tensor,4*k,1)
    x = bn_relu_conv(x,k,3)
    print(tensor.shape)
    print(x.shape)
    tensor = Concatenate()([tensor,x])
  return tensor

def transition(x,theta):
  f = int(tensorflow.keras.backend.int_shape(x)[-1]*theta)
  x = bn_relu_conv(x,f,1)
  x = AvgPool2D(2,strides=2,padding='same')(x)
  return x

import tensorflow as tf


model=Sequential()
k=24
theta = 0.5
repetitions = 4, 4, 4, 4
#repetitions = 6, 12, 24, 16
model = Sequential()

inputs = tf.keras.Input(shape=(224,224,3))

#x = Conv2D(2*k,7,strides=2, padding='same')(inputs)
x = Conv2D(64,7,strides=2, padding='same')(inputs)
x = MaxPool2D(3,strides=2,padding='same')(x)
for reps in repetitions:
  d=dense_block(x,k,reps)
  x=transition(d,theta)
  #print(x.shape)

x=GlobalAvgPool2D()(d)
ououtputst=Dense(2,activation='softmax')(x)
#model = Sequential()
model = tf.keras.Model(inputs=[inputs], outputs=ououtputst)
model.summary()
#%%
import keras
from keras.optimizers import SGD
from keras.optimizers import Adam
OBJECTIVE_FUNCTION = 'categorical_crossentropy'
LOSS_METRICS = ['accuracy']
#adam=Adam(lr=0.001)
sgd = SGD(lr = 0.1, decay = 1e-4, momentum = 0.9, nesterov = True)
model.compile(optimizer = sgd, loss = OBJECTIVE_FUNCTION, metrics = LOSS_METRICS)

#%%
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.applications import imagenet_utils

data_generator = ImageDataGenerator(validation_split=0.2,rescale=1./255)
train_generator = data_generator.flow_from_directory(directory="C:/Users/tonim/cnn/train",target_size=(224,224), subset="training",class_mode='categorical',shuffle=True)
valid_generator = data_generator.flow_from_directory(directory="C:/Users/tonim/cnn/train",target_size=(224,224),subset="validation",class_mode='categorical',shuffle=True)
#%%
from keras.callbacks import ModelCheckpoint, EarlyStopping
checkpoint=ModelCheckpoint("DenseNet2.h5", monitor='val_acc',verbose=1,save_best_only=(True),save_weights_only=(False),mode='auto',save_freq=1)
early=EarlyStopping(monitor='val_acc',min_delta=0, patience=25, verbose=1, mode='auto')
#%%

hist=model.fit_generator(steps_per_epoch=(100),generator=train_generator,validation_data=(valid_generator), validation_steps=(10),epochs=10,shuffle=True)

#%%
from matplotlib import pyplot as plt
# summarize history for accuracy
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('DenseNet2 accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','validation'], loc='upper left')
plt.show()
#%%
# summarize history for loss
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('DenseNet2 loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','validation'], loc='upper left')
plt.show()
#%%

gener = ImageDataGenerator()
testdata = gener.flow_from_directory(directory="C:/Users/tonim/cnn/test",target_size=(224,224))


acc = model.evaluate(train_generator, steps=len(train_generator), verbose=0)
print('> %.3f' % (acc[1] * 100.0))
print(acc[0])

acc = model.evaluate(valid_generator, steps=len(valid_generator), verbose=0)
print('> %.3f' % (acc[1] * 100.0))
print(acc[0])

acc = model.evaluate_generator(testdata, steps=len(testdata), verbose=0)
print('> %.3f' % (acc[1] * 100.0))
print(acc[0])
#%%
#new
import os, cv2, itertools
import numpy as np
import pandas as pd
import cv2

import matplotlib.pyplot as plt

from keras.utils.np_utils import to_categorical

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout

from sklearn.utils import shuffle

import sklearn
from sklearn.model_selection import train_test_split

#TEST_DIR_cat = 'C:/Users/PC/Desktop/Fatemeh/test_Fatemeh/cat/'
TEST_DIR_cat = 'C:/Users/tonim/cnn/test/cat/'
TEST_DIR_dog = 'C:/Users/tonim/cnn/test/dog/'

ROWS = 224
COLS = 224
CHANNELS = 3
test_image_cat=[TEST_DIR_cat+i for i in os.listdir(TEST_DIR_cat)]
print(len(test_image_cat))
test_image_dog=[TEST_DIR_dog+i for i in os.listdir(TEST_DIR_dog)]
print(len(test_image_dog))

def read_image(file_path):
  #print(file_path)
  img = cv2.imread(file_path, cv2.IMREAD_COLOR)
  #print(img)
  return cv2.resize(img, (ROWS, COLS), interpolation=cv2.INTER_CUBIC)

def prep_data(test_image_cat, test_image_dog):
  m = len(test_image_cat)+len(test_image_dog)
  n_x = ROWS*COLS*CHANNELS
  
  X = np.ndarray((m,ROWS,COLS,CHANNELS), dtype=np.uint8)
  y = np.zeros((m,1))
  print("X.shape is {}".format(X.shape))

  for i,image_file in enumerate(test_image_cat):
    #print(image_file)
    image = read_image(image_file)
    X[i,:] = np.squeeze(image.reshape((ROWS, COLS, CHANNELS)))
    y[i,0] = 0

  i=i+1
  print(i)
  for j,image_file in enumerate(test_image_dog):
    #print(image_file)
    image = read_image(image_file)
    X[i,:] = np.squeeze(image.reshape((ROWS, COLS, CHANNELS)))
    y[i,0] = 1
    i = i+1
    
  if j%100 == 0 :
    print("Proceed {} of {}".format(i, m))
    
  print(i)
  return X,y

X_test, test_label = prep_data(test_image_cat, test_image_dog)
#%%
y_prediction=np.zeros((len(X_test),2))
for i in range(len(X_test)):
   image = X_test[i].reshape(1,224,224,3)
   image = image/255
   y_prediction[i] = model.predict(image)

y_predict = np.argmax(y_prediction,axis=1)

from sklearn.metrics import classification_report
print("Precision, Recall, F1-score:")
print(classification_report(test_label, y_predict))

from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
cf_matrix = confusion_matrix(test_label, y_predict)
import seaborn as sns
sns.heatmap(cf_matrix, annot=True,fmt='g',cmap='Blues')

#%%
model.save("DenseNet2.h5")
#%%