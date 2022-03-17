# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 12:41:53 2020

@author: tonim
"""
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import keras 
from keras.models import Model
from keras.layers import Input,Conv2D,Convolution2D, Dense, MaxPool2D, Dropout, Flatten, Concatenate, AvgPool2D, Dropout
from keras.optimizers import Adam, Adagrad, SGD, RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.utils import plot_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import cv2, glob
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
from os import listdir
from os.path import isfile, join
from kerastuner import HyperParameters
from keras import utils
from kerastuner.tuners import Hyperband, RandomSearch
#%% edge detection
path=Path(".")
mypath="D:/x ray/#CLASSES/knee/TPA suitable"
onlyfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]
images = np.empty(len(onlyfiles), dtype=object)
edges=np.zeros(len(onlyfiles), dtype=object)
im1=np.zeros(len(onlyfiles), dtype=object)
im2=np.zeros(len(onlyfiles), dtype=object)
op=np.zeros(len(onlyfiles), dtype=object)
mini=50
maxi=225
for n in range(0, len(onlyfiles)):
  images[n] = cv2.imread( join(mypath,onlyfiles[n]) )
  #im1[n]=cv2.fastNlMeansDenoisingColored(images[n],None,10,10,7,21)
  #im2[n]=cv2.bilateralFilter(im1[n],10,100,100)
  #edges[n]=cv2.Canny(im2[n],mini,maxi,apertureSize = 5,L2gradient = True)
  #op[n]=cv2.morphologyEx(edges[n],cv2.MORPH_OPEN, kernel)
  cv2.imwrite('D:/x ray/#CLASSES/knee/TPAedge/edges %.2f.jpg'%(n),images[n])
  # plt.subplot(121),plt.imshow(images[n],cmap = 'gray')
  # plt.title('Original Image'), plt.xticks([]), plt.yticks([])
  # plt.subplot(122),plt.imshow(edges[n],cmap = 'gray')
  # plt.title('Edge Image %.2f to %.2f pix val'%(mini,maxi)), plt.xticks([]), plt.yticks([])
#%% model training
print('Model making is under process')
def model_builder2(hp):    
    hp_units=hp.Int('units', min_value = 60, max_value = 120, step = 4)#optimal value from min to max is chosen
    hp_units2=hp.Int('units', min_value = 42, max_value = 84, step = 2)#optimal value from min to max is chosen
    hp_lr2=hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, step=1e-4)
    k_1x1="kernal_1x1"
    k_1x3="kernal_1x3"
    k_3x1="kernal_3x1"
    k_3x3="kernal_3x3"
    mx="maxpool"
    conc="concatenate"
    sq="squeeze"
    ex="expand"
    
    def conv3_1_3(x,filters,act):
        
        c1=Conv2D(filters=filters[0], kernel_size=(1,3), activation=act,padding='same')(x)
        c2=Conv2D(filters=filters[0], kernel_size=(3,1), activation=act,padding='same')(c1)
        return c2

    def conv3_1_conc(x,filters,act):
        
        c1=Conv2D(filters=filters, kernel_size=(1,3), activation=act,padding='same')(x)
        c2=Conv2D(filters=filters, kernel_size=(1,3), activation=act,padding='same')(x)
        conc=Concatenate(axis=3)([c1,c2])
        return conc
    def reduc_block(x):
        
        x=Conv2D(8,(1,1), activation=act, padding='same')(x)
        c1=Conv2D(filters=32, kernel_size=(1,1),activation=act,padding='same')(x)
        c11=Conv2D(filters=32, kernel_size=(3,3),activation=act,padding='same')(c1)
        c12=Conv2D(filters=32, kernel_size=(3,3),activation=act,padding='same')(c11)
        c2=Conv2D(filters=32, kernel_size=(3,3),activation=act,padding='same')(x)
        c3=MaxPool2D((1))(c2)
        out=Concatenate(axis=3)([c12,c2,c3])
        out=MaxPool2D(4,4)(out)
        return out
    def fire_module(x):
        
        # Squeeze layer
        x1 = Convolution2D(4,(1,1),activation=(act),padding='same')(x)
        # Expand layer 1x1 filters
        c1 = Conv2D(8, (1,1), activation=(act),padding='same')(x1)
        # Expand layer 3x3 filters
        c2 = Conv2D(8, (3,3),activation=(act), padding='same')(x1)
        # concatenate outputs
        y = Concatenate(axis=3)([c1,c2])
        return y
    act='swish'
    ins1=Input(shape=(224,224,1))
    ins=fire_module(ins1)
    l1a=conv3_1_3(ins, [32], act)
    l1b=conv3_1_3(ins, [32], act)
    l1=Concatenate(axis=3)([l1a,l1b])
    l2=MaxPool2D(pool_size=(4,4) )(l1)
    l3=Conv2D(filters=32,kernel_size=3,strides=1,padding='same')(l2)
    l3=reduc_block(l3)
    l4=conv3_1_conc(l3, 32, act)
    l5=AvgPool2D(pool_size=(2,2),strides=4)(l4)
    l5=(Flatten())(l5)
    l5=Dropout(0.2)(l5)
    l6=(Dense(units=hp_units,activation='relu'))(l5)
    l7=(Dense(units=hp_units2, activation='relu'))(l6)
    l8=(Dense(4, activation='softmax'))(l7)
    M2=Model(inputs=ins1, outputs=l8)
    print(M2.summary())
    M2.compile(optimizer=Adagrad(learning_rate=hp_lr2), loss='categorical_crossentropy',metrics=['accuracy'])
    return M2
tuner2 = RandomSearch(model_builder2,tune_new_entries=True,objective='val_accuracy',executions_per_trial=2, max_trials=2,overwrite=(True))
hist=tuner2.search(x = traindata, epochs = 3, verbose =1, validation_data=valdata, steps_per_epoch=20)
print('Model built')
best_model2 = tuner2.get_best_models(num_models=1)[0]
best_model2.save('bestmodel2.h5')
#%% training
from pytictoc import TicToc
t = TicToc() #create instance of class
t.tic() #Start timer
print(best_model2.summary())
callback=EarlyStopping(monitor="val_loss",min_delta=0,patience=40,verbose=1,mode="auto",baseline=None,restore_best_weights=False)
hist2 = best_model2.fit(x = traindata, epochs =100, verbose =1, validation_data=valdata,steps_per_epoch = 50)
t.toc()            

plt.plot(hist2.history["accuracy"])#
plt.plot(hist2.history['val_accuracy'])
plt.plot(hist.history["loss"])
plt.plot(hist.history["val_loss"])
plt.title("model")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(["Accuracy","Validation Accuracy","loss", "val_loss"])
plt.show()
max(hist2.history["accuracy"])
max(hist2.history['val_accuracy'])

#%% testing

from keras.utils.np_utils import to_categorical
import os, cv2
from sklearn.utils import shuffle
import sklearn
from sklearn.model_selection import train_test_split

plt.show()
inputs=
data, tlabel= inputs.next()
meh=numpy.array([[1,1,1,1]])#used to generate % of class belonging

prediction1=np.zeros((len(data),4))
prediction2=np.zeros((len(data),4))
for i in range(len(data)):
   image = data[i].reshape(1,224,224,1)
   #image = image/255
   prediction2[i] = best_model2.predict(image)
p2 = np.argmax(prediction2,axis=1)
test_label=np.argmax(tlabel,axis=1)
cf_matrix2=confusion_matrix(test_label,p2)
disp2=ConfusionMatrixDisplay(cf_matrix2,display_labels=(class_names))
disp2 = disp2.plot()
plt.show()
print("Precision, Recall, F1-score:")
print(classification_report(test_label,p2, target_names=class_names))
