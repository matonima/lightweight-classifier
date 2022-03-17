# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 18:59:50 2020

@author: tonim
"""
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import keras 
from keras.models import Model
from keras.layers import Input,Conv2D, Dense, MaxPool2D, Dropout, Flatten, Concatenate, AvgPool2D, Dropout
from keras.optimizers import Adam, Adagrad, SGD, Nadam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.utils import plot_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
def conv3_1_3(x,filters,act):
    c1=Conv2D(filters=filters[0], kernel_size=(1,3), activation=act,padding='same')(x)
    c2=Conv2D(filters=filters[0], kernel_size=(3,1), activation=act,padding='same')(c1)
    return c2

def conv3_1_conc(x,filters,act):
    c1=Conv2D(filters=filters, kernel_size=(1,3), activation=act,padding='same')(x)
    c2=Conv2D(filters=filters, kernel_size=(1,3), activation=act,padding='same')(x)
    conc=Concatenate(axis=3)([c1,c2])
    return conc
trdata = ImageDataGenerator(validation_split=0.2)
traindata = trdata.flow_from_directory(directory="train",batch_size=4,target_size=(224,224), shuffle=True, subset=('training'), class_mode='categorical')
valdata = trdata.flow_from_directory(directory="train",batch_size=4,target_size=(224,224), shuffle=True, subset=('validation'), class_mode='categorical')
tsdata = ImageDataGenerator()
testdata = tsdata.flow_from_directory(directory="test",batch_size=1476, shuffle=True, target_size=(224,224), class_mode='categorical')
class_names=["cat", "dog"]
print('Model making is under process')
act='tanh'
ins=Input(shape=(224,224,3))

#----------------------------------------
l1a=conv3_1_3(ins, [64], act)
l1b=conv3_1_3(ins, [64], act)
l1=Concatenate(axis=3)([l1a,l1b])
l2=MaxPool2D(pool_size=(4,4))(l1)
l3=Conv2D(filters=64,kernel_size=3,strides=1,padding='same')(l2)
l4=conv3_1_conc(l3, 64, act)
l5=AvgPool2D(pool_size=(2,2),strides=4)(l4)
l5=(Flatten())(l5)
l6=(Dense(120,activation='relu'))(l5)
l7=(Dense(84, activation='relu'))(l6)
l8=(Dense(2, activation='softmax'))(l7)
M1=Model(inputs=ins, outputs=l8)
print(M1.summary())
#plot_model(M1)
#('Failed to import pydot. You must `pip install pydot` and install graphviz (https://graphviz.gitlab.io/download/), ', 'for `pydotprint` to work.'
M1.compile(optimizer=Adagrad(learning_rate=0.009),loss='categorical_crossentropy',metrics=['accuracy'])
print('Model built')
#%%
callback=EarlyStopping(monitor="val_loss",min_delta=0,patience=40,verbose=1,mode="auto",baseline=None,restore_best_weights=False)
hist = M1.fit(x = traindata, epochs =500, verbose =1, validation_data=valdata,steps_per_epoch = 100)
            
plt.plot(hist.history["accuracy"])#
plt.plot(hist.history['val_accuracy'])
#plt.plot(hist.history["loss"])
#plt.plot(hist.history["val_loss"])
plt.title("model")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(["Accuracy","Validation Accuracy","loss", "val_loss"])
plt.show()
#%%

from keras.utils.np_utils import to_categorical
import os, cv2
from sklearn.utils import shuffle
import sklearn
from sklearn.model_selection import train_test_split

#TEST_DIR_cat = 'C:/Users/PC/Desktop/Fatemeh/test_Fatemeh/cat/'
TEST_DIR_cat = 'C:/Users/tonim/cnn/test/cat/'
TEST_DIR_dog = 'C:/Users/tonim/cnn/test/dog/'
direct=('C:/Users/tonim/cnn/test')
ROWS = 224
COLS = 224
CHANNELS = 3
test_image_cat=[TEST_DIR_cat+i for i in os.listdir(TEST_DIR_cat)]
print('number of cat images=')
print(len(test_image_cat))
test_image_dog=[TEST_DIR_dog+i for i in os.listdir(TEST_DIR_dog)]
print('number of dog images=')
print(len(test_image_dog))
m = len(test_image_cat)+len(test_image_dog)

inputs=testdata
#test_label = inputs.classes
data, tlabel= inputs.next()

prediction=np.zeros((len(data),2))
for i in range(len(data)):
   image = data[i].reshape(1,224,224,3)
   #image = image/255
   prediction[i] = M1.predict(image)

#prediction=M1.predict(inputs)
# p=np.zeros((m,2))
# for i in range(m):
#     if (prediction[i,0])>0.5:
#         p[i,0]=1 
#     else:
#         p[i,1]=1
p = np.argmax(prediction,axis=1)
test_label=np.argmax(tlabel,axis=1)
cf_matrix=confusion_matrix(test_label,p)
disp=ConfusionMatrixDisplay(cf_matrix,display_labels=(class_names))
disp = disp.plot()
plt.show()
print("-----------------------------------------------------------------------") 
print(cf_matrix) 
print("-----------------------------------------------------------------------")

print("Precision, Recall, F1-score:")
print(classification_report(test_label,p, target_names=class_names))
