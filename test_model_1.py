# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 19:51:28 2020

@author: tonim
"""
#trial model 1:
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import keras 
from keras.models import Model
from keras.layers import Input,Conv2D, Dense, MaxPool2D, Dropout, Flatten, Concatenate, AvgPool2D, Dropout
from keras.optimizers import Adam, Adagrad, SGD, RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.utils import plot_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

trdata = ImageDataGenerator(validation_split=0.2)
traindata = trdata.flow_from_directory(directory="train",target_size=(224,224),batch_size=4, shuffle=True, subset=('training'), class_mode='categorical')
valdata = trdata.flow_from_directory(directory="train",target_size=(224,224), batch_size=4,shuffle=True, subset=('validation'), class_mode='categorical')
tsdata = ImageDataGenerator()
# testdata = tsdata.flow_from_directory(directory="test", shuffle= False, target_size=(224,224), class_mode='categorical')
testdata = tsdata.flow_from_directory(directory="test",batch_size=1476, shuffle=True, target_size=(224,224), class_mode='categorical')
class_names=["cat", "dog"]
print('Model making is under process')
act='tanh'
def conv_block(x,filters):
    c1=Conv2D(filters=filters[0], kernel_size=(1,5), activation=act,padding='same')(x)
    c2=Conv2D(filters=filters[1], kernel_size=(5,1), activation=act,padding='same')(x)
    c_o=Concatenate(axis=3)([c1,c2])
    return c_o
ins=Input(shape=(224,224,3))

l1=(conv_block(ins,[32,32]))# if filters=[32,32]-->filter[0]=32, filter[1]=32
l2=(AvgPool2D(pool_size=(2,2), strides=(2)))(l1)
l3=(conv_block(l2,filters=[32,32]))
l4=(AvgPool2D(pool_size=(2,2), strides=(2)))(l3)
l5=(Flatten())(l4)
l6=(Dense(120,activation='relu'))(l5)
l7=(Dense(84, activation='relu'))(l6)
l8=(Dense(2, activation='softmax'))(l7)# 1 can be replaced with 2 or more when it itsnt a binary classification anymore
M1=Model(inputs=ins, outputs=l8)
print(M1.summary())
#plot_model(M1)
#('Failed to import pydot. You must `pip install pydot` and install graphviz (https://graphviz.gitlab.io/download/), ', 'for `pydotprint` to work.'
M1.compile(optimizer=RMSprop(learning_rate=0.001),loss='categorical_crossentropy',metrics=['accuracy'])
print('Model built')
#%%---------------------------------------------------------------------------------------------------------
#model = M1(weights='imagenet')

# train the model on the new data for a few epochs
callback=EarlyStopping(
    monitor="val_loss",
    min_delta=0,
    patience=40,
    verbose=1,
    mode="auto",
    baseline=None,
    restore_best_weights=False,
)
hist = M1.fit( x = traindata, epochs = 100, verbose =1, validation_data=valdata,steps_per_epoch = 100)
#hist=M1.fit_generator(steps_per_epoch=(100),generator=traindata,validation_data=valdata, validation_steps=(10),epochs=500, verbose=1, callbacks=callback)
              
plt.plot(hist.history["accuracy"])#
plt.plot(hist.history['val_accuracy'])
plt.plot(hist.history["loss"])
plt.plot(hist.history["val_loss"])
plt.title("model")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(["Accuracy","Validation Accuracy","loss", "val_loss"])
plt.show()
#%%------------------------------------------------------------------
# #evaluate
#hist=M1.evaluate(x=testdata,verbose=1)
# plt.plot(hist.history["accuracy"])
# plt.show()

#
#evaluate model
# predict what the test data is
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
