# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 12:49:59 2020

@author: tonim
"""
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 09:01:01 2020
Models 1 and 2
@author: Tonima
"""
#%% Data import and prep
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

trdata = ImageDataGenerator(validation_split=0.2, rescale=(1./255),featurewise_center=True)
traindata = trdata.flow_from_directory(directory="Torso",target_size=(224,224),batch_size=4, shuffle=True, subset=('training'), class_mode='categorical',color_mode='grayscale')
valdata = trdata.flow_from_directory(directory="Torso",target_size=(224,224), batch_size=4,shuffle=True, subset=('validation'), class_mode='categorical', color_mode='grayscale')
tsdata = ImageDataGenerator(rescale=(1./255))
# testdata = tsdata.flow_from_directory(directory="test", shuffle= False, target_size=(224,224), class_mode='categorical')
testdata = tsdata.flow_from_directory(directory="Torso_test", shuffle=True,batch_size=786, target_size=(224,224), class_mode='categorical',color_mode='grayscale')
class_names=[ "DV lower","DV Upper", "LAT Lower","LAT Upper",]

#%% trial model 1
from kerastuner import HyperParameters
from kerastuner.tuners import Hyperband, RandomSearch
print('Model making is under process')
def model_builder(hp):
    act='swish'
    hp_units=hp.Int('units', min_value = 60, max_value = 120, step = 4)#optimal value from min to max is chosen
    hp_units2=hp.Int('units', min_value = 42, max_value = 84, step = 2)#optimal value from min to max is chosen
    hp_lr=hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, step=1e-4)

    def conv_block(x,filters):
        c1=Conv2D(filters=filters[0], kernel_size=(1,5), activation=act,padding='same')(x)
        c2=Conv2D(filters=filters[1], kernel_size=(5,1), activation=act,padding='same')(x)
        c_o=Concatenate(axis=3)([c1,c2])
        return c_o
    def reduc_block(x):
        x=Conv2D(8,(1,1), activation=act, padding='same')(x)
        c1=Conv2D(filters=32, kernel_size=(1,1),activation=act,padding='same')(x)
        c11=Conv2D(filters=32, kernel_size=(3,3),activation=act,padding='same')(c1)
        c12=Conv2D(filters=32, kernel_size=(3,3),activation=act,padding='same')(c11)
        c2=Conv2D(filters=32, kernel_size=(3,3),activation=act,padding='same')(x)
        c3=MaxPool2D((1))(c2)
        out=Concatenate(axis=1)([c12,c2,c3])
        #out=MaxPool2D(3,3)(out)
        return out
    def fire_module(x):
        # Squeeze layer
        x1 = Convolution2D(4,(1,1),activation=(act),padding='same')(x)
        # Expand layer 1x1 filters
        c1 = Conv2D(16, (1,1), activation=(act),padding='same')(x1)
        # Expand layer 3x3 filters
        c2 = Conv2D(16, (3,3),activation=(act), padding='same')(x1)
        # concatenate outputs
        y = Concatenate(axis=1)([c1,c2])
        return y
    ins=Input(shape=(224,224,1))

    l1=(conv_block(ins,[32,32]))# if filters=[32,32]-->filter[0]=32, filter[1]=32
    l1=fire_module(l1)
    l2=(AvgPool2D(pool_size=(2,2), strides=(2)))(l1)
    l31=(conv_block(l2,filters=[32,32]))
    l32=(reduc_block(l31))
    l4=(AvgPool2D(pool_size=(2,2), strides=(2)))(l32)
    l5=(Flatten())(l4)
    l5=Dropout(0.2)(l5)
    l6=(Dense(units=hp_units,activation='relu'))(l5)
    l7=(Dense(units=hp_units2, activation='relu'))(l6)
    l8=(Dense(4, activation='softmax'))(l7)# 1 can be replaced with 2 or more when it itsnt a binary classification anymore
    M1=Model(inputs=ins, outputs=l8)
    print(M1.summary())
    M1.compile(optimizer=Adagrad(learning_rate=hp_lr), loss='categorical_crossentropy',metrics=['accuracy'])
    return M1
tuner = RandomSearch(model_builder,tune_new_entries=True,objective='val_accuracy',executions_per_trial=1, max_trials=2,overwrite=(True))
#tuner.search_space_summary()
hist=tuner.search(x = traindata, epochs =2, verbose =1, validation_data=valdata, steps_per_epoch=10)
#hist=M1.fit(traindata, epochs = 1, verbose =1, validation_data=valdata, steps_per_epoch=50)
print('Model built')
best_model = tuner.get_best_models(num_models=1)[0]
best_model.save('bestmodel.h5')
#%% trial model 2
import numpy 
from kerastuner import HyperParameters
from keras import utils
from kerastuner.tuners import Hyperband, RandomSearch
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
    #plot_model(M1)
    #('Failed to import pydot. You must `pip install pydot` and install graphviz (https://graphviz.gitlab.io/download/), ', 'for `pydotprint` to work.'
    M2.compile(optimizer=Adagrad(learning_rate=hp_lr2), loss='categorical_crossentropy',metrics=['accuracy'])
    return M2
tuner2 = RandomSearch(model_builder2,tune_new_entries=True,objective='val_accuracy',executions_per_trial=2, max_trials=2,overwrite=(True))
#tuner.search_space_summary()
hist=tuner2.search(x = traindata, epochs = 3, verbose =1, validation_data=valdata, steps_per_epoch=20)
#hist=M1.fit(traindata, epochs = 1, verbose =1, validation_data=valdata, steps_per_epoch=50)
print('Model built')
best_model2 = tuner2.get_best_models(num_models=1)[0]
best_model2.save('bestmodel2.h5')
#%% training
from pytictoc import TicToc
t = TicToc() #create instance of class
t.tic() #Start timer
print(best_model2.summary())
callback=EarlyStopping(monitor="val_loss",min_delta=0,patience=40,verbose=1,mode="auto",baseline=None,restore_best_weights=False)
#hist1 = best_model.fit(x = traindata, epochs =100, verbose =1, validation_data=valdata,steps_per_epoch = 25)

hist2 = best_model2.fit(x = traindata, epochs =100, verbose =1, validation_data=valdata,steps_per_epoch = 50)
t.toc()            
# plt.plot(hist1.history["accuracy"])#
# plt.plot(hist1.history['val_accuracy'])
# #plt.plot(hist.history["loss"])
# #plt.plot(hist.history["val_loss"])
# plt.title("model 1")
# plt.ylabel("Accuracy")
# plt.xlabel("Epoch")
# plt.legend(["Accuracy","Validation Accuracy","loss", "val_loss"])
# plt.show()
plt.plot(hist2.history["accuracy"])#
plt.plot(hist2.history['val_accuracy'])
#plt.plot(hist.history["loss"])
#plt.plot(hist.history["val_loss"])
plt.title("model 2")
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
inputs=testdata
data, tlabel= inputs.next()
meh=numpy.array([[1,1,1,1]])#used to generate % of class belonging

prediction1=np.zeros((len(data),4))
prediction2=np.zeros((len(data),4))
for i in range(len(data)):
   image = data[i].reshape(1,224,224,1)
   #image = image/255
   #prediction1[i] = best_model.predict(image)
   prediction2[i] = best_model2.predict(image)
#p1 = np.argmax(prediction1,axis=1)
p2 = np.argmax(prediction2,axis=1)
test_label=np.argmax(tlabel,axis=1)
#cf_matrix1=confusion_matrix(test_label,p1)
cf_matrix2=confusion_matrix(test_label,p2)
#disp1=ConfusionMatrixDisplay(cf_matrix1,display_labels=(class_names))
disp2=ConfusionMatrixDisplay(cf_matrix2,display_labels=(class_names))
#disp1 = disp1.plot()
disp2 = disp2.plot()
plt.show()
# print("-----------------------------------------------------------------------") 
# print(cf_matrix1, cf_matrix2) 
# print("-----------------------------------------------------------------------")

# print("Precision, Recall, F1-score:")
# print(classification_report(test_label,p1, target_names=class_names))

print("Precision, Recall, F1-score:")
print(classification_report(test_label,p2, target_names=class_names))

misclassification2=test_label-p2
def condition(misclassification2):return misclassification2!=0
output=[idx for idx, element in enumerate(misclassification2) if condition (element)]
print(output)
plt.figure(figsize=(4,4))
for images,labels in testdata:
    for i in range (output):
        ax=plt.subplot(3,3,i+1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(int(labels[i]))
        #plt.axis("off")
#%%
#%% testing single image
from keras.preprocessing import image
import numpy as np

# select a sample image path
img_path8="D:/x ray/#CLASSES/Torso_test/LAT Upper/1.2.826.0.1.3680043.2.876.17357.1.6.0.20200817105801.2.1.jpg"
#img_path8="D:/x ray/#CLASSES/Torso_test/DV Lower/1.2.826.0.1.3680043.2.876.8248.1.3.1.20170511212235.2.39.jpg" 
#img_path8="D:/x ray/#CLASSES/Torso_test/DV Upper/1.2.276.0.7230010.3.0.3.5.1.11072342.4025358845.jpg"
#img_path8="D:/x ray/#CLASSES/Torso_test/LAT Lower/1.2.276.0.7230010.3.0.3.5.1.11215350.830898152.jpg" #sample DV upper image
imgxray=image.load_img(img_path8, target_size=(224,224,1),color_mode='grayscale')
imgxray.show()

img = cv2.imread(img_path8,0)
edges = cv2.Canny(img,100,200)

plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

imgxray=np.asarray(imgxray)
imgxray=np.expand_dims(imgxray,axis=0)
imgxray=np.rot90(imgxray,1,(1,2))
#resize or not to resize
#imgxray=imgxray.reshape(1,224,224,1)
y=best_model2.predict(imgxray)
print('prediction probability is: ')
print(y)
class_labels = list(traindata.class_indices.keys())  
0
y_predict = np.argmax(y,axis=1)
print("model prediction is: ",class_labels[y_predict[0]])
#%%
import tensorflow as tf
#meh=numpy.array([[1,1,1,1]])
#or
meh1=numpy.array([[1,0,0,0]])
meh2=numpy.array([[0,1,0,0]])
meh3=numpy.array([[0,0,1,0]])
meh4=numpy.array([[0,0,0,1]])
testlocal="D:/x ray/#CLASSES/Torso_test/LAT Upper/1.2.826.0.1.3680043.2.876.17440.2.5.1.20201101122147.2.3.jpg"
img = keras.preprocessing.image.load_img(
    testlocal, target_size=(224,224), color_mode='grayscale')
img_array = keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)  # Create batch axis

PRED = best_model2.predict(img_array)
score = PRED[0]
a1=np.max(100*np.multiply(meh1,score))
a2=np.max(100*np.multiply(meh2,score))
a3=np.max(100*np.multiply(meh3,score))
a4=np.max(100*np.multiply(meh4,score))
print(
    "This image is %.2f percent DV Lower, %.2f percent DV Upper,%.2f percent LAT Lower and %.2f percent LAT Upper."
    % (a1,a2,a3,a4)
)
#or
# ans=(np.argmax(np.multiply(meh,score)))
# print("The image belongs to class %.2f"
#       % (ans)
# )