#trial model 1:
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
#import tensorflow as tf
#from tf
import keras 
from keras import Model
from keras.models import Sequential
from keras.layers import Input, Conv2D, Dense, MaxPool2D, Dropout, Flatten,Concatenate, AvgPool2D, Dropout, Flatten, Dense 
from keras.optimizers import Adam, Adagrad
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
#from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
trdata = ImageDataGenerator(validation_split=0.35)
traindata = trdata.flow_from_directory(directory="train",color_mode='rgb', batch_size=32, target_size=(224,224,3), shuffle=True, subset=('training'), class_mode='categorical')
valdata = trdata.flow_from_directory(directory="train",target_size=(224,224), shuffle=True, subset=('validation'), class_mode='categorical')
tsdata = ImageDataGenerator()
testdata = tsdata.flow_from_directory(directory="test", target_size=(224,224), class_mode='categorical')
class_names=["cat", "dog"]

act='tanh'
#act='relu'
k=5 #kernal size
model=Sequential()
def conv_block(x,filters):
    c1=Conv2D(filters=filters[0], kernel_size=(1,5), activation=act,padding='same')(x)
    c2= Conv2D(filters=filters[1], kernel_size=(5,1), activation=act,padding='same')(x)
    c_o=Concatenate()([c1,c2])
    return c_o
inputs=Input(shape=(224,224,3))
import tensorflow
print(tensorflow.keras.backend.int_shape(inputs))
l1=conv_block(inputs, [32,32])
l2=AvgPool2D(pool_size=2, strides=2)(l1)
l3=conv_block(l2,[32,32])
l4=AvgPool2D(pool_size=2,strides=2)(l3)
l4_a=Flatten()(l4)
l5=Dense(units=120,activation='tanh')(l4_a)#---dense layer 1
l5_a=Dense(units=84, activation='tanh')(l5)#---dense layer 2
l6=Dense(units=1000,activation='softmax')(l5)
model.build(input_shape=(224,224,3))
model.summary()
#trail 1 model
opt=Adagrad(learning_rate=0.001)
M1=Model(inputs=traindata,outputs=l6)
M1.build()
M1.summary()
M1.compile()(optimizer=opt, loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])

print(M1.summary())
from tf.keras.utils import plot_model 
plot_model(M1)


#model.add()#