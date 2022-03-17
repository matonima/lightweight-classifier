# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 13:49:55 2020
tuner example
@author: tonim
"""
import tensorflow
from tensorflow import keras
from tensorflow.keras import layers
from kerastuner.applications import HyperResNet
from kerastuner.tuners import RandomSearch, Hyperband
from kerastuner import HyperParameters
from keras.preprocessing.image import ImageDataGenerator
trdata = ImageDataGenerator(validation_split=0.2)
traindata = trdata.flow_from_directory(directory="train",batch_size=32,target_size=(224,224), shuffle=True, subset=('training'), class_mode='categorical')
valdata = trdata.flow_from_directory(directory="train",batch_size=32,target_size=(224,224), shuffle=True, subset=('validation'), class_mode='categorical')
tsdata = ImageDataGenerator()
testdata = tsdata.flow_from_directory(directory="test",batch_size=1476, shuffle=True, target_size=(224,224), class_mode='categorical')
class_names=["cat", "dog"]
model=HyperResNet(input_shape=(224,224,3), include_top=True, classes=(2))
hp = HyperParameters()
hp.Fixed('learning_rate', value=1e-4)

tuner = Hyperband(
    model,hyperparameters=hp,
    tune_new_entries=True,
    objective='val_accuracy',
    executions_per_trial=1,max_epochs=2, overwrite=(True))
#tuner.search_space_summary()
hist=tuner.search(x = traindata, epochs = 1, verbose =1, validation_data=valdata, steps_per_epoch=50)
models = tuner.get_best_models(num_models=2)
tuner.results_summary()

config = tensorflow.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
tensorflow.compat.v1.keras.backend.set_session(tensorflow.compat.v1.Session(config=config));