# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 09:36:23 2020

@author: tonim
"""
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np

model=VGG16(include_top=False, weights='imagenet')
img_path = '1.jpg'
img=image.load_img(img_path,grayscale=False, color_mode='rgb', target_size=(224,224) )
x=image.img_to_array(img)
x=np.expand_dims(x, axis=0)
x=preprocess_input(x)
f=model.predict(x)
