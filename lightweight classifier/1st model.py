# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 19:27:15 2020

@author: tonim
"""
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from sklearn.datasets.samples_generator import make_circles
X, y = make_circles(n_samples=1000,
                    noise=0.1,
                    factor=0.2,
                    random_state=0)
X
X.shape
plt.figure(figsize=(5, 5))
plt.plot(X[y==0, 0], X[y==0, 1], 'ob', alpha=0.5)
plt.plot(X[y==1, 0], X[y==1, 1], 'xr', alpha=0.5)
plt.xlim(-1.5, 1.5)
plt.ylim(-1.5, 1.5)
plt.legend(['0', '1'])
plt.title("Blue circles and Red crosses")

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD

model=Sequential()
model.add(Dense(4, input_shape=(2,), activation='tanh'))##gmoidmodel.add(Dense(4, input_shape=(2,), activation='softmax'))
model.add(Dense(1, activation='sigmoid'))
model.compile(SGD(lr=0.5), 'binary_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=20)

hticks = np.linspace(-1.5, 1.5, 101)
vticks = np.linspace(-1.5, 1.5, 101)
aa, bb = np.meshgrid(hticks, vticks)
ab = np.c_[aa.ravel(), bb.ravel()]
c = model.predict(ab)
cc= c.reshape(aa.shape)
c1=np.resize(c,(aa.shape))

plt.figure(figsize=(5, 5))
plt.contourf(aa, bb,cc,map='bwr', alpha=0.2)
plt.plot(X[y==0, 0], X[y==0, 1], 'ob', alpha=0.5)
plt.plot(X[y==1, 0], X[y==1, 1], 'xr', alpha=0.5)
plt.xlim(-1.5, 1.5)
plt.ylim(-1.5, 1.5)
plt.legend(['0', '1'])
plt.title("Blue circles and Red crosse with reshape")

plt.figure(figsize=(5, 5))
plt.contourf(aa, bb,c1,map='bwr', alpha=0.2)
plt.plot(X[y==0, 0], X[y==0, 1], 'ob', alpha=0.5)
plt.plot(X[y==1, 0], X[y==1, 1], 'xr', alpha=0.5)
plt.xlim(-1.5, 1.5)
plt.ylim(-1.5, 1.5)
plt.legend(['0', '1'])
plt.title("Blue circles and Red crosseswith resize")