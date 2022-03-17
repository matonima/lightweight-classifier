# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 12:40:18 2020

@author: tonim
"""
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
# create the base pre-trained model
base_model = InceptionV3(weights='imagenet', include_top=False)

 # add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
# and a logistic layer -- let's say we have 200 classes
predictions = Dense(2, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
from keras.optimizers import Adagrad
opt=Adagrad(learning_rate=0.001)
#'rmsprop')
model.compile(optimizer=opt, loss='categorical_crossentropy',metrics='accuracy')
trdata = ImageDataGenerator(validation_split=0.35)
traindata = trdata.flow_from_directory(directory="train",target_size=(224,224), shuffle=True, subset=('training'), class_mode='categorical')
valdata = trdata.flow_from_directory(directory="train",target_size=(224,224), shuffle=True, subset=('validation'), class_mode='categorical')

tsdata = ImageDataGenerator()
testdata = tsdata.flow_from_directory(directory="test", target_size=(224,224), class_mode='categorical')
class_names=["cat", "dog"]

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
#hist = model.fit( x = traindata, epochs = 10, verbose =1, validation_data=valdata, shuffle =True ,steps_per_epoch = 100)
hist=model.fit_generator(steps_per_epoch=(100),generator=traindata,validation_data=valdata, validation_steps=(10),epochs=500, verbose=1, callbacks=callback)
              
from matplotlib import pyplot as plt
plt.plot(hist.history["accuracy"])#
plt.plot(hist.history['val_accuracy'])
#plt.plot(hist.history["loss"])
#plt.plot(hist.history["val_loss"])
plt.title("model")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(["Accuracy","Validation Accuracy","loss", "val_loss"])
plt.show()
  
# #evaluate
# hist=model.evaluate(x=testdata,verbose=1)
# plt.plot(hist.history["accuracy"])
# plt.show()

  
#evaluate model
from sklearn.metrics import classification_report,confusion_matrix,plot_confusion_matrix, ConfusionMatrixDisplay
# predict what the test data is
inputs=testdata
prediction=model.predict(inputs)
pred=prediction #for vector format, extra matrix created
y_test = inputs.classes #classes from the test data set
y_label=pred # y_test==in vector form
p=[1]*(len(pred))
for i in range(len(pred)):
    if (prediction[i,1])>0.5:
       p[i]=0 #scalar
       #print(p[i])
       #pred[i,:]=[1,0]#vector
       #print(pred[i,:])from tensorflow
    else:
       p[i]=1
       #pred[i,:]=[0,1]
       #print(pred[i,:])
#vector test class assignment       
# for i in range(len(pred)):
#    # print(y_test[i])
#     if (y_test[i])<1:
#         y_label[i]=[1,0]
#     else:
#         y_label[i]=[0,1]
# print(y_label)
#predi=np.argmax(pred)
#display=plot_confusion_matrix(estimator=InceptionV3 ,X=p, y_true=y_test, display_labels=('class_names'))
c=confusion_matrix(p,y_test)
disp=ConfusionMatrixDisplay(c,display_labels=(class_names))

disp = disp.plot()
plt.show()
print("-----------------------------------------------------------------------") 
print(c) 
print("-----------------------------------------------------------------------")
TP=np.diag(c) #diagonal is the true positive
FP=np.sum(c,axis=0)-TP #sum of all column -true positives
FN=np.sum(c,axis=1)-TP #sum of rows -TP
TN=[]
for i in range(2):
	temp=np.delete(c,i,0) #delete ith row
	temp=np.delete(temp,i,1) #delete ith column
	TN.append(sum(sum(temp)))
    
for i in range(2):
	print(TP[i]+FP[i]+FN[i]+TN[i]==len(prediction))
precision=TP/(TP+FP)
recall=TP/(TP+FN)
specificity=TN/(TN+FP)
accuracy=(TP+TN)/(TP+TN+FP+FN)
f1score=2*(precision*recall)/(precision+recall)
#classification report from calculated values:
# from astropy.table import QTable, Table, 
# classification=QTable()
# classification['precision']=precision
# classification['f1score']=f1score
# classification=Table(names='precision','f1score'),dtype=('f4','i4')
# print(classification)
print(classification_report(y_test,p, target_names=class_names))
# Visualization of confusion matrix
# import pandas as pd
# df_cm = pd.DataFrame(c, class_names)
# fig = plt.figure() 
# for i in range(1): 
#  plt.subplot(3,3,i+1) 
#  plt.tight_layout() 
#  plt.imshow(testdata[i], cmap='gray', interpolation='none') 
#  title = "Class: "+class_names[int(predi[i])] 
#  plt.title(title) 
#  plt.xticks([]) 
#plt.yticks([])
