# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 15:33:16 2020

@author: shamik.tiwari
"""

import os
import numpy as np
import pandas as pd
#from scipy.misc import imread
from sklearn.metrics import accuracy_score

import tensorflow as tf
import keras
import keras.backend as K
from keras.layers.advanced_activations import LeakyReLU, PReLU

from keras import layers, models, optimizers
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.utils import to_categorical
import cv2
import glob

from scipy.stats import kurtosis
from scipy.stats import skew
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from scipy.stats import moment
from skimage.feature import greycomatrix, greycoprops
x1 = []
files = glob.glob ("E:\\SC\\data\\train\\benign\\*.jpg")
for myFile in files:
    #print(myFile)
    image = cv2.imread (myFile)
    image = cv2.cvtColor( image, cv2.COLOR_BGR2RGB )
    image=image[80:144,80:144, :]
    x1.append (image)
x2 = []
files = glob.glob ("E:\\SC\\data\\train\\malignant\\*.jpg")
for myFile in files:
    #print(myFile)
    image = cv2.imread (myFile)
    image = cv2.cvtColor( image, cv2.COLOR_BGR2RGB )
    image=image[80:144,80:144, :]
    x2.append (image)
    
x3 = []
files = glob.glob ("E:\\SC\\data\\test\\malignant\\*.jpg")
for myFile in files:
    #print(myFile)
    image = cv2.imread (myFile)
    image = cv2.cvtColor( image, cv2.COLOR_BGR2RGB )
    image=image[80:144,80:144, :]
    x3.append (image)

x4 = []
files = glob.glob ("E:\\SC\\data\\test\\malignant\\*.jpg")
for myFile in files:
    #print(myFile)
    image = cv2.imread (myFile)
    image = cv2.cvtColor( image, cv2.COLOR_BGR2RGB )
    image=image[80:144,80:144, :]
    x4.append (image)
    
data=np.concatenate((x1,x3,x2,x4),axis = 0)
t1=np.zeros(1740)[:,None]
t2=np.ones(1497)[:,None]
t=np.concatenate((t1,t2),axis=0)

data=np.array(data)
for i in range(3237):
 a=data[i,:,:,:]
 a = np.array(a, dtype=np.uint8)
 data[i,:,:,:]=cv2.cvtColor(a, cv2.COLOR_BGR2HSV)
fe=[]
for i in range(3237):
 a=data[i,:,:,:]
 ar=a[:,:,0]
 ag=a[:,:,1]
 ab=a[:,:,2]
 f1=skew(ar,axis=None)
 f2=kurtosis(ar,axis=None)
 f3=ar.mean()
 f4=ar.std()
 f5=moment(ar,moment=5,axis=None)
 f6=skew(ag,axis=None)
 f7=kurtosis(ag,axis=None)
 f8=ag.mean()
 f9=ag.std()
 f10=moment(ar,moment=5,axis=None)
 f11=skew(ab,axis=None)
 f12=kurtosis(ab,axis=None)
 f13=ab.mean()
 f14=ab.std()
 f15=moment(ar,moment=5,axis=None)
 a=data[i,:,:,:]
 a = np.array(a, dtype=np.uint8)
 a=cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
 f16=skew(a,axis=None)
 f17=kurtosis(a,axis=None)
 f18=a.mean()
 f19=a.std()
 f20=moment(a,moment=5,axis=None)
 glcm = greycomatrix(ar,distances=[5], angles=[0], levels=256,
                        symmetric=True, normed=True)
 f21=greycoprops(glcm, 'dissimilarity')[0][0]
 f22=greycoprops(glcm, 'correlation')[0][0]
 f23=greycoprops(glcm, 'contrast')[0][0]
 f24=greycoprops(glcm, 'energy')[0][0]
 f25=greycoprops(glcm,'homogeneity')[0][0]
 glcm = greycomatrix(ab,distances=[5], angles=[0], levels=256,
                        symmetric=True, normed=True)
 f26=greycoprops(glcm, 'dissimilarity')[0][0]
 f27=greycoprops(glcm, 'correlation')[0][0]
 f28=greycoprops(glcm, 'contrast')[0][0]
 f29=greycoprops(glcm, 'energy')[0][0]
 f30=greycoprops(glcm,'homogeneity')[0][0]
 glcm = greycomatrix(ag,distances=[5], angles=[0], levels=256,
                        symmetric=True, normed=True)
 f31=greycoprops(glcm, 'dissimilarity')[0][0]
 f32=greycoprops(glcm, 'correlation')[0][0]
 f33=greycoprops(glcm, 'contrast')[0][0]
 f34=greycoprops(glcm, 'energy')[0][0]
 f35=greycoprops(glcm,'homogeneity')[0][0]
 f=[f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14,f15,f21,f22,f23,f24,f25,f26,f27,f28,f29,f30,f31,f32,f33,f34]
 fe.append(f)
fe=np.asarray(fe)


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
fe = scaler.fit_transform(fe)
trainX,testX,trainY,testY = train_test_split(data,t,test_size=0.2,random_state=43)
trainX,validX,trainY,validY = train_test_split(trainX,trainY,test_size=0.1,random_state=17)
ftrainX,ftestX,ftrainY,ftestY = train_test_split(fe,t,test_size=0.2,random_state=17)
ftrainX,fvalidX,ftrainY,fvalidY = train_test_split(ftrainX,ftrainY,test_size=0.1,random_state=21)
trainX = trainX.astype('float64') / 255.0
testX =  testX.astype('float64') / 255.0
validX = validX.astype('float64') / 255.0



from keras.layers import Flatten, Dense, Activation, Conv2D, MaxPooling2D, Dropout, BatchNormalization
from keras.models import load_model, Sequential
from keras.callbacks import EarlyStopping,ReduceLROnPlateau, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import GaussianNoise
from keras.layers import Dropout
import matplotlib.pyplot as plt
from keras.layers import Conv2D, GlobalMaxPooling2D, Input, Dense, Flatten, concatenate
from keras.models import Model
import numpy as np
del model
img_input = Input(shape=(64,64, 3))  ## branch 1 with image input
x = Conv2D(16, (3, 3),activation='relu')(img_input)
x = BatchNormalization()(x)
x = Dropout(0.1)(x)
x = Conv2D(32, (3, 3),activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)
x = Conv2D(64, (3, 3),activation='relu')(x)
x = BatchNormalization()(x)
x = Conv2D(32, (3, 3),activation='relu')(x)
x=BatchNormalization()(x)
x = Dropout(0.2)(x)
x = Conv2D(16, (3, 3),activation='relu')(x)
x = Dropout(0.1)(x)
x = Flatten()(x)
out_a = Dense(1)(x)

num_input = Input(shape=(29,))        ## branch 2 with numerical input
x1 = Dense(40, activation='relu')(num_input)
x1 = Dropout(0.1)(x1)
x1= Dense(80, activation='relu')(x1)
x1 = Dropout(0.2)(x1)
x1= Dense(100, activation='relu')(x1)
x1 = Dropout(0.5)(x1)
x1= Dense(120, activation='relu')(x1)
x1 = Dropout(0.1)(x1)
out_b = Dense(1)(x1)

concatenated = concatenate([out_a,out_b])    ## concatenate the two branches
out = Dense(1, activation='sigmoid')(concatenated)
model = Model([img_input,num_input], out)
print(model.summary())
from keras.optimizers import Adam
optimizer = adam = Adam(lr=0.0025, epsilon = 1e-8, beta_1 = .9, beta_2 = .999)
# Compile the model
model.compile(optimizer = optimizer , loss = "binary_crossentropy", metrics=["accuracy"])
# Set a learning rate annealer
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=2, 
                                            verbose=1, 
                                            factor=0.2, 
                                            min_lr=0.00001)

### Just for sanity check
X = [trainX,ftrainX]
y = trainY
X1 = [validX,fvalidX]
y1 = validY
X2 = [testX,ftestX]
y2 = testY
history=model.fit(X, y,batch_size=64, epochs=100, verbose=1, validation_data = (X1,y1), callbacks=[learning_rate_reduction])
pred=model.predict(X2)
pred=pred.round()
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y2, pred))
print(classification_report(y2,pred))
plt.plot(history.history['acc'],'g--', linewidth=1, markersize=2)
plt.plot(history.history['val_acc'],'^k:', linewidth=1, markersize=2)
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train accuracy', 'test accuracy'], loc='upper left')
plt.grid()
plt.show()
# summarize history for loss
plt.plot(history.history['loss'],'b--', linewidth=1, markersize=2)
plt.plot(history.history['val_loss'],'^k:', linewidth=1, markersize=2)
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train loss', 'test loss'], loc='upper left')
plt.grid()
plt.show()

