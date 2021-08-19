# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 12:52:36 2020

@author: shamik.tiwari
"""
import cv2
import numpy as np
import pandas as pd
import os
import pandas as pd
from scipy.stats import kurtosis
from scipy.stats import skew
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from scipy.stats import moment
from skimage.feature import greycomatrix, greycoprops
test = pd.read_csv('E:/skincancer/hmnist_28_28_RGB.csv')
fe=[]
test.head(10)
X = test.iloc[:,0:-1]
Y = test.iloc[:,-1]
#Y=Y-1
from collections import Counter
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler(random_state=0)
X, Y = ros.fit_resample(X, Y)
X.shape, Y.shape
X = np.array(X)
Y = np.array(Y)
X = X.reshape(X.shape[0],28,28,3)

for i in range(46935):
 a=X[i,:,:,:]
 a = np.array(a, dtype=np.uint8)
 X[i,:,:,:]=cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
for i in range(46935):
 a=X[i,:,:,:]
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
 a=X[i,:,:,:]
 a = np.array(a, dtype=np.uint8)
 a=cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
 f16=skew(a,axis=None)
 f17=kurtosis(a,axis=None)
 f18=a.mean()
 f19=a.std()
 f20=moment(a,moment=5,axis=None)
 glcm = greycomatrix(a,distances=[1], angles=[0, 45, 90], levels=256,
                    symmetric=True, normed=True)
 f21=greycoprops(glcm, 'dissimilarity')[0][0]
 f22=greycoprops(glcm, 'correlation')[0][0]
 f23=greycoprops(glcm, 'contrast')[0][0]
 f24=greycoprops(glcm, 'energy')[0][0]
 f25=greycoprops(glcm,'homogeneity')[0][0]
 glcm = greycomatrix(ab,distances=[1], angles=[0, 45, 90], levels=256,
                        symmetric=True, normed=True)
 f26=greycoprops(glcm, 'dissimilarity')[0][0]
 f27=greycoprops(glcm, 'correlation')[0][0]
 f28=greycoprops(glcm, 'contrast')[0][0]
 f29=greycoprops(glcm, 'energy')[0][0]
 f30=greycoprops(glcm,'homogeneity')[0][0]
 glcm = greycomatrix(ag,distances=[1], angles=[0, 45, 90], levels=256,
                        symmetric=True, normed=True)
 f31=greycoprops(glcm, 'dissimilarity')[0][0]
 f32=greycoprops(glcm, 'correlation')[0][0]
 f33=greycoprops(glcm, 'contrast')[0][0]
 f34=greycoprops(glcm, 'energy')[0][0]
 f35=greycoprops(glcm,'homogeneity')[0][0]
 f=[f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14,f15,f21,f22,f23,f24,f25]
 fe.append(f)
fe=np.asarray(fe)


from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
# Feature extraction
model = LogisticRegression()
rfe = RFE(model, 15)
fit = rfe.fit(fe, Y)
print("Num Features: %s" % (fit.n_features_))
print("Selected Features: %s" % (fit.support_))
print("Feature Ranking: %s" % (fit.ranking_))

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
fe = scaler.fit_transform(fe)

trainX,testX,trainY,testY = train_test_split(X,Y,test_size=0.2,random_state=17)
trainX,validX,trainY,validY = train_test_split(trainX,trainY,test_size=0.2,random_state=17)
ftrainX,ftestX,ftrainY,ftestY = train_test_split(fe,Y,test_size=0.2,random_state=17)
ftrainX,fvalidX,ftrainY,fvalidY = train_test_split(ftrainX,ftrainY,test_size=0.2,random_state=21)
trainX = trainX.astype('float64') / 255.0
testX =  testX.astype('float64') / 255.0
trainY = to_categorical(trainY)
testY = to_categorical(testY)

validX = validX.astype('float64') / 255.0
validY = to_categorical(validY)

from keras.layers import Flatten, Dense, Activation, Conv2D, MaxPooling2D, Dropout, BatchNormalization
from keras.models import load_model, Sequential
from keras.callbacks import EarlyStopping,ReduceLROnPlateau, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import GaussianNoise
from keras.layers import Dropout

def plot_model_history(model_history):
    fig, axs = plt.subplots(1,2,figsize=(15,5))
    # summarize history for accuracy
    axs[0].plot(range(1,len(model_history.history['acc'])+1),model_history.history['acc'])
    axs[0].plot(range(1,len(model_history.history['val_acc'])+1),model_history.history['val_acc'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1,len(model_history.history['acc'])+1),len(model_history.history['acc'])/10)
    axs[0].legend(['train', 'val'], loc='best')
    # summarize history for loss
    axs[1].plot(range(1,len(model_history.history['loss'])+1),model_history.history['loss'])
    axs[1].plot(range(1,len(model_history.history['val_loss'])+1),model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1,len(model_history.history['loss'])+1),len(model_history.history['loss'])/10)
    axs[1].legend(['train', 'val'], loc='best')
    plt.show()
    plot_model_history(history)




import matplotlib.pyplot as plt
from keras.layers import Conv2D, MaxPooling2D, Input, Dense, Flatten, concatenate
from keras.models import Model
import numpy as np
del model
img_input = Input(shape=(28, 28, 3))  ## branch 1 with image input
x = Conv2D(64, (3, 3),activation='relu')(img_input)
x=MaxPooling2D(2,2)(x)
x = Dropout(0.1)(x)
x = Conv2D(128, (3, 3),activation='relu')(x)
x=MaxPooling2D(2,2)(x)
x = Dropout(0.3)(x)
x = Conv2D(128, (3, 3),activation='relu')(x)
x = Dropout(0.2)(x)
x=BatchNormalization()(x)
x = Conv2D(256, (3, 3),activation='relu')(x)
x = Dropout(0.4)(x)
x=BatchNormalization()(x)
x = Flatten()(x)
out_a = Dense(7)(x)

num_input = Input(shape=(20,))        ## branch 2 with numerical input
x1 = Dense(40, activation='relu')(num_input)
x1 = Dropout(0.3)(x1)
x1= Dense(80, activation='relu')(x1)
x1 = Dropout(0.4)(x1)
x1= Dense(100, activation='relu')(x1)
x1 = Dropout(0.3)(x1)
x1= Dense(200, activation='relu')(x1)
x1 = Dropout(0.5)(x1)
x=BatchNormalization()(x)
out_b = Dense(7)(x1)

concatenated = concatenate([out_a,out_b]) 
out = Dense(7, activation='softmax')(concatenated)

model = Model([img_input,num_input,], out)
print(model.summary())
from keras.optimizers import Adam
optimizer = Adam(lr=0.00125, epsilon = 1e-8, beta_1 = .9, beta_2 = .999)
# Compile the model
model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])
# Set a learning rate annealer
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=2, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.0001)

### Just for sanity check
trainX1=np.concatenate((trainX,trainX,trainX), axis=0)
ftrainX1=np.concatenate((ftrainX,ftrainX,ftrainX), axis=0)
trainY1=np.concatenate((trainY,trainY,trainY), axis=0)
X1 = [trainX1,ftrainX1]
y1 = trainY1
X2 = [validX,fvalidX]
y2 = validY
testX=np.concatenate((trainX,validX,testX), axis=0)
ftestX=np.concatenate((ftrainX,fvalidX,ftestX), axis=0)
testY=np.concatenate((trainY,validY,testY), axis=0)
X3 = [testX,ftestX]
y3 = testY

history=model.fit(X1, y1,batch_size=32, epochs=20, verbose=1, validation_data = (X2,y2), callbacks=[learning_rate_reduction])
pred=np.round(model.predict(X3),0)
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y3.argmax(axis=1), pred.argmax(axis=1)))
print(classification_report(y3,pred))
plt.plot(history.history['acc'],'g--', linewidth=2, markersize=6)
plt.plot(history.history['val_acc'],'^k:', linewidth=2, markersize=6)
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train accuracy', 'test accuracy'], loc='upper left')
plt.grid()
plt.show()
# summarize history for loss
plt.plot(history.history['loss'],'b--', linewidth=2, markersize=6)
plt.plot(history.history['val_loss'],'^k:', linewidth=2, markersize=6)
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train loss', 'test loss'], loc='upper left')
plt.grid()
plt.show()

score = model.evaluate(X3, y3,verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])   
from sklearn.metrics import roc_curve,auc
from scipy import interp
from itertools import cycle
x_test = X3
n_classes=7
lw=1
y_score =pred

### MACRO
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y3[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y3.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
# Plot ROC curves for the multiclass problem

# Compute macro-average ROC curve and ROC area

# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
plt.figure()
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=1)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=1)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue','red'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve for skin cancer classification')
plt.legend(loc="lower right")
plt.show()


del model
img_input = Input(shape=(28, 28, 3))  ## branch 1 with image input
x = Conv2D(64, (3, 3),activation='relu')(img_input)
x=MaxPooling2D(2,2)(x)
x = Dropout(0.1)(x)
x = Conv2D(128, (3, 3),activation='relu')(x)
x=MaxPooling2D(2,2)(x)
x = Dropout(0.3)(x)
x = Conv2D(128, (3, 3),activation='relu')(x)
x = Dropout(0.2)(x)
x=BatchNormalization()(x)
x = Conv2D(256, (3, 3),activation='relu')(x)
x = Dropout(0.4)(x)
x=BatchNormalization()(x)
x = Flatten()(x)
out = Dense(7,activation='softmax')(x)
model = Model(img_input, out)
print(model.summary())
from keras.optimizers import Adam
optimizer = Adam(lr=0.00125, epsilon = 1e-8, beta_1 = .9, beta_2 = .999)
# Compile the model
model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])
# Set a learning rate annealer
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)


history=model.fit(X1, y1,batch_size=8, epochs=50, verbose=1, validation_data = (X2,y2), callbacks=[learning_rate_reduction])
pred=np.round(model.predict(X3),0)
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y3.argmax(axis=1), pred.argmax(axis=1)))
print(classification_report(y3,pred))
plt.plot(history.history['acc'],'g--', linewidth=2, markersize=6)
plt.plot(history.history['val_acc'],'^k:', linewidth=2, markersize=6)
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train accuracy', 'test accuracy'], loc='upper left')
plt.grid()
plt.show()
# summarize history for loss
plt.plot(history.history['loss'],'b--', linewidth=2, markersize=6)
plt.plot(history.history['val_loss'],'^k:', linewidth=2, markersize=6)
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train loss', 'test loss'], loc='upper left')
plt.grid()
plt.show()

score = model.evaluate(X3, y3,verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])   
from sklearn.metrics import roc_curve,auc
from scipy import interp
from itertools import cycle
x_test = X3
n_classes=7
lw=1
y_score =pred

### MACRO
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y3[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y3.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
# Plot ROC curves for the multiclass problem

# Compute macro-average ROC curve and ROC area

# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
plt.figure()
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=1)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=1)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue','red'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve for skin cancer classification')
plt.legend(loc="lower right")
plt.show()
