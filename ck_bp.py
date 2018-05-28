# -*- coding:utf-8 -*-

from os import path as ps
from os import listdir
import numpy as np
import scipy.io as sio
from keras import initializers
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation
from keras.optimizers import SGD
train_path="./CK/CKTrain/"
test_path="./CK/CKTest/"
Trainlist=listdir(train_path)
Testlist=listdir(test_path)
data_train=np.zeros(shape=(0,21))
data_test=np.zeros(shape=(0,21))
for fpath in Trainlist:
	file=sio.loadmat(ps.join(train_path,fpath))
	for data in file[fpath.split('.')[0]][0]:
	 	data_train=np.vstack((data_train,data))
for fpath in Testlist:
	file=sio.loadmat(ps.join(test_path,fpath))
	for data in file[fpath.split('.')[0]][0]:
	 	data_test=np.vstack((data_test,data))

x_train=np.zeros(shape=(data_train.__len__(),20))
y_train=np.zeros(shape=(data_train.__len__(),))
x_test=np.zeros(shape=(data_test.__len__(),20))
y_test=np.zeros(shape=(data_test.__len__(),))
for i in range(data_train.__len__()):
	x_train[i]=data_train[i][:-1]
	y_train[i]=data_train[i][-1]
for i in range(data_test.__len__()):
	x_test[i]=data_test[i][:-1]
	y_test[i]=data_test[i][-1]
for i in range(y_train.__len__()):
	y_train[i]=1 if y_train[i]==1 else 0
for i in range(y_test.__len__()):
	y_test[i]=1 if y_test[i]==1 else 0

print(x_train.shape)
print(y_train.shape)
model = Sequential()
model.add(Dense(
    units=64,
    activation='relu',
    input_dim=20,
    ))
model.add(Dropout(0.25))
model.add(Dense(units=32,activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(units=32,activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(units=1,activation='softmax'))
model.summary()
sgd=SGD(lr=0.01,decay=1e-6,momentum=0.9,nesterov=True)
model.compile(
        loss='binary_crossentropy',
        optimizer='rmsprop',
        metrics=['accuracy'])
model.fit(x_train,y_train,epochs=8)
score=model.evaluate(x_test,y_test)
print(score)
