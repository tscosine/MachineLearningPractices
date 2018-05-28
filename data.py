# -*- coding: utf-8 -*-  
from os import path as ps
from os import listdir
import numpy as np
import scipy.io as sio
train_path="./CK/CKTrain/"
test_path="./CK/CKTest/"
def getCKdata():
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

	x_test_Positive=np.zeros(shape=(20))
	x_test_Negative=np.zeros(shape=(20))
	y_test_Positive=np.zeros(shape=(1))
	y_test_Negative=np.zeros(shape=(1))
	for i in range(x_test.__len__()):
		if y_test[i]==1:#Negative
			x_test_Positive=np.vstack((x_test_Positive,x_test[i]))
			y_test_Positive=np.vstack((y_test_Positive,y_test[i]))
		else:
			x_test_Negative=np.vstack((x_test_Negative,x_test[i]))
			y_test_Negative=np.vstack((y_test_Negative,y_test[i]))
	return x_train,y_train,x_test_Positive,y_test_Positive,x_test_Negative,y_test_Negative