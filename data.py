# -*- coding: utf-8 -*-  
from os import path as ps
from os import listdir
import numpy as np
import scipy.io as sio
from sklearn import preprocessing
def normalizetion(x_train):
	featureNum=x_train.shape[1]
	fmax=np.zeros(shape=(featureNum,))
	fmin=np.zeros(shape=(featureNum,))
	for f in fmin:
		f=float("inf")
	for f in fmax:
		f=float("-inf")
	for x in x_train:
		for i in range(featureNum):
			if x[i]>fmax[i]:
				fmax[i]=x[i]
			if x[i]<fmin[i]:
				fmin[i]=x[i]
	for x in x_train:
		for i in range(featureNum):
			x[i]=(x[i]-fmin[i])/(fmax[i]-fmin[i])
def NASAData(index):
	assert(index<=7)
	train_path="./NASA/NASATrain/"
	test_path="./NASA/NASATest/"
	datalen=[38,40,40,38,38,38,38,39]
	dataname=['cm1','kc3','mc2','mw1','pc1','pc3','pc4','pc5']
	data_train=np.zeros(shape=(0,datalen[index]))
	data_test=np.zeros(shape=(0,datalen[index]))

	file=sio.loadmat(train_path+dataname[index]+'train.mat')
	for data in file[dataname[index]+'train'][0]:
		data_train=np.vstack((data_train,data))

	file=sio.loadmat(test_path+dataname[index]+'test.mat')
	for data in file[dataname[index]+'test'][0]:
		data_test=np.vstack((data_test,data))
	x_train=np.zeros(shape=(data_train.__len__(),datalen[index]-1))
	y_train=np.zeros(shape=(data_train.__len__(),))
	x_test=np.zeros(shape=(data_test.__len__(),datalen[index]-1))
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
	xp=np.zeros(shape=(datalen[index]-1))
	xn=np.zeros(shape=(datalen[index]-1))
	yp=np.zeros(shape=(1))
	yn=np.zeros(shape=(1))
	for i in range(x_test.__len__()):
		if y_test[i]==1:#Negative
			xp=np.vstack((xp,x_test[i]))
			yp=np.vstack((yp,y_test[i]))
		else:
			xn=np.vstack((xn,x_test[i]))
			yn=np.vstack((yn,y_test[i]))
	return x_train,x_test,y_train,y_test,xp,xn,yp,yn
def CKData():
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
	xp=np.zeros(shape=(20))
	xn=np.zeros(shape=(20))
	yp=np.zeros(shape=(1))
	yn=np.zeros(shape=(1))
	for i in range(x_test.__len__()):
		if y_test[i]==1:#Negative
			xp=np.vstack((xp,x_test[i]))
			yp=np.vstack((yp,y_test[i]))
		else:
			xn=np.vstack((xn,x_test[i]))
			yn=np.vstack((yn,y_test[i]))
	# normalizetion(x_train)
	# normalizetion(xp)
	# normalizetion(xn)
	x_test=np.vstack((xp,xn))
	y_test=np.vstack((yp,yn))
	print(x_test.shape)
	print(y_test.shape)
	return x_train,y_train,x_test,y_test,xp,yp,xn,yn
if __name__== '__main__':
	for i in range(7):
		x_train,x_test,y_train,y_test,xp,xn,yp,yn=NASAData(i)
		print(x_train.shape)
		print(x_test.shape)
		print(y_train.shape)
		print(y_test.shape)