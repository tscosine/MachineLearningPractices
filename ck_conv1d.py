# -*- coding:utf-8 -*-
import numpy as np
from keras import initializers
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation
from keras.layers import Conv1D,GlobalAveragePooling1D,MaxPooling1D
from keras.optimizers import RMSprop,Adam
from data import CKData
from sklearn import metrics 
x_train,y_train,x_test_P,y_test_P,x_test_N,y_test_N=CKData()

featureNum=20
batch=128
epochs=2000

x_test=np.vstack((x_test_P,x_test_N))
y_test=np.vstack((y_test_P,y_test_N))
x_train=x_train.reshape(-1,featureNum,1)
x_test_P=x_test_P.reshape(-1,featureNum,1)
x_test_N=x_test_N.reshape(-1,featureNum,1)
x_test=x_test.reshape(-1,featureNum,1)
P_num=x_test_P.shape[0]
N_num=x_test_N.shape[0]

model = Sequential()
model.add(Conv1D(256, 3, 
	activation='relu',
	padding='same',
	input_shape=(featureNum,1)))
model.add(Conv1D(256, 3, activation='relu',padding='same'))
model.add(MaxPooling1D(2))
model.add(Conv1D(256, 3, activation='relu',padding='same'))
model.add(MaxPooling1D(2))
model.add(Conv1D(256, 3, activation='relu',padding='same'))
model.add(GlobalAveragePooling1D())
model.add(Dropout(0.5))
model.add(Dense(64,activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()
#opt=RMSprop(lr=0.001)
opt=Adam(lr=0.001)
model.compile(
	loss='binary_crossentropy',
	#loss='mean_squared_error',
	optimizer=opt,
	metrics=['accuracy'])
model.fit(x_train,y_train,batch_size=batch,epochs=epochs)
P_score = model.evaluate(
	x_test_P, y_test_P,batch_size=batch)
N_score = model.evaluate(
	x_test_N, y_test_N,batch_size=batch)
pred=model.predict(
	x_test,batch_size=batch,verbose=1)
TP=P_num*P_score[1]
TN=N_num*N_score[1]
FP=N_num-TN
FN=P_num-TP
Precision=TP/(TP+FP)
Recall=TP/(TP+FN)
pf=FP/(FP+TN)
F_measure=2*Recall*Precision/(Recall+Precision)
fpr,tpr,thresholds = metrics.roc_curve(y_test, pred, pos_label=1)
auc=metrics.auc(fpr, tpr)
print('Auc=',auc)
print('Positive test accuracy=', P_score[1])
print('Negative test accuracy=', N_score[1])
print('Total accuracy=',
	(P_score[1]*P_num+N_score[1]*N_num)/(P_num+N_num))
print('Precision=',Precision)
print('Recall=',Recall)
print('F-measure=',F_measure)
print('pf=',pf)
