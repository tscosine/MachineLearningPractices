# -*- coding:utf-8 -*-
import numpy as np
from keras import initializers
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation
from keras.layers import Embedding,Conv1D,GlobalAveragePooling1D,MaxPooling1D
from keras.optimizers import SGD
from data import getCKdata
x_train,y_train,x_test_Positive,y_test_Positive,x_test_Negative,y_test_Negative=getCKdata()
x_train=x_train.reshape(-1,20,1)
x_test_Positive=x_test_Positive.reshape(-1,20,1)
x_test_Negative=x_test_Negative.reshape(-1,20,1)
P_num=x_test_Positive.shape[0]
N_num=x_test_Negative.shape[0]

model = Sequential()
model.add(Conv1D(64, 3, activation='relu', input_shape=(20,1)))
model.add(Conv1D(64, 3, activation='relu'))
model.add(MaxPooling1D(3))
model.add(Conv1D(128, 3, activation='relu'))
model.add(Conv1D(128, 3, activation='relu'))
model.add(GlobalAveragePooling1D())
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
model.summary()
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
model.fit(x_train,y_train,
	batch_size=128,
	epochs=1024)
P_score = model.evaluate(
	x_test_Positive, y_test_Positive,batch_size=64)
N_score = model.evaluate(
	x_test_Negative, y_test_Negative,batch_size=64)

TP=P_num*P_score[1]
TN=N_num*N_score[1]
FP=N_num-TN
FN=P_num-TP
Precision=TP/(TP+FP)
Recall=TP/(TP+FN)
pf=FP/(FP+TN)
F_measure=2*Recall*Precision/(Recall+Precision)
print('Positive test accuracy:', P_score[1])
print('Negative test accuracy:', N_score[1])
print('Total accuracy',
	(P_score[1]*P_num+N_score[1]*N_num)/(P_num+N_num))
print('Precision=',Precision)
print('Recall=',Recall)
print('F-measure=',F_measure)
print('pf=',pf)
