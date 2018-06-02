# -*- coding:utf-8 -*-
import numpy as np
from keras import initializers
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation
from keras.layers import Embedding,Conv1D,GlobalAveragePooling1D,MaxPooling1D
from keras.optimizers import Adam
from sklearn import metrics 
import data

DataSetN=5
x_train,x_test,y_train,y_test=data.NASAData(DataSetN)
featureNum=[37,39,39,37,37,37,37,37][DataSetN]
batch=64
epochs=100

x_train=x_train.reshape(-1,featureNum,1)
x_test=x_test.reshape(-1,featureNum,1)

print('data number=',x_train.shape[0])
print('feature number=',x_train.shape[1])
model = Sequential()
model.add(Conv1D(64, 5, 
	activation='relu',
	padding='same',
	input_shape=(featureNum,1)))
model.add(Conv1D(128, 5, activation='relu',padding='same'))
model.add(MaxPooling1D(2))
model.add(Conv1D(256, 5, activation='relu',padding='same'))
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
	optimizer=opt,
	metrics=['accuracy'])
model.fit(x_train,y_train,batch_size=batch,epochs=epochs)

y_pred=model.predict(x_test,verbose=1)
auc=metrics.roc_auc_score(y_test,y_pred)
print('Auc=',auc)