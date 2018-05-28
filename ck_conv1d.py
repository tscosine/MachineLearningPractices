# -*- coding:utf-8 -*-
import numpy as np
from keras import initializers
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation
from keras.layers import Embedding,Conv1D,GlobalAveragePooling1D,MaxPooling1D
from keras.optimizers import SGD
from CKdata import getCKdata
x_train,y_train,x_test_Positive,y_test_Positive,x_test_Negative,y_test_Negative=getCKdata()
x_train=x_train.reshape(-1,20,1)
x_test_Positive=x_test_Positive.reshape(-1,20,1)
x_test_Negative=x_test_Negative.reshape(-1,20,1)

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
model.fit(x_train, y_train, batch_size=16, epochs=10)

P_score = model.evaluate(
	x_test_Positive, y_test_Positive, batch_size=16)
N_score = model.evaluate(
	x_test_Negative, y_test_Negative, batch_size=16)
print('Positive test accuracy:', P_score[1]) #loss
print('Negative test accuracy:', N_score[1])