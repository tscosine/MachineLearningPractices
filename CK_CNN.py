# -*- coding: utf-8 -*-  
import numpy as np
from keras import initializers
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Conv2D,MaxPooling2D,Flatten
import keras.optimizers
from CKdata import getCKdata
x_train,y_train,x_test_Positive,y_test_Positive,x_test_Negative,y_test_Negative=getCKdata()

batch_size = 32
epochs = 20
learning_rate = 1e-7

x_train=x_train.reshape(-1,5,4,1)
x_test_Positive=x_test_Positive.reshape(-1,5,4,1)
x_test_Negative=x_test_Negative.reshape(-1,5,4,1)
y_train = to_categorical(y_train,2)
y_test_Positive = to_categorical(y_test_Positive,2)
y_test_Negative = to_categorical(y_test_Negative,2)

model=Sequential()
model.add(Conv2D(
	filters=64,
	kernel_size=(2,2),
	activation='relu',
	input_shape=(5,4,1),
	padding='same'
	))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Conv2D(
	filters=64,
	kernel_size=(2,2),
	activation='relu',
	padding='same'
	))
# model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Dense(2))
model.add(Activation('softmax'))
model.summary()
opt=keras.optimizers.SGD(lr=learning_rate)
model.compile(
	optimizer=opt,
	loss='categorical_crossentropy',
	metrics=['accuracy'])
model.fit(x_train, y_train,
	#batch_size=batch_size,
	epochs=epochs,
	verbose=1, #0不显示 1显示
	validation_data=(x_train, y_train))

P_scores = model.evaluate(x_test_Positive, y_test_Positive, verbose=1)
N_scores = model.evaluate(x_test_Negative, y_test_Negative, verbose=1)
print('Positive test accuracy:', P_scores[1]) #loss
print('Negative test accuracy:', N_scores[1])