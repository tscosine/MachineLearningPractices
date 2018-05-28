# -*- coding: utf-8 -*- 
import numpy as np
from keras import initializers
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,LSTM,Embedding
import keras.optimizers
from CKdata import getCKdata
n_input=1
n_step=20
n_classes=2
x_train,y_train,x_test_Positive,y_test_Positive,x_test_Negative,y_test_Negative=getCKdata()
x_train=x_train.reshape(-1,n_step,n_input)
y_train = to_categorical(y_train,n_classes)
x_test_Positive=x_test_Positive.reshape(-1,n_step,n_input)
y_test_Positive = to_categorical(y_test_Positive,n_classes)
x_test_Negative=x_test_Negative.reshape(-1,n_step,n_input)
y_test_Negative = to_categorical(y_test_Negative,n_classes)

P_num=x_test_Positive.shape[0]
N_num=x_test_Negative.shape[0]

batch_size = 16
epochs = 10
learning_rate = 1e-6

model = Sequential()
model.add(LSTM(
	256,
	batch_input_shape=(None, n_step, n_input),
	unroll=True))
model.add(Dense(256))
model.add(Dropout(0.25))
model.add(Dense(128))
model.add(Dropout(0.25))
model.add(Dense(n_classes))
model.add(Activation('softmax'))
#显示模型细节
model.summary()
model.compile(
	optimizer=keras.optimizers.Adam(lr=learning_rate),
	loss='categorical_crossentropy',
	metrics=['accuracy'])
model.fit(x_train, y_train,
	batch_size=batch_size,
	epochs=epochs,
	verbose=1,
	validation_data=(x_train, y_train))

P_scores = model.evaluate(
	x_test_Positive,y_test_Positive,verbose=1)
N_scores = model.evaluate(
	x_test_Negative,y_test_Negative,verbose=1)
print('Positive test accuracy:', P_scores[1])
print('Negative test accuracy:', N_scores[1])