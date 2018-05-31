from keras.layers import Input,Dense,Embedding,LSTM,Average
from keras.models import Model
from data import CKData
from keras.utils import to_categorical
from sklearn import metrics 
import numpy as np
x_train,y_train,xp,yp,xn,yn=CKData()

x_train=x_train.reshape((-1,20,1))
y_train=to_categorical(y_train,2)
xp=xp.reshape((-1,20,1))
xn=xn.reshape((-1,20,1))
yp=to_categorical(yp,2)
yn=to_categorical(yn,2)
# x_test=np.vstack((xp,xn))
# y_test=np.vstack((yp,yn))
# y_test=to_categorical(y_test,2)
P_num=xp.shape[0]
N_num=xn.shape[0]
inputs = Input(shape=(20,1))
lstm_forward=LSTM(256)(inputs)
lstm_back=LSTM(256,go_backwards=True)(inputs)
merge=Average()([lstm_forward,lstm_back])
dense=Dense(128,activation='relu')(merge)
outputs=Dense(2,activation='softmax')(dense)
model = Model(inputs=inputs, outputs=outputs)
model.summary()
model.compile(
	loss='categorical_crossentropy',
	optimizer='adam',
	metrics=['accuracy'])
model.fit(x_train,y_train,epochs=128,batch_size=32)
P_score = model.evaluate(
	xp,yp,verbose=1)
N_score = model.evaluate(
	xn,yn,verbose=1)

# TP=P_num*P_score[1]
# TN=N_num*N_score[1]
# FP=N_num-TN
# FN=P_num-TP
# Precision=TP/(TP+FP)
# Recall=TP/(TP+FN)
# pf=FP/(FP+TN)
# F_measure=2*Recall*Precision/(Recall+Precision)

# # print('Auc=',auc)
print('Positive test accuracy=', P_score[1])
print('Negative test accuracy=', N_score[1])
# print('Total accuracy=',
# 	(P_score[1]*P_num+N_score[1]*N_num)/(P_num+N_num))
# print('Precision=',Precision)
# print('Recall=',Recall)
# print('F-measure=',F_measure)
# print('pf=',pf)