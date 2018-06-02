from keras.layers import Input,Dense,Embedding,LSTM,Average
from keras.models import Model
from keras.utils import to_categorical
from sklearn import metrics 
from keras.callbacks import Callback
import numpy as np
import data
batch=32
feature=37
x_train,x_test,y_train,y_test,xp,xn,yp,yn=data.NASAData(4)
x_train=x_train.reshape((-1,feature,1))
xp=xp.reshape((-1,feature,1))
xn=xn.reshape((-1,feature,1))
P_num=xp.shape[0]
N_num=xn.shape[0]
inputs = Input(shape=(feature,1))
lstm_forward=LSTM(256)(inputs)
lstm_back=LSTM(256,go_backwards=True)(inputs)
merge=Average()([lstm_forward,lstm_back])
dense=Dense(128,activation='relu')(merge)
outputs=Dense(1,activation='sigmoid')(dense)
model = Model(inputs=inputs, outputs=outputs)
model.summary()
model.compile(
	loss='binary_crossentropy',
	optimizer='adam',
	metrics=['accuracy'])
model.fit(
	x_train,
	y_train,
	epochs=1024,
	batch_size=batch)
P_score = model.evaluate(
	xp,yp,verbose=1)
N_score = model.evaluate(
	xn,yn,verbose=1)
y_pred=model.predict(x_test,verbose=1)
auc=metrics.roc_auc_score(y_test,y_pred)
TP=P_num*P_score[1]
TN=N_num*N_score[1]
FP=N_num-TN
FN=P_num-TP
Precision=TP/(TP+FP)
Recall=TP/(TP+FN)
pf=FP/(FP+TN)
F_measure=2*Recall*Precision/(Recall+Precision)
print('Auc=',auc)
print('Positive test accuracy=', P_score[1])
print('Negative test accuracy=', N_score[1])
print('Total accuracy=',
	(P_score[1]*P_num+N_score[1]*N_num)/(P_num+N_num))
print('Precision=',Precision)
print('Recall=',Recall)
print('F-measure=',F_measure)
print('pf=',pf)