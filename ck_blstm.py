from keras.layers import Input,Dense,Embedding,LSTM,Average
from keras.models import Model
from keras.utils import to_categorical
from sklearn import metrics 
from keras.callbacks import Callback
import numpy as np
import data
import matplotlib.pyplot as plt
from keras.models import load_model  
#parameter
batch=32
epochs=512
hidden_units=128
feature=20
#get data set 
x_train,y_train,x_test,y_test,xp,yp,xn,yn=data.CKData()
x_train=x_train.reshape((-1,feature,1))
x_test=x_test.reshape((-1,feature,1))
xp=xp.reshape((-1,feature,1))
xn=xn.reshape((-1,feature,1))
P_num=xp.shape[0]
N_num=xn.shape[0]

# Functional model BLSTM
inputs = Input(shape=(feature,1))
lstm_forward=LSTM(hidden_units)(inputs)
lstm_back=LSTM(hidden_units,go_backwards=True)(inputs)
merge=Average()([lstm_forward,lstm_back])
dense=Dense(hidden_units,activation='relu')(merge)
outputs=Dense(1,activation='sigmoid')(dense)
model = Model(inputs=inputs, outputs=outputs)
model.summary()

#Train
model.compile(
	loss='binary_crossentropy',
	optimizer='adam',
	metrics=['accuracy'])

model.fit(
	x_train,
	y_train,
	epochs=epochs,
	batch_size=batch)
model.save('./ck_blstm.h5')
# model=load_model('./ck_blstm.h5')
#Evaluate model
P_score = model.evaluate(
	xp,yp,verbose=1)
N_score = model.evaluate(
	xn,yn,verbose=1)

y_pred=model.predict(x_test,verbose=1)
fpr,tpr,threshold=metrics.roc_curve(y_test,y_pred) ###计算真正率和假正率  
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

lw=2
plt.figure(figsize=(10,10))  
plt.plot(
	fpr, tpr,
	color='darkorange',  
	lw=lw, 
	label='ROC curve (area = %0.2f)' % auc) 
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')  
plt.xlim([0.0, 1.0])  
plt.ylim([0.0, 1.05])  
plt.xlabel('False Positive Rate')  
plt.ylabel('True Positive Rate')  
plt.title('Receiver operating characteristic example')  
plt.legend(loc="lower right")  
plt.show()  