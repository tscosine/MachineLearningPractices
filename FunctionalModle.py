from keras.layers import Input, Dense
from keras.models import Model
from CKdata import getCKdata
from keras.utils import to_categorical
from sklearn import metrics 
import numpy as np
x_train,y_train,xp,yp,xn,yn=getCKdata()
x_test=np.vstack((xp,xn))
y_test=np.vstack((yp,yn))
P_num=xp.shape[0]
N_num=xn.shape[0]
yp=to_categorical(yp,2)
yn=to_categorical(yn,2)
y_test=to_categorical(y_test,2)
y_train=to_categorical(y_train,2)

inputs = Input(shape=(20,))
x = Dense(128,activation='relu')(inputs)
x = Dense(128,activation='relu')(x)
predictions = Dense(2, activation='softmax')(x)
# This creates a model that includes
# the Input layer and three Dense layers
model = Model(inputs=inputs, outputs=predictions)
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x_train,y_train,epochs=2000,batch_size=16)  # starts training
P_score = model.evaluate(
	xp,yp,verbose=1)
N_score = model.evaluate(
	xn,yn,verbose=1)

TP=P_num*P_score[1]
TN=N_num*N_score[1]
FP=N_num-TN
FN=P_num-TP
Precision=TP/(TP+FP)
Recall=TP/(TP+FN)
pf=FP/(FP+TN)
F_measure=2*Recall*Precision/(Recall+Precision)

# print('Auc=',auc)
print('Positive test accuracy=', P_score[1])
print('Negative test accuracy=', N_score[1])
print('Total accuracy=',
	(P_score[1]*P_num+N_score[1]*N_num)/(P_num+N_num))
print('Precision=',Precision)
print('Recall=',Recall)
print('F-measure=',F_measure)
print('pf=',pf)