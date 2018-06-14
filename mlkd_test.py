from keras.models import load_model
import numpy as np
import os
from keras.preprocessing.image import load_img, img_to_array 
modelPath='./Model_ResNet/InceptionResNetV2_.h5'
testDataPath='./testdata'
predictFilePath='./predict.txt'

def getn(y):
	y=y.tolist()
	result = np.zeros((len(y)))
	for i in range(len(y)):
		result[i]=y[i].index(max(y[i]))
	return result

feature=224
model=load_model(modelPath)
predictFile=open(predictFilePath,'w')

x_tests=os.listdir(testDataPath)
for x_test in x_tests:
	img = load_img(testDataPath+'/'+x_test,
		target_size=(feature,feature))
	img = img_to_array(img)
	img = np.expand_dims(img,axis=0)
	img = img.astype('float32')
	img /= 255
	predict_y = model.predict(img, batch_size = 16)
	s=x_test.split('.')[0]+' '+str(1+int(getn(predict_y)))+'\n'
	print(s)
	predictFile.write(s)
predictFile.close()
