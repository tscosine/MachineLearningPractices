from PIL import Image
import os
import numpy as np
from sklearn.cross_validation import train_test_split
from keras.layers import MaxPooling2D,Conv2D,Input,Flatten,Dense,Dropout
from keras.models import Sequential,Model
from keras.utils import to_categorical
from keras.optimizers import Adam,RMSprop,SGD
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from MLKDDATA import getdata,generate_arr
from keras.callbacks import ModelCheckpoint
def ResNet50_model(lr=0.005, decay=1e-6,
	momentum=0.9, nb_classes=20, img_rows=197,
	img_cols=197): 
	base_model = InceptionResNetV2	(
		weights='imagenet', 
		include_top=False,
		pooling=None, 
		input_shape=(img_rows, img_cols, 3),
		classes=nb_classes)  
	#冻结base_model所有层，这样就可以正确获得bottleneck特征  
	for layer in base_model.layers:  
		layer.trainable = False  
	x = base_model.output  
	#添加自己的全链接分类层  
	x = Flatten()(x)  
	#x = GlobalAveragePooling2D()(x)  
	x = Dense(1024, activation='relu')(x)  
	predictions = Dense(nb_classes, activation='softmax')(x)  
	model = Model(inputs=base_model.input, outputs=predictions)  
	sgd = SGD(lr=lr, decay=decay, momentum=momentum, nesterov=True)  
	model.compile(
		loss='categorical_crossentropy',
		# optimizer='Adam',
		optimizer=sgd,
		metrics=['accuracy'])
	return model
if __name__ == '__main__':
	feature=224
	batch_size=64
	epochs=512
	classnum=20
	gen=getdata(batch_size=batch_size,feature=feature,test=False)
	testset=getdata(batch_size=128,feature=feature,test=True)
	model = ResNet50_model(nb_classes=classnum,
		img_rows=feature, img_cols=feature)
	checkpoint = ModelCheckpoint(
		filepath='./Model_ResNet/InceptionResNetV2_.h5', 
		monitor='val_acc',
		verbose=1,
		save_best_only='True',
		mode='auto',
		period=1)
	model.fit_generator(
		generator=gen,  
		samples_per_epoch=batch_size,
		epochs=epochs,
		verbose=1,
		callbacks=[checkpoint],
		validation_data=testset,
		validation_steps=10)