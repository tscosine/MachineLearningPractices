from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array 
import os
from keras.utils import to_categorical
import numpy as np
from PIL import Image
from time import time
import random
imgpath = './MLKD-Final-Project/ic-data/train_data/'
labelpath = './MLKD-Final-Project/ic-data/train.label'
trainpath='./data/train_src/'
testpath='./data/test_src/'
def getdata(batch_size,feature,test=False):
	if test:
		path = testpath
	else:
		path = trainpath
	x_train=np.zeros((0,feature,feature,3))
	y_train=np.zeros((batch_size))
	batchcount = 0
	count = 0
	fresh=True
	subdir=os.listdir(path)
	while 1:
		for sdir in subdir:
			ran=random.randint(0,9)
			if ran >= 7 and not test:
				continue
			label=int(sdir)
			imgs=os.listdir(path+sdir)
			if int(count/20) >= len(imgs):
				count = 0
			imgname=imgs[int(count/20)]
			img = load_img(path+sdir+'/'+imgname,
				target_size=(feature,feature))
			img = img_to_array(img)
			img = np.expand_dims(img,axis=0)
			img = img.astype('float32')
			img /= 255
			x_train=np.vstack((x_train,img))
			y_train[batchcount]=label-1
			batchcount += 1
			count += 1
			if batchcount >= batch_size:
				y_train=to_categorical(y_train,20)
				yield (x_train,y_train)
				batchcount = 0
				x_train=np.zeros((0,feature,feature,3))
				y_train=np.zeros((batch_size))

def getlabel(imgname):
	labelfile=open(labelpath)
	index = int(imgname.split('.')[0])
	labels=labelfile.readlines()
	result=labels[index].split(' ')[-1].split('\n')[0]
	labelfile.close()
	return result
def generate_arr(batch_size=32,feature=64,test=False):
	imgpath='./MLKD-Final-Project/ic-data/train_data/'
	if test:
		imgpath='./MLKD-Final-Project/ic-data/test_data/'
	while 1:
		labelfile=open(labelpath)
		imglist=os.listdir(imgpath)
		count = 0
		img_data=np.zeros((batch_size,feature,feature,3))
		y_train=np.zeros(batch_size,dtype=int)
		for imgname in imglist:
			label=int(getlabel(imgname))
			if label == 15 and not test:
				ran=random.randint(0,9)
				if ran > 2:
					continue
			path = os.path.join(imgpath,imgname)
			img = Image.open(path)
			img = img.resize((feature,feature),Image.ANTIALIAS)
			r,g,b = img.split()
			r_arr = np.array(r)[:,:,np.newaxis]
			g_arr = np.array(g)[:,:,np.newaxis]
			b_arr = np.array(b)[:,:,np.newaxis]
			img_arr = np.concatenate((r_arr,g_arr,b_arr),axis=2)
			img_data[count] = img_arr
			y_train[count]=label-1
			count += 1
			if count >= batch_size:
				count = 0
				y_train = to_categorical(y_train,num_classes=20)
				print(y_train)
				yield(img_data,y_train)
				img_data=np.zeros((batch_size,feature,feature,3))
				y_train=np.zeros(batch_size,dtype=int)
def gettestdata(feature):
	imgpath = './MLKD-Final-Project/ic-data/test_data'
	labelpath = './MLKD-Final-Project/ic-data/train.label'
	labelfile=open(labelpath)
	imglist=os.listdir(imgpath)
	img_data=np.zeros((len(imglist),feature,feature,3))
	y_test=np.zeros(len(imglist),dtype=int)
	count=0
	for imgname in imglist:
		path = os.path.join(imgpath,imgname)
		img = Image.open(path)
		img = img.resize((feature,feature),Image.ANTIALIAS)
		r,g,b = img.split()
		r_arr = np.array(r)[:,:,np.newaxis]
		g_arr = np.array(g)[:,:,np.newaxis]
		b_arr = np.array(b)[:,:,np.newaxis]
		img_arr = np.concatenate((r_arr,g_arr,b_arr),axis=2)
		img_data[count] = img_arr
		label=int(getlabel(imgname))
		y_test[count]=label-1
		count+=1
	y_test=to_categorical(y_test,20)
	return img_data,y_test

if __name__ == '__main__':
	gen=getdata(32,64,True)
	print(next(gen))