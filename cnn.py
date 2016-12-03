
from scipy.misc import imsave
import numpy as np
import time
from keras.applications import vgg16

import os
from skimage import io,transform
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D
from keras.layers import MaxPooling2D, Flatten, Dropout


num_epoch = 10
image_size = 50
data = "data"
train_fraction = 0.8

def data_split(data,labels,f):
	test_size = int(len(data)*f)
	return data[:test_size], labels[:test_size], data[test_size:], labels[test_size:]


def cnn():
	model = Sequential()
	model.add(Convolution2D(8,3,3,border_mode='same',
		input_shape=(image_size,image_size,3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2,2),border_mode='same'))
	model.add(Convolution2D(16,3,3,border_mode='same'))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2,2),border_mode='same'))
	model.add(Flatten())
	model.add(Dense(2))
	model.add(Activation('softmax'))
	model.compile(optimizer='adam',
		loss='binary_crossentropy',
		metrics=['accuracy'])
	return model


def transform_image(image):
	return transform.resize(image,(image_size,image_size,image.shape[2]))


def load_images():
	files = os.listdir(data)
	images=[]
	labels=[]
	for file in files:
		if file[-4:]=='jpeg':
			image = io.imread(data+"/"+file)
			transformed_image = transform_image(image)
			images.append(transformed_image)
			label_file = file[:-5]+".txt"
			with open(data+"/"+label_file) as f:
				label = int(float(f.readlines()[0]))
				l=[0,0]
				l[label]=1
				labels.append(l)
	return np.array(images),np.array(labels)

images,labels = load_images()
train_data,train_labels,test_data,test_labels=data_split(images,labels,train_fraction)


print "Train size is: ", len(train_data)
print "Test size is: ", len(test_data)


model = cnn()
model.fit(train_data,train_labels,nb_epoch=num_epoch)

preds = model.predict(test_data)

predictions = np.argmax(preds,axis=1)
actual_res = np.argmax(test_labels,axis=1)
print accuracy_score(actual_res,predictions)