import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu,floatX=float32"

import numpy as np
import keras

from keras.applications.vgg16  import VGG16, preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.layers              import Input, Flatten, Dense,Dropout
from keras.models              import Model
from keras.utils               import np_utils
from keras.optimizers          import Adam
from keras.datasets            import cifar10
from keras.callbacks           import ReduceLROnPlateau, CSVLogger,EarlyStopping
from keras                     import regularizers,optimizers

from scipy.misc import toimage, imresize

from utilites import *

IMAGE_SIZE = 64
CHANNELS = 3

class_names  = get_class_names()
print(class_names)
num_classes = len(class_names)
print(num_classes)
images_train, labels_train, class_train = get_train_data()

images_test, labels_test, class_test = get_test_data()

images_train = Resize(images_train)
images_test  = Resize(images_test)

mean = np.mean(images_train,axis=(0,1,2,3))
std  = np.std(images_train,axis=(0,1,2,3))

x_train = (images_train-mean)/(std+1e-7)
x_test = (images_test-mean)/(std+1e-7)

print("Training set: %d images" % len(images_train))
print("Testing  set: %d images" % len(images_test))

def vgg_model(num_of_classes) :
	vgg_16_model = VGG16(weights=None,include_top= False,input_shape=(IMAGE_SIZE,IMAGE_SIZE,CHANNELS))
	vgg_16_model.summary()
	
	vgg_16_model.load_weights('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')
	
	print(len(vgg_16_model.layers))
	
	inputs = Input(shape = (IMAGE_SIZE,IMAGE_SIZE,CHANNELS), name = "image_input")
	
	output_vgg16_model = vgg_16_model(inputs)
	
	x = Flatten(name='flatten')(output_vgg16_model)
	x = Dense(2048, activation='relu', name='fc1')(x)
	x = Dropout(0.5)(x)
	x = Dense(num_of_classes, activation='softmax', name='predictions')(x)

	cifar10_vgg = Model(inputs=inputs, outputs=x)
	cifar10_vgg.summary()
	
	print(len(cifar10_vgg.layers))
	
	return cifar10_vgg

model = vgg_model(num_classes)

batch_size = 64
epochs = 10

opt = optimizers.rmsprop(lr=0.0001, decay=1e-6)

model.compile(
	loss='categorical_crossentropy',
	optimizer=Adam(lr=1.0e-4),
	metrics = ['accuracy']
)

lr_reducer = ReduceLROnPlateau(
	factor=np.sqrt(0.1), 
	cooldown=0, 
	patience=2, 
	min_lr=0.5e-6
)

csv_logger = CSVLogger('./vgg16imagenetpretrained_upsampleimage_cifar10_data_argumentation.csv')

early_stopper = EarlyStopping(min_delta=0.001, patience=10)

model.fit(
	images_train, 
	class_train,
	batch_size=batch_size*4,
	epochs=epochs,
	validation_data=(images_test, class_test),
	shuffle=True,
	callbacks=[lr_reducer, early_stopper, csv_logger]
)