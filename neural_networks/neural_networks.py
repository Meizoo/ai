import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu,floatX=float32"

import numpy as np
import matplotlib

from matplotlib import pyplot as plt

from keras.models     import Sequential
from keras.optimizers import Adam
from keras.callbacks  import ModelCheckpoint
from keras.models     import load_model
from keras.layers     import Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten, Activation

from helper import *

matplotlib.style.use('ggplot')

IMAGE_SIZE  = 32
CHANNELS    = 3
class_names = get_class_names()

print(class_names)

num_classes = len(class_names)

print(num_classes)

images_train, labels_train, class_train = get_train_data()

images_test, labels_test, class_test = get_test_data()

print("Training set: %d images" % len(images_train))
print("Testing  set: %d images" % len(images_test))

def cnn_model():
	
	model = Sequential()
	
	model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(IMAGE_SIZE,IMAGE_SIZE,CHANNELS)))	
	model.add(Conv2D(32, (3, 3), activation='relu'))	
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))	
	model.add(Conv2D(64, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	model.add(Flatten())
	
	model.add(Dense(512, activation='relu'))
	model.add(Dropout(0.5))
	
	model.add(Dense(num_classes, activation='softmax'))
	
	model.summary()
	
	return model

model = cnn_model()

checkpoint = ModelCheckpoint(
	'best_model_simple.h5', # model filename
	monitor='val_loss', # quantity to monitor
	verbose=0, # verbosity - 0 or 1
	save_best_only= True, # The latest best model will not be overwritten
	mode='auto' # The decision to overwrite model is made
) 

# automatically depending on the quantity to monitor
model.compile(
	loss='categorical_crossentropy', # Better loss function for neural networks
	optimizer=Adam(lr=1.0e-4), # Adam optimizer with 1.0e-4 learning rate
	metrics= ['accuracy'] # Metrics to be evaluated by the model
)

model_details = model.fit(
	images_train, 
	class_train,
	batch_size=128, # number of samples per gradient update
	epochs=20, # number of iterations
	validation_data=(images_test, class_test),
	callbacks=[checkpoint],
	verbose=1
)

scores = model.evaluate(images_test, class_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
plot_model(model_details)

class_pred  = model.predict(images_test, batch_size=32)
labels_pred = np.argmax(class_pred,axis=1)
correct     = (labels_pred == labels_test)
num_images  = len(correct)

print("Correct: %d" % sum(correct))

incorrect = (correct == False)

# Images of the test-set that have been incorrectly classified.
images_error = images_test[incorrect]

# Get predicted classes for those images
labels_error = labels_pred[incorrect]

# Get true classes for those images
labels_true = labels_test[incorrect]

plot_images(
	images     =images_error[0:9],
	labels_true=labels_true [0:9],
	class_names=class_names,
	labels_pred=labels_error[0:9]
)
