import pickle
import numpy as np

from keras.utils import np_utils
from matplotlib import pyplot as plt

#  Used dataset: https://www.cs.toronto.edu/~kriz/cifar.html
#  Constants:
# Path to dataset
PATH = "data/"

# Width and height of all pictures
SIZE = 32 

# RGB
CHANNELS = 3  

# Number of classification classes
NUM_CLASSES = 10 

# Number of images in one file
IMAGE_BATCH = 10000 

# Number of files
NUM_FILES_TRAIN = 5  

# Number of all images
IMAGES_TRAIN = IMAGE_BATCH * NUM_FILES_TRAIN


def unpickle(file):  
	"""Loads dataset file."""
	with open(PATH + file,'rb') as fo:
		print("Preparing %s..." % (PATH+file))
		dict = pickle.load(fo, encoding='bytes')
	return dict

def convert_images(raw_images):
	"""Converts images to numpy arrays"""
	return (np.array(raw_images, dtype = float) / 255.0).reshape([-1, CHANNELS, SIZE, SIZE]).transpose([0, 2, 3, 1])

def load_data(file):
	"""Load file, unpickle it and return images with their labels"""
	data = unpickle(file)
	return convert_images(data[b'data']), np.array(data[b'labels'])

def get_test_data():
	"""Load all test data"""	
	images, labels = load_data(file = "test_batch")	
	# Images, their labels and corresponding one-hot vectors in form of numpy arrays
	return images, labels, np_utils.to_categorical(labels,NUM_CLASSES)

def get_train_data():
	"""Load all training data from files"""
	
	# Pre-allocate arrays
	images = np.zeros(shape = [IMAGES_TRAIN, SIZE, SIZE, CHANNELS], dtype = float)
	labels = np.zeros(shape=[IMAGES_TRAIN],dtype = int)
	
	# Starting index of training dataset
	start = 0
	
	# For all 5 files
	for i in range(NUM_FILES_TRAIN):
		
		# Load images and labels
		images_batch, labels_batch = load_data(file = "data_batch_" + str(i+1))
		
		# Calculate end index for current batch
		end = start + IMAGE_BATCH
		
		# Store data to corresponding arrays
		images[start:end,:] = images_batch		
		labels[start:end] = labels_batch
		
		# Update starting index of next batch
		start = end
	
	# Images, their labels and 
	# corresponding one-hot vectors in form of np arrays
	return images, labels, np_utils.to_categorical(labels,NUM_CLASSES)
		
def get_class_names():

	# Load class names
	raw = unpickle("batches.meta")[b'label_names']

	# Convert from binary strings
	names = [x.decode('utf-8') for x in raw]

	# Class names
	return names

def plot_images(images, labels_true, class_names, labels_pred=None):
	"""Creates and shows plots for given images"""
	assert len(images) == len(labels_true)

	# Create a figure with sub-plots
	fig, axes = plt.subplots(3, 3, figSIZE = (8,8))

	# Adjust the vertical spacing
	if labels_pred is None:
		hspace = 0.2
	else:
		hspace = 0.5
	fig.subplots_adjust(hspace=hspace, wspace=0.3)

	for i, ax in enumerate(axes.flat):
		# Fix crash when less than 9 images
		if i < len(images):
			# Plot the image
			ax.imshow(images[i], interpolation='spline16')
			
			# Name of the true class
			labels_true_name = class_names[labels_true[i]]

			# Show true and predicted classes
			if labels_pred is None:
				xlabel = "True: "+labels_true_name
			else:
				# Name of the predicted class
				labels_pred_name = class_names[labels_pred[i]]

				xlabel = "True: "+labels_true_name+"\nPredicted: "+ labels_pred_name

			# Show the class on the x-axis
			ax.set_xlabel(xlabel)
		
		# Remove ticks from the plot
		ax.set_xticks([])
		ax.set_yticks([])
	
	# Show the plot
	plt.show()
	

def plot_model(model_details):
	"""Creates and shows plot for given model"""
	# Create sub-plots
	fig, axs = plt.subplots(1,2,figSIZE=(15,5))
	
	# Summarize history for accuracy
	axs[0].plot(range(1,len(model_details.history['acc'])+1),model_details.history['acc'])
	axs[0].plot(range(1,len(model_details.history['val_acc'])+1),model_details.history['val_acc'])
	axs[0].set_title('Model Accuracy')
	axs[0].set_ylabel('Accuracy')
	axs[0].set_xlabel('Epoch')
	axs[0].set_xticks(np.arange(1,len(model_details.history['acc'])+1),len(model_details.history['acc'])/10)
	axs[0].legend(['train', 'val'], loc='best')
	
	# Summarize history for loss
	axs[1].plot(range(1,len(model_details.history['loss'])+1),model_details.history['loss'])
	axs[1].plot(range(1,len(model_details.history['val_loss'])+1),model_details.history['val_loss'])
	axs[1].set_title('Model Loss')
	axs[1].set_ylabel('Loss')
	axs[1].set_xlabel('Epoch')
	axs[1].set_xticks(np.arange(1,len(model_details.history['loss'])+1),len(model_details.history['loss'])/10)
	axs[1].legend(['train', 'val'], loc='best')
	
	# Show the plot
	plt.show()

def visualize_errors(images_test, labels_test, class_names, labels_pred, correct):
	"""Visualization of error using plot_images"""
	incorrect = (correct == False)

	# Images of the test-set that have been incorrectly classified.
	images_error = images_test[incorrect]	

	# Get predicted classes for those images
	labels_error = labels_pred[incorrect]

	# Get true classes for those images
	labels_true = labels_test[incorrect]
	
	# Plot the first 9 images.
	plot_images(images=images_error[0:9],
				labels_true=labels_true[0:9],
				class_names=class_names,
				labels_pred=labels_error[0:9])
	
	
def predict_classes(model, images_test, labels_test):
	"""Predict class of image using model"""
	class_pred = model.predict(images_test, batch_SIZE=32)

	# Convert vector to a label
	labels_pred = np.argmax(class_pred,axis=1)

	# Boolean array that tell if predicted label is the true label
	correct = (labels_pred == labels_test)

	# Array which tells if the prediction is correct or not
	# And predicted labels
	return correct, labels_pred

def Resize(images, size=64):
	"""Resizes images to given size"""
	X = np.zeros((images.shape[0],size,size,3))
	for i in range(images.shape[0]):
		X[i]= imresize(images[i], (size,size,CHANNELS), interp='bilinear', mode=None)
	return X
