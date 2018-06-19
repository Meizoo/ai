
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import os
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu,floatX=float32"
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.layers import Input, Flatten, Dense,Dropout
from keras.models import Model
from keras.utils import np_utils
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras import regularizers,optimizers
import numpy as np
import keras
from keras.datasets import cifar10
from keras.callbacks import ReduceLROnPlateau, CSVLogger,EarlyStopping
from scipy.misc import toimage, imresize
from helper import get_class_names, get_train_data, get_test_data, plot_images, plot_model
# Hight and width of the images
IMAGE_SIZE = 64
# 3 channels, Red, Green and Blue
CHANNELS = 3
def Resize(images) :
    X = np.zeros((images.shape[0],IMAGE_SIZE,IMAGE_SIZE,3))
    for i in range(images.shape[0]):
        X[i]= imresize(images[i], (IMAGE_SIZE,IMAGE_SIZE,CHANNELS), interp='bilinear', mode=None)
    return X
class_names  = get_class_names()
print(class_names)
num_classes = len(class_names)
print(num_classes)
images_train, labels_train, class_train = get_train_data()

images_test, labels_test, class_test = get_test_data()

#Changing the shape of CIFAR10 dataset from 32*32*3 to 64*64*3#Changin 
images_train = Resize(images_train)
images_test = Resize(images_test)
mean = np.mean(images_train,axis=(0,1,2,3))
std = np.std(images_train,axis=(0,1,2,3))
x_train = (images_train-mean)/(std+1e-7)
x_test = (images_test-mean)/(std+1e-7)
print("Training set size:\t",len(images_train))
print("Testing set size:\t",len(images_test))

#Resizing the dataset#Resizi 

def PretrainedModel(num_of_classes) :
    
    #load vgg model from keras
    vgg_16_model = VGG16(weights=None,include_top= False,input_shape=(IMAGE_SIZE,IMAGE_SIZE,CHANNELS))
    vgg_16_model.summary()
    
    #get weights from pretrained model on imagenet
    vgg_16_model.load_weights('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')
    
    print(len(vgg_16_model.layers))
    
    #Freeze all layers
    #for i in range(19):
     # vgg_16_model.layers[i].trainable = False
    
    #change model input layer according to cifar-10 dataset
    inputs = Input(shape = (IMAGE_SIZE,IMAGE_SIZE,CHANNELS), name = "image_input")
    
    #create dummy layer
    output_vgg16_model = vgg_16_model(inputs)
    
    #Add the fully-connected layers 
    #Adding one fully connected layer instead of 2 to decrease overfitting
    x = Flatten(name='flatten')(output_vgg16_model)
    x = Dense(2048, activation='relu', name='fc1')(x)
    x = Dropout(0.5)(x)
    x = Dense(num_of_classes, activation='softmax', name='predictions')(x)

    #Create custom model
    cifar10_vgg = Model(inputs=inputs, outputs=x)

    #In the summary, weights and layers from VGG part will be hidden, but they will be fit during the training
    cifar10_vgg.summary()
    
    print(len(cifar10_vgg.layers))
    
    return cifar10_vgg

model = PretrainedModel(num_classes)

batch_size = 64
epochs = 10

opt = optimizers.rmsprop(lr=0.0001, decay=1e-6)
model.compile(loss='categorical_crossentropy', # Better loss function for neural networks
              optimizer=Adam(lr=1.0e-4), # Adam optimizer with 1.0e-4 learning rate
              metrics = ['accuracy']) # Metrics to be evaluated by the model
lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=2, min_lr=0.5e-6)
csv_logger = CSVLogger('./vgg16imagenetpretrained_upsampleimage_cifar10_data_argumentation.csv')
early_stopper = EarlyStopping(min_delta=0.001, patience=10)
model.fit(images_train, class_train,
	batch_size=batch_size*4,
	epochs=epochs,
	validation_data=(images_test, class_test),
	shuffle=True,callbacks=[lr_reducer, early_stopper, csv_logger])