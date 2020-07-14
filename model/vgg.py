# import the libraries as shown below

from tensorflow.keras.layers import Input, Lambda, Dense, Flatten,Conv2D
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img
from tensorflow.keras.models import Sequential
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
print('Import successful')


# re-size all the images to this
IMAGE_SIZE = [224, 224]

train_path = './data/Train'
valid_path = './data/Test'


# Import the Vgg 16 library and add preprocessing layer to the front of VGG
# Here we will be using imagenet weights

mobilnet = VGG19(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)

# don't train existing weights
for layer in mobilnet.layers:
    layer.trainable = False

# useful for getting number of output classes
folders = glob('./data/Train/*')
print(folders)


# our layers -
x = Flatten()(mobilnet.output)

prediction = Dense(len(folders), activation='softmax')(x)

# create a model object
model = Model(inputs=mobilnet.input, outputs=prediction)


# to check model summary 
model.summary()

# compiling model 
model.compile(
  loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)


# Use the Image Data Generator to import the images from the dataset
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

# preparing training set for model 
training_set = train_datagen.flow_from_directory('./data/Train',
                                                 target_size = (224, 224),
                                                 batch_size = 10,
                                                 class_mode = 'categorical')
#preparing the  test set                                            
test_set = test_datagen.flow_from_directory('./data/Test',
                                            target_size = (224, 224),
                                            batch_size = 10,
                                            class_mode = 'categorical')

