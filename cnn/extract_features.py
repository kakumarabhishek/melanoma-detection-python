# import the necessary packages
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.applications import VGG16
from tensorflow.keras import models
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from sklearn.model_selection import train_test_split
from pyimagesearch import config
from imutils import paths
import numpy as np
import pickle
import random
import os
import cv2

import time

import json
from skimage.color import rgb2gray

from keras.models import Model
from keras.layers import Flatten
from keras.applications.resnet50 import ResNet50
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.utils.np_utils import to_categorical
from keras import layers
import tensorflow as tf

# load the network
print("[INFO] loading network...")
pre_trained_model = ResNet50(weights="imagenet", include_top=False, input_shape=(256, 256, 3))
# pre_trained_model = VGG16(weights="imagenet", include_top=False, input_shape=(256, 256, 3))

for layer in pre_trained_model.layers:
    print(layer.name)
    layer.trainable = False

# Used to specifiy directory name "all"
split = config.ALL_DATA_PATH

print("[INFO] processing '{} split'...".format(split))

# get skin lesion image paths
p = os.path.sep.join([config.BASE_PATH, split])
imagePaths = list(paths.list_images(p))

imageData = []
labels = []
imageNames = []

desired_size = 256

count = 0
print("[INFO] processing images...")
for imagePath in imagePaths:
	# load the input image using the Keras helper utility
	image = load_img(imagePath)
	image = np.array(image)

	# resize image to desired_size while padding to maintain aspect ratio
	old_size = image.shape
	ratio = float(desired_size)/max(old_size)
	new_size = tuple([int(x*ratio) for x in old_size])

	image = cv2.resize(image, (new_size[1], new_size[0]))

	delta_w = desired_size - new_size[1]
	delta_h = desired_size - new_size[0]
	top, bottom = delta_h//2, delta_h-(delta_h//2)
	left, right = delta_w//2, delta_w-(delta_w//2)

	color = [0, 0, 0]
	image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT,
    value=color)

	image = img_to_array(image)
	image = np.array(image)

	# Conversion to greyscale
	image = rgb2gray(image)
	image = np.repeat(image[..., np.newaxis], 3, -1)

	# preprocess the image by (1) expanding the dimensions and
	# (2) subtracting the mean RGB pixel intensity from the
	# ImageNet dataset
	# image = np.expand_dims(image, axis=0)
	# image = preprocess_input(image)

	# add the image to the batch
	imageData.append(image)

	# get feature we want to predict (D1)
	imageName = imagePath.split(os.path.sep)[7]
	file_of_feature_to_predict = os.path.sep.join(["../results",
												   "{}.json".format(os.path.splitext(imageName)[0])])
	imageNames.append(imageName)

	with open(file_of_feature_to_predict) as f:
		label_data = json.load(f)

	# Get Diameter data from json
	d1 = label_data['D1']
	# Program that generates json divides the pixel diameter by 72. Multiplying by 72 to obtain original pixel diameter
	d1 = (72 * d1) * ratio
	# print("d1: " + str(d1) + ", ratio: " + str(ratio) + ", image:" + imageName)
	d1 = round(d1)
	labels.append(d1)

imageNames = np.array(imageNames)
labels = np.array(labels)

numLabels = len(set(labels))

print("[INFO] splitting data...")
trainX, testX, trainY, testY = train_test_split(imageData, labels, test_size=0.2, random_state=1)
trainX, valX, trainY, valY = train_test_split(trainX, trainY, test_size=0.25, random_state=1)

# Used to keep track of imageNames for test labels
imagetrainX, imagetestX, imagetrainY, imagetestY = train_test_split(imageData, imageNames, test_size=0.2, random_state=1)
imagetrainX, imagevalX, imagetrainY, imagevalY = train_test_split(imagetrainX, imagetrainY, test_size=0.25, random_state=1)

trainX = np.array(trainX)
trainY = np.array(trainY)

testX = np.array(testX)
testY = np.array(testY)

valX = np.array(valX)
valY = np.array(valY)

print("[INFO] adding to network...")

# last_layer = pre_trained_model.get_layer('block5_pool')
last_layer = pre_trained_model.get_layer('conv5_block3_out')
# print('last layer output shape:', last_layer.output_shape)
last_output = last_layer.output

# Flatten the output layer to 1 dimension
x = layers.GlobalMaxPooling2D()(last_output)
# Add a fully connected layer with 512 hidden units and ReLU activation
x = layers.Dense(512, activation='relu')(x)
# Add a dropout rate of 0.5
x = layers.Dropout(0.5)(x)
# Add a final linear layer for regression
x = layers.Dense(1, activation="linear")(x)

model = Model(pre_trained_model.input, x)


print("[INFO] optimizing...")
optimizer = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True)
model.compile(loss='mean_absolute_percentage_error', optimizer=optimizer, metrics=['accuracy'])

model.summary()

print("[INFO] feature extraction 1/2...")
train_datagen = ImageDataGenerator(rotation_range=60, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, fill_mode='nearest')

train_datagen.fit(trainX)

val_datagen = ImageDataGenerator()
val_datagen.fit(valX)

print("[INFO] feature extraction 2/2...")
batch_size = 64
epochs = 3
history = model.fit_generator(train_datagen.flow(trainX,trainY, batch_size=batch_size),
                              epochs = epochs, validation_data = val_datagen.flow(valX, valY),
                              verbose = 1, steps_per_epoch=(trainX.shape[0] // batch_size), 
                              validation_steps=(valX.shape[0] // batch_size))

for layer in model.layers[:15]:
	layer.trainable = False

for layer in model.layers[15:]:
	layer.trainable = True

print("[INFO] fine tune...")
optimizer = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(loss='mean_absolute_percentage_error',
              optimizer=optimizer,
              metrics=['acc'])

model.summary()

print("[INFO] learning rate...")
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', patience=3, verbose=1, factor=0.5, min_lr=0.000001, cooldown=3)

batch_size = 64
epochs = 30
history = model.fit_generator(train_datagen.flow(trainX,trainY, batch_size=batch_size),
                              epochs = epochs, validation_data = val_datagen.flow(valX, valY),
                              verbose = 1, steps_per_epoch=(trainX.shape[0] // batch_size),
                              validation_steps=(valX.shape[0] // batch_size), callbacks=[learning_rate_reduction])

model.save('model/diameterNormalizedRegression')
# model = models.load_model('model/diameterNormalizedRegression')

loss_val, acc_val = model.evaluate(valX, valY, verbose=1)
print("Validation: accuracy = %f  ;  loss_v = %f" % (acc_val, loss_val))

loss_test, acc_test = model.evaluate(testX, testY, verbose=1)
print("Test: accuracy = %f  ;  loss = %f" % (acc_test, loss_test))

classes = model.predict(testX, verbose=1)

classes = np.array(classes)
testLabels = np.array(testY)
imageLabels = np.array(imagetestY)

print("[INFO] printing classes ...")
print(classes.shape)
print(classes[1])
print(classes[2])
print(classes[3])

print("[INFO] printing testLabels ...")
print(testLabels.shape)
print(testLabels[1])
print(testLabels[2])
print(testLabels[3])


output = np.column_stack((classes.flatten(),testLabels.flatten(), imageLabels.flatten()))
np.savetxt('outputRegressionNames.csv', output, delimiter=',', header="Predicted, Label, Image", fmt='%s')

