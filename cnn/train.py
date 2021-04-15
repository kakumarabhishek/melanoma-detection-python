# set the matplotlib backend so figures can be saved in the background
# import matplotlib
# matplotlib.use("Agg")
# import the necessary packages
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from imutils import paths
# import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import pickle
import cv2
import os

import json

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset of images")
ap.add_argument("-m", "--model", required=True,
	help="path to output trained model")
ap.add_argument("-l", "--label-bin", required=True,
	help="path to output label binarizer")
# ap.add_argument("-p", "--plot", required=True,
# 	help="path to output accuracy/loss plot")
args = vars(ap.parse_args())

# initialize the data and labels
print("[INFO] loading images...")
data = []
labels = []
# grab the image paths and randomly shuffle them
imagePaths = list(paths.list_images(args["dataset"]))
random.seed(42)
random.shuffle(imagePaths)
# loop over the input images

i = 1
for imagePath in imagePaths:
	print("[INFO] image " + str(i) + " (1/3) ...")
	# load the image, resize the image to be 32x32 pixels (ignoring
	# aspect ratio), flatten the image into 32x32x3=3072 pixel image
	# into a list, and store the image in the data list
	image = cv2.imread(imagePath)
	image = cv2.resize(image, (32, 32)).flatten()
	data.append(image)
	# extract the class label from the image path and update the
	# labels list
	imageName = imagePath.split(os.path.sep)[6]
	print("[INFO] image " + str(i) + " (2/3) ...")
	# print(imagePath)
	# print(imagePath.split(os.path.sep)[5])
	# exit()

	# derive path to json file containing the ground truth features to predict
	file_of_feature_to_predict = os.path.sep.join(["../results",
												   "{}.json".format(os.path.splitext(imageName)[0])])

	with open(file_of_feature_to_predict) as f:
		label_data = json.load(f)
	# derive the D1 ground truth feature and add it to label list
	labels.append(label_data['D1'])
	print("[INFO] image " + str(i) + " (3/3) ...")
	i = i +1

	# print("imageName:" + imageName + "  label:" + str(label_data['D1']))
# print(sorted(set(labels)))
# exit()
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
(trainX, testX, trainY, testY) = train_test_split(data,
	labels, test_size=0.25, random_state=42)

# convert the labels from integers to vectors (for 2-class, binary
# classification you should use Keras' to_categorical function
# instead as the scikit-learn's LabelBinarizer will not return a
# vector)
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

# define the 3072-1024-512-3 architecture using Keras
model = Sequential()
model.add(Dense(1024, input_shape=(3072,), activation="sigmoid"))
model.add(Dense(512, activation="sigmoid"))
model.add(Dense(len(lb.classes_), activation="softmax"))

# initialize our initial learning rate and # of epochs to train for
INIT_LR = 0.01
EPOCHS = 80
# compile the model using SGD as our optimizer and categorical
# cross-entropy loss (you'll want to use binary_crossentropy
# for 2-class classification)
print("[INFO] training network...")
opt = SGD(lr=INIT_LR)
model.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# train the neural network
H = model.fit(x=trainX, y=trainY, validation_data=(testX, testY),
	epochs=EPOCHS, batch_size=32)

# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(x=testX, batch_size=32)
print(classification_report(testY.argmax(axis=1),
	predictions.argmax(axis=1)))