# import the necessary packages
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from pyimagesearch import config
from imutils import paths
import numpy as np
import pickle
import random
import os

from skimage.color import rgb2gray

from keras.models import Model
from keras.layers import Flatten
from keras.applications.resnet50 import ResNet50

# load the network and initialize the label encoder
print("[INFO] loading network...")
model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
flat1 = Flatten()(model.layers[-1].output)
model = Model(inputs=model.inputs, outputs=flat1)
# summarize the model
model.summary()


split = config.ALL_DATA_PATH

print("[INFO] processing '{} split'...".format(split))

# get skin lesion images
p = os.path.sep.join([config.BASE_PATH, split])

imagePaths = list(paths.list_images(p))

# get file names of skin lesion images
labels = [p.split(os.path.sep)[7] for p in imagePaths]

# open the output CSV file for writing
csvPath = os.path.sep.join([config.BASE_CSV_PATH,
	"{}.csv".format(split)])
csv = open(csvPath, "w")


batchImages = []
for imagePath in imagePaths:
	# load the input image using the Keras helper utility
	# while ensuring the image is resized to 224x224 pixels
	image = load_img(imagePath, target_size=(224, 224))

	image = img_to_array(image)
	image = np.array(image)

	# Testing conversion to greyscale
	image = rgb2gray(image)
	image = np.repeat(image[..., np.newaxis], 3, -1)

	# preprocess the image by (1) expanding the dimensions and
	# (2) subtracting the mean RGB pixel intensity from the
	# ImageNet dataset
	image = np.expand_dims(image, axis=0)
	image = preprocess_input(image)

	# add the image to the batch
	batchImages.append(image)

# pass the images through the network and use the outputs as
# our actual features
batchImages = np.vstack(batchImages)
# labels = np.vstack(labels)
features = model.predict(batchImages)
print("------- Dimensions -----")
print(features.shape)

# loop over the labels and extracted features
for (label, vec) in zip(labels, features):
	# construct a row that exists of the class label and
	# extracted features
	vec = ",".join([str(v) for v in vec])
	csv.write("{},{}\n".format(label, vec))
# close the CSV file
csv.close()
