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
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.utils.np_utils import to_categorical

# load the network and initialize the label encoder
print("[INFO] loading network...")
pre_trained_model = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

for layer in pre_trained_model.layers:
    print(layer.name)
    layer.trainable = False

print(len(pre_trained_model.layers))
exit()
# flat1 = Flatten()(model.layers[-1].output)

# model = Model(inputs=model.inputs, outputs=flat1)
# summarize the model


split = config.ALL_DATA_PATH

print("[INFO] processing '{} split'...".format(split))

# get skin lesion images
p = os.path.sep.join([config.BASE_PATH, split])

imagePaths = list(paths.list_images(p))

# get file names of skin lesion images
# labels = [p.split(os.path.sep)[7] for p in imagePaths]

# open the output CSV file for writing
csvPath = os.path.sep.join([config.BASE_CSV_PATH,
	"{}.csv".format(split)])
csv = open(csvPath, "w")


batchImages = []
labels = []

print("[INFO] processing images...")
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

	imageName = imagePath.split(os.path.sep)[6]
	file_of_feature_to_predict = os.path.sep.join(["../results",
												   "{}.json".format(os.path.splitext(imageName)[0])])

	with open(file_of_feature_to_predict) as f:
		label_data = json.load(f)
	# derive the D1 ground truth feature and add it to label list
	d1 = label_data['D1']
	labels.append(d1)


print("[INFO] splitting data...")
# (trainX, testX, trainY, testY) = train_test_split(batchImages,
# 	labels, test_size=0.25, random_state=42)

trainX, testX, trainY, testY = train_test_split(batchImages, labels, test_size=0.2, random_state=1)

trainX, valX, trainY, valY = train_test_split(trainX, trainY, test_size=0.25, random_state=1)

print("trainX: " + trainX.shape + "   trainY: " +  trainY.shape)
print("testX: " + testX.shape + "   testY: " +  testY.shape)
print("valX: " + valX.shape + "   valY: " +  valY.shape)

trainY = to_categorical(trainY)
valY = to_categorical(valY)
testY = to_categorical(testY)

print("[INFO] adding to network...")

# Flatten the output layer to 1 dimension
x = layers.GlobalMaxPooling2D()(last_output)
# Add a fully connected layer with 512 hidden units and ReLU activation
x = layers.Dense(512, activation='relu')(x)
# Add a dropout rate of 0.5
x = layers.Dropout(0.5)(x)
# Add a final sigmoid layer for classification
x = layers.Dense(len(lb.classes_), activation="softmax")(x)

model = Model(pre_trained_model.input, x)

print("[INFO] optimizing...")
optimizer = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

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
model.compile(loss='categorical_crossentropy',
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

loss_val, acc_val = model.evaluate(valX, valY, verbose=1)
print("Validation: accuracy = %f  ;  loss_v = %f" % (acc_val, loss_val))

loss_test, acc_test = model.evaluate(testX, testY, verbose=1)
print("Test: accuracy = %f  ;  loss = %f" % (acc_test, loss_test))

# print("[INFO] fitting...")
# model.fit(trainX, trainY, batch_size=32, epochs=50)



# print("[INFO] testing...")
# loss_test, acc_test = model.evaluate(testX, testY, verbose=1)
# print("Test: accuracy = %f  ;  loss = %f" % (acc_test, loss_test))
# pass the images through the network and use the outputs as
# our actual features
# batchImages = np.vstack(batchImages)
# labels = np.vstack(labels)
# features = model.predict(batchImages)
# print("------- Dimensions -----")
# print(features.shape)

# # loop over the labels and extracted features
# for (label, vec) in zip(labels, features):
# 	# construct a row that exists of the class label and
# 	# extracted features
# 	vec = ",".join([str(v) for v in vec])
# 	csv.write("{},{}\n".format(label, vec))
# # close the CSV file
# csv.close()
