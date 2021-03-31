# import the necessary packages
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from pyimagesearch import config
import numpy as np
import pickle
import os
import json
from sklearn.model_selection import train_test_split
import random

# derive the paths to the CSV file containing CNN feature data
csvPath = os.path.sep.join([config.BASE_CSV_PATH,
	"{}.csv".format(config.ALL_DATA_PATH)])

data = []
labels = []

for row in open(csvPath):
	row = row.strip().split(",")
	imagePath = row[0]

	# derive path to json file containing the ground truth features to predict
	file_of_feature_to_predict = os.path.sep.join([config.BASE_FEATURE_PATH,
	"{}.json".format(os.path.splitext(imagePath)[0])])

	with open(file_of_feature_to_predict) as f:
		label_data = json.load(f)
  	# derive the D1 ground truth feature and add it to label list
	labels.append(label_data['D1'])
	# random.shuffle(labels)
	# print(feature_data['D1'])
	# exit()

	features = np.array(row[1:], dtype="float")
	data.append(features)

# load the data from disk
print("[INFO] loading data...")
# (trainX, trainY) = load_data_split(trainingPath)
# (testX, testY) = load_data_split(testingPath)

trainX, testX, trainY, testY = train_test_split(data, labels, test_size=0.33, random_state=8)
# load the label encoder from disk
# le = pickle.loads(open(config.LE_PATH, "rb").read())
from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
predY = gnb.fit(trainX, trainY).predict(testX)
print("Number of mislabeled points out of a total %d points : %d" % (np.shape(testX)[0], (testY != predY).sum()))
# print(testY)
# print(predY)

import matplotlib.pyplot as plt
x_coordinates = list(range(0, 84))

plt.plot(x_coordinates, testY, 'o', color='red')
plt.plot(x_coordinates, predY, 'o', color='blue')
plt.show()

# train the model
# print("[INFO] training model...")
# model = LogisticRegression(solver="lbfgs", multi_class="auto",
# 	max_iter=150)
# model.fit(trainX, trainY)
# # evaluate the model
# print("[INFO] evaluating...")
# preds = model.predict(testX)
# print(classification_report(testY, preds))
# # serialize the model to disk
# print("[INFO] saving model...")
# f = open(config.MODEL_PATH, "wb")
# f.write(pickle.dumps(model))
# f.close()