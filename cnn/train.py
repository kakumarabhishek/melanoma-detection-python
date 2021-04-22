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
from sklearn.decomposition import PCA

from xgboost import XGBClassifier


# save numpy array as csv file
from numpy import savetxt

READ_DATA = True
newData = []
labels = []
if not READ_DATA:
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
		file_of_feature_to_predict = file_of_feature_to_predict.replace("[", "")
		file_of_feature_to_predict = file_of_feature_to_predict.replace("'", "")
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
	
	# print(np.shape(data))
	# exit()
	# pca = PCA(n_components=1028)
	# pca.fit(data)
	# newData = pca.transform(data)
	newData = np.array(data)
	labels = np.array(labels)


	savetxt('data.csv', newData, delimiter=',')
	savetxt('labels.csv', labels, delimiter=',')


if READ_DATA:
	print("[INFO] reading data...")

	from numpy import genfromtxt
	newData = genfromtxt('data.csv', delimiter=',')
	labels = genfromtxt('labels.csv', delimiter=',')
	print("[SHAPES]...")
	print(newData.shape)
	print(labels.shape)
	# exit()
	# print("[PCA]...")
	# pca = PCA(n_components = 0.99)
	# pca.fit(data)
	# newData = pca.transform(data)
	# savetxt('reducedData.csv', newData, delimiter=',')

print("[SHAPE]...")
print(newData.shape)
print("[SPLITTING]...")
trainX, testX, trainY, testY = train_test_split(newData, labels, test_size=0.33, random_state=8)
# load the label encoder from disk
# le = pickle.loads(open(config.LE_PATH, "rb").read())
from sklearn.naive_bayes import GaussianNB

# gnb = XGBClassifier()
# predY = gnb.fit(trainX, trainY).predict(testX)
# print("Number of mislabeled points out of a total %d points : %d" % (np.shape(testX)[0], (testY != predY).sum()))
#

print("[MODEL]...")
model = XGBClassifier()
model.fit(trainX, trainY)
print("[PRED]...")
# make predictions for test data
predY = model.predict(testX)
predictions = [round(value) for value in predY]
# evaluate predictions
from sklearn.metrics import accuracy_score

print("[RESULTS]...")
accuracy = accuracy_score(testY, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

from sklearn.metrics import mean_absolute_error
print("Mean absolute error : %d" % (mean_absolute_error(testY, predY)))
# for x in range(75):
#     print("pred:" + str(predY[x]) + "  test:" + str(testY[x]) + "\n")

# print(testY)
# print(predY)

# import matplotlib.pyplot as plt
# x_coordinates = list(range(0, 84))

# plt.plot(x_coordinates, testY, 'o', color='red')
# plt.plot(x_coordinates, predY, 'o', color='blue')
# plt.show()

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
