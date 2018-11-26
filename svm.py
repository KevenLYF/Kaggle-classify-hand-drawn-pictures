import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import csv
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
import warnings; warnings.simplefilter('ignore')
from utility import cleanNoise, TrimImage, moveToMid, cleanNoise3, sharpening, pre_clean

images = np.load('./data/train_images.npy', encoding="latin1")

def readLabel(filepath):
    with open(filepath) as csvFile:
        dataSet = csv.reader(csvFile, delimiter=',')
        label = []
        for row in dataSet:
            label.append(row[1])
    return label[1:]


trainX = list(images[:, 1])
trainY = readLabel('./data/train_labels.csv')

c_range = []
d_range = [True, False]
c = 1
for i in range(5):
    c *= 0.8
    c_range.append(c)

def gridSearch_SVM(X_train, y_train):

    param_grid = dict(C=c_range, dual=d_range)
    svm = LinearSVC(C=c_range, dual=d_range)
    gs = GridSearchCV(svm, param_grid, scoring='f1_micro', n_jobs=-1, verbose=50)
    gs.fit(X_train, y_train)
    best_score = gs.best_score_
    best_param = gs.best_params_
    print("The F1 measure = {} \nC = {}\nDual = {}".format(best_score, best_param.get('C'), best_param.get('dual')))

# count = 0
# countAll = 0
for i in range(10000):
    trainX[i] = cleanNoise3(trainX[i])
    trainX[i] = pre_clean(trainX[i])
    trainX[i] = TrimImage(trainX[i])
    trainX[i] = sharpening(trainX[i])
    # trainX[i] = moveToMid(trainX[i])
    trainX[i] = trainX[i].flatten()
    # if (trainX[i] == 0).all():
    #     if (trainY[i] == 'empty'):
    #         print(str(i))
    #         count += 1
    #     countAll += 1
    # plt.imshow(trainX[i])
    # plt.show()

#
# print(count)
# print(countAll)

gridSearch_SVM(trainX, trainY)
