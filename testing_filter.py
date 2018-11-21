import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utility import cleanNoise, TrimImage
import cv2

def cleanNoise3(img):
    img=img.astype(np.uint8)
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=4)
    sizes = stats[:, -1]

    max_label = 1
    max_size = sizes[1]
    for i in range(2, nb_components):
        if sizes[i] > max_size:
            max_label = i
            max_size = sizes[i]

    img2 = np.array(img)
    img2[output != max_label] = 0

    for i in range(100):
        for j in range(100):
            if (img2[i][j] > 200):
                img2[i][j] = 255
            else:
                img2[i][j] = 0

    min_value = 24 * 255
    if (img2.sum() < min_value):
        img2.fill(0)
    return img2


training_label = pd.read_csv('input/train_labels.csv')
category_index = {}
current_index = 0
for index, row in training_label.iterrows():
    if row['Category'] not in category_index:
        category_index[row['Category']] = current_index
        current_index += 1

targets = []
for index, row in training_label.iterrows():
    targets.append(category_index[row['Category']])
targets = np.array(targets)

training = np.load('input/train_images.npy', encoding='bytes')
images = training[:, 1]
count = 0

for i in range(10):
# for i in range(100):
    image = images[i]
    img = cleanNoise3(image.reshape(100, 100))
    img = TrimImage(img)
    cv2.imshow("Biggest component", img)
    cv2.waitKey()

    # if (img.sum() == 0):
    #     count += 1
    #     print(i)
# print(count)

