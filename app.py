import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras.utils
from keras.models import Sequential
from keras.layers import GlobalMaxPooling2D, LeakyReLU, Dense, Conv2D, MaxPooling2D, Dropout, Flatten, Lambda, ELU, Activation, BatchNormalization
from keras.layers.convolutional import Convolution2D, Cropping2D, ZeroPadding2D, MaxPooling2D
from keras.optimizers import SGD, Adam, RMSprop
from keras.layers.core import Activation
from keras import backend as K
from keras.utils import np_utils
from utility import cleanNoise, cleanNoise3, TrimImage, AugmentImages

IMG_SIZE = 56
SPLIT_RATIO = 0.8

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
#targets = np.concatenate((targets, targets))
print(targets.shape)


training = np.load('input/train_images.npy', encoding='bytes')

features = np.zeros(shape=(10000, IMG_SIZE, IMG_SIZE, 1), dtype=float)
testing_feature_origin = training[int(len(training)*SPLIT_RATIO):len(training), 1]
for i in range(10000):
    temp_img = cleanNoise3(training[i, 1])
    temp_img = TrimImage(temp_img)
    features[i, :, :, 0] = temp_img
    #features[i, :, :, 1] = temp_img
    #features[i, :, :, 2] = temp_img

#features = AugmentImages(features)

features /= 255

training_feature = features[0:int(len(features)*SPLIT_RATIO), :, :]
testing_feature = features[int(len(features)*SPLIT_RATIO):len(features), :, :]
training_target = targets[0:int(len(features)*SPLIT_RATIO)]
testing_target = targets[int(len(features)*SPLIT_RATIO):len(features)]

#training_feature = AugmentImages(training_feature)

#training_target = np.concatenate((training_target, training_target))
training_target_one_hot = keras.utils.to_categorical(training_target)
testing_target_one_hot = keras.utils.to_categorical(testing_target)
#targets_one_hot = keras.utils.to_categorical(targets)


model = Sequential()
row, col, ch = IMG_SIZE, IMG_SIZE, 1

model.add(ZeroPadding2D((1, 1), input_shape=(row, col, ch)))


# CNN model - Building the model suggested in paper

model.add(Convolution2D(filters=32, kernel_size =(3,3), strides= (1,1),
padding='same', name='conv1')) #32, 5
#model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2), name='pool1'))

model.add(Convolution2D(filters=64, kernel_size =(3,3), strides= (1,1),
padding='same', name='conv2'))  #64, 3
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2), name='pool2'))

model.add(Convolution2D(filters=128, kernel_size =(1,1), strides= (1,1),
padding='same', name='conv3'))  #128, 3
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2), name='pool3'))

model.add(Convolution2D(filters=256, kernel_size =(1,1), strides= (1,1),
padding='same', name='conv4'))  #128, 3
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2), name='pool4'))

model.add(Flatten())
model.add(Dropout(0.5))

model.add(Dense(512, name='dense1'))  #1024
# model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5)) #0.5

model.add(Dense(256, name='dense2'))  #256, 0.5
model.add(Activation('relu'))
model.add(Dropout(0.5)) #0.5

model.add(Dense(31,name='output'))
model.add(Activation('softmax'))  #softmax since output is within 50 classes

model.compile(loss='categorical_crossentropy', optimizer=Adam(),
              metrics=['accuracy'])



history = model.fit(training_feature, training_target_one_hot, batch_size=64,
                    epochs=50, verbose=1, validation_data=(testing_feature,
                                                           testing_target_one_hot))
'''
row, col, ch = IMG_SIZE, IMG_SIZE, 1
model = Sequential()
model.add(Conv2D(32, (5, 5), padding='same',
                 input_shape=(row, col, 1)))
model.add(LeakyReLU(alpha=0.02))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Conv2D(196, (5, 5)))
model.add(LeakyReLU(alpha=0.02))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(GlobalMaxPooling2D())
model.add(Dense(1024))
model.add(LeakyReLU(alpha=0.02))
model.add(Dropout(0.5)) 
model.add(Dense(31))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])

history = model.fit(features, targets_one_hot, batch_size=256, nb_epoch=80,
          verbose=1,validation_split=0.2)
'''
print(model.evaluate(testing_feature, testing_target_one_hot))


keys=list(category_index.keys())
values=list(category_index.values())
incorrects = np.nonzero(model.predict_classes(testing_feature).reshape((-1,)) !=
                        testing_target)
for i in incorrects[0]:
    fig = plt.figure(figsize=[8,6])
    fig.add_subplot(1,2,1)
    plt.imshow(testing_feature_origin[i].reshape(100, 100))
    fig.add_subplot(1,2,2)
    plt.imshow(testing_feature[i].reshape(IMG_SIZE, IMG_SIZE))
    right = keys[values.index(testing_target[i])]
    current_feature = np.zeros((1, 56, 56, 1))
    current_feature[0, :, :, :] = testing_feature[i]
    wrong = keys[values.index(model.predict_classes(current_feature))]
    plt.title('{}/{}'.format(right, wrong))
    plt.savefig('wrong/{}.jpg'.format(i))
    plt.close(fig)


plt.figure(figsize=[8,6])
plt.plot(history.history['loss'],'r',linewidth=3.0)
plt.plot(history.history['val_loss'],'b',linewidth=3.0)
plt.legend(['Training loss', 'Validation Loss'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Loss',fontsize=16)
plt.title('Loss Curves',fontsize=16)
plt.savefig('lost.png')

plt.figure(figsize=[8,6])
plt.plot(history.history['acc'],'r',linewidth=3.0)
plt.plot(history.history['val_acc'],'b',linewidth=3.0)
plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Accuracy',fontsize=16)
plt.title('Accuracy Curves',fontsize=16)
plt.savefig('accuracy.png')


# prediction
testing = np.load('input/test_images.npy', encoding='bytes')

testing_features = np.zeros(shape=(10000, IMG_SIZE, IMG_SIZE, 1), dtype=float)
for i in range(len(testing)):
    temp_img = cleanNoise3(testing[i, 1])
    temp_img = TrimImage(temp_img)
    testing_features[i, :, :, 0] = temp_img
    #testing_features[i, :, :, 1] = temp_img
    #testing_features[i, :, :, 2] = temp_img

testing_features /= 255

prediction = model.predict_classes(testing_features)
np.savetxt("prediction.csv", prediction, delimiter=",")
prediction = prediction.astype('int')
try:
    file = open("output.csv",'w') 
    file.write('Id,Category\n')
    keys=list(category_index.keys())
    values=list(category_index.values())
    for i in range(len(prediction)):
        cate = keys[values.index(prediction[i])]
        file.write(str(i)+','+cate+'\n')

    file.close()
except:
    pass
