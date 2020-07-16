# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 09:39:29 2020

@author: mnerurkar
"""
# Import required libraries

import cv2
import math
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from skimage.transform import resize
from keras.utils import np_utils
from keras.applications.imagenet_utils import preprocess_input
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.applications.vgg16 import VGG16
from keras.layers import Dense, InputLayer, Dropout
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import ModelCheckpoint

# Creating images from Training video

count = 0   # count: for image number
videoFile = "Tom and jerry.mp4"     # Training video file name
cap = cv2.VideoCapture(videoFile)   # Capturing the video
frameRate = cap.get(5)              # Frame rate: frames per 5 milliseconds
while(cap.isOpened()):
    frameId = cap.get(1)            #current frame number
    ret, frame = cap.read()
    if (ret != True):
        break
    if (frameId % math.floor(frameRate) == 0):
        filename ="frame%d.jpg" % count;count+=1
        cv2.imwrite(filename, frame)
cap.release()
print ("Done!")


# Creating images from Testing video

count = 0
videoFile = "Tom and Jerry 3.mp4"
cap = cv2.VideoCapture(videoFile)
frameRate = cap.get(5) 
while(cap.isOpened()):
    frameId = cap.get(1)
    ret, frame = cap.read()
    if (ret != True):
        break
    if (frameId % math.floor(frameRate) == 0):
        filename ="test%d.jpg" % count;count+=1
        cv2.imwrite(filename, frame)
cap.release()
print ("Done!")

# Reading data created by Training and Testing images

data = pd.read_csv('mapping.csv')
test = pd.read_csv('testing.csv')

# Read the training images and store it into an array

X = []
for img_name in data.Image_ID:
    img = plt.imread('' + img_name)
    X.append(img)
X = np.array(X)

# Read the testing images and store it into an array

test_image = []
for img_name in test.Image_ID:
    img = plt.imread('' + img_name)
    test_image.append(img)
test_img = np.array(test_image)

# One hot encoding of Class column from training and testing dataset as there are 4 different classes
#	0: none are present
#	1: only Jerry is present
#	2: only Tom is present
#	3: both are present

train_y = np_utils.to_categorical(data.Class)
test_y = np_utils.to_categorical(test.Class)

# Reshaping all the training and testing images as we will be using 
# VGG16 pretrained model which takes input
# image of shape (224 x 224 X 3)

image = []
for i in range(0,X.shape[0]):
    a = resize(X[i], preserve_range=True, output_shape=(224,224,3)).astype(int)
    image.append(a)
X = np.array(image)

test_image = []
for i in range(0,test_img.shape[0]):
    a = resize(test_img[i], preserve_range=True, output_shape=(224,224)).astype(int)
    test_image.append(a)
test_image = np.array(test_image)

# Scaling of pixels between -1 and 1

X = preprocess_input(X, mode='tf')
test_image = preprocess_input(test_image, mode='tf')

# Splitting of training and validation datasets

X_train, X_valid, y_train, y_valid = train_test_split(X, train_y, test_size=0.3, random_state=42)

# using VGG16 (a pretrained model) as a starting point for our model
# using pretrained weights of imagenet
# include_top = False is to exclude the output layers of the model as
# we will be fitting our own model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# using prediction features of base model for our own model

X_train = base_model.predict(X_train)
X_valid = base_model.predict(X_valid)
test_image = base_model.predict(test_image)

# converting 2-D to 1-D for our neural network

X_train = X_train.reshape(208, 7*7*512)
X_valid = X_valid.reshape(90, 7*7*512)
test_image = test_image.reshape(186, 7*7*512)

# Preprocess the images and make them zero-centered 
# which helps the model to converge faster

train = X_train/X_train.max()
X_valid = X_valid/X_train.max()
test_image = test_image/test_image.max()

# Added dropout as the model was overfitting
# Increased the number of layers for better accuracy

model = Sequential()
model.add(InputLayer((7*7*512,)))    # input layer
model.add(Dense(units=1024, activation='sigmoid'))   # hidden layer
model.add(Dropout(0.5))      # adding dropout
model.add(Dense(units=512, activation='sigmoid'))    # hidden layer
model.add(Dropout(0.5))      # adding dropout
model.add(Dense(units=256, activation='sigmoid'))    # hidden layer
model.add(Dropout(0.5))      # adding dropout
model.add(Dense(4, activation='softmax'))            # output layer

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# computing weights of different classes, to make them balanced
class_weights = compute_class_weight('balanced',np.unique(data.Class), data.Class)
# converting numpy array to dictionary
class_weights = {i : class_weights[i] for i in range(class_weights.size)}

# used ModelCheckpoint to save the best model, with lowest validation loss
filepath="weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]      # model check pointing based on validation loss


model.fit(train, y_train, epochs=100, validation_data=(X_valid, y_valid), class_weight=class_weights, callbacks=callbacks_list)

# loading the best weights for final predictions
model.load_weights("weights.best.hdf5")

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

scores = model.evaluate(test_image, test_y)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

predictions = model.predict_classes(test_image)

print("The screen time of JERRY is", predictions[predictions==1].shape[0], "seconds")
print("The screen time of TOM is", predictions[predictions==2].shape[0], "seconds")
print("The screen time of TOM and JERRY is", predictions[predictions==3].shape[0], "seconds")
