# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 14:56:43 2020

@author: soderdahl
""" 

import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import pathlib
import cv2
import os
import numpy as np
import random
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier


# Get paths to data
DIR1 = 'C:/Users/soder/Desktop/dippakoodit/katodit/training_data/1/'
DIR9 = 'C:/Users/soder/Desktop/dippakoodit/katodit/training_data/9/'
DIR_romu = 'C:/Users/soder/Desktop/dippakoodit/katodit/training_data/romu/'
DIR1_ = 'C:/Users/soder/Desktop/dippakoodit/katodit/test_data/1/'
DIR9_ = 'C:/Users/soder/Desktop/dippakoodit/katodit/test_data/9/'
DIR_romu_ = 'C:/Users/soder/Desktop/dippakoodit/katodit/test_data/romu/'

class_names = ['9', '1', 'romu']

# parhaat tulokset: 400+100 kuvaa, (64,64) koko, Dense (128), Epochs 40 ------- 72%


# %% Functions

# gets data from path, resizes images and add labels to them
def get_data(data, path, label):
    lbl = label
    for image in os.listdir(path):
        img = cv2.imread(path + '/' + image, 1)
        img = cv2.resize(img, (32, 32))
        img_and_label = []
        img_and_label.append(img)
        img_and_label.append(lbl)
        data.append(img_and_label)
    
    return data


# separate images and labels to different arrays
def separate_img_and_lab(list):
    img = []
    label = []
    i = 0
    for element in list:
        img.append(list[i][0])
        label.append(list[i][1])
        i = i + 1
    return img, label


# change labels to integer values
def labels_to_int(labels):
    integer = []
    i = 0
    for elements in labels:
        if labels[i] == '1':
            integer.append(1)
        elif labels[i] == '9':
            integer.append(0)
        else:
            integer.append(2)
        i = i + 1
    return integer


# plots image and label that says what is predicted and its %, and what is correct label
def plot_image(i, predictions_array, true_label, img):
      true_label, img = true_label[i], img[i]
      plt.grid(False)
      plt.xticks([])
      plt.yticks([])
    
      plt.imshow(img, cmap=plt.cm.binary)
    
      predicted_label = np.argmax(predictions_array)
      if predicted_label == true_label:
        color = 'blue'
      else:
        color = 'red'
    
      plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                     100*np.max(predictions_array),
                                     class_names[true_label]),
                                     color=color)


# plots bar graph of predictions
def plot_value_array(i, predictions_array, true_label):
      true_label = true_label[i]
      plt.grid(False)
      plt.xticks(range(3))
      plt.yticks([])
    
      thisplot = plt.bar(range(3), predictions_array, color="#777777", tick_label = class_names)
      plt.ylim([0, 1])
      predicted_label = np.argmax(predictions_array)
    
      thisplot[predicted_label].set_color('red')
      thisplot[true_label].set_color('blue')


# %% Get data

# get training data
img_and_lab = []
img_and_lab = get_data(img_and_lab, DIR1, '1')
img_and_lab = get_data(img_and_lab, DIR9, '9')
img_and_lab = get_data(img_and_lab, DIR_romu, 'romu')

# shuffle training data
random.shuffle(img_and_lab)
# separate images and labels and change labels to binary
train_images, train_labels = separate_img_and_lab(img_and_lab)
train_labels = labels_to_int(train_labels)
# change lists to arrays
train_images = np.array(train_images)
train_labels = np.array(train_labels)

# same things for test data
img_and_lab = []
img_and_lab = get_data(img_and_lab, DIR1_, '1')
img_and_lab = get_data(img_and_lab, DIR9_, '9')
img_and_lab = get_data(img_and_lab, DIR_romu_, 'romu')

random.shuffle(img_and_lab)
test_images, test_labels = separate_img_and_lab(img_and_lab)
test_images = np.array(test_images)
test_labels = labels_to_int(test_labels)
test_labels = np.array(test_labels)

# %% Plot 25 first images in training data with labels
plt.figure(figsize = (10,10))
for i in range(9):
    plt.subplot(3,3,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap = plt.cm.binary)
    label = class_names[train_labels[i]]
    plt.xlabel(label)
plt.show()

# %% Data augmentation
train_generator = ImageDataGenerator(vertical_flip = True)
train_generator = train_generator.flow(train_images,train_labels)
vali_generator = ImageDataGenerator(vertical_flip = True)
vali_generator = vali_generator.flow(test_images,test_labels)

plt.figure(figsize = (10,10))
for i in range(9):
    plt.subplot(3,3,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_generator[i][0][0].astype('uint8'), cmap = plt.cm.binary)
    label = class_names[train_generator[i][1][0]]
    plt.xlabel(label)
plt.show()

# %% CNN

# Conv2D(filters, kernel_size, activation, input)
# Alkuperäne
# model = models.Sequential()
# model.add(layers.Conv2D(32, (3, 3), activation = 'relu', input_shape = (32, 32, 3)))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(64, (3, 3), activation = 'relu'))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(32, (3, 3), activation = 'relu'))
# model.add(layers.Flatten())
# model.add(layers.Dense(64, activation = 'relu')) # aluks 64
# model.add(layers.Dense(3))

model = models.Sequential()
model.add(layers.Conv2D(64, (3, 3), activation = 'relu', input_shape = (32, 32, 3)))
model.add(layers.Conv2D(64, (3, 3), activation = 'relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation = 'relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(32, (3, 3), activation = 'relu'))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation = 'relu')) # aluks 64
model.add(layers.Dense(3))

# model = models.Sequential()
# model.add(layers.Conv2D(32, (3, 3), activation = 'relu', input_shape = (32, 32, 3))) # padding
# model.add(layers.Conv2D(32, (3, 3), activation = 'relu'))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Dropout(0.1))

# model.add(layers.Conv2D(64, (3, 3), activation = 'relu')) # padding
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(64, (3, 3), activation = 'relu'))
# model.add(layers.MaxPooling2D((2, 2))) # tää ehkä huono
# model.add(layers.Dropout(0.1))

# model.add(layers.Flatten())
# model.add(layers.Dense(128, activation = 'relu')) # aluks 64
# model.add(layers.Dropout(0.25))
# model.add(layers.Dense(3))#, activation = 'softmax'))

model.output_shape
model.summary()

# %% Compile and train
model.compile(optimizer = 'adam',
              loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
              metrics = ['accuracy'])

history = model.fit(train_images, train_labels, epochs = 10, 
                    validation_data = (test_images, test_labels))
# history = model.fit_generator(train_generator, epochs = 30, 
#                     validation_data = (test_images, test_labels))
# history = model.fit_generator(train_generator, epochs = 100, 
#                     validation_data = vali_generator)

# %% Evaluate and plot accuracies
plt.plot(history.history['acc'], label = 'accuracy')
plt.plot(history.history['val_acc'], label = 'validation_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc = 'lower right')

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose = 2)

# %%
probability_model = tf.keras.Sequential([model, 
                                         tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_images)  

# Plots first image, predicted label and true label
i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i],  test_labels)
plt.show()

# Plot the first 15 test images, predicted labels and true labels
# Correct predictions in blue and incorrect predictions in red
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions[i], test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.show()