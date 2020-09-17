# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 13:17:44 2020

@author: soderdahl
"""


# import tensorflow as tf
# from tensorflow.keras import datasets, layers, models
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# import matplotlib.pyplot as plt
# import cv2
# import os
# import numpy as np
# import random

import tensorflow as tf
from tensorflow.keras.optimizers import Adam, SGD

import functions as fun
import architectures as cnn

# Get paths to data folders
DIR1 = 'C:/Users/soder/Desktop/dippakoodit/katodit/training_data/1/'
DIR9 = 'C:/Users/soder/Desktop/dippakoodit/katodit/training_data/0/'
DIRR = 'C:/Users/soder/Desktop/dippakoodit/katodit/training_data/2/'
DIR1_ = 'C:/Users/soder/Desktop/dippakoodit/katodit/test_data/1/'
DIR9_ = 'C:/Users/soder/Desktop/dippakoodit/katodit/test_data/0/'
DIRR_ = 'C:/Users/soder/Desktop/dippakoodit/katodit/test_data/2/'

TRAIN_PATHS = [DIR1, DIR9, DIRR]
TEST_PATHS = [DIR1_, DIR9_, DIRR_]

CLASS_NAMES = ['9', '1', 'romu']

# PARAMETERS
IMG_SIZE = 227
EPOCHS = 20
BATCH = 50
LEARNING_RATE = 0.0000000001
N_CLASSES = 3 # ei käytössä
CLR = 0 # 0 rgb, 1 gray # ei käytössä
VERTICAL_FLIP = True
HORIZONTAL_FLIP = False
PADDING = 'valid' # ei käytössä
POOLING = 'max' # ei käytössä
OPTIMIZER = 'adam' # ei käytössä

# %% Get data and model
train_images, train_labels = fun.pre_process(TRAIN_PATHS, CLASS_NAMES, IMG_SIZE)
test_images, test_labels = fun.pre_process(TEST_PATHS, CLASS_NAMES, IMG_SIZE)

fun.plot_images(train_images, train_labels, CLASS_NAMES)

train_generator = fun.augmentation(train_images, train_labels, VERTICAL_FLIP, HORIZONTAL_FLIP, BATCH)
test_generator = fun.augmentation(test_images, test_labels, VERTICAL_FLIP, HORIZONTAL_FLIP, BATCH)

fun.plot_images(train_generator, train_labels, CLASS_NAMES)

model = cnn.AlexNet()


# %% Compile and train model
model.compile(optimizer = Adam(lr=LEARNING_RATE),
              loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
              metrics = ['accuracy'])


# treenaus ilman augmentointia
history = model.fit(train_images, train_labels, batch_size = BATCH, epochs = EPOCHS, 
                    validation_data = (test_images, test_labels))


# treenaus training datan augmentoinnilla
# history = model.fit_generator(train_generator, epochs = EPOCHS, 
#                     validation_data = (test_images, test_labels))


# treenaus molempien datojen augmentoinnilla
# history = model.fit_generator(train_generator, epochs = EPOCHS, 
#                     validation_data = test_generator)


# %% Evaluate and plotting
fun.evaluate_and_plot(history, model, test_images, test_labels, CLASS_NAMES)


# %%
# def main():
#     # get training and testing data
#     train_images, train_labels = fun.pre_process(TRAIN_PATHS, CLASS_NAMES)

# if __name__ == "__main__":
#     main()
    