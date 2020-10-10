# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 13:17:44 2020

@author: soderdahl
"""


import tensorflow as tf
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras import datasets, layers, models, applications, Input, Model

import functions as fun
import architectures as cnn

# Get paths to data folders
# If data is not expanded
# DIR1 = 'C:/Users/soder/Desktop/dippakoodit/katodit/training_data/1/'
# DIR9 = 'C:/Users/soder/Desktop/dippakoodit/katodit/training_data/0/'
# DIRR = 'C:/Users/soder/Desktop/dippakoodit/katodit/training_data/2/'
# DIR1_ = 'C:/Users/soder/Desktop/dippakoodit/katodit/test_data/1/'
# DIR9_ = 'C:/Users/soder/Desktop/dippakoodit/katodit/test_data/0/'
# DIRR_ = 'C:/Users/soder/Desktop/dippakoodit/katodit/test_data/2/'

# If data is expanded
DIR1 = 'C:/Users/soder/Desktop/dippakoodit/katodit/data/1/'
DIR9 = 'C:/Users/soder/Desktop/dippakoodit/katodit/data/0/'
DIRR = 'C:/Users/soder/Desktop/dippakoodit/katodit/data/2/'

TRAIN_PATHS = [DIR9, DIR1, DIRR]
# TEST_PATHS = [DIR9_, DIR1_, DIRR_]

CLASS_NAMES = ['priima', 'std', 'romu']

# PARAMETERS
IMG_SIZE = 32
EPOCHS = 5
BATCH = 500
LEARNING_RATE = 0.00001
VERTICAL_FLIP = True
HORIZONTAL_FLIP = True

# %% Get data and model
# Get data without expanded data
# train_images, train_labels = fun.pre_process(TRAIN_PATHS, CLASS_NAMES, IMG_SIZE)
# test_images, test_labels = fun.pre_process(TEST_PATHS, CLASS_NAMES, IMG_SIZE)

# Get data with expanded data
train_images, test_images, train_labels, test_labels = fun.pre_process2(TRAIN_PATHS, CLASS_NAMES, IMG_SIZE)

#fun.plot_images(train_images, train_labels, CLASS_NAMES)

# Augment data
train_generator = fun.augmentation(train_images, train_labels, VERTICAL_FLIP, HORIZONTAL_FLIP, BATCH)
test_generator = fun.augmentation(test_images, test_labels, VERTICAL_FLIP, HORIZONTAL_FLIP, BATCH)

#fun.plot_images(test_generator, train_labels, CLASS_NAMES)

# Choose model
model = cnn.LeNet()
# model = cnn.CustomNet()

# %% Transfer learning
# pretrainedNet = applications.ResNet50(weights='imagenet', include_top=False, input_shape=(32,32,3))
# pretrainedNet.trainable = False
# inputs = Input(shape=(32,32,3))
# x = pretrainedNet(inputs, training=False)
# x = layers.GlobalAveragePooling2D()(x)
# outputs = layers.Dense(3)(x)
# model = Model(inputs, outputs)    

# %% Compile and train model
model.output_shape
model.summary()    

# model.compile(optimizer = 'adam',
#               loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
#               metrics = ['accuracy'])
model.compile(optimizer = Adam(),
              loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
              metrics = ['accuracy'])

CSV = tf.keras.callbacks.CSVLogger('test', append=True)

# treenaus ilman augmentointia
# history = model.fit(train_images, train_labels, batch_size = BATCH, epochs = EPOCHS, 
#                     validation_data = (test_images, test_labels))

# treenaus training datan augmentoinnilla
history = model.fit_generator(train_generator, epochs = EPOCHS, 
                    validation_data = (test_images, test_labels), callbacks=[CSV])

# treenaus molempien datojen augmentoinnilla
# history = model.fit_generator(train_generator, epochs = EPOCHS, 
#                     validation_data = test_generator)

# %% Fine-tuning
# EPOCHS = 5
# pretrainedNet.trainable = True
# model.summary()
# model.compile(optimizer = Adam(1e-5),
#               loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
#               metrics = ['accuracy'])

# treenaus ilman augmentointia
# history = model.fit(train_images, train_labels, batch_size = BATCH, epochs = EPOCHS, 
#                     validation_data = (test_images, test_labels))

# treenaus training datan augmentoinnilla
history = model.fit_generator(train_generator, epochs = EPOCHS, 
                    validation_data = (test_images, test_labels), callbacks=[CSV])

# %%
fun.evaluate_and_plot(history, model, test_images, test_labels, CLASS_NAMES)
