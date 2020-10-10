# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 14:27:54 2020

@author: soderdahl
"""

from tensorflow.keras import datasets, layers, models, applications, Input, Model


def LeNet():
    model = models.Sequential()
    model.add(layers.Conv2D(6, (5, 5), strides=(1,1), activation = 'tanh', input_shape = (32, 32, 3)))
    model.add(layers.AveragePooling2D((2,2), strides=(2,2)))
    #model.add(layers.MaxPooling2D((2,2), strides=(2,2)))
    model.add(layers.Conv2D(16, (5, 5), strides=(1,1), activation = 'tanh'))
    model.add(layers.AveragePooling2D((2,2), strides=(2,2)))
    #model.add(layers.MaxPooling2D((2,2), strides=(2,2)))
    
    model.add(layers.Flatten())
    model.add(layers.Dense(120, activation = 'tanh'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(84, activation = 'tanh'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(3, activation = 'softmax'))
    
    model.output_shape
    model.summary()
    
    return model


def AlexNet():
    model = models.Sequential()
    model.add(layers.Conv2D(96, (11,11), strides=(4,4), activation='relu', input_shape=(227,227,3)))
    model.add(layers.MaxPooling2D((3,3), strides=(2,2)))
    model.add(layers.Conv2D(256, (5,5), strides=(1,1), activation='relu'))
    model.add(layers.MaxPooling2D((3,3), strides=(2,2)))
    model.add(layers.Conv2D(384, (3,3), strides=(1,1), activation='relu'))
    model.add(layers.Conv2D(384, (3,3), strides=(1,1), activation='relu'))
    model.add(layers.Conv2D(256, (3,3), strides=(1,1), activation='relu'))
    model.add(layers.MaxPooling2D((3,3), strides=(2,2)))
    
    model.add(layers.Flatten())
    model.add(layers.Dense(4096, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(4096, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1000, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(3, activation='softmax'))
    
    model.output_shape
    model.summary()
    
    return model


def CustomNet():
    # Conv2D(filters, kernel_size, activation, input)

    model = models.Sequential()
    model.add(layers.Conv2D(16, (3, 3), activation = 'tanh', input_shape = (32, 32, 3)))
    model.add(layers.Conv2D(16, (3, 3), activation = 'tanh'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.1))
    
    model.add(layers.Conv2D(16, (3, 3), activation = 'tanh'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(16, (3, 3), activation = 'tanh'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.1))
    
    model.add(layers.Flatten())
    model.add(layers.Dense(100, activation = 'tanh'))
    model.add(layers.Dropout(0.25))
    model.add(layers.Dense(3, activation = 'softmax'))
    
    return model