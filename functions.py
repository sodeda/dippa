# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 13:32:02 2020

@author: soderdahl
"""

import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# %% Functions

def pre_process(paths, class_names, size):
    img_and_lab = []
    img_and_lab = get_data(img_and_lab, paths[0], '1', size)
    img_and_lab = get_data(img_and_lab, paths[1], '9', size)
    img_and_lab = get_data(img_and_lab, paths[2], 'romu', size)
    
    # shuffle data
    random.shuffle(img_and_lab)
    
    # separate images and labels and change labels to binary
    images, labels = separate_img_and_lab(img_and_lab)
    labels = labels_to_int(labels)
    
    # change lists to arrays
    images = np.array(images)
    labels = np.array(labels)
    
    return images, labels
    

# gets data from path, resizes images and add labels to them
def get_data(data, path, label, size):
    lbl = label
    for image in os.listdir(path):
        img = cv2.imread(path + '/' + image, 1) # 0 gray, 1 rgb
        img = cv2.resize(img, (size, size))
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


def augmentation(images, labels, vert, hori, batch):
    generator = ImageDataGenerator(vertical_flip = vert, horizontal_flip = hori)
    generator = generator.flow(images, labels, batch)
    
    return generator


def plot_images(images, labels, class_names):
    # Plot 9 first images in training data with labels
    plt.figure(figsize = (10,10))
    for i in range(9):
        plt.subplot(3,3,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        
        try:
            plt.imshow(images[i], cmap = plt.cm.binary)
            label = class_names[labels[i]]
        except:
            plt.imshow(images[i][0][0].astype('uint8'), cmap = plt.cm.binary)
            label = class_names[images[i][1][0]]
                
        plt.xlabel(label)
    plt.show()


# plots image and label that says what is predicted and its %, and what is correct label
def plot_image(i, predictions_array, true_label, img, class_names):
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
def plot_value_array(i, predictions_array, true_label, class_names):
      true_label = true_label[i]
      plt.grid(False)
      plt.xticks(range(3))
      plt.yticks([])
    
      thisplot = plt.bar(range(3), predictions_array, color="#777777", tick_label = class_names)
      plt.ylim([0, 1])
      predicted_label = np.argmax(predictions_array)
    
      thisplot[predicted_label].set_color('red')
      thisplot[true_label].set_color('blue')
      
     
def evaluate_and_plot(history, model, images, labels, classes):
    plt.plot(history.history['acc'], label = 'accuracy')
    plt.plot(history.history['val_acc'], label = 'validation_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    plt.legend(loc = 'lower right')
    
    print('best accuracy', max(history.history['val_acc']))
    
    # %%
    probability_model = tf.keras.Sequential([model, 
                                             tf.keras.layers.Softmax()])
    predictions = probability_model.predict(images)  
    
    # Plots first image, predicted label and true label
    i = 0
    plt.figure(figsize=(6,3))
    plt.subplot(1,2,1)
    plot_image(i, predictions[i], labels, images, classes)
    plt.subplot(1,2,2)
    plot_value_array(i, predictions[i], labels, classes)
    plt.show()
    
    # Plot the first 15 test images, predicted labels and true labels
    # Correct predictions in blue and incorrect predictions in red
    num_rows = 5
    num_cols = 3
    num_images = num_rows*num_cols
    plt.figure(figsize=(2*2*num_cols, 2*num_rows))
    for i in range(num_images):
      plt.subplot(num_rows, 2*num_cols, 2*i+1)
      plot_image(i, predictions[i], labels, images, classes)
      plt.subplot(num_rows, 2*num_cols, 2*i+2)
      plot_value_array(i, predictions[i], labels, classes)
    plt.tight_layout()
    plt.show()