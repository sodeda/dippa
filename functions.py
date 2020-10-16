# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 13:32:02 2020

@author: soderdahl
"""

import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# %% Functions

# If data is not expanded
def pre_process(paths, class_names, size):
    # gets data from every path
    img_and_lab = []
    img_and_lab = get_data(img_and_lab, paths[0], class_names[0], size, exp=False)
    img_and_lab = get_data(img_and_lab, paths[1], class_names[1], size, exp=False)
    img_and_lab = get_data(img_and_lab, paths[2], class_names[2], size, exp=False)
    
    # shuffle data
    random.shuffle(img_and_lab)
    
    # separate images and labels and change labels to binary
    images, labels = separate_img_and_lab(img_and_lab)
    labels = labels_to_int(labels)
    
    # change lists to arrays
    images = np.array(images)
    labels = np.array(labels)
    
    return images, labels


# If data is expanded
def pre_process2(paths, class_names, size):
    # gets data from every path to different lists
    img_and_lab = []
    img_and_lab2 = []
    img_and_lab3 = []
    img_and_lab = get_data(img_and_lab, paths[0], class_names[0], size, exp=True)
    img_and_lab2 = get_data(img_and_lab2, paths[1], class_names[1], size, exp=True)
    img_and_lab3 = get_data(img_and_lab3, paths[2], class_names[2], size, exp=True)
    
    # shuffle all lists
    random.shuffle(img_and_lab)
    random.shuffle(img_and_lab2)
    random.shuffle(img_and_lab3)
    
    # separate lists to training and testing data
    tr0 = img_and_lab[120:]
    tr1 = img_and_lab2[120:]
    tr2 = img_and_lab3[40:]
    te0 = img_and_lab[:120]
    te1 = img_and_lab2[:120]
    te2 = img_and_lab3[:40]
    # combine training and test data lists
    tr = tr0+tr1+tr2
    te = te0+te1+te2
    
    # separate images and labels and change labels to binary
    trimages, trlabels = separate_img_and_lab(tr)
    trlabels = labels_to_int(trlabels)
    teimages, telabels = separate_img_and_lab(te)
    telabels = labels_to_int(telabels)
    
    # change lists to arrays
    trimages = np.array(trimages)
    trlabels = np.array(trlabels)
    teimages = np.array(teimages)
    telabels = np.array(telabels)
        
    return trimages, teimages, trlabels, telabels
 

# Gets data from path, resizes images and adds labels to them
def get_data(data, path, label, size, exp):
    for image in os.listdir(path):
        img = cv2.imread(path + '/' + image, 1) # 0 gray, 1 rgb
        img = cv2.resize(img, (size, size))
        img_and_label = []
        img_and_label.append(img)
        img_and_label.append(label)
        data.append(img_and_label)

        # img2 used when data is expanded
        if exp == True:
            img2 = augment(img)
            img_and_label2 = []
            img_and_label2.append(img2)
            img_and_label2.append(label)
            data.append(img_and_label2)
    
    return data


def augment(img):
    N = random.randint(1, 3)
    M = random.randint(1, 3)

    # choose flip based on randomized number
    if N == 1:
        img = cv2.flip(img, 0)
    elif N == 2:
        img = cv2.flip(img, 1)
    else:
        img = cv2.flip(img, -1)
       
    # choose brightness based on randomized number    
    if M == 1:
        hsvImg = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        hsvImg[...,2] = hsvImg[...,2]*1.1
        img = cv2.cvtColor(hsvImg, cv2.COLOR_HSV2BGR)
    elif M == 2:
        hsvImg = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        hsvImg[...,2] = hsvImg[...,2]*0.8
        img = cv2.cvtColor(hsvImg, cv2.COLOR_HSV2BGR)
   
    return img


# Separate images and labels to different arrays
def separate_img_and_lab(list):
    img = []
    label = []
    i = 0
    for element in list:
        img.append(list[i][0])
        label.append(list[i][1])
        i = i + 1
    return img, label


# Change labels to integer values
def labels_to_int(labels):
    integer = []
    i = 0
    for elements in labels:
        if labels[i] == 'std':
            integer.append(1)
        elif labels[i] == 'priima':
            integer.append(0)
        else:
            integer.append(2)
        i = i + 1
    return integer


def augmentation(images, labels, vert, hori, batch):
    # brightness_range = [0.75, 1.0], zoom_range = [0.9, 1.0], 
    # height_shift_range = [-1, 1], width_shift_range = [-1, 1]
    generator = ImageDataGenerator(vertical_flip = vert, 
                                   horizontal_flip = hori,
                                   brightness_range = [0.75, 1.0])
    generator = generator.flow(images, labels, batch_size = batch)
    
    return generator


# Plots 9 first images in training data with labels
def plot_images(images, labels, class_names):
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
            plt.imshow(images[0][0][i].astype('uint8'), cmap = plt.cm.binary)
            label = class_names[images[0][1][i]]
                
        plt.xlabel(label)
    plt.show()


# Plots image and label that says what is predicted and its %, and what is correct label
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


# Plots bar graph of predictions
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
      

# Plots training accuracies, losses and images with predicted and correct labels     
def evaluate_and_plot(history, model, images, labels, classes):
    # plots training accuracies
    csv = pd.read_csv('test')
    plt.plot(csv['acc'], label = 'accuracy')
    plt.plot(csv['val_acc'], label = 'validation_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    plt.legend(loc = 'lower right')
    
    # plots training losses
    plt.figure()
    csv = pd.read_csv('test')
    plt.plot(csv['loss'], label = 'loss')
    plt.plot(csv['val_loss'], label = 'validation_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.ylim([0, 1.5])
    plt.legend(loc = 'lower right')
    
    print('best accuracy', max(history.history['val_acc']))
    
    probability_model = tf.keras.Sequential([model, 
                                             tf.keras.layers.Softmax()])
    predictions = probability_model.predict(images)  
    
    # plot the first  test images, predicted labels and true labels
    # correct predictions in blue and incorrect predictions in red
    num_rows = 3
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