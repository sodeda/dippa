# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 11:27:38 2020

@author: soderdahl
"""

# https://towardsdatascience.com/implementing-yann-lecuns-lenet-5-in-pytorch-5e05a0911320


import numpy as np
from datetime import datetime 

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torchvision import datasets, transforms
from torch.utils.data.dataset import Dataset  # For custom datasets

import matplotlib.pyplot as plt

from PIL import Image
import glob
from torch.autograd import Variable


# check device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# parameters
RANDOM_SEED = 42
LEARNING_RATE = 0.001
BATCH_SIZE = 5
N_EPOCHS = 10

IMG_SIZE = 32
N_CLASSES = 3


# %% FUNCTIONS
def get_accuracy(model, data_loader, device):
    '''
    Function for computing the accuracy of the predictions over the entire data_loader
    '''
    
    correct_pred = 0 
    n = 0
    
    with torch.no_grad():
        model.eval()
        for X, y_true in data_loader:

            X = X.to(device)
            y_true = y_true.to(device)

            _, y_prob = model(X)
            _, predicted_labels = torch.max(y_prob, 1)

            n += y_true.size(0)
            correct_pred += (predicted_labels == y_true).sum()

    return correct_pred.float() / n


def plot_losses(train_losses, valid_losses):
    '''
    Function for plotting training and validation losses
    '''
    
    # temporarily change the style of the plots to seaborn 
    plt.style.use('seaborn')

    train_losses = np.array(train_losses) 
    valid_losses = np.array(valid_losses)

    fig, ax = plt.subplots(figsize = (8, 4.5))

    ax.plot(train_losses, color='blue', label='Training loss') 
    ax.plot(valid_losses, color='red', label='Validation loss')
    ax.set(title="Loss over epochs", 
            xlabel='Epoch',
            ylabel='Loss') 
    ax.legend()
    fig.show()
    
    # change the plot style to default
    plt.style.use('default')
    
    
def train(train_loader, model, criterion, optimizer, device):
    '''
    Function for the training step of the training loop
    '''

    model.train()
    running_loss = 0
    
    for X, y_true in train_loader:

        optimizer.zero_grad()
        
        X = X.to(device)
        # miks -1?
        y_true = y_true.to(device)
        #print(y_true)
        # Forward pass
        y_hat, _ = model(X)
        #print(y_hat)
        
        # size (N, C) (batch, classes)
        loss = criterion(y_hat, y_true) 
        running_loss += loss.item() * X.size(0)

        # Backward pass
        loss.backward()
        optimizer.step()
        
    epoch_loss = running_loss / len(train_loader.dataset)
    return model, optimizer, epoch_loss



def validate(valid_loader, model, criterion, device):
    '''
    Function for the validation step of the training loop
    '''
   
    model.eval()
    running_loss = 0
    
    for X, y_true in valid_loader:
    
        X = X.to(device)
        y_true = y_true.to(device) - 1

        # Forward pass and record loss
        y_hat, _ = model(X) 
        loss = criterion(y_hat, y_true) 
        running_loss += loss.item() * X.size(0)

    epoch_loss = running_loss / len(valid_loader.dataset)
        
    return model, epoch_loss


def training_loop(model, criterion, optimizer, train_loader, valid_loader, epochs, device, print_every=1):
    '''
    Function defining the entire training loop
    '''
    epochs = N_EPOCHS
    device = DEVICE
    # set objects for storing metrics
    best_loss = 1e10
    train_losses = []
    valid_losses = []
 
    # Train model
    for epoch in range(0, epochs):

        # training
        model, optimizer, train_loss = train(train_loader, model, criterion, optimizer, device)
        train_losses.append(train_loss)

        # validation
        with torch.no_grad():
            model, valid_loss = validate(valid_loader, model, criterion, device)
            valid_losses.append(valid_loss)

        if epoch % print_every == (print_every - 1):
            
            train_acc = get_accuracy(model, train_loader, device=device)
            valid_acc = get_accuracy(model, valid_loader, device=device)
                
            print(f'{datetime.now().time().replace(microsecond=0)} --- '
                  f'Epoch: {epoch}\t'
                  f'Train loss: {train_loss:.4f}\t'
                  f'Valid loss: {valid_loss:.4f}\t'
                  f'Train accuracy: {100 * train_acc:.2f}\t'
                  f'Valid accuracy: {100 * valid_acc:.2f}')

    plot_losses(train_losses, valid_losses)
    
    return model, optimizer, (train_losses, valid_losses)

# %% Custom dataset
class CustomDatasetFromFile(Dataset):
    def __init__(self, folder_paths, h, w, transform=None):
        """
        A dataset example where the class is embedded in the file names
        This data example also does not use any torch transforms
        Args:
            folder_path (string): path to image folder
        """
        # Get image list
        self.image_list = glob.glob(folder_paths[0]+'*')+glob.glob(folder_paths[1]+'*')+glob.glob(folder_paths[0]+'*')
        # Calculate len
        self.data_len = len(self.image_list)
        self.h = h
        self.w = w
        self.transform = transform

    def __getitem__(self, index):
        # Get image name from the pandas df
        single_image_path = self.image_list[index]
        # Open image
        im_as_im = Image.open(single_image_path)
        #im_as_im = im_as_im.resize((28,28))
        # Do some operations on image
        # Convert to numpy, dim = 28x28
        #im_as_np = np.asarray(im_as_im).reshape(28, 28).astype('uint8')
        # Add channel dimension, dim = 1x28x28
        # Note: You do not need to do this if you are reading RGB images
        # or i there is already channel dimension
        #im_as_np = np.expand_dims(im_as_im, 0)
        # Some preprocessing operations on numpy array
        # ...
        # ...
        # ...

        #img_as_np = np.asarray(self.data.iloc[index][1:]).reshape(28, 28).astype('uint8')

        # Transform image to tensor, change data type
        #im_as_ten = torch.from_numpy(im_as_np).float()
        
        
        if self.transform is not None:
            im_as_ten = self.transform(im_as_im)
            
        # Get label(class) of the image based on the folder name
        class_name = single_image_path
        class_name = class_name.split('/')
        label = class_name[7]
        label = label.split('\\')
        try:
            label = int(label[0])        
        except:
            label = int(0)
        
        return (im_as_ten, label)

    def __len__(self):
        return self.data_len
    
    
    def add_images(self, path):
        images = glob.glob(path+'*')
        self.image_list = self.image_list + images
        self.data_len = len(self.image_list)


# %% DATA
# define transforms
# transforms.ToTensor() automatically scales the images to [0,1] range
transforms = transforms.Compose([transforms.Resize((32, 32)), transforms.Grayscale(),
                                 transforms.ToTensor()])

paths = ['C:/Users/soder/Desktop/dippakoodit/katodit/training_data/1/',
         'C:/Users/soder/Desktop/dippakoodit/katodit/training_data/9/',
         'C:/Users/soder/Desktop/dippakoodit/katodit/training_data/romu/']

# download and create datasets
train_dataset = CustomDatasetFromFile(paths, 32, 32, transforms)

valid_dataset = CustomDatasetFromFile('C:/Users/soder/Desktop/dippakoodit/katodit/test_data/1/', 32, 32, transforms)
valid_dataset.add_images('C:/Users/soder/Desktop/dippakoodit/katodit/test_data/9/')
valid_dataset.add_images('C:/Users/soder/Desktop/dippakoodit/katodit/test_data/romu/')

valid_dataset = train_dataset
# define the data loaders
train_loader = DataLoader(dataset=train_dataset, 
                          batch_size=BATCH_SIZE, 
                          shuffle=True)

valid_loader = DataLoader(dataset=valid_dataset, 
                          batch_size=BATCH_SIZE, 
                          shuffle=False)

#
dataiter = iter(train_loader)
images, labels = dataiter.next()

# print(images.shape)
# print(labels.shape)

# %% PLOTS
# ROW_IMG = 10
# N_ROWS = 5

# # värillisenä
# fig = plt.figure()
# for index in range(1, ROW_IMG * N_ROWS + 1):
#     plt.subplot(N_ROWS, ROW_IMG, index)
#     plt.axis('off')
#     plt.imshow(train_dataset.data[index])
# fig.suptitle('MNIST Dataset - preview');

# # harmaana
# fig = plt.figure()
# for index in range(1, ROW_IMG * N_ROWS + 1):
#     plt.subplot(N_ROWS, ROW_IMG, index)
#     plt.axis('off')
#     plt.imshow(train_dataset.data[index], cmap='gray_r')
# fig.suptitle('MNIST Dataset - preview');


# %% LeNet-5
class LeNet5(nn.Module):

    def __init__(self, n_classes):
        super(LeNet5, self).__init__()
        
        self.feature_extractor = nn.Sequential(            
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1),
            nn.Tanh()
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=120, out_features=84),
            nn.Tanh(),
            nn.Linear(in_features=84, out_features=n_classes),
        )


    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        probs = F.softmax(logits, dim=1)
        return logits, probs
    
# %%
torch.manual_seed(RANDOM_SEED)

model = LeNet5(N_CLASSES).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

# %%  TRAIN
model, optimizer, _ = training_loop(model, criterion, optimizer, train_loader, valid_loader, N_EPOCHS, DEVICE)

# %% EVALUATING
ROW_IMG = 10
N_ROWS = 5

fig = plt.figure()
for index in range(1, ROW_IMG * N_ROWS + 1):
    plt.subplot(N_ROWS, ROW_IMG, index)
    plt.axis('off')
    plt.imshow(valid_dataset.data[index], cmap='gray_r')
    
    with torch.no_grad():
        model.eval()
        _, probs = model(valid_dataset[index][0].unsqueeze(0))
        
    title = f'{torch.argmax(probs)} ({torch.max(probs * 100):.0f}%)'
    
    plt.title(title, fontsize=7)
fig.suptitle('LeNet-5 - predictions');