#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 30 10:51:01 2018

@author: ssw
"""

import scipy.io
import numpy as np
from random import sample, randint
import scipy.ndimage
import os
from sklearn.preprocessing import label_binarize

#load dataset
DATA_PATH = os.path.join(os.getcwd(),"Data")
data_filename = ['Indianpines_train_1d.mat',
                 'Indianpines_train_2d.mat']
input_train_patch_1d = scipy.io.loadmat(os.path.join(DATA_PATH, data_filename[0]))['train_patch_1d']
train_labels = scipy.io.loadmat(os.path.join(DATA_PATH, data_filename[0]))['train_labels']
input_train_patch_2d = scipy.io.loadmat(os.path.join(DATA_PATH, data_filename[1]))['train_patch_2d']

#define global variables
train_patch_1d,train_patch_2d,test_patch_1d,test_patch_2d,test_labels = [],[],[],[],[]
count = 300 #the  training samples
ANGLES = [0.0, -90.0, 90.0, 180.0]
class_num = train_labels.shape[1]

'''
1d data augment
'''
#add noise to 1d_data
train_patch_1d_noise = input_train_patch_1d + 5e-6*np.random.normal(0.0, 1.0, input_train_patch_1d.shape)
input_train_patch_1d = np.vstack((input_train_patch_1d, train_patch_1d_noise,))
train_labels = np.vstack((train_labels, train_labels))
for i in range(class_num):
    train_patch_1d.append([])
for i in range(input_train_patch_1d.shape[0]):
    label = np.argmax(train_labels[i,:])
    train_patch_1d[label].append(input_train_patch_1d[i])

del input_train_patch_1d

#generate training data to count
for i in range(len(train_patch_1d)):
    while len(train_patch_1d[i]) < count:
        index = sample(range(len(train_patch_1d[i])), 2)
        tmp = (train_patch_1d[i][index[0]] + train_patch_1d[i][index[0]])/2.0
        train_patch_1d[i].append(tmp)
for i in range(len(train_patch_1d)):
    train_patch_1d[i] = train_patch_1d[i][0:count]
#finish 1d training data
train_labels_1d = []
for i in range(len(train_patch_1d)):
    train_labels_1d.extend(np.full(count, i, dtype=int))
train_patch_1d_flat = train_patch_1d[0]
for i in range(1, len(train_patch_1d)):
    train_patch_1d_flat = np.vstack((train_patch_1d_flat, train_patch_1d[i]))
train_labels_1d_onehot = label_binarize(train_labels_1d, classes=range(class_num))
#save 1d-data
train_1d = {}
file_name = 'Indianpines_train_1d_wa.mat' 
train_1d['train_patch_1d'] = train_patch_1d_flat
train_1d['train_labels'] = train_labels_1d_onehot
scipy.io.savemat(os.path.join(DATA_PATH, file_name), train_1d)

del train_patch_1d, train_patch_1d_flat

'''
2d data augment
'''     

for i in range(class_num):
    train_patch_2d.append([])
for i in range(input_train_patch_2d.shape[0]):
    label = np.argmax(train_labels[i,:])
    train_patch_2d[label].append(input_train_patch_2d[i])

#2d data augment
for i in range(class_num):
    while  len(train_patch_2d[i]) < count:
        #rotate the patches
        j = randint(0, len(train_patch_2d[i])-1)
        for angles in ANGLES[1:]:
            rotated_patch = scipy.ndimage.interpolation.rotate(train_patch_2d[i][j], 
                                                               angles,reshape=False, output=None, prefilter=False)
            train_patch_2d[i] = np.vstack((train_patch_2d[i], rotated_patch[np.newaxis, :]))
        #flip up and down
        flipped_patch = np.flipud(train_patch_2d[i][j])
        #rotate the patches
        for angles in ANGLES:
            rotated_patch = scipy.ndimage.interpolation.rotate(flipped_patch, 
                                                               angles,reshape=False, output=None, prefilter=False)
            train_patch_2d[i] = np.vstack((train_patch_2d[i], rotated_patch[np.newaxis, :]))

for i in range(len(train_patch_2d)):
    train_patch_2d[i] = train_patch_2d[i][0:count]

#finish 2d data
train_labels_2d = []
for i in range(len(train_patch_2d)):
    train_labels_2d.extend(np.full(count, i, dtype=int))
train_patch_2d_flat = train_patch_2d[0]
for i in range(1, len(train_patch_2d)):
    train_patch_2d_flat = np.vstack((train_patch_2d_flat, train_patch_2d[i]))
train_labels_2d_onehot = label_binarize(train_labels_2d, classes=range(class_num))
#save 2d data
train_2d = {}
file_name = 'Indianpines_train_2d_wa.mat'
train_2d['train_patch_2d'] = train_patch_2d_flat
train_2d['train_labels'] = train_labels_2d_onehot 
scipy.io.savemat(os.path.join(DATA_PATH, file_name), train_2d) 

