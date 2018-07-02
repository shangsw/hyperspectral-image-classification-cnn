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
import patch_size
from sklearn.preprocessing import label_binarize

#load dataset
DATA_PATH = "/home/ssw/Hyperspectral_classification_CNN/v5/Data"
input_mat = scipy.io.loadmat(os.path.join(DATA_PATH, 'Indian_pines_corrected.mat'))['indian_pines_corrected']
target_mat = scipy.io.loadmat(os.path.join(DATA_PATH, 'Indian_pines_gt.mat'))['indian_pines_gt']
#define global variables
HEIGHT = input_mat.shape[0]
WIDTH = input_mat.shape[1]
BAND = input_mat.shape[2]
PATCH_SIZE = patch_size.patch_size
train_patch_1d,train_patch_2d,test_patch_1d,test_patch_2d,test_labels = [],[],[],[],[]
classes_for_2d = [] 
classes_for_1d = []
OUTPUT_CLASSES = 16
count = 200 #the smallest training samples' number
TEST_FRAC = 0.9 #Fraction of data to be used for testing
ANGLES = [0.0, -90.0, 90.0, 180.0]

'''change the scale method if needed'''
#Scale the input between [0,1]
input_mat = input_mat.astype(float)
input_mat -= np.min(input_mat)
input_mat /= np.max(input_mat)
#Calculate the mean of each channel for normalization
MEAN_ARRAY = np.ndarray(shape=(BAND,),dtype=float)
for i in range(BAND):
    MEAN_ARRAY[i] = np.mean(input_mat[:,:,i])
transpose_array = np.transpose(input_mat,(2,0,1))

def Patch_1d(height_index, width_index):
    patch_1d = transpose_array[:, height_index, width_index]
    mean_normalized_patch = []
    for i in range(patch_1d.shape[0]):
        mean_normalized_patch.append(patch_1d[i] - MEAN_ARRAY[i]) 
    
    return np.array(mean_normalized_patch)

def Patch_2d(height_index,width_index):
    """
    Returns a mean-normalized patch, the top left corner of which 
    is at (height_index, width_index)
    
    Inputs: 
    height_index - row index of the top left corner of the image patch
    width_index - column index of the top left corner of the image patch
    
    Outputs:
    mean_normalized_patch - mean normalized patch of size (PATCH_SIZE, PATCH_SIZE) 
    whose top left corner is at (height_index, width_index)
    """
    height_slice = slice(height_index, height_index+PATCH_SIZE)
    width_slice = slice(width_index, width_index+PATCH_SIZE)
    patch_2d = transpose_array[:, height_slice, width_slice]
    mean_normalized_patch = []
    for i in range(patch_2d.shape[0]):
        mean_normalized_patch.append(patch_2d[i] - MEAN_ARRAY[i]) 
    
    return np.array(mean_normalized_patch)

#Collect all available patches of each class from the given image
for i in range(OUTPUT_CLASSES):
    classes_for_2d.append([])
    classes_for_1d.append([])
for i in range(HEIGHT - PATCH_SIZE + 1):
    for j in range(WIDTH - PATCH_SIZE + 1):
        curr_for_2d = Patch_2d(i,j)
        curr_tar = target_mat[i + int((PATCH_SIZE - 1)/2), j + int((PATCH_SIZE - 1)/2)]
        curr_for_1d = Patch_1d(i + int((PATCH_SIZE - 1)/2), j + int((PATCH_SIZE - 1)/2))
        if(curr_tar!=0): #Ignore patches with unknown landcover type for the central pixel
            classes_for_2d[curr_tar-1].append(curr_for_2d)
            classes_for_1d[curr_tar-1].append(curr_for_1d)
#Make a test split with TEST_FRAC data from each class
for c in range(OUTPUT_CLASSES): #for each class
    class_population = len(classes_for_1d[c])
    test_split_size = int(class_population*TEST_FRAC)
        
    perm = np.arange(class_population)
    np.random.shuffle(perm)
    patches_of_current_class_1d = np.array(classes_for_1d[c])[perm]
    patches_of_current_class_2d = np.array(classes_for_2d[c])[perm]
    
    #Make training and test splits
    train_patch_1d.append(patches_of_current_class_1d[:-test_split_size])
    train_patch_2d.append(patches_of_current_class_2d[:-test_split_size])
    train_patch_2d[c] = np.transpose(train_patch_2d[c], (0, 2, 3, 1))
    
    test_patch_1d.extend(patches_of_current_class_1d[-test_split_size:])
    test_patch_2d.extend(patches_of_current_class_2d[-test_split_size:])
    test_labels.extend(np.full(test_split_size, c, dtype=int))

del patches_of_current_class_2d, patches_of_current_class_1d
del classes_for_1d, classes_for_2d
#save test data
test_patch_final_1d = np.reshape(test_patch_1d, (-1, 200, 1, 1))
test_patch_final_2d = np.transpose(test_patch_2d, (0, 2, 3, 1))
test_labels_onehot = label_binarize(test_labels, classes=range(OUTPUT_CLASSES))

test_dict = {}
file_name = 'Test_'+str(PATCH_SIZE)+'x'+str(PATCH_SIZE)+'.mat'
test_dict["test_patch_1d"] = test_patch_final_1d
test_dict["test_patch_2d"] = test_patch_final_2d
test_dict["test_labels"] = test_labels_onehot
scipy.io.savemat(os.path.join(DATA_PATH, file_name),test_dict)

del test_patch_final_1d, test_patch_final_2d, test_labels_onehot, test_patch_1d, test_patch_2d, test_labels

'''
1d data augment
'''
#add noise to 1d_data

for i in range(len(train_patch_1d)):
    randmatrix = np.random.normal(0.0, 1.0, train_patch_1d[i].shape)
    train_patch_1d[i] = np.vstack((train_patch_1d[i], train_patch_1d[i] + 1e-6*randmatrix))
#generate training data to count
for i in range(len(train_patch_1d)):
    while train_patch_1d[i].shape[0] < count:
        index = sample(range(train_patch_1d[i].shape[0]), 2)
        tmp = (train_patch_1d[i][index[0],:] + train_patch_1d[i][index[1],:])/2.0
        train_patch_1d[i] = np.vstack((train_patch_1d[i], tmp))
#finish 1d training data
train_labels_1d = []
for i in range(len(train_patch_1d)):
    class_pop = train_patch_1d[i].shape[0]
    train_labels_1d.extend(np.full(class_pop, i, dtype=int))
train_patch_1d_flat = train_patch_1d[0]
for i in range(len(train_patch_1d)-1):
    class_pop = train_patch_1d[i+1].shape[0]
    train_patch_1d_flat = np.vstack((train_patch_1d_flat, train_patch_1d[i+1]))
train_patch_1d_final = np.reshape(train_patch_1d_flat, (-1, 200, 1, 1))
train_labels_1d_onehot = label_binarize(train_labels_1d, classes=range(OUTPUT_CLASSES))
#save 1d-data
train_1d = {}
file_name = 'Train_'+str(PATCH_SIZE)+'x'+str(PATCH_SIZE)+'_1d.mat' 
train_1d['train_patch_1d'] = train_patch_1d_final
train_1d['train_labels_1d'] = train_labels_1d_onehot
scipy.io.savemat(os.path.join(DATA_PATH, file_name), train_1d)

del train_patch_1d, train_patch_1d_final, train_patch_1d_flat

'''
2d data augment
'''     
#2d data augment
for i in range(OUTPUT_CLASSES):
    num = train_patch_2d[i].shape[0]
    while  train_patch_2d[i].shape[0] < count:
        #rotate the patches
        j = randint(0, num-1)
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
       
#finish 2d data
train_labels_2d = []
for i in range(len(train_patch_2d)):
    class_pop = train_patch_2d[i].shape[0]
    train_labels_2d.extend(np.full(class_pop, i, dtype=int))
train_patch_2d_flat = train_patch_2d[0]
for i in range(len(train_patch_2d)-1):
    class_pop = train_patch_2d[i+1].shape[0]
    train_patch_2d_flat = np.vstack((train_patch_2d_flat, train_patch_2d[i+1]))
train_labels_2d_onehot = label_binarize(train_labels_2d, classes=range(OUTPUT_CLASSES))
#save 2d data
train_2d = {}
file_name = 'Train_'+str(PATCH_SIZE)+'x'+str(PATCH_SIZE)+'_2d.mat'
train_2d['train_patch_2d'] = train_patch_2d_flat
train_2d['train_labels_2d'] = train_labels_2d_onehot 
scipy.io.savemat(os.path.join(DATA_PATH, file_name), train_2d) 

