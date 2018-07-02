#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 21 16:11:57 2018

@author: ssw
"""

import scipy.io
import numpy as np
from random import shuffle
import scipy.ndimage
import os
import patch_size
from sklearn.preprocessing import label_binarize

#load dataset
DATA_PATH = os.path.join(os.getcwd(),"Data")
input_mat = scipy.io.loadmat(os.path.join(DATA_PATH, 'PaviaU.mat'))['paviaU']
target_mat = scipy.io.loadmat(os.path.join(DATA_PATH, 'PaviaU_gt.mat'))['paviaU_gt']
#define global variables
HEIGHT = input_mat.shape[0]
WIDTH = input_mat.shape[1]
BAND = input_mat.shape[2]
PATCH_SIZE = patch_size.patch_size
TRAIN_PATCH,TRAIN_LABELS,TEST_PATCH,TEST_LABELS = [],[],[],[]
CLASSES = [] 
COUNT = 200 #Number of patches of each class
OUTPUT_CLASSES = 9
TEST_FRAC = 0.9 #Fraction of data to be used for testing

'''shuju guiyihua fangshi, xuyao shi ke zixing xiugai'''
#Scale the input between [0,1]
input_mat = input_mat.astype(float)
input_mat -= np.min(input_mat)
input_mat /= np.max(input_mat)
#Calculate the mean of each channel for normalization
MEAN_ARRAY = np.ndarray(shape=(BAND,),dtype=float)
for i in range(BAND):
    MEAN_ARRAY[i] = np.mean(input_mat[:,:,i])

def Patch(height_index,width_index):
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
    transpose_array = np.transpose(input_mat,(2,0,1))
    height_slice = slice(height_index, height_index+PATCH_SIZE)
    width_slice = slice(width_index, width_index+PATCH_SIZE)
    patch = transpose_array[:, height_slice, width_slice]
    mean_normalized_patch = []
    for i in range(patch.shape[0]):
        mean_normalized_patch.append(patch[i] - MEAN_ARRAY[i]) 
    
    return np.array(mean_normalized_patch)

#Collect all available patches of each class from the given image
for i in range(OUTPUT_CLASSES):
    CLASSES.append([])
for i in range(HEIGHT - PATCH_SIZE + 1):
    for j in range(WIDTH - PATCH_SIZE + 1):
        curr_inp = Patch(i,j)
        curr_tar = target_mat[i + int((PATCH_SIZE - 1)/2), j + int((PATCH_SIZE - 1)/2)]
        if(curr_tar!=0): #Ignore patches with unknown landcover type for the central pixel
            CLASSES[curr_tar-1].append(curr_inp)
#Make a test split with TEST_FRAC data from each class
for c in range(OUTPUT_CLASSES): #for each class
    class_population = len(CLASSES[c])
    test_split_size = int(class_population*TEST_FRAC)
        
    patches_of_current_class = CLASSES[c]
    shuffle(patches_of_current_class)
    
    #Make training and test splits
    TRAIN_PATCH.extend(patches_of_current_class[:-test_split_size])
    TRAIN_LABELS.extend(np.full(class_population - test_split_size, c, dtype=int))
    
    TEST_PATCH.extend(patches_of_current_class[-test_split_size:])
    TEST_LABELS.extend(np.full(test_split_size, c, dtype=int))
    
TRAIN_PATCH_FINAL = np.transpose(TRAIN_PATCH, (0, 2, 3, 1))
TEST_PATCH_FINAL = np.transpose(TEST_PATCH, (0, 2, 3, 1))
#label one-hot encoding
train_labels_onehot = label_binarize(TRAIN_LABELS, classes=range(OUTPUT_CLASSES))
test_labels_onehot = label_binarize(TEST_LABELS, classes=range(OUTPUT_CLASSES))

#Save the patches in segments
#1.Training data
train_dict = {}
file_name = 'Train_'+str(PATCH_SIZE)+'x'+str(PATCH_SIZE)+'.mat'
train_dict["train_patch"] = TRAIN_PATCH_FINAL
train_dict["train_labels"] = train_labels_onehot
scipy.io.savemat(os.path.join(DATA_PATH, file_name),train_dict)
#2.Test data
test_dict = {}
file_name = 'Test_'+str(PATCH_SIZE)+'x'+str(PATCH_SIZE)+'.mat'
test_dict["test_patch"] = TEST_PATCH_FINAL
test_dict["test_labels"] = test_labels_onehot
scipy.io.savemat(os.path.join(DATA_PATH, file_name),test_dict)





