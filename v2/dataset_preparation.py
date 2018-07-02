#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 21 16:11:57 2018

@author: ssw
"""

import scipy.io
import numpy as np
#from random import shuffle
import scipy.ndimage
import os
import patch_size
from sklearn.preprocessing import label_binarize

#load dataset
DATA_PATH = os.path.join(os.getcwd(),"Data")
input_mat = scipy.io.loadmat(os.path.join(DATA_PATH, 'Indian_pines_corrected.mat'))['indian_pines_corrected']
target_mat = scipy.io.loadmat(os.path.join(DATA_PATH, 'Indian_pines_gt.mat'))['indian_pines_gt']
#define global variables
HEIGHT = input_mat.shape[0]
WIDTH = input_mat.shape[1]
BAND = input_mat.shape[2]
PATCH_SIZE = patch_size.patch_size
TRAIN_PATCH_1D,TRAIN_PATCH_2D,TRAIN_LABELS,TEST_PATCH_1D,TEST_PATCH_2D,TEST_LABELS = [],[],[],[],[],[]
classes_for_2d = [] 
classes_for_1d = []
OUTPUT_CLASSES = 16
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
    TRAIN_PATCH_1D.extend(patches_of_current_class_1d[:-test_split_size])
    TRAIN_PATCH_2D.extend(patches_of_current_class_2d[:-test_split_size])
    TRAIN_LABELS.extend(np.full(class_population - test_split_size, c, dtype=int))
    
    TEST_PATCH_1D.extend(patches_of_current_class_1d[-test_split_size:])
    TEST_PATCH_2D.extend(patches_of_current_class_2d[-test_split_size:])
    TEST_LABELS.extend(np.full(test_split_size, c, dtype=int))
TRAIN_PATCH_FINAL_1D = np.reshape(TRAIN_PATCH_1D, (-1, 200, 1, 1)) 
TEST_PATCH_FINAL_1D = np.reshape(TEST_PATCH_1D, (-1, 200, 1, 1)) 
TRAIN_PATCH_FINAL_2D = np.transpose(TRAIN_PATCH_2D, (0, 2, 3, 1))
TEST_PATCH_FINAL_2D = np.transpose(TEST_PATCH_2D, (0, 2, 3, 1))
#label one-hot encoding
train_labels_onehot = label_binarize(TRAIN_LABELS, classes=range(OUTPUT_CLASSES))
test_labels_onehot = label_binarize(TEST_LABELS, classes=range(OUTPUT_CLASSES))

#Save the patches in segments
#1.Training data
train_dict_1d = {}
file_name_1d = 'Train_'+str(PATCH_SIZE)+'x'+str(PATCH_SIZE)+'_1d.mat'
train_dict_1d["train_patch_1d"] = TRAIN_PATCH_FINAL_1D
train_dict_1d["train_labels"] = train_labels_onehot
scipy.io.savemat(os.path.join(DATA_PATH, file_name_1d),train_dict_1d)
train_dict_2d = {}
file_name_2d = 'Train_'+str(PATCH_SIZE)+'x'+str(PATCH_SIZE)+'_2d.mat'
train_dict_2d["train_patch_2d"] = TRAIN_PATCH_FINAL_2D
train_dict_2d["train_labels"] = train_labels_onehot
scipy.io.savemat(os.path.join(DATA_PATH, file_name_2d),train_dict_2d)
#2.Test data
test_dict_1d = {}
file_name_1d = 'Test_'+str(PATCH_SIZE)+'x'+str(PATCH_SIZE)+'_1d.mat'
test_dict_1d["test_patch_1d"] = TEST_PATCH_FINAL_1D
test_dict_1d["test_labels"] = test_labels_onehot
scipy.io.savemat(os.path.join(DATA_PATH, file_name_1d),test_dict_1d)
test_dict_2d = {}
file_name_2d = 'Test_'+str(PATCH_SIZE)+'x'+str(PATCH_SIZE)+'_2d.mat'
test_dict_2d["test_patch_2d"] = TEST_PATCH_FINAL_2D
test_dict_2d["test_labels"] = test_labels_onehot
scipy.io.savemat(os.path.join(DATA_PATH, file_name_2d),test_dict_2d)





