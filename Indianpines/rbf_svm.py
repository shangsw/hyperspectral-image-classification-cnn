#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri May 11 18:36:39 2018

@author: ssw
"""


import os
import scipy.io
import numpy as np
from sklearn import svm
import result_eval



data_path = "/home/ssw/Hyperspectral_classification_CNN/v4/Data"
data_filename = ['Indianpines_train_1d_wa.mat',
                 'Indianpines_test_1d.mat']

train = scipy.io.loadmat(os.path.join(data_path, data_filename[0]))
train_data = train['train_patch_1d']
train_data = np.reshape(train_data, (-1, train_data.shape[1]))
train_labels = np.argmax(train['train_labels'], axis=1)

test = scipy.io.loadmat(os.path.join(data_path, data_filename[1]))
test_data = test['test_patch_1d']
test_data = np.reshape(test_data, (-1, test_data.shape[1]))
test_labels = np.argmax(test['test_labels'], axis=1)

#svm classification
clf = svm.SVC(C=12.0, kernel='rbf', gamma=6.0, decision_function_shape='ovr')
clf.fit(train_data, train_labels)
predicted_labels = clf.predict(test_data)
print clf.score(train_data, train_labels)
print clf.score(test_data,test_labels)

#result
cnf_matrix = result_eval.my_cnf_matrix(test_labels, predicted_labels)
kappa = result_eval.my_kappa(test_labels, predicted_labels)
oa = result_eval.my_oa(test_labels, predicted_labels)
(accuracy, aa) = result_eval.my_aa(cnf_matrix)

#groundtruth
input_mat = scipy.io.loadmat(os.path.join(data_path, 'Indian_pines_corrected.mat'))['indian_pines_corrected']
svm_gt = scipy.io.loadmat(os.path.join(data_path, 'Indian_pines_gt.mat'))['indian_pines_gt']
#Scale the input between [0,1]
input_mat = input_mat.astype(float)
input_mat -= np.min(input_mat)
input_mat /= np.max(input_mat)
#Calculate the mean of each channel for normalization
MEAN_ARRAY = np.ndarray(shape=(input_mat.shape[2],),dtype=float)
for i in range(input_mat.shape[2]):
    MEAN_ARRAY[i] = np.mean(input_mat[:,:,i])

transpose_array = np.transpose(input_mat,(2,0,1))

def Patch_1d(height_index, width_index):
    patch_1d = transpose_array[:, height_index, width_index]
    mean_normalized_patch = []
    for i in range(patch_1d.shape[0]):
        mean_normalized_patch.append(patch_1d[i] - MEAN_ARRAY[i]) 
    
    return np.array(mean_normalized_patch)

for i in range(svm_gt.shape[0]):
    for j in range(svm_gt.shape[1]):
        if svm_gt[i][j] == 0:
            continue
        else:
            svm_gt[i][j]=clf.predict(Patch_1d(i, j)[np.newaxis,:]) + 1
#save result
rbf_svm_result = {}
file_name = 'Indianpines_rbf_svm_result_wa.mat'
rbf_svm_result["test_labels"] = test_labels
rbf_svm_result["predict_labels"] = predicted_labels
rbf_svm_result["kappa"] = kappa
rbf_svm_result["OA"] = oa
rbf_svm_result["AA"] = aa
rbf_svm_result["Accuracy"] = accuracy
rbf_svm_result["svm_gt"] = svm_gt
scipy.io.savemat(os.path.join(data_path, file_name), rbf_svm_result)

