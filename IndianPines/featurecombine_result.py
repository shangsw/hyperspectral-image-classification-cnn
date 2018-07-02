#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun May 13 18:37:36 2018

@author: ssw
"""

import tensorflow as tf
import patch_size
import os
import scipy.io
from next_batch import Dataset
import numpy as np
import result_eval

DATA_PATH = "/home/ssw/Hyperspectral_classification_CNN/v4/Data"

DATA_FILENAME = 'Indianpines_test_'+str(patch_size.patch_size)+'x'+str(patch_size.patch_size)+'_1d.mat'

evl_data = Dataset(scipy.io.loadmat(os.path.join(DATA_PATH, DATA_FILENAME))['test_patch_1d'], 
                   scipy.io.loadmat(os.path.join(DATA_PATH, DATA_FILENAME))['test_labels']) 

batch_size_for_test = 1
test_labels = np.argmax(scipy.io.loadmat(os.path.join(DATA_PATH, DATA_FILENAME))['test_labels'], axis=1)
predict_labels=np.zeros(evl_data.sample_num, dtype=int)

with tf.Session() as sess:
    #load model
    saver = tf.train.import_meta_graph('./model/1dcnn_model/1dcnn-model-20000.meta')
    saver.restore(sess, tf.train.latest_checkpoint("./model/1dcnn_model/"))
    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name("x:0")
    y_ = graph.get_tensor_by_name("y_:0")
    predict_y = graph.get_tensor_by_name("predict_y:0")
    keep_prob = graph.get_tensor_by_name("keep_prob:0")
    for i in range(evl_data.sample_num):
        test_batch = evl_data.next_batch(batch_size_for_test)
        predict_labels[i]=np.argmax(sess.run(predict_y, feed_dict={
            x: test_batch[0], y_: test_batch[1], keep_prob: 1.0}))

cnf_matrix = result_eval.my_cnf_matrix(test_labels, predict_labels)
kappa = result_eval.my_kappa(test_labels, predict_labels)
oa = result_eval.my_oa(test_labels, predict_labels)
(accuracy, aa) = result_eval.my_aa(cnf_matrix)

#groundtruth
input_mat = scipy.io.loadmat(os.path.join(DATA_PATH, 'Indian_pines_corrected.mat'))['indian_pines_corrected']
gt_1dcnn = scipy.io.loadmat(os.path.join(DATA_PATH, 'Indian_pines_gt.mat'))['indian_pines_gt']
gt =  scipy.io.loadmat(os.path.join(DATA_PATH, 'Indian_pines_gt.mat'))['indian_pines_gt']
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
        
with tf.Session() as sess:
    #load model
    saver = tf.train.import_meta_graph('./model/1dcnn_model/1dcnn-model-20000.meta')
    saver.restore(sess, tf.train.latest_checkpoint("./model/1dcnn_model/"))
    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name("x:0")
    y_ = graph.get_tensor_by_name("y_:0")
    keep_prob = graph.get_tensor_by_name("keep_prob:0")
    for i in range(gt_1dcnn.shape[0]):
        for j in range(gt_1dcnn.shape[1]):
            if gt_1dcnn[i][j] == 0:
                continue
            else:
                gt_1dcnn[i][j]=np.argmax(sess.run(predict_y, feed_dict={
                        x: Patch_1d(i, j)[np.newaxis, :, np.newaxis, np.newaxis], y_: test_batch[1], keep_prob: 1.0})) + 1
#save result
result_1dcnn = {}
file_name = 'Indianpines_1dcnn_result.mat'
result_1dcnn["test_labels"] = test_labels
result_1dcnn["predict_labels"] = predict_labels
result_1dcnn["kappa"] = kappa
result_1dcnn["OA"] = oa
result_1dcnn["AA"] = aa
result_1dcnn["Accuracy"] = accuracy
result_1dcnn["gt_1dcnn"] = gt_1dcnn
scipy.io.savemat(os.path.join(DATA_PATH, file_name), result_1dcnn)
