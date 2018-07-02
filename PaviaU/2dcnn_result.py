#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May 14 12:17:19 2018

@author: ssw
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May 14 10:19:35 2018

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

DATA_FILENAME = 'PaviaU_test_2d'

test_data, test_labels=[],[]
for i in range(8):
    test_data_filename = DATA_FILENAME+'_'+str(i)+'.mat'
    test = scipy.io.loadmat(os.path.join(DATA_PATH, test_data_filename))
    test_data.extend(test['test_patch_2d'])
    test_labels.extend(test['test_labels'])
test_data = np.array(test_data)
test_labels = np.array(test_labels)
evl_data = Dataset(test_data, test_labels)    

batch_size_for_test = 1
tf.reset_default_graph() 

test_labels = np.argmax(test_labels, axis=1)
predict_labels=np.zeros(evl_data.sample_num, dtype=int)
with tf.Session() as sess:
    #load model
    saver = tf.train.import_meta_graph('./model/2dcnn_model/2dcnn-model-20000.meta')
    saver.restore(sess, tf.train.latest_checkpoint("./model/2dcnn_model/"))
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
del test_data, evl_data
#groundtruth
input_mat = scipy.io.loadmat(os.path.join(DATA_PATH, 'PaviaU.mat'))['paviaU']
gt_2dcnn = scipy.io.loadmat(os.path.join(DATA_PATH, 'PaviaU_gt.mat'))['paviaU_gt']
gt =  scipy.io.loadmat(os.path.join(DATA_PATH, 'PaviaU_gt.mat'))['paviaU_gt']
#Scale the input between [0,1]
input_mat = input_mat.astype(float)
input_mat -= np.min(input_mat)
input_mat /= np.max(input_mat)
#Calculate the mean of each channel for normalization
MEAN_ARRAY = np.ndarray(shape=(input_mat.shape[2],),dtype=float)
for i in range(input_mat.shape[2]):
    MEAN_ARRAY[i] = np.mean(input_mat[:,:,i])

transpose_array = np.transpose(input_mat,(2,0,1))

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
    height_slice = slice(height_index, height_index+patch_size.patch_size)
    width_slice = slice(width_index, width_index+patch_size.patch_size)
    patch_2d = transpose_array[:, height_slice, width_slice]
    mean_normalized_patch = []
    for i in range(patch_2d.shape[0]):
        mean_normalized_patch.append(patch_2d[i] - MEAN_ARRAY[i]) 
    
    return np.array(mean_normalized_patch)
    
with tf.Session() as sess:
    #load model
    saver = tf.train.import_meta_graph('./model/2dcnn_model/2dcnn-model-20000.meta')
    saver.restore(sess, tf.train.latest_checkpoint("./model/2dcnn_model/"))
    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name("x:0")
    y_ = graph.get_tensor_by_name("y_:0")
    keep_prob = graph.get_tensor_by_name("keep_prob:0")
    for i in range(gt_2dcnn.shape[0]-patch_size.patch_size):
        for j in range(gt_2dcnn.shape[1]-patch_size.patch_size):
            i_tar = i + int((patch_size.patch_size - 1)/2)
            j_tar = j + int((patch_size.patch_size - 1)/2)
            if gt_2dcnn[i_tar][j_tar] == 0:
                continue
            else:
                gt_2dcnn[i_tar][j_tar]=np.argmax(sess.run(predict_y, feed_dict={
                        x: np.transpose(Patch_2d(i, j),(1,2,0))[np.newaxis, :], y_: test_batch[1], keep_prob: 1.0})) + 1
#save result
result_2dcnn = {}
file_name = 'PaviaU_2dcnn_result_0.05.mat'
result_2dcnn["test_labels"] = test_labels
result_2dcnn["predict_labels"] = predict_labels
result_2dcnn["kappa"] = kappa
result_2dcnn["OA"] = oa
result_2dcnn["AA"] = aa
result_2dcnn["Accuracy"] = accuracy
result_2dcnn["gt_2dcnn"] = gt_2dcnn
scipy.io.savemat(os.path.join(DATA_PATH, file_name), result_2dcnn)