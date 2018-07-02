#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat May 19 16:04:08 2018

@author: ssw
"""


import tensorflow as tf
import os
import scipy.io
from next_batch_for_combinenet import Dataset_for_combinenet
import numpy as np
import result_eval

DATA_PATH = "/home/ssw/Hyperspectral_classification_CNN/v5/Data"

data_filename = 'PaviaU_test_feature_'
predict_labels = []
test_labels = []
for file_num in range(8):
    test_data_filename = data_filename+str(file_num)+'.mat'
    eval_data_1d = scipy.io.loadmat(os.path.join(DATA_PATH, test_data_filename))['test_feature_1d']
    eval_data_2d = scipy.io.loadmat(os.path.join(DATA_PATH, test_data_filename))['test_feature_2d']
    eval_labels = scipy.io.loadmat(os.path.join(DATA_PATH, test_data_filename))['test_labels']
    eval_data = np.hstack((eval_data_1d, eval_data_2d))
    eval_dataset = Dataset_for_combinenet(eval_data, eval_labels)
    
    batch_size_for_test = 1
    test_labels.extend(np.argmax(eval_labels, axis=1))
    
    
    with tf.Session() as sess:
        #load model
        saver = tf.train.import_meta_graph('./model/dcn/dcn-model-20000.meta')
        saver.restore(sess, tf.train.latest_checkpoint("./model/dcn/"))
        graph = tf.get_default_graph()
        x = graph.get_tensor_by_name("x:0")
        y_ = graph.get_tensor_by_name("y_:0")
        predict_y = graph.get_tensor_by_name("predict_y:0")
        keep_prob = graph.get_tensor_by_name("keep_prob:0")
        for i in range(eval_dataset.sample_num):
            test_batch = eval_dataset.next_batch(batch_size_for_test)
            tmp = np.argmax(sess.run(predict_y, feed_dict={
                x: test_batch[0], y_: test_batch[1], keep_prob: 1.0}))
            predict_labels.append(tmp)
    
cnf_matrix = result_eval.my_cnf_matrix(test_labels, predict_labels)
kappa = result_eval.my_kappa(test_labels, predict_labels)
oa = result_eval.my_oa(test_labels, predict_labels)
(accuracy, aa) = result_eval.my_aa(cnf_matrix)

'''
#save result
result_1dcnn = {}
file_name = 'PaviaU_dcn_result_0.05(5).mat'
result_1dcnn["test_labels"] = test_labels
result_1dcnn["predict_labels"] = predict_labels
result_1dcnn["kappa"] = kappa
result_1dcnn["OA"] = oa
result_1dcnn["AA"] = aa
result_1dcnn["Accuracy"] = accuracy
scipy.io.savemat(os.path.join(DATA_PATH, file_name), result_1dcnn)
'''