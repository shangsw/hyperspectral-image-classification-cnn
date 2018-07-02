#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May 17 16:42:56 2018

@author: ssw
"""

import tensorflow as tf
import os
import scipy.io
from next_batch_for_combinenet import Dataset_for_combinenet
import numpy as np
import result_eval

DATA_PATH = "/home/ssw/Hyperspectral_classification_CNN/v4/Data"

data_filename = 'Indianpines_test_feature.mat'


eval_data_1d = scipy.io.loadmat(os.path.join(DATA_PATH, data_filename))['test_feature_1d']
eval_data_2d = scipy.io.loadmat(os.path.join(DATA_PATH, data_filename))['test_feature_2d']
eval_labels = scipy.io.loadmat(os.path.join(DATA_PATH, data_filename))['test_labels']
eval_dataset_1d = Dataset_for_combinenet(eval_data_1d, eval_labels)
eval_dataset_2d = Dataset_for_combinenet(eval_data_2d, eval_labels)

batch_size_for_test = 1
test_labels = np.argmax(eval_labels, axis=1)
predict_labels=np.zeros(eval_dataset_1d.sample_num, dtype=int)

with tf.Session() as sess:
    #load model
    saver = tf.train.import_meta_graph('./model/fusion_net/fusion_net-model-20000.meta')
    saver.restore(sess, tf.train.latest_checkpoint("./model/fusion_net/"))
    graph = tf.get_default_graph()
    x_1d = graph.get_tensor_by_name("x_1d:0")
    x_2d = graph.get_tensor_by_name("x_2d:0")
    y_ = graph.get_tensor_by_name("y_:0")
    predict_y = graph.get_tensor_by_name("predict_y:0")
    keep_prob = graph.get_tensor_by_name("keep_prob:0")
    for i in range(eval_dataset_1d.sample_num):
        test_batch_1d = eval_dataset_1d.next_batch(batch_size_for_test)
        test_batch_2d = eval_dataset_2d.next_batch(batch_size_for_test)
        predict_labels[i]=np.argmax(sess.run(predict_y, feed_dict={
            x_1d: test_batch_1d[0], x_2d: test_batch_2d[0], y_: test_batch_2d[1], keep_prob: 1.0}))

cnf_matrix = result_eval.my_cnf_matrix(test_labels, predict_labels)
kappa = result_eval.my_kappa(test_labels, predict_labels)
oa = result_eval.my_oa(test_labels, predict_labels)
(accuracy, aa) = result_eval.my_aa(cnf_matrix)


#save result
result_1dcnn = {}
file_name = 'Indianpines_fn_result_4.mat'
result_1dcnn["test_labels"] = test_labels
result_1dcnn["predict_labels"] = predict_labels
result_1dcnn["kappa"] = kappa
result_1dcnn["OA"] = oa
result_1dcnn["AA"] = aa
result_1dcnn["Accuracy"] = accuracy
scipy.io.savemat(os.path.join(DATA_PATH, file_name), result_1dcnn)