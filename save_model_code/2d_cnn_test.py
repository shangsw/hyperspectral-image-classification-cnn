#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 20:58:20 2018

@author: ssw
"""

import tensorflow as tf
import patch_size
import os
import scipy.io
from next_batch import Dataset
import numpy as np

DATA_PATH = os.path.join(os.getcwd(),"Data")
IMAGE_SIZE = patch_size.patch_size

DATA_FILENAME = ['Train_'+str(IMAGE_SIZE)+'x'+str(IMAGE_SIZE)+'.mat',
                 'Test_'+str(IMAGE_SIZE)+'x'+str(IMAGE_SIZE)+'.mat']

FILTER_SIZE = 5
conv1_channels = 32
conv2_channels = 64
fc1_units = 1024
batch_size = 50
batch_size_for_test = 1000
test_accuracy = []

#load preparaed data

evl_data = Dataset(scipy.io.loadmat(os.path.join(DATA_PATH, DATA_FILENAME[1]))['test_patch'], 
                   scipy.io.loadmat(os.path.join(DATA_PATH, DATA_FILENAME[1]))['test_labels'])        

with tf.Session() as sess:
    #load model
    saver = tf.train.import_meta_graph('./model/cnn-model-2000.meta')
    saver.restore(sess, tf.train.latest_checkpoint("model/"))
    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name("x:0")
    y_ = graph.get_tensor_by_name("y_:0")
    keep_prob = graph.get_tensor_by_name("keep_prob:0")
    accuracy = graph.get_tensor_by_name("accuracy:0")
    for i in range(evl_data.sample_num/batch_size_for_test + 1):
        test_batch = evl_data.next_batch(batch_size_for_test)
        test_accuracy.append(sess.run(accuracy, feed_dict={
            x: test_batch[0], y_: test_batch[1], keep_prob: 1.0}))
    
print('Test accuracy: %g' % np.mean(test_accuracy))