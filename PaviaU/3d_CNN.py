#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 10:57:37 2018

@author: ssw
"""
import tensorflow as tf
import patch_size
import os
import scipy.io
import math
from next_batch_3dcnn import Dataset
import numpy as np

DATA_PATH = os.path.join(os.getcwd(),"Data")
IMAGE_SIZE = patch_size.patch_size

DATA_FILENAME = ['Train_'+str(IMAGE_SIZE)+'x'+str(IMAGE_SIZE)+'_3d.mat',
                 'Test_'+str(IMAGE_SIZE)+'x'+str(IMAGE_SIZE)+'_3d.mat']

FILTER_SIZE = 3
FILTER_DEPTH = 7
conv1_filter_num = 32
conv2_filter_num = 64
fc1_units = 1024
batch_size = 50
batch_size_for_test = 100
test_accuracy = []

#load preparaed data
'''
train_data = scipy.io.loadmat(os.path.join(
        DATA_PATH, 'Train_'+str(IMAGE_SIZE)+'x'+str(IMAGE_SIZE)+'_3d.mat'))
test_data = scipy.io.loadmat(os.path.join(
        DATA_PATH, 'Test_'+str(IMAGE_SIZE)+'x'+str(IMAGE_SIZE)+'_3d.mat'))

input_data = Dataset(train_data['train_patch'], train_data['train_labels'])
evl_data = Dataset(test_data['test_patch'], test_data['test_labels'])
'''
input_data = Dataset(scipy.io.loadmat(os.path.join(DATA_PATH, DATA_FILENAME[0]))['train_patch'], 
                     scipy.io.loadmat(os.path.join(DATA_PATH, DATA_FILENAME[0]))['train_labels']) 
evl_data = Dataset(scipy.io.loadmat(os.path.join(DATA_PATH, DATA_FILENAME[1]))['test_patch'], 
                   scipy.io.loadmat(os.path.join(DATA_PATH, DATA_FILENAME[1]))['test_labels'])        

BAND = input_data.channels
CLASS_NUM = input_data.class_num
tf.reset_default_graph()
#model CNN 
x = tf.placeholder(tf.float32, shape=[None, BAND, IMAGE_SIZE, IMAGE_SIZE, 1])
y_ = tf.placeholder(tf.float32, shape=[None, CLASS_NUM])

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv3d(x, w):
    return tf.nn.conv3d(x, w, strides=[1, 1, 1, 1, 1], padding='VALID')

def max_pool_2x2x2(x):
    return tf.nn.max_pool3d(x, ksize=[1, 2, 2, 2, 1],strides=[1, 2, 2, 2, 1], padding='SAME')

#first convolutional layer
w_conv1 = weight_variable([FILTER_DEPTH, FILTER_SIZE, FILTER_SIZE, 1, conv1_filter_num])
b_conv1 = bias_variable([conv1_filter_num])

h_conv1 = tf.nn.relu(conv3d(x, w_conv1)+ b_conv1)
h_pool1 = max_pool_2x2x2(h_conv1)
#second convolutional layer
w_conv2 = weight_variable([FILTER_DEPTH, FILTER_SIZE, FILTER_SIZE, conv1_filter_num, conv2_filter_num])
b_conv2 = bias_variable([conv2_filter_num])

h_conv2 = tf.nn.relu(conv3d(h_pool1, w_conv2) + b_conv2)
h_pool2 = max_pool_2x2x2(h_conv2)
#full-connected layer
'''
#with conv_padding='SAME'
size_after_conv_and_pool = int(math.ceil(math.ceil(IMAGE_SIZE/2.0)/2.0))     
depth_after_conv_and_pool = int(math.ceil(math.ceil(BAND)/2.0)/2.0)
'''
#with conv_padding='VALID'
size_after_conv_and_pool = int(math.ceil((math.ceil((IMAGE_SIZE - FILTER_SIZE + 1)/2.0)- FILTER_SIZE + 1)/2.0))  #change with net construction
depth_after_conv_and_pool = int(math.ceil((math.ceil((BAND - FILTER_DEPTH + 1)/2.0) - FILTER_DEPTH + 1)/2.0))
#FC1
w_fc1 = weight_variable([depth_after_conv_and_pool*(size_after_conv_and_pool**2)*conv2_filter_num, fc1_units])
b_fc1 = bias_variable([fc1_units])

h_pool2_flat = tf.reshape(h_pool2, [-1, depth_after_conv_and_pool*(size_after_conv_and_pool**2)*conv2_filter_num])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)

#dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
#readout layer
w_fc2 = weight_variable([fc1_units, CLASS_NUM])
b_fc2 = bias_variable([CLASS_NUM])

y_conv = tf.matmul(h_fc1_drop, w_fc2) + b_fc2

#loss and evluation
cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(5e-3).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(5001):
        batch = input_data.next_batch(batch_size)
        _, loss, train_accuracy = sess.run([train_step,cross_entropy,accuracy], 
                           feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
        if i % 100 == 0:
            print('Num of epcho: %d, Training accuracy: %g' % (i, train_accuracy))
            print('loss: %g' % loss)      
    
    #evluation
    for i in range(evl_data.sample_num/batch_size_for_test + 1):
        test_batch = evl_data.next_batch(batch_size_for_test)
        test_accuracy.append(accuracy.eval(feed_dict={
            x: test_batch[0], y_: test_batch[1], keep_prob: 1.0}))
    
print('Test accuracy: %g' % np.mean(test_accuracy))