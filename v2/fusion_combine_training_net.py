#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 15:07:47 2018

@author: ssw
"""

import tensorflow as tf
import patch_size
import os
import scipy.io
import math
from next_batch import Dataset
import numpy as np

DATA_PATH = os.path.join(os.getcwd(),"Data")

data_filename = ['Train_'+str(patch_size.patch_size)+'x'+str(patch_size.patch_size)+'_1d.mat',
                 'Train_'+str(patch_size.patch_size)+'x'+str(patch_size.patch_size)+'_2d.mat',
                 'Test_'+str(patch_size.patch_size)+'x'+str(patch_size.patch_size)+'_1d.mat',
                 'Test_'+str(patch_size.patch_size)+'x'+str(patch_size.patch_size)+'_2d.mat']

filter_size_1d = 5
filter_size_2d = 5
conv1_channels = 32
conv2_channels = 64
fc1_units = 1024
batch_size = 100
batch_size_for_test = 1000
test_accuracy = []
tf.reset_default_graph()

input_data_1d = Dataset(scipy.io.loadmat(os.path.join(DATA_PATH, data_filename[0]))['train_patch_1d'], 
                     scipy.io.loadmat(os.path.join(DATA_PATH, data_filename[0]))['train_labels']) 
input_data_2d = Dataset(scipy.io.loadmat(os.path.join(DATA_PATH, data_filename[1]))['train_patch_2d'], 
                     scipy.io.loadmat(os.path.join(DATA_PATH, data_filename[1]))['train_labels']) 
eval_data_1d = scipy.io.loadmat(os.path.join(DATA_PATH, data_filename[2]))['test_patch_1d']
eval_data_2d = scipy.io.loadmat(os.path.join(DATA_PATH, data_filename[3]))['test_patch_2d']
eval_labels = scipy.io.loadmat(os.path.join(DATA_PATH, data_filename[2]))['test_labels']
                      
class_num = input_data_1d.class_num
#variables for 1dcnn
height = input_data_1d.height
#variables for 2dcnn
band = input_data_2d.channels

#build fudion net
x_1d = tf.placeholder(tf.float32, shape=[None, height, 1, 1], name='input_1d')
x_2d = tf.placeholder(tf.float32, shape=[None, patch_size.patch_size, patch_size.patch_size, band], name='input_2d')
y_ = tf.placeholder(tf.float32, shape=[None, class_num], name='y_')

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name='weights')

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name='bias')

def conv1d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x1(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 1, 1],strides=[1, 2, 1, 1], padding='SAME', name='pool')

def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME', name='pool')
#1d-net build
with tf.name_scope('net_1d'):
    #first convolutional layer
    with tf.variable_scope('conv1'):
        w_conv1 = weight_variable([filter_size_1d, 1, 1, conv1_channels])
        b_conv1 = bias_variable([conv1_channels])
        
        h_conv1 = tf.nn.relu(conv1d(x_1d, w_conv1)+ b_conv1, name='h_conv1')
        h_pool1 = max_pool_2x1(h_conv1)
    #second convolutional layer
    with tf.variable_scope('conv2'):
        w_conv2 = weight_variable([filter_size_1d, 1, conv1_channels, conv2_channels])
        b_conv2 = bias_variable([conv2_channels])
        
        h_conv2 = tf.nn.relu(conv1d(h_pool1, w_conv2) + b_conv2, name='h_conv2')
        h_pool2 = max_pool_2x1(h_conv2)

    #feature flat
    size_after_cp_1d = int(math.ceil(math.ceil(height/2.0)/2.0))     #with padding 'SAME'
    h_pool2_flat_1d = tf.reshape(h_pool2, [-1, size_after_cp_1d*conv2_channels], name='1dfeature')

#2d-net build
with tf.name_scope('net_2d'):
    #first convolutional layer
    with tf.variable_scope('conv1'):
        w_conv1 = weight_variable([filter_size_2d, filter_size_2d, band, conv1_channels])
        b_conv1 = bias_variable([conv1_channels])
        
        h_conv1 = tf.nn.relu(conv2d(x_2d, w_conv1)+ b_conv1, name='h_conv1')
        h_pool1 = max_pool_2x2(h_conv1)
    #second convolutional layer
    with tf.variable_scope('conv2'):
        w_conv2 = weight_variable([filter_size_2d, filter_size_2d, conv1_channels, conv2_channels])
        b_conv2 = bias_variable([conv2_channels])
        
        h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2, name='h_conv2')
        h_pool2 = max_pool_2x2(h_conv2)

    #feature flat
    size_after_cp_2d = int(math.ceil(math.ceil(patch_size.patch_size/2.0)/2.0))     #with padding 'SAME'
    h_pool2_flat_2d = tf.reshape(h_pool2, [-1, (size_after_cp_2d**2)*conv2_channels], name='2dfeature')

#fusion net
h_feature = tf.concat([h_pool2_flat_1d, h_pool2_flat_2d], 1, name='combine-feature')
#full-connect layer
h_feature_shape = h_feature.get_shape().as_list()
#FC1
with tf.variable_scope('FC1'):
    w_fc1 = weight_variable([h_feature_shape[1], fc1_units])
    b_fc1 = bias_variable([fc1_units])
    
    h_fc1 = tf.nn.relu(tf.matmul(h_feature, w_fc1) + b_fc1, name='h_fc1')
#dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob, name='keep_prob')
#readout layer
with tf.variable_scope('FC2'):
    w_fc2 = weight_variable([fc1_units, class_num])
    b_fc2 = bias_variable([class_num])
    
    y_conv = tf.matmul(h_fc1_drop, w_fc2) + b_fc2
#loss and evluation
cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(5e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')

#train model
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(10001):
        batch_1d = input_data_1d.next_batch(batch_size)
        batch_2d = input_data_2d.next_batch(batch_size)
        #when 1d-net is training,2d-net feed zeros
        batch_1d_zeros = input_data_1d.zeros_batch(batch_2d[0].shape[0])
        batch_2d_zeros = input_data_2d.zeros_batch(batch_1d[0].shape[0])
        _, loss_1d, train_accuracy_1d = sess.run([train_step,cross_entropy,accuracy], 
                           feed_dict={x_1d: batch_1d[0], x_2d: batch_2d_zeros, y_: batch_1d[1], keep_prob: 0.5})
        _, loss_2d, train_accuracy_2d = sess.run([train_step,cross_entropy,accuracy], 
                           feed_dict={x_1d: batch_1d_zeros, x_2d: batch_2d[0], y_: batch_2d[1], keep_prob: 0.5})
        if i % 1000 == 0:
            print('(Num of epcho: %d)' % i)
            print('Training accuracy_1d: %g, accuracy_2d: %g' % (train_accuracy_1d, train_accuracy_2d))
            print('loss_1d: %g, loss_2d: %g' % (loss_1d, loss_2d))   
    
    #evluation
    for i in range(int(eval_labels.shape[0]/batch_size_for_test)+1):
        start = i * batch_size_for_test 
        end = start + batch_size_for_test
        if end > eval_labels.shape[0]:
            end = eval_labels.shape[0]
        test_accuracy.append(accuracy.eval(feed_dict={
            x_1d: eval_data_1d[start:end], x_2d: eval_data_2d[start:end], y_: eval_labels[start:end], keep_prob: 1.0}))

print('Test accuracy: %g' % np.mean(test_accuracy))