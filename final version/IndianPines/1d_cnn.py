#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 09:09:26 2018

@author: ssw
"""

import tensorflow as tf
import os
import scipy.io
import math
from next_batch import Dataset
import numpy as np

DATA_PATH = os.path.join(os.getcwd(),"Data")

data_filename = ['Indianpines_train_1d_wa.mat',
                 'Indianpines_test_1d.mat']

filter_size_1 = 16
filter_size_2 = 4
conv1_channels = 53
conv2_channels = 64
conv3_channels = 128
conv4_channels = 128

fc1_units = 1024
batch_size = 100
batch_size_for_test = 1000
test_accuracy = []
a=scipy.io.loadmat(os.path.join(DATA_PATH, data_filename[0]))
input_data = Dataset(scipy.io.loadmat(os.path.join(DATA_PATH, data_filename[0]))['train_patch_1d'], 
                     scipy.io.loadmat(os.path.join(DATA_PATH, data_filename[0]))['train_labels']) 
evl_data = Dataset(scipy.io.loadmat(os.path.join(DATA_PATH, data_filename[1]))['test_patch_1d'], 
                   scipy.io.loadmat(os.path.join(DATA_PATH, data_filename[1]))['test_labels'])        

#load preparaed data
height = input_data.height
class_num = input_data.class_num
tf.reset_default_graph()

#model CNN
x = tf.placeholder(tf.float32, shape=[None, height, 1, 1], name='x')
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
    return tf.nn.max_pool(x, ksize=[1, 3, 1, 1],strides=[1, 3, 1, 1], padding='SAME', name='pool')

#first convolutional layer
with tf.variable_scope('conv1'):
    w_conv1 = weight_variable([filter_size_1, 1, 1, conv1_channels])
    b_conv1 = bias_variable([conv1_channels])
    
    h_conv1 = tf.nn.relu(conv1d(x, w_conv1)+ b_conv1, name='h_conv1')
    h_pool1 = max_pool_2x1(h_conv1)

#second convolutional layer
with tf.variable_scope('conv2'):
    w_conv2 = weight_variable([filter_size_1, 1, conv1_channels, conv2_channels])
    b_conv2 = bias_variable([conv2_channels])
    
    h_conv2 = tf.nn.relu(conv1d(h_pool1, w_conv2) + b_conv2, name='h_conv2')
    h_pool2 = max_pool_2x1(h_conv2)

#third convolutional layer
with tf.variable_scope('conv3'):
    w_conv3 = weight_variable([filter_size_2, 1, conv2_channels, conv3_channels])
    b_conv3 = bias_variable([conv3_channels])
    
    h_conv3 = tf.nn.relu(conv1d(h_pool2, w_conv3)+ b_conv3, name='h_conv3')
    h_pool3 = max_pool_2x1(h_conv3)

#forth convolutional layer
with tf.variable_scope('conv4'):
    w_conv4 = weight_variable([filter_size_2, 1, conv3_channels, conv4_channels])
    b_conv4 = bias_variable([conv4_channels])
    
    h_conv4 = tf.nn.relu(conv1d(h_pool3, w_conv4)+ b_conv4, name='h_conv4')

#full-connected layer
size_after_conv_and_pool = int(math.ceil(math.ceil(math.ceil(height/3.0)/3.0)/3.0))     #with padding 'SAME'
h_pool2_flat = tf.reshape(h_conv4, [-1, size_after_conv_and_pool*conv4_channels], name='h_feature')

#FC1
with tf.variable_scope('FC1'):
    w_fc1 = weight_variable([size_after_conv_and_pool*conv4_channels, fc1_units])
    b_fc1 = bias_variable([fc1_units])
    
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1, name='h_fc1')
#dropout
keep_prob = tf.placeholder(tf.float32, name='keep_prob')
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
#readout layer
with tf.variable_scope('FC2'):
    w_fc2 = weight_variable([fc1_units, class_num])
    b_fc2 = bias_variable([class_num])
    
    y_conv = tf.matmul(h_fc1_drop, w_fc2) + b_fc2
#loss and evluation
softmax_out = tf.nn.softmax(logits=y_conv, name='predict_y')
cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(5e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')
# save model
saver = tf.train.Saver(max_to_keep=1)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(20001):
        batch = input_data.next_batch(batch_size)
        _, loss, train_accuracy = sess.run([train_step,cross_entropy,accuracy], 
                           feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
        if i % 2000 == 0:
            saver.save(sess, "model/1dcnn_model/1dcnn", global_step=i)
            print('Num of epoch: %d, Training accuracy: %g' % (i, train_accuracy))
            print('loss: %g' % loss)      
    
    #evluation
    for i in range(evl_data.sample_num/batch_size_for_test + 1):
        test_batch = evl_data.next_batch(batch_size_for_test)
        test_accuracy.append(accuracy.eval(feed_dict={
            x: test_batch[0], y_: test_batch[1], keep_prob: 1.0}))
    
print('Test accuracy: %g' % np.mean(test_accuracy))



