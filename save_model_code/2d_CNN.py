#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 10:05:59 2018

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
'''
train_data = scipy.io.loadmat(os.path.join(
        DATA_PATH, 'Train_'+str(IMAGE_SIZE)+'x'+str(IMAGE_SIZE)+'.mat'))
test_data = scipy.io.loadmat(os.path.join(
        DATA_PATH, 'Test_'+str(IMAGE_SIZE)+'x'+str(IMAGE_SIZE)+'.mat'))

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
x = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, BAND], name='x')
y_ = tf.placeholder(tf.float32, shape=[None, CLASS_NUM], name='y_')

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name='weights')

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name='bias')

def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME', name='pool')

#first convolutional layer
with tf.variable_scope('conv1'):
    w_conv1 = weight_variable([FILTER_SIZE, FILTER_SIZE, BAND, conv1_channels])
    b_conv1 = bias_variable([conv1_channels])
    
    h_conv1 = tf.nn.relu(conv2d(x, w_conv1)+ b_conv1, name='h_conv1')
    h_pool1 = max_pool_2x2(h_conv1)
#second convolutional layer
with tf.variable_scope('conv2'):
    w_conv2 = weight_variable([FILTER_SIZE, FILTER_SIZE, conv1_channels, conv2_channels])
    b_conv2 = bias_variable([conv2_channels])
    
    h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2, name='h_conv2')
    h_pool2 = max_pool_2x2(h_conv2)

#full-connected layer
size_after_conv_and_pool = int(math.ceil(math.ceil(IMAGE_SIZE/2.0)/2.0))     #with padding 'SAME'
h_pool2_flat = tf.reshape(h_pool2, [-1, (size_after_conv_and_pool**2)*conv2_channels])
#FC1
with tf.variable_scope('FC1'):
    w_fc1 = weight_variable([(size_after_conv_and_pool**2)*conv2_channels, fc1_units])
    b_fc1 = bias_variable([fc1_units])
    
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1, name='h_fc1')
#dropout
keep_prob = tf.placeholder(tf.float32, name='keep_prob')
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
#readout layer
with tf.variable_scope('FC2'):
    w_fc2 = weight_variable([fc1_units, CLASS_NUM])
    b_fc2 = bias_variable([CLASS_NUM])
    
    y_conv = tf.matmul(h_fc1_drop, w_fc2) + b_fc2
#loss and evluation
cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')
# save model
saver = tf.train.Saver(max_to_keep=4)

with tf.Session() as sess:
    
    sess.run(tf.global_variables_initializer())
    for i in range(2001):
        batch = input_data.next_batch(batch_size)
        _, loss, train_accuracy = sess.run([train_step,cross_entropy,accuracy], 
                           feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
        if i % 100 == 0:
            saver.save(sess, "model/cnn-model", global_step=i)
            print('Num of epcho: %d, Training accuracy: %g' % (i, train_accuracy))
            print('loss: %g' % loss)      
    
    #evluation
    for i in range(evl_data.sample_num/batch_size_for_test + 1):
        test_batch = evl_data.next_batch(batch_size_for_test)
        test_accuracy.append(accuracy.eval(feed_dict={
            x: test_batch[0], y_: test_batch[1], keep_prob: 1.0}))
    
print('Test accuracy: %g' % np.mean(test_accuracy))