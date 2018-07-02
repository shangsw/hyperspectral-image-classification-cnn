#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 18:37:59 2018

@author: ssw
"""

import tensorflow as tf
import os
import scipy.io
from next_batch_for_combinenet import Dataset_for_combinenet
import numpy as np

DATA_PATH = os.path.join(os.getcwd(),"Data")
data_filename = ['Train_feature.mat','Test_feature.mat', 'input_1d.mat']

fc1_units = 2048
'''
fc2_units = 1024
fc3_units = 500
'''
batch_size = 100
batch_size_for_test = 1000
test_accuracy = []
tf.reset_default_graph()

#load data

input_labels = scipy.io.loadmat(os.path.join(DATA_PATH, data_filename[0]))['train_labels']
eval_data_1d = scipy.io.loadmat(os.path.join(DATA_PATH, data_filename[1]))['test_feature_1d']
#eval_data_1d = scipy.io.loadmat(os.path.join(DATA_PATH, data_filename[2]))['test_original_1d']
eval_data_2d = scipy.io.loadmat(os.path.join(DATA_PATH, data_filename[1]))['test_feature_2d']
eval_labels = scipy.io.loadmat(os.path.join(DATA_PATH, data_filename[1]))['test_labels']
#if use original 1d, change every input_data_1d with input_1d_original
'''
input_data_1d = Dataset_for_combinenet(scipy.io.loadmat(os.path.join(DATA_PATH, data_filename[2]))['train_original_1d'], 
                            input_labels)
'''
input_data_1d = Dataset_for_combinenet(scipy.io.loadmat(os.path.join(DATA_PATH, data_filename[0]))['train_feature_1d'], 
                        input_labels)

input_data_2d = Dataset_for_combinenet(scipy.io.loadmat(os.path.join(DATA_PATH, data_filename[0]))['train_feature_2d'],
                        input_labels)

feature_1d_num = input_data_1d.feature_shape
feature_2d_num = input_data_2d.feature_shape
class_num = input_data_1d.class_num
#model fudion net
x_1d = tf.placeholder(tf.float32, shape=[None, feature_1d_num])
x_2d = tf.placeholder(tf.float32, shape=[None, feature_2d_num])
y_ = tf.placeholder(tf.float32, shape=[None, class_num])

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name='weights')

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name='bias')

h_input = tf.concat([x_1d, x_2d], 1, name='combine_input')
#full-connet layers
#FC1
with tf.variable_scope('FC1'):
    w_fc1 = weight_variable([feature_1d_num + feature_2d_num, fc1_units])
    b_fc1 = bias_variable([fc1_units])
    
    h_fc1 = tf.nn.relu(tf.matmul(h_input, w_fc1) + b_fc1, name='h_fc1')
#dropout
keep_prob = tf.placeholder(tf.float32, name='keep_prob')
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
'''
#FC2
with tf.variable_scope('FC2'):
    w_fc2 = weight_variable([fc1_units, fc2_units])
    b_fc2 = bias_variable([fc2_units])
    
    h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, w_fc2) + b_fc2, name='h_fc2')
#dropout
h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)
#FC3
with tf.variable_scope('FC3'):
    w_fc4 = weight_variable([fc2_units, fc3_units])
    b_fc4 = bias_variable([fc3_units])
    
    h_fc4 = tf.nn.relu(tf.matmul(h_fc2_drop, w_fc4) + b_fc4, name='h_fc4')
#dropout
h_fc3_drop = tf.nn.dropout(h_fc4, keep_prob)
'''
#readout layer
with tf.variable_scope('FC3'):
    w_fc3 = weight_variable([fc1_units, class_num])
    b_fc3 = bias_variable([class_num])
    
    y_output = tf.matmul(h_fc1_drop, w_fc3) + b_fc3
#loss and evluation
cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_output))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_output, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')

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
            print('Num of epcho: %d)' % i)
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