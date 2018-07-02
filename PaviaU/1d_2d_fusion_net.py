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


DATA_PATH = "/home/ssw/Hyperspectral_classification_CNN/v5/Data"
data_filename = 'PaviaU_train_feature.mat'
 
fc1_units = 1024
fc2_units = 512
#fc3_units = 256





batch_size = 100
batch_size_for_test = 1000
test_accuracy = []
tf.reset_default_graph()

#load data

input_labels_1d = scipy.io.loadmat(os.path.join(DATA_PATH, data_filename))['train_labels_1d']
input_labels_2d = scipy.io.loadmat(os.path.join(DATA_PATH, data_filename))['train_labels_2d']

input_data_1d = Dataset_for_combinenet(scipy.io.loadmat(os.path.join(DATA_PATH, data_filename))['train_feature_1d'], 
                        input_labels_1d)

input_data_2d = Dataset_for_combinenet(scipy.io.loadmat(os.path.join(DATA_PATH, data_filename))['train_feature_2d'],
                        input_labels_2d)

feature_1d_num = input_data_1d.feature_shape
feature_2d_num = input_data_2d.feature_shape
class_num = input_data_1d.class_num
#model fudion net
x_1d = tf.placeholder(tf.float32, shape=[None, feature_1d_num], name='x_1d')
x_2d = tf.placeholder(tf.float32, shape=[None, feature_2d_num], name='x_2d')
y_ = tf.placeholder(tf.float32, shape=[None, class_num], name='y_')

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

#FC2
with tf.variable_scope('FC2'):
    w_fc2 = weight_variable([fc1_units, fc2_units])
    b_fc2 = bias_variable([fc2_units])
    
    h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, w_fc2) + b_fc2, name='h_fc2')
#dropout
h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)
'''
#FC3
with tf.variable_scope('FC3'):
    w_fc4 = weight_variable([fc2_units, fc3_units])
    b_fc4 = bias_variable([fc3_units])
    
    h_fc4 = tf.nn.relu(tf.matmul(h_fc2_drop, w_fc4) + b_fc4, name='h_fc3')
#dropout
h_fc3_drop = tf.nn.dropout(h_fc4, keep_prob)

#FC4
with tf.variable_scope('FC4'):
    w_fc5 = weight_variable([fc3_units, fc4_units])
    b_fc5 = bias_variable([fc4_units])
    
    h_fc5 = tf.nn.relu(tf.matmul(h_fc3_drop, w_fc5) + b_fc5, name='h_fc4')
#dropout
h_fc4_drop = tf.nn.dropout(h_fc5, keep_prob)

#FC5
with tf.variable_scope('FC5'):
    w_fc6 = weight_variable([fc4_units, fc5_units])
    b_fc6 = bias_variable([fc5_units])
    
    h_fc6 = tf.nn.relu(tf.matmul(h_fc4_drop, w_fc6) + b_fc6, name='h_fc5')
#dropout
h_fc5_drop = tf.nn.dropout(h_fc6, keep_prob)
'''
#readout layer
with tf.variable_scope('FC4'):
    w_fc3 = weight_variable([fc2_units, class_num])
    b_fc3 = bias_variable([class_num])
    
    y_output = tf.matmul(h_fc2_drop, w_fc3) + b_fc3
#loss and evluation
softmax_out = tf.nn.softmax(logits=y_output, name='predict_y')
cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_output))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_output, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')
#save net
saver = tf.train.Saver(max_to_keep=1)

acc_1d, acc_2d, loss_1d_get, loss_2d_get = [], [], [], []

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(20001):
        batch_1d = input_data_1d.next_batch(batch_size)
        batch_2d = input_data_2d.next_batch(batch_size)
        #when 1d-net is training,2d-net feed zeros
        batch_1d_zeros = input_data_1d.zeros_batch(batch_2d[0].shape[0])
        batch_2d_zeros = input_data_2d.zeros_batch(batch_1d[0].shape[0])
        _, loss_1d, train_accuracy_1d = sess.run([train_step,cross_entropy,accuracy], 
                           feed_dict={x_1d: batch_1d[0], x_2d: batch_2d_zeros, y_: batch_1d[1], keep_prob: 0.5})
        acc_1d.append(train_accuracy_1d)
        loss_1d_get.append(loss_1d)
        _, loss_2d, train_accuracy_2d = sess.run([train_step,cross_entropy,accuracy], 
                           feed_dict={x_1d: batch_1d_zeros, x_2d: batch_2d[0], y_: batch_2d[1], keep_prob: 0.5})
        acc_2d.append(train_accuracy_2d)
        loss_2d_get.append(loss_2d)
        
        if i % 2000 == 0:
            saver.save(sess, "model/fusion_net/fusion_net-model", global_step=i)
            print('(Num of epoch: %d)' % i)
            print('Training accuracy_1d: %g, accuracy_2d: %g' % (train_accuracy_1d, train_accuracy_2d))
            print('loss_1d: %g, loss_2d: %g' % (loss_1d, loss_2d))   
'''
import numpy as np    
result={}
file_name = 'result'
result['accuracy_1d'] = np.array(acc_1d)
result['accuracy_2d'] = np.array(acc_2d)
result['loss_1d'] = np.array(loss_1d_get)
result['loss_2d'] = np.array(loss_2d_get)
scipy.io.savemat(os.path.join(DATA_PATH, file_name), result) 
'''   