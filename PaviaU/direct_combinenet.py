#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat May 19 15:59:15 2018

@author: ssw
"""


import tensorflow as tf
import os
import scipy.io
from next_batch_for_combinenet import Dataset_for_combinenet
import numpy as np

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
input_data_1d = scipy.io.loadmat(os.path.join(DATA_PATH, data_filename))['train_feature_1d']
input_data_2d = scipy.io.loadmat(os.path.join(DATA_PATH, data_filename))['train_feature_2d']
input_labels = scipy.io.loadmat(os.path.join(DATA_PATH, data_filename))['train_labels_1d']
input_data = np.hstack((input_data_1d, input_data_2d))
input_dataset = Dataset_for_combinenet(input_data, input_labels)
del input_data_1d, input_data_2d

feature_train_num = input_dataset.feature_shape
class_num = input_dataset.class_num

#model direct-combine net
x = tf.placeholder(tf.float32, shape=[None, feature_train_num], name='x')
y_ = tf.placeholder(tf.float32, shape=[None, class_num], name='y_')

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name='weights')

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name='bias')

#full-connet layers
#FC1
with tf.variable_scope('FC1'):
    w_fc1 = weight_variable([feature_train_num, fc1_units])
    b_fc1 = bias_variable([fc1_units])
    
    h_fc1 = tf.nn.relu(tf.matmul(x, w_fc1) + b_fc1, name='h_fc1')
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
    w_fc3 = weight_variable([fc2_units, fc3_units])
    b_fc3 = bias_variable([fc3_units])
    
    h_fc3 = tf.nn.relu(tf.matmul(h_fc2_drop, w_fc3) + b_fc3, name='h_fc3')
#dropout
h_fc3_drop = tf.nn.dropout(h_fc3, keep_prob)

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
    w_fc4 = weight_variable([fc2_units, class_num])
    b_fc4 = bias_variable([class_num])
    
    y_output = tf.matmul(h_fc2_drop, w_fc4) + b_fc4
#loss and evluation
softmax_out = tf.nn.softmax(logits=y_output, name='predict_y')
cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_output))
train_step = tf.train.AdamOptimizer(1e-5).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_output, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')
#save graph
saver = tf.train.Saver(max_to_keep=1)
loss_get = []
with tf.Session() as sess:
    #sess.run(tf.global_variables_initializer())
    saver.restore(sess, "model/dcn/dcn-model-20000")
    for i in range(10001):
        batch = input_dataset.next_batch(batch_size)
        #when 1d-net is training,2d-net feed zeros
        _, loss, train_accuracy = sess.run([train_step,cross_entropy,accuracy], 
                           feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
        loss_get.append(loss)
        if i % 2000 == 0:
            saver.save(sess, "model/dcn/dcn-model", global_step=i)
            print('(Num of epoch: %d)' % i)
            print('Training accuracy: %g' % train_accuracy)
            print('loss: %g' % loss)   
'''
loss_result={}
loss_result['dcn_loss'] = np.array(loss_get)

file_name = 'dcn_loss'
scipy.io.savemat(os.path.join(DATA_PATH, file_name), loss_result)     
'''