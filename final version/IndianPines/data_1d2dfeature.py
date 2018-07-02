#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 10:20:36 2018

@author: ssw
"""
'''
Use trained 1d-net and 2d-net for fune-tunning, training the fusion net 
'''
import tensorflow as tf
import os
import scipy.io


DATA_PATH = os.path.join(os.getcwd(),"Data")

data_filename = ['Indianpines_train_1d_wa.mat',
                 'Indianpines_train_2d_wa.mat',
                 'Indianpines_test_1d.mat',
                 'Indianpines_test_2d.mat']

batch_size = 100
batch_size_for_test = 1000

a=scipy.io.loadmat(os.path.join(DATA_PATH, data_filename[2]))
#load preparaed data
input_data_1d = scipy.io.loadmat(os.path.join(DATA_PATH, data_filename[0]))['train_patch_1d']     
input_data_2d = scipy.io.loadmat(os.path.join(DATA_PATH, data_filename[1]))['train_patch_2d']
input_labels_1d = scipy.io.loadmat(os.path.join(DATA_PATH, data_filename[0]))['train_labels']
input_labels_2d = scipy.io.loadmat(os.path.join(DATA_PATH, data_filename[1]))['train_labels']
eval_data_1d = scipy.io.loadmat(os.path.join(DATA_PATH, data_filename[2]))['test_patch_1d']
eval_data_2d = scipy.io.loadmat(os.path.join(DATA_PATH, data_filename[3]))['test_patch_2d']
eval_labels = scipy.io.loadmat(os.path.join(DATA_PATH, data_filename[2]))['test_labels']

train_feature_1d = []
test_feature_1d = []
train_feature_2d = []
test_feature_2d = []
accuracy_1d = []
accuracy_2d = []
#load trained model
#load 1d-net and get 1d-feature
tf.reset_default_graph()
with tf.Session() as sess:
    saver = tf.train.import_meta_graph('./model/1dcnn_model/1dcnn-20000.meta')
    saver.restore(sess, './model/1dcnn_model/1dcnn-20000')
    graph1d = tf.get_default_graph()
    x_1d = graph1d.get_tensor_by_name("x:0")
    y_1d = graph1d.get_tensor_by_name("y_:0")
    h_feature_1d = graph1d.get_tensor_by_name("h_feature:0")
    keep_prob = graph1d.get_tensor_by_name('keep_prob:0')
    h_accuracy_1d = graph1d.get_tensor_by_name('accuracy:0')
    #get training data's 1d-net feature 
    for i in range(int(input_labels_1d.shape[0]/batch_size)+1):
        start = i * batch_size
        end = start + batch_size
        if end > input_labels_1d.shape[0]:
            end = input_labels_1d.shape[0]
        train_feature_1d.extend(h_feature_1d.eval(feed_dict={
            x_1d: input_data_1d[start:end], y_1d: input_labels_1d[start:end], keep_prob: 1.0}))
    #get test data's 1d-ne feature
    for i in range(int(eval_labels.shape[0]/batch_size_for_test)+1):
        start = i * batch_size_for_test 
        end = start + batch_size_for_test
        if end > eval_labels.shape[0]:
            end = eval_labels.shape[0]
        test_feature_1d.extend(h_feature_1d.eval(feed_dict={
            x_1d: eval_data_1d[start:end], y_1d: eval_labels[start:end], keep_prob: 1.0}))

#load 2d-net model and get 2d-feature

tf.reset_default_graph()
with tf.Session() as sess:
    saver = tf.train.import_meta_graph('./model/2dcnn_model/2dcnn-20000.meta')
    saver.restore(sess, tf.train.latest_checkpoint("model/2dcnn_model/"))
    graph2d = tf.get_default_graph()
    x_2d = graph2d.get_tensor_by_name('x:0')
    y_2d = graph2d.get_tensor_by_name('y_:0')
    h_feature_2d = graph2d.get_tensor_by_name('h_feature:0')
    keep_prob = graph2d.get_tensor_by_name('keep_prob:0')
    h_accuracy_2d = graph2d.get_tensor_by_name('accuracy:0')
    #get training data's 1d-net feature 
    for i in range(int(input_labels_2d.shape[0]/batch_size)+1):
        start = i * batch_size
        end = start + batch_size
        if end > input_labels_2d.shape[0]:
            end = input_labels_2d.shape[0]
        train_feature_2d.extend(h_feature_2d.eval(feed_dict={
            x_2d: input_data_2d[start:end], y_2d: input_labels_2d[start:end], keep_prob: 1.0}))
    #get test data's 1d-ne feature
    for i in range(int(eval_labels.shape[0]/batch_size_for_test)+1):
        start = i * batch_size_for_test 
        end = start + batch_size_for_test
        if end > eval_labels.shape[0]:
            end = eval_labels.shape[0]
        test_feature_2d.extend(h_feature_2d.eval(feed_dict={
            x_2d: eval_data_2d[start:end], y_2d: eval_labels[start:end], keep_prob: 1.0}))
        
#save data for fudion net
#1.Training data
train_dict = {}
file_name_train= 'Indianpines_train_feature.mat'
train_dict["train_feature_1d"] = train_feature_1d
train_dict['train_feature_2d'] = train_feature_2d
train_dict["train_labels"] = input_labels_1d
scipy.io.savemat(os.path.join(DATA_PATH, file_name_train),train_dict)
#2.Test data
test_dict = {}
file_name_test = 'Indianpines_test_feature.mat'
test_dict["test_feature_1d"] = test_feature_1d
test_dict["test_feature_2d"] = test_feature_2d
test_dict["test_labels"] = eval_labels
scipy.io.savemat(os.path.join(DATA_PATH, file_name_test),test_dict)
