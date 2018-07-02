#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 19:15:51 2018

@author: ssw
"""

import numpy as np

class Dataset_for_combinenet(object):
    def __init__(self, input_data, labels):
        '''
        input_data_1d/2d[number of samples, feature]
        labels[number of samples, onehot label]
        '''
        self._images = input_data
        self._labels = labels
        self.sample_num = input_data.shape[0]
        self.feature_shape = input_data.shape[1]
        self.class_num = labels.shape[1]
        self._index = 0
        self._complete = 0
        

    def next_batch(self, batch_size):
        start = self._index
        if self._complete == 1:
            # Shuffle the data
            perm = np.arange(self.sample_num)
            np.random.shuffle(perm)
            self._images = self._images[perm]
            self._labels = self._labels[perm]
            self._complete = 0
            start = 0
        end = start + batch_size 
        if end >= self.sample_num:
            self._complete = 1
            end = self.sample_num 
        self._index = end
        return (self._images[start:end], self._labels[start:end])
    
    def zeros_batch(self, batch_size):
        
        return np.zeros([batch_size, self.feature_shape]) 