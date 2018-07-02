#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 16:31:11 2018

@author: ssw
"""
import numpy as np

class Dataset(object):
    '''
    with image[number of batches, depth, heigh, width, channels]
    labels[number of batches, onehot label]
    '''
    def __init__(self, images, labels):
        
        self._images = images
        self._labels = labels
        self.sample_num = images.shape[0]
        self.channels = images.shape[1]
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
        if end > self.sample_num:
            self._complete = 1
            end = self.sample_num 
        self._index = end
        return (self._images[start:end], self._labels[start:end])