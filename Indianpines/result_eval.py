#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun May 13 11:01:32 2018

@author: ssw
"""
from sklearn.metrics import confusion_matrix, cohen_kappa_score, precision_score
import numpy as np

def my_cnf_matrix(true_labels, predict_labels):
    cnf_matrix = confusion_matrix(true_labels, predict_labels)
    return cnf_matrix

def my_kappa(true_labels, predict_labels):
    kappa = cohen_kappa_score(true_labels, predict_labels)
    return kappa

def my_oa(true_labels, predict_labels):
    oa = precision_score(true_labels, predict_labels, average='micro')
    return oa

def my_aa(cnf_matrix):
    accuracy = []
    for i in range(cnf_matrix.shape[0]):
        acc = cnf_matrix[i][i].astype(float)/np.sum(cnf_matrix[i,:])
        accuracy.append(acc)
    aa = np.mean(accuracy)
    return (accuracy, aa)