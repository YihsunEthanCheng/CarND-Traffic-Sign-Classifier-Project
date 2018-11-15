#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 23:08:39 2018

@author: Ethan Cheng
"""
%load_ext autoreload
%autoreload 2
import matplotlib.pyplot as plt
from modules.dataLoader import dataLoader
import cv2

Pool = {
       'ksize': [1, 2, 2, 1],
       'strides': [1, 2, 2, 1]        
        }
C1 = {
       'ksize' : [5, 5],
       'n': 6,
       'strides': [1, 1, 1, 1],Tf
       'pool': Pool
       }

C2 = {
       'ksize' : [5, 5],
       'n': 16,
       'strides': [1, 1, 1, 1],
       'pool': Pool
       }

params = {
        # Arguments used for tf.truncated_normal, randomly defines variables 
        # for the weights and biases for each layer
        'mu' : 0,
        'sigma' :  0.1,
        'c1': C1, 
        'c2': C2,
        'nFC1' : 120,
        'nFC2' : 84,
        'nCls' : 43,
        'imsize' : [32, 32, 3],
        'lr' : 0.001,
        'BATCH_SIZE' : 256,
        'EPOCHS' : 1
        } 

#%%
from modules.LeNet import LeNet
import tensorflow as tf

data = dataLoader()
X = tf.placeholder(tf.float32, tuple([None] + params['imsize']))
Y = tf.placeholder(tf.int32, (None, params['nCls']))
model = LeNet(X, Y, params)
