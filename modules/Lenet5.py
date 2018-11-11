#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 00:12:19 2018

@author: Ethan Cheng
"""
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.contrib.layers import flatten
import numpy as np

Pool = {
       'ksize': [1, 2, 2, 1],
       'strides': [1, 2, 2, 1]        
        }
C1 = {
       'ksize' : [5, 5],
       'n': 6,
       'strides': [1, 1, 1, 1],
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
        'imsize' : [32, 32, 3]
        } 

class LeNet(object):
    
    def __init__(self, params = params):
        self.__dict__ = params.copy()
        # init weights        
        shape = self.c1['ksize'] + [self.imsize[2], self.c1['n']]
        # conv1 weights
        self.conv1_W = tf.Variable(tf.truncated_normal(shape = shape, mean = self.mu, stddev = self.sigma))
        self.conv1_b = tf.Variable(tf.zeros(self.c1['n']))
        # Conv1 ouput dimensions
        self.size_c1 = (np.array(self.imsize[:2]) - (np.array(self.c1['ksize'])-1))/np.array(self.c1['strides'][1:3])
        self.size_p1 = self.size_c1/np.array(self.c1['pool']['strides'][1:3])
        shape =  self.c2['ksize'] + [self.c1['n'], self.c2['n']]
        # conv2 weights
        self.conv2_W = tf.Variable(tf.truncated_normal(shape = shape, mean = self.mu, stddev = self.sigma))
        self.conv2_b = tf.Variable(tf.zeros(self.c2['n']))
        # conv2 output dimensions
        self.size_c2 = (self.size_p1 - (np.array(self.c2['ksize'])-1))/np.array(self.c2['strides'][1:3])
        self.size_p2 = (self.size_c2/np.array(self.c2['pool']['strides'][1:3])).astype(int)
        shape = (np.prod(self.size_p2)*self.c2['n'], self.nFC1)
        # fully connected layers
        self.fc1_W = tf.Variable(tf.truncated_normal(shape = shape, mean = self.mu, stddev = self.sigma))
        self.fc1_b = tf.Variable(tf.zeros(self.nFC1))
        self.fc2_W  = tf.Variable(tf.truncated_normal(shape=(self.nFC1, self.nFC2), mean = self.mu, stddev = self.sigma))
        self.fc2_b  = tf.Variable(tf.zeros(self.nFC2))
        self.fc3_W = tf.Variable(tf.truncated_normal(shape=(self.nFC2, self.nCls), mean = self.mu, stddev = self.sigma))
        self.fc3_b = tf.Variable(tf.zeros(self.nCls))
               
    def feed(self, x):           
        # Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
        conv1 = tf.nn.conv2d(x, self.conv1_W, strides = self.c1.strides, padding='VALID') + self.conv1_b
        conv1 = tf.nn.relu(conv1)
        conv1 = tf.nn.max_pool(conv1, ksize = self.c1.pool.ksize, strides = self.c1.pool.strides, padding='VALID')
        # Layer 2: Convolutional. Output = 10x10x16.
        conv2 = tf.nn.conv2d(conv1, self.conv2_W, strides = self.c2.strides, padding='VALID') + self.conv2_b
        conv2 = tf.nn.relu(conv2)
        conv2 = tf.nn.max_pool(conv2, ksize= self.c2.pool.ksize, strides = self.c1.pool.strides, padding='VALID')
        fc0 = flatten(conv2)
        # Layer 3: Fully Connected. Input = 400. Output = 120.
        fc1 = tf.matmul(fc0, self.fc1_W) + self.fc1_b
        fc1 = tf.nn.relu(fc1)
        # Layer 4: Fully Connected. Input = 120. Output = 84.
        fc2 = tf.matmul(fc1, self.fc2_W) + self.fc2_b
        fc2 = tf.nn.relu(fc2)
    
        return tf.matmul(fc2, self.fc3_W) + self.fc3_b

