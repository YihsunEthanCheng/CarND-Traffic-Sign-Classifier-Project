#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 23:42:05 2018

@author: Ethan Cheng
"""

import functools
import tensorflow as tf
from tensorflow.contrib.layers import flatten
import numpy as np

def doublewrap(function):
    """
    A decorator decorator, allowing to use the decorator to be used without
    parentheses if no arguments are provided. All arguments must be optional.
    """
    @functools.wraps(function)
    def decorator(*args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            return function(args[0])
        else:
            return lambda wrapee: function(wrapee, *args, **kwargs)
    return decorator


@doublewrap
def define_scope(function, scope=None, *args, **kwargs):
    """
    A decorator for functions that define TensorFlow operations. The wrapped
    function will only be executed once. Subsequent calls to it will directly
    return the result so that operations are added to the graph only once.
    The operations added by the function live within a tf.variable_scope(). If
    this decorator is used with arguments, they will be forwarded to the
    variable scope. TfThe scope name defaults to the name of the wrapped
    function.
    """
    attribute = '_cache_' + function.__name__
    name = scope or function.__name__
    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            with tf.variable_scope(name, *args, **kwargs):
                setattr(self, attribute, function(self))
        return getattr(self, attribute)
    return decorator


class LeNet(object):

    def __init__(self, params):
        self.__dict__ = params.copy()
        self.X = tf.placeholder(tf.float32, (None,) + self.imsize)
        self.Y = tf.placeholder(tf.int32, (None))
        self.prob_keep = tf.placeholder(tf.float32)
        self.one_hot_Y = tf.one_hot(self.Y, self.nCls)
        self.inference
        self.optimize
        self.accuracy
        self.saver = tf.train.Saver()
        #Dropout

    def train(self, data):
        ii = np.arange(len(data.y_train))
        self.valid_accuracy = []
        self.best_accuracy = 0
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for ep in range(self.EPOCHS):
                np.random.shuffle(ii)
                for offset in range(0, len(data.x_train), self.BATCH_SIZE):
                    jj = ii[offset:offset + self.BATCH_SIZE]
                    sess.run(self.optimize, {self.X: data.x_train[jj], self.Y: data.y_train[jj], self.prob_keep: self.keep_prob})
                    self.valid_accuracy += [sess.run(self.accuracy, {self.X: data.x_valid, self.Y: data.y_valid, self.prob_keep: 1.0})]
                    print('Validation error {:6.2f}%'.format(100 * self.valid_accuracy[-1]))
                    if self.valid_accuracy[-1] > self.best_accuracy:
                        self.saver.save(sess, 'checkpoints/lenet5_64-32-256-256')
                        self.best_accuracy = self.valid_accuracy[-1]
                        print('best accuracy @ ep#{} = {}'.format(ep, self.best_accuracy))
            self.saver.restore(sess, tf.train.latest_checkpoint('checkpoints'))        
            self.test_accuracy = sess.run(self.accuracy, {self.X: data.x_test, self.Y: data.y_test, self.prob_keep: 1.0})
            print('Test accuracy : {}'.format(self.test_accuracy))

    def eval(self, x, y):
        with tf.Session() as sess:
            self.saver.restore(sess, tf.train.latest_checkpoint('checkpoints'))
            accuracy = sess.run(self.accuracy, {self.X: x, self.Y: y, self.prob_keep: self.keep_prob})
            print("Accuracy = {:.3f}".format(accuracy))
        return accuracy


    @define_scope(initializer=tf.global_variables_initializer())
    def inference(self):
        shape = self.c1['ksize'] + [self.imsize[2], self.c1['n']]
        self.conv1_W = tf.Variable(tf.truncated_normal(shape = shape, mean = self.mu, stddev = self.sigma))
        self.conv1_b = tf.Variable(tf.zeros(self.c1['n']))
        conv1 = tf.nn.conv2d(self.X, self.conv1_W, strides = self.c1['strides'], padding='VALID') + self.conv1_b
        conv1 = tf.nn.relu(conv1)
        conv1 = tf.nn.max_pool(conv1, ksize = self.c1['pool']['ksize'], strides = self.c1['pool']['strides'], padding='VALID')
        # Conv1 ouput dimensions
        self.size_c1 = ((np.array(self.imsize[:2]) - (np.array(self.c1['ksize'])-1))/np.array(self.c1['strides'][1:3])).astype(np.int32)
        self.size_p1 = (self.size_c1/np.array(self.c1['pool']['strides'][1:3])).astype(np.int32)

        # Layer 2: Convolutional. Output = 10x10x16.
        shape =  self.c2['ksize'] + [self.c1['n'], self.c2['n']]
        self.conv2_W = tf.Variable(tf.truncated_normal(shape = shape, mean = self.mu, stddev = self.sigma))
        self.conv2_b = tf.Variable(tf.zeros(self.c2['n']))
        conv2 = tf.nn.conv2d(conv1, self.conv2_W, strides = self.c2['strides'], padding='VALID') + self.conv2_b
        conv2 = tf.nn.relu(conv2)
        conv2 = tf.nn.max_pool(conv2, ksize= self.c2['pool']['ksize'], strides = self.c1['pool']['strides'], padding='VALID')
         # conv2 output dimensions
        self.size_c2 = ((self.size_p1 - (np.array(self.c2['ksize'])-1))/np.array(self.c2['strides'][1:3])).astype(np.int32)
        self.size_p2 = ((self.size_c2/np.array(self.c2['pool']['strides'][1:3])).astype(int)).astype(np.int32)
         
        # Layer 3: Fully Connected. Input = 400. Output = 120.
        fc0 = flatten(conv2)
        shape = (np.prod(self.size_p2)*self.c2['n'], self.nFC1)
         # fully connected layers
        self.fc1_W = tf.Variable(tf.truncated_normal(shape = shape, mean = self.mu, stddev = self.sigma))
        self.fc1_b = tf.Variable(tf.zeros(self.nFC1))
        fc1 = tf.matmul(fc0, self.fc1_W) + self.fc1_b
        fc1 = tf.nn.relu(fc1)       
        fc1 = tf.nn.dropout(fc1, self.prob_keep)

        # Layer 4: Fully Connected. Input = 120. Output = 84.
        self.fc2_W  = tf.Variable(tf.truncated_normal(shape=(self.nFC1, self.nFC2), mean = self.mu, stddev = self.sigma))
        self.fc2_b  = tf.Variable(tf.zeros(self.nFC2))
        fc2 = tf.matmul(fc1, self.fc2_W) + self.fc2_b
        fc2 = tf.nn.relu(fc2)
        
        self.fc3_W = tf.Variable(tf.truncated_normal(shape=(self.nFC2, self.nCls), mean = self.mu, stddev = self.sigma))
        self.fc3_b = tf.Variable(tf.zeros(self.nCls))
   
        return tf.matmul(fc2, self.fc3_W) + self.fc3_b

    @define_scope
    def optimize(self):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels= self.one_hot_Y, logits= self.inference)
        loss_operation = tf.reduce_mean(cross_entropy)
        optimizer = tf.train.AdamOptimizer(learning_rate = self.lr)
        return optimizer.minimize(loss_operation)
        
# ================== alternative ===========================================================
#         logprob = tf.log(self.prediction + 1e-12)
#         cross_entropy = -tf.reduce_sum(self.label * logprob)
#         optimizer = tf.train.RMSPropOptimizer(0.03)
#         return optimizer.minimize(cross_entropy)
# 
# =============================================================================
    
    @define_scope
    def accuracy(self):
        correct_prediction = tf.equal(tf.argmax(self.inference, 1), tf.argmax(self.one_hot_Y, 1))
        return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#        mistakes = tf.not_equal(
#            tf.argmax(self.label, 1), tf.argmax(self.prediction, 1))
#        return tf.reduce_mean(tf.cast(mistakes, tf.float32))


#%%