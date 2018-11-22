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
from time import gmtime, strftime
import matplotlib.pyplot as plt
import pickle

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
    function will only be executed once. Subvalid_accuracysequent calls to it will directly
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
        self.is_training = tf.placeholder(tf.bool)
        self.prob_keep = tf.placeholder(tf.float32)
        self.one_hot_Y = tf.one_hot(self.Y, self.nCls)
        self.inference
        self.optimize
        self.accuracy
        self.saver = tf.train.Saver()
        self.valid_accuracy = []


    def train(self, data, resume = False):
        self.best_accuracy = 0
        self.checkptFn = 'checkpoints/lenet5_' + strftime("%m%d%H%M%S", gmtime())
        with tf.Session() as sess:
            if resume:
                try:    
                    self.saver.restore(sess, tf.train.latest_checkpoint('checkpoints'))
                except:
                    raise ValueError('checkpoint resume error..')
                try:
                    lines = open('checkpoints/checkpoint','r').readlines()
                    self.valid_accuracy = pickle.load(open('checkpoints/' + lines[-1].split('"')[-2] + '_trainCrv.pickle', 'rb'))
                except:
                    self.valid_accuracy = []            
            else:
                sess.run(tf.global_variables_initializer())
                self.valid_accuracy = []
    
            best_at = 0
            for ep in range(self.EPOCHS):
                for jj in range(len(data.x_train) // self.BATCH_SIZE):
                    x_batch, y_batch = data.nextBatch()
                    sess.run(self.optimize, {self.X: x_batch, self.Y: y_batch, 
                        self.prob_keep: self.param_keep_prob, self.is_training: True})
                        
                    self.valid_accuracy += [ 100.0*sess.run(self.accuracy, 
                        {self.X: data.x_valid, self.Y: data.y_valid, 
                         self.prob_keep: 1.0, self.is_training: False})]
                
                    # evaluate epoch validation error
                    if jj%self.validate_every_n_batch == 0:
                        if self.valid_accuracy[-1] >= self.best_accuracy:
                            self.saver.save(sess, self.checkptFn)
                            self.best_accuracy = self.valid_accuracy[-1]
                            best_at = ep
                        pickle.dump(self.valid_accuracy, open(self.checkptFn + '_trainCrv.pickle','wb'))
                        print('Ep#{}-{} validation Rate = {:6.2f}% best @ #{} = {:6.2f}%'.
                              format(ep,jj,self.valid_accuracy[-1],best_at,self.best_accuracy))
            self.saver.restore(sess, tf.train.latest_checkpoint('checkpoints'))        
            self.test_accuracy = sess.run(self.accuracy, {self.X: data.x_test, 
                self.Y: data.y_test, self.prob_keep: 1.0, self.is_training: False})
            print('Test accuracy : {}'.format(self.test_accuracy))
        pickle.dump(self.valid_accuracy, open(self.checkptFn + '_trainCrv.pickle','wb'))

    def eval(self, x, y):
        with tf.Session() as sess:
            self.saver.restore(sess, tf.train.latest_checkpoint('checkpoints'))
            accuracy = sess.run(self.accuracy, {self.X: x, self.Y: y, 
                self.prob_keep: 1.0, self.is_training: False})
            print("Accuracy = {}".format(accuracy))
        return accuracy
    
    def plotTrainingCurve(self, n_train_img, dump = False):
        if len(self.valid_accuracy) == 0:
            lines = open('checkpoints/checkpoint','r').readlines()
            self.checkptFn = 'checkpoints/' + lines[-1].split('"')[-2]
            self.valid_accuracy = pickle.load(open(self.checkptFn + '_trainCrv.pickle', 'rb'))
        fig = plt.figure()
        fig.clf()
        nbatch = n_train_img //self.BATCH_SIZE + 1 
        plt.plot(np.arange(len(self.valid_accuracy[::nbatch])),self.valid_accuracy[::nbatch])
        plt.grid()
        plt.title('Valiation during training')
        plt.xlabel('Epoch #')
        plt.ylabel('Accuracy (%)')
        if dump:
            fig.savefig(self.checkptFn + '_training_curve')

    @define_scope(initializer=tf.global_variables_initializer())
    def test(self, is_training = True, a = 100):
        print(is_training)
        return a
    
    @define_scope(initializer=tf.global_variables_initializer())
    def inference(self):
        shape = self.c1['ksize'] + [self.imsize[2], self.c1['n']]
        self.conv1_W = tf.Variable(tf.truncated_normal(shape = shape, mean = self.mu, stddev = self.sigma))
        self.conv1_b = tf.Variable(tf.zeros(self.c1['n']))
        self.conv1 = tf.nn.conv2d(self.X, self.conv1_W, strides = self.c1['strides'], padding='VALID') + self.conv1_b
        conv1 = tf.nn.relu(self.conv1)
#        conv1 = tf.layers.batch_normalization(conv1, training = self.is_training)
        conv1 = tf.nn.max_pool(conv1, ksize = self.c1['pool']['ksize'], strides = self.c1['pool']['strides'], padding='VALID')
        # Conv1 ouput dimensions
        self.size_c1 = ((np.array(self.imsize[:2]) - (np.array(self.c1['ksize'])-1))/np.array(self.c1['strides'][1:3])).astype(np.int32)
        self.size_p1 = (self.size_c1/np.array(self.c1['pool']['strides'][1:3])).astype(np.int32)

        # Layer 2: Convolutional. Output = 10x10x16.
        shape =  self.c2['ksize'] + [self.c1['n'], self.c2['n']]
        self.conv2_W = tf.Variable(tf.truncated_normal(shape = shape, mean = self.mu, stddev = self.sigma))
        self.conv2_b = tf.Variable(tf.zeros(self.c2['n']))
        self.conv2 = tf.nn.conv2d(conv1, self.conv2_W, strides = self.c2['strides'], padding='VALID') + self.conv2_b
        conv2 = tf.nn.relu(self.conv2)
#        conv2 = tf.layers.batch_normalization(conv2, training = self.is_training)
        conv2 = tf.nn.max_pool(conv2, ksize= self.c2['pool']['ksize'], strides = self.c1['pool']['strides'], padding='VALID')
         # conv2 output dimensions
        self.size_c2 = ((self.size_p1 - (np.array(self.c2['ksize'])-1))/np.array(self.c2['strides'][1:3])).astype(np.int32)
        self.size_p2 = ((self.size_c2/np.array(self.c2['pool']['strides'][1:3])).astype(int)).astype(np.int32)
         
        # Layer 3: Fully Connected
        fc0 = flatten(conv2)
        shape = (np.prod(self.size_p2)*self.c2['n'], self.nFC1)
        fc0 = tf.nn.dropout(fc0, self.prob_keep)

         # fully connected layers
        self.fc1_W = tf.Variable(tf.truncated_normal(shape = shape, mean = self.mu, stddev = self.sigma))
        self.fc1_b = tf.Variable(tf.zeros(self.nFC1))
        fc1 = tf.matmul(fc0, self.fc1_W) + self.fc1_b
        fc1 = tf.nn.relu(fc1)       
        fc1 = tf.nn.dropout(fc1, self.prob_keep)

        # Layer 4: Fully Connected
        self.fc2_W  = tf.Variable(tf.truncated_normal(shape=(self.nFC1, self.nFC2), mean = self.mu, stddev = self.sigma))
        self.fc2_b  = tf.Variable(tf.zeros(self.nFC2))
        fc2 = tf.matmul(fc1, self.fc2_W) + self.fc2_b
        fc2 = tf.nn.relu(fc2)
        fc2 = tf.nn.dropout(fc2, self.prob_keep)

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


    def predict(self, x_unlabeled, nTop = 5):
        """
        predict unlabeled images
        """           
        with tf.Session() as sess:
            self.saver.restore(sess, tf.train.latest_checkpoint('checkpoints'))        
            y_softmax = tf.nn.softmax(self.inference)
            p = sess.run(y_softmax, {self.X: x_unlabeled, 
                    self.prob_keep: 1.0, self.is_training: False})
        yTop = []
        pTop = []
        for r in p:
            ii = np.argsort(r)[::-1]
            pTop += [ ii[:nTop] ]
            yTop += [ r[ii[:nTop]]]
            
        return pTop, yTop      
    
    def getKernel(self, x_):
        with tf.Session() as sess:
            self.saver.restore(sess, tf.train.latest_checkpoint('checkpoints'))        
            conv1 = sess.run(self.conv1, {self.X: x_, self.prob_keep: 1.0, self.is_training: False})
            conv2 = sess.run(self.conv2, {self.X: x_, self.prob_keep: 1.0, self.is_training: False})
        return np.squeeze(conv1), np.squeeze(conv2)
    
    
    
#%%