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
       'n': 64,
       'strides': [1, 1, 1, 1],
       'pool': Pool
       }

C2 = {
       'ksize' : [5, 5],
       'n': 32,
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
        'nFC1' : 256,
        'nFC2' : 256,
        'lr' : 0.001,
        'BATCH_SIZE' : 256,
        'EPOCHS' : 500
        } 

#%%
from modules.LeNet import LeNet
import tensorflow as tf

#%%def main():
    
data = dataLoader()
params['nCls'] = data.nCls
params['imsize'] = data.imsize
X = tf.placeholder(tf.float32, (None,) + params['imsize'])
Y = tf.placeholder(tf.int32, (None))
model = LeNet(X, Y, params)
saver = tf.train.Saver()
ii = np.arange(len(data.y_train))
valid_accuracy = []
best_accuracy = 0
#%%

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for ep in range(params['EPOCHS']):
        np.random.shuffle(ii)
        for offset in range(0, len(data.x_train), params['BATCH_SIZE']):
            jj = ii[offset:offset + params['BATCH_SIZE']]
            sess.run(model.optimize, {X: data.x_train[jj], Y: data.y_train[jj]})
        valid_accuracy += [sess.run(model.accuracy, {X: data.x_valid, Y: data.y_valid})]
        print('Validation error {:6.2f}%'.format(100 * valid_accuracy[-1]))
        if valid_accuracy[-1] > best_accuracy:
            saver.save(sess, 'checkpoints/lenet5')
            best_accuracy = valid_accuracy[-1]
            print('best accuracy @ ep#{} = {}'.format(ep, best_accuracy))

#%%
with tf.Session() as sess:
     test_accuracy = sess.run(model.accuracy, {X: data.x_test, Y: data.y_test})
print('Test accuracy : {}'.format(test_accuracy))
#if __name__ == '__main__':
#    main()


