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
from modules.LeNet import LeNet
import tensorflow as tf

Pool = {
       'ksize': [1, 2, 2, 1],
       'strides': [1, 2, 2, 1]        
        }
C1 = {
       'ksize' : [5, 5],
       'n': 32,
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
        'nFC1' : 128,
        'nFC2' : 64,
        'lr' : 0.001,
        'BATCH_SIZE' : 256,
        'EPOCHS' : 500
        } 


#%%def main():
  
data = dataLoader()
params['nCls'] = data.nCls
params['imsize'] = data.imsize
model = LeNet(params)
model.train(data)


#%%
#with tf.Session() as sess:
#if __name__ == '__main__':
#    main()


