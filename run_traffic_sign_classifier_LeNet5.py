#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 23:08:39 2018

@author: Ethan Cheng
"""
%load_ext autoreload
%autoreload 2
import matplotlib.pyplot as plt
from modules.dataLoader import TrafficSignData, scaleForShow, tileForShow, warpImageShow
from modules.LeNet import LeNet
#import tensorflow as tf

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
       'n': 64,      
       'strides': [1, 1, 1, 1],
       'pool': Pool
       }

# data augmentation parameters
augment = {
        'rotate': 15, # +/- degrees
        'translate': 3, # +/- pixels
        'scale': 0.1, # 1.0 +/- ratio
        'Gaussian_blur': 1.2, # Gaussian blur kernel
        'Gaussian_speckle' : 0.08, # +/- pepper and salt noise
        'warpFreq': 0.75 # warp the image at 50% of the time
        }

params = {
        # Arguments used for tf.truncated_normal, randomly defines variables 
        # for the weights and biases for each layer
        'mu' : 0,
        'sigma' :  0.1,
        'c1': C1, 
        'c2': C2,
        'nFC1' : 512,
        'nFC2' : 128,
        'lr' : 0.001,
        'BATCH_SIZE' : 256,
        'EPOCHS' : 200,
        'param_keep_prob': 0.5,
        'augment': augment
        } 
                                                                                                                                                                                                                                                                                    

#%def main():
data = TrafficSignData(params)

#%%
model = LeNet(params)
model.train(data)

#%%
model.eval(data.x_test, data.y_test)
#%% dumping for visualizing dataset
data.viewSample()
#dump training sample without perturbation
data.BATCH_SIZE = 16
img, _ = data.nextBatch()
plt.figure(1)
plt.clf()
imshow(scaleForShow(tileForShow(img)))

#%% show warped images for the same sample
plt.figure()
imgWarped = data.warpImage(img)
imshow(scaleForShow(tileForShow(imgWarped)))
#%% show steps of image warping for one single image
warpImageShow(img[9], augment)


#%%
#with tf.Session() as sess:
#if __name__ == '__main__':
#    main()


