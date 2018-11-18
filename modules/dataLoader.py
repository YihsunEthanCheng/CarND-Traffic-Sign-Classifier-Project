#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 10 13:20:29 2018

@author: Ethan Cheng
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt
import skimage.transform as skt
from skimage import color, exposure, transform

class dataLoader(object):
    
    def __init__(self, params, path = 'data'):
        try:
            train = pickle.load(open(path + '/train.p','rb'))
            test = pickle.load(open(path + '/test.p','rb'))
            valid = pickle.load(open(path + '/valid.p','rb'))
        except:
            raise ValueError('Data loading errors..')
        
        # update params with data spec
        self.BATCH_SIZE = params['BATCH_SIZE'] 
        self.nCls = len(np.unique(train['labels']))
        self.imsize = train['features'].shape[1:]
        params.update(self.__dict__)
       
        # data stats
        self.y_train, self.y_valid, self.y_test = train['labels'], valid['labels'], test['labels']
        self.labs = np.unique(self.y_valid)
        self.nPerCls = [ np.sum(self.y_train == lab) for lab in self.labs]
        self.maxN =  np.max(self.nPerCls)

        # process image
        self.x_train, self.mu_train, self.sigma_train = self.processImage(train['features'])
        self.x_valid, _ , _ = self.processImage(valid['features'], self.mu_train, self.sigma_train)
        self.x_test,  _ , _ = self.processImage(test['features'],  self.mu_train, self.sigma_train) 
        
        # create lookup table for random sampling for training batch
        augIdx = []
        for lab in self.labs:
            idx_i = np.where(self.y_train == lab)[0] 
            np.random.shuffle(idx_i)
            idx_j = idx_i.tolist() * (self.maxN//len(idx_i)+1)
            augIdx += idx_j[:self.maxN]
        self.augIdx = np.array(augIdx)
        np.random.shuffle(self.augIdx)
        self.augiNext = 0 # circular starting point for the next batch 
        self.showDataStats()
              
    def nextBatch(self):
        ii = self.augIdx[np.arange(self.augiNext, self.augiNext + self.BATCH_SIZE) % len(self.augIdx)]
        # move pointer 
        self.augiNext =  (self.augiNext + self.BATCH_SIZE) % len(self.augIdx)
        if self.augiNext < self.BATCH_SIZE:
            np.random.shuffle(self.augIdx)
        return self.x_train[ii], self.y_train[ii]
        
    def showDataStats(self):
        print("Number of training examples =", len(self.y_train))
        print("Number of testing examples =", len(self.y_test))
        print("Image data shape =", self.imsize)
        print("Number of classes =", self.nCls )
        print("num of Training patterns", self.nPerCls)
        
    def viewSample(self, nPerCls = 20):
        mosaic = []
        idx = np.vstack([np.where(self.y_train == lab_i)[0][:nPerCls] for lab_i in self.labs])
        mosaic = np.vstack([np.hstack([ self.x_train[i] for i in row]) for row in idx])
        if mosaic.dtype != np.uint8:
            L, H = np.min(mosaic), np.max(mosaic)
            mosaic = (mosaic - L) / (H - L )
        plt.imshow(mosaic)
        return mosaic
            
    def processImage(self, img, mu = None, sigma = None):
        isSingleImage = False
        if len(img.shape) == 3:
            img = [img]
            isSingleImage = True
        imo = []
        for ii, imi in enumerate(img):
            hsv = color.rgb2hsv(imi)
            hsv[:, :, 2] = exposure.equalize_hist(hsv[:, :, 2])
            imo += [color.hsv2rgb(hsv)]
        imo = np.array(imo)
        if mu is None or sigma is None:    
            mu = np.mean(imo)
            sigma = np.std(imo)
        imo = (imo-mu)/sigma
        if isSingleImage:
            imo = imo[0]
        return imo, mu, sigma        
        
        

        
        