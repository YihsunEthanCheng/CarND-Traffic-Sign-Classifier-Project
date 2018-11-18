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
        
        # rename dataset
        self.x_train, self.y_train = train['features'], train['labels']
        self.x_valid, self.y_valid = valid['features'], valid['labels']
        self.x_test,  self.y_test  = test['features'],  test['labels']
        self.labs = np.unique(self.y_valid)
        self.nPerCls = [ np.sum(self.y_train == lab) for lab in self.labs]
        self.maxN =  np.max(self.nPerCls)
        
        # create lookup table for random sampling for batch training
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
        canvas = []
        labels = np.unique(self.y_train)
        idx = np.vstack([np.where(self.y_train == lab_i)[0][:nPerCls] for lab_i in labels])
        canvas = np.vstack([np.hstack([ self.x_train[i] for i in row]) for row in idx])
        return canvas
            
 #%%           
    def enhance_contrast(img, tail = 0.005):    
        img.astype(np.float)
        imgO=[]
        for ii in range(img.shape[-1]):
            ch_sort = np.sort(img[:,:,ii], axis = None)
            jth = int(len(ch_sort)*tail)
            L, H = ch_sort[jth], ch_sort[-jth]
            imgO += [255.*(img[:,:,ii] - L)/(H-L)]
        imgO = np.dstack(imgO)
        return 
            

        
        