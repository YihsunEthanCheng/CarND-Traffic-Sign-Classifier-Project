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
    
    def __init__(self, path = 'data'):
        try:
            train = pickle.load(open(path + '/train.p','rb'))
            test = pickle.load(open(path + '/test.p','rb'))
            valid = pickle.load(open(path + '/valid.p','rb'))
        except:
            raise ValueError('Data loading errors..')
            
        self.x_train, self.y_train = train['features'], train['labels']
        self.x_valid, self.y_valid = valid['features'], valid['labels']
        self.x_test, self.y_test = test['features'], test['labels']
        self.nCls = len(np.unique(self.y_train))
        self.imsize = self.x_train.shape[1:]
        self.dumpDataStats()
        
    def dumpDataStats(self):
        print("Number of training examples =", len(self.y_train))
        print("Number of testing examples =", len(self.y_test))
        print("Image data shape =", self.imsize)
        print("Number of classes =", self.nCls )
        
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
            

        
        