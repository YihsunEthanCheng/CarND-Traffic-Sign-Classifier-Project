#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 10 13:20:29 2018

@author: Ethan Cheng
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt
from skimage import color, exposure, transform, util, filters
import pandas as pd
from matplotlib import image as mpimg


noise = lambda span: 2*(np.random.rand()-0.5)*span
    
class TrafficSignData(object):
    
    def __init__(self, params, path = 'data'):
        try:
            train = pickle.load(open(path + '/train.p','rb'))
            test = pickle.load(open(path + '/test.p','rb'))
            valid = pickle.load(open(path + '/valid.p','rb'))
        except:
            raise ValueError('Data loading errors..')
        
        # export params 
        self.BATCH_SIZE = params['BATCH_SIZE'] 
        self.nCls = len(np.unique(train['labels']))
        self.imsize = train['features'].shape[1:]
        params.update(self.__dict__)
        
        # import params
        self.__dict__.update(params['augment'])
        
        # data stats
        self.y_train, self.y_valid, self.y_test = train['labels'], valid['labels'], test['labels']
        self.labs = np.unique(self.y_valid)
        self.nPerCls = [ np.sum(self.y_train == lab) for lab in self.labs]
        self.maxN =  np.max(self.nPerCls)

        # process image
        self.x_train, self.mu_train, self.sigma_train = self.processImage(train['features'])
        try:
            self.x_valid = pickle.load( open('data/x_valid.pickle','rb'))
        except:
            self.x_valid, _ , _ = self.processImage(valid['features'], self.mu_train, self.sigma_train)
        try:
            self.x_test = pickle.load( open('data/x_test.pickle','rb'))           
        except:
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
        self.signNames = np.array(pd.read_csv('signnames.csv').SignName)
      
        
    def nextBatch(self):
        ii = self.augIdx[np.arange(self.augiNext, self.augiNext + self.BATCH_SIZE) % len(self.augIdx)]
        # move pointer 
        self.augiNext =  (self.augiNext + self.BATCH_SIZE) % len(self.augIdx)
        if self.augiNext < self.BATCH_SIZE:
            np.random.shuffle(self.augIdx)
        if np.random.rand() < self.warpFreq: 
            return self.warpImage(self.x_train[ii]), self.y_train[ii]
        else:
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
        plt.imshow(scaleForShow(mosaic))
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
        print('image is processed')    
        return imo, mu, sigma        


    def warpImage(self, img):
        for ii in range(len(img)):
            # add scaling
            im1 = transform.rescale(img[ii], 1+noise(self.scale), 
                multichannel = True, anti_aliasing = True, mode = 'reflect')
            # padding to get smae size
            pw = (np.array(self.imsize[:2]) + self.translate*2 - np.array(im1.shape[:2])+1)//2
            im1 = util.pad(im1,(tuple([pw[0]]*2), tuple([pw[1]]*2), (0,0)), mode = 'constant', constant_values = 0)
            # add rotation
            im1 = transform.rotate(im1,noise(self.rotate))
            # add bluring
            im1 = filters.gaussian(im1, np.random.rand()*self.Gaussian_blur,  multichannel = True)
            # add gaussian noise
            im1 = im1 + np.random.randn(im1.shape[0],im1.shape[1], im1.shape[2]) * self.Gaussian_speckle
            # cropping
            LT = np.int_([noise(self.translate), noise(self.translate)]) + self.translate
            img[ii] = im1[LT[0]:(LT[0]+self.imsize[0]), LT[1]:(LT[1]+self.imsize[1]), :]
        return img    
    
    #  load the online testing set that is not used in the training 
    
    def mapIndex2TrafficSignName(self, idx):
        return self.signNames[idx]
    
    
    def normalizeImage(self, im0, ROI):
        """
        Normalize a novel image to be the same size of training set
        """
        scale = float(28.0/np.max(np.diff(ROI,axis = 0)))
        im1 = transform.rescale(im0,scale, multichannel = True, anti_aliasing= True, mode= 'reflect')
        center = (np.mean(ROI, axis = 0)*scale).round().astype(int)
        LT = np.max([center - 32//2,[0,0]], axis = 0)
        im1 = im1[LT[0]:min(LT[0]+32,im1.shape[0]), LT[1]:min(LT[1]+32,im1.shape[1]), :]
        im1pad = np.zeros((32,32,3), dtype = im1.dtype) + np.median(im1)
        im1pad[:im1.shape[0],:im1.shape[1]] = im1
        return im1pad
    
    
    def loadUnlabeledImage(self, sel):
        imStack = []
        df = pd.read_csv('data/GTSRB_Online-Test-Images/GT-online_test.test.csv', delimiter = ';')
        for i, row in df.iterrows():
            if i in sel:
                im0 = mpimg.imread('data/GTSRB_Online-Test-Images/{}'.format(row.Filename))
                ROI = np.float_(row[['Roi.Y1','Roi.X1','Roi.Y2','Roi.X2']]).reshape(2,2)
                im1 = self.normalizeImage(im0,ROI)
                imStack += [ im1 ]
        return imStack

# =============================================================================
# Drawing utilities
# =============================================================================

def tileForShow(img):
    mosaic = []
    n = int(np.sqrt(len(img)))
    row = []
    for ii, im in enumerate(img):
        row += [im]
        if (ii+1)%n == 0:
            mosaic += [np.hstack(row)]
            row = []
    mosaic = np.vstack(mosaic)    
    return mosaic    

def scaleForShow(img):    
    L, H = np.min(img), np.max(img)
    img = (img - L) / (H - L )
    return img
        
def warpImageShow(im, augment):

    fig, axes = plt.subplots(4,2)
    ax = axes.T.flatten()
    # original
    i = 0
    ax[i].imshow(scaleForShow(im))
    ax[i].set_title('{}. Ogional'.format(i))
    i+=1
    # add scaling
    scale = 1.2 #1+noise(augment['scale'])
    im1 = transform.rescale(im, scale, 
            multichannel = True, anti_aliasing = True, mode = 'reflect')
    ax[i].imshow(scaleForShow(im1))
    ax[i].set_title('{}. Scaled {:3.2f}X'.format(i, scale))
    i+=1
    # padding to get same size
    pw = (np.array(im.shape[:2]) + augment['translate']*2 - np.array(im1.shape[:2])+1)//2
    im2 = util.pad(im1,(tuple([pw[0]]*2), tuple([pw[1]]*2), (0,0)), mode = 'constant', constant_values = 0)
    ax[i].imshow(scaleForShow(im2))
    ax[i].set_title('{}. Padded'.format(i))
    i+=1
    # add rotation
    deg = 8.5 #noise(augment['rotate'])
    im3 = transform.rotate(im2,deg)
    ax[i].imshow(scaleForShow(im3))       
    ax[i].set_title('{}. Roatated by {:3.2f} degree'.format(i, deg))
    i+=1
    # add bluring
    sigma = np.random.rand()*augment['Gaussian_blur']
    im4 = filters.gaussian(im3, sigma, multichannel = True)
    ax[i].imshow(scaleForShow(im4))       
    ax[i].set_title('{}. Gaussian Blured, sigma = {:3.2f}'.format(i, sigma))
    i+=1
    # add gaussian noise
    a,b,c = im4.shape
    im5 = im4 + np.random.randn(a,b,c) * augment['Gaussian_speckle']
    ax[i].imshow(scaleForShow(im5))       
    ax[i].set_title('{}. Gaussian \"Sprinkle\" Noise, sigma = {:3.2f}'.format(i, augment['Gaussian_speckle']))
    i+=1
    # cropping
    LT = np.int_([noise(augment['translate']), noise(augment['translate'])]) + augment['translate']
    im6 = im5[LT[0]:(LT[0]+im.shape[0]), LT[1]:(LT[1]+im.shape[1]), :]
    ax[i].imshow(scaleForShow(im6))       
    ax[i].set_title('{}. Translated (Cropped) by {} pixels'.format(i, LT - augment['translate']))   
    i+=1
    ax[i].axis('off')
    margin = 0.02
    plt.subplots_adjust(margin, margin*2, 1-margin, 1- margin*2, wspace = margin)
    fig.set_figheight(15.5)
    fig.set_figwidth(9)
    

