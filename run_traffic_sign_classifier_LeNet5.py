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
from skimage import transform
from matplotlib import image as mpimg

#import tensorflow as tf

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
       'n': 128,      
       'strides': [1, 1, 1, 1],
       'pool': Pool
       }

# data augmentation parameters
augment = {
        'rotate': 15, # +/- degrees
        'translate': 3, # +/- pixels
        'scale': 0.12, # 1.0 +/- ratio
        'Gaussian_blur': 1.25, # Gaussian blur kernel
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
        'nFC2' : 256,
        'lr' : 0.001,
        'BATCH_SIZE' : 512,
        'validate_every_n_batch': 8,
        'EPOCHS' : 800,
        'param_keep_prob': 0.5,
        'augment': augment
        } 

#%%
#%def main():

data = TrafficSignData(params)
model = LeNet(params)
model.train(data) # training the model, commented this line run the classifier with the trained weight
model.plotTrainingCurve(len(data.y_train))


#%% plot downloaded images for testing
Germ_signs = [ '100_1607_small.jpg', 'Stop_sign_small.jpg', 'Arterial_small1.jpg',
    'Radfahrer_Absteigen_small.jpg', 'Do-Not-Enter_small.jpg', 'speed_30.jpg', 
    'no_passing.jpg','Share-Path-1_small.jpg','Bike-Path-Ends_small.jpg' ]

imgs = []
for gs_i in Germ_signs:
    im0 = mpimg.imread('examples/' +gs_i)
    sc = 32.0/np.min(im0.shape[:2])
    im1 = transform.rescale(im0, sc, multichannel = True, anti_aliasing = True, mode = 'reflect')
    imgs += [im1[im1.shape[0]//2-16:im1.shape[0]//2+16, im1.shape[1]//2-16:im1.shape[1]//2+16, :]]
    

#%%
img_processed, _, _ = data.processImage(np.array(imgs))
plt.figure(1)
plt.imshow(scaleForShow(np.hstack(img_processed.tolist())))
yTop, pTop = model.predict(img_processed)
labels = data.mapIndex2TrafficSignName(np.array(yTop))

#%%
for labi, pi in zip(labels, pTop):
    for ii in range(len(labi)):
        print('| {} | {:5.4f} |'.format(labi[ii], pi[ii]))
    print('|-----------|---------------|')


#%% Viewing featuremaps
for jj in range(5):
    conv1, conv2 = model.getKernel(np.array([img_processed[jj]]))
    conv1 = scaleForShow(conv1)
    mosaic = []
    kk = 0
    conv1 = conv1[:,:,:16]
    for ri in range(int(np.sqrt(conv1.shape[-1]))):
        row =[]
        for cj in range(int(np.sqrt(conv1.shape[-1]))):
            row += [conv1[:,:,kk]]
        mosaic += [np.hstack(row)]
    mosaic = np.vstack(mosaic)
    fig, ax = plt.subplots(1,2)
    ax[0].imshow(imgs[jj])
    ax[1].imshow(mosaic, cmap = 'gray')
    fig.suptitle('Feature Map of ' + Germ_signs[jj])
    fig.savefig('examples/' + Germ_signs[jj][:-4] + '_featuremap')
    
#%% find training error per class
train_accu_cls, test_accu_cls, valid_accu_cls = [], [], []
for lab in range(data.nCls):
    ii = data.y_train == lab
    train_accu_cls += [model.eval(data.x_train[ii], data.y_train[ii])]
    ii = data.y_test == lab
    test_accu_cls += [model.eval(data.x_test[ii], data.y_test[ii]) ]
    ii = data.y_valid == lab
    valid_accu_cls += [ model.eval(data.x_valid[ii], data.y_valid[ii])]
train_accu = np.sum(np.float_(train_accu_cls)*np.float_(data.nPerCls))/np.sum(data.nPerCls)
valid_accu = model.eval(data.x_valid, data.y_valid)
test_accu =  model.eval(data.x_test, data.y_test)

#%% draw the class recognition rate
plt.figure(1)
plt.clf()
fig, ax1 = plt.subplots()
ax1.plot(np.arange(43),train_accu_cls,':.')
ax1.plot(np.arange(43),test_accu_cls, ':.')
ax1.plot(np.arange(43),valid_accu_cls,':.')
ax1.set_ylim([0.65,1.01])
ax1.set_ylabel('%')
ax1.set_xlabel('class ID')
plt.legend(['Train set', 'Test test', 'Valid set'])
fig.savefig(model.checkptFn + '_Recognition_Accuracy_Per_Class')

#%% plot correlation between training sample population and the testing accuracy
plt.figure(2)
plt.clf()
plt.plot(data.nPerCls, test_accu_cls, '*')
plt.title('# of training samples vs. Test Set Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('# of training samples')  
plt.savefig(model.checkptFn + '_n_vs_recog_rate')


#%% plot number of pattern per class
fig =figure(2)
plt.clf()
plt.bar(range(data.nCls), data.nPerCls)
plt.title('Number of Trining Images Per Class')
plt.xlabel('Class ID')
plt.ylabel('N')
fig.savefig('examples/nTrainImagePerCls')
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


