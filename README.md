# **Traffic Sign Recognition** 

___

## Outlines ##

* How to run this project
* Dataset overview
* Image processing pipeline
* Data augmentation pipeline
* Neural network model
* Training 
* Testing on novel data
* Summary
___

[//]: # "Image References"

[image1]: ./examples/nTrainImagePerCls.png "Training data size"
[image2]: ./examples/00000.png "Traffic Sign 1"
[image3]: ./examples/00002.png "Traffic Sign 2"
[image4]: ./examples/00003.png "Traffic Sign 3"
[image5]: ./examples/00009.png "Traffic Sign 4"
[image6]: ./examples/00010.png "Traffic Sign 5"
[image7]: ./examples/data_overview.png "Size normalized"
[image8]: ./examples/data_overview_processed.png "processed"
[image9]: ./examples/augmentation_example_1.png "augmentation example1"
[image10]: ./examples/augmentation_example_2.png "augmentation example2"
[image11]: ./checkpoints/lenet5_1120165543_training_curve.png "training curve"
[image12]: ./checkpoints/lenet5_1120165543_Recognition_Accuracy_Per_Class.png "Accuracy Per Class"
[image13]: ./checkpoints/lenet5_1120165543_n_vs_recog_rate.png "N vs Accuracy"
[image14]: ./examples/100_1607_small.jpg "download_0"
[image15]: ./examples/Stop_sign_small.jpg "download_1"
[image16]: ./examples/Arterial_small1.jpg "download_2"
[image17]: ./examples/Radfahrer_Absteigen_small.jpg "download_3"
[image18]: ./examples/Do-Not-Enter_small.jpg "download_4"
[image19]: ./examples/speed_30.jpg "download_5"
[image20]: ./examples/no_passing.jpg "download_6"
[image21]: ./examples/Share-Path-1_small.jpg "download_7"
[image22]: ./examples/Bike-Path-Ends_small.jpg "download_8"
[image23]: ./examples/Radfahrer_Absteigen_small_featuremap.png "Feauturemap 1"
[image24]: ./examples/Stop_sign_small_featuremap.png "featuremap 2"

___
## How to run this project
The project can be executed by running the "run_traffic_sign_classifier_LeNet5.py" including a lengthy training. You may turn off the training and run the classifier with the line of code below commented out.

   Line # 68    model.train(data)
___
## Dataset overview

* Dataset source: [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset)

* Dataset statistics

  |     Data Specification      | Training set |
  | :-------------------------: | :----------: |
  |         Image type          |     RGB      |
  |         Image size          |    Varies    |
  |       Number of Class       |      43      |
  |  Number of training images  |    34799     |
  |  Number of testing images   |    12630     |
  | Number of validation images |     4410     |
  | Names of Traffic signs      | [signnames.csv](./signnames.csv)|


* Samples of images from the data set

  * Major variations in data set are seen in size and lighting condition as shown below.

  * ROI are annotated in a csv file

  | Traffic sign images  | Traffic sign name |
  | :-------------------:|:-----------------:|
  | ![Alt text][image2]  | Speed limit (100km/h)|
  | ![Alt text][image3]  |No passing        |
  | ![Alt text][image4]  | Ahead only       |
  | ![Alt text][image5]  | Stop             |
  | ![Alt text][image6]  | Children crossing|

* Uneven population
  * Number of training samples are found to be uneven in the following plot. Without equalizing the number of training samples, the neural networks would be trained to weight on populated classes. To make all classes equal weighted during training, each class is made to appear in the same frequency in each training epoch. This normalization is done through a "sampling pool" (lookup table) of image indices, in which some images are repeated multiple times to appear in the random drawings.
  * Validation and testing sets are kept as-is, no equalization in testing/validation frequency per class.

    ![example #1][image1]

---
## Image Processing Pipeline

This image processing module resides in the "[TrafficSignData](./modules/dataLoader.py)" class. The module consists of three components to process each raw images.

* **Size normalization**

  * Images are scaled into 32x32 with the ROI scaled into the 28x28 center region.

* **Lighting equalization**

  * RGB to HLS color space conversion.
  * Histogram equalization on L-Channel.
  * HLS to RGB conversion

* **Intensity normalization**
  * Mean and standard deviation of the training set are computed and stored as parameters to normalize validation and testing sets. The normalization could be performed within a pool of data. But here I am assuming once the training is done, the module is queried one image a time and the parameters from training set will be our best prior information.    
  * This step normalizes all image to be ready for neural net training.

* **Results** 
  * As shown below, the dark images are now *normalized* to look similar to one another in lighting.

| After Size Normalization | After Lighting equalization |
| :-------------------:|:-----------------:|
| ![Alt text][image7] | ![Alt text][image8] |

___
## Data Augmentation
 Data augmentation is a stochastic step performed prior to each batch of training. Depending on the parameter (i.e.,75% of the time), each batch of training images are injected with the designed perturbations to create variations to the training set. As the examples shown below, the augmentation module does the following steps to manufacture different looks of the same image for the neural net.

* **Scaling (0.88~0.12)** 
   * The raw image is magnified/shrunk within a small range such that the majority of ROI stays within the default image size 32x32.
* **Padding (32x32 -> 38x38)**
   * The image is padded by 3 pixels each side to provide wiggle room for holding additional information from interpolation due to scaling and rotation.
   * The padding size is dictated by the range of translation, which moves the center of an image.
* **Rotation (-15~15 degree)**
   * The image is rotated within a small range to angles for the robustness to rotation variations.
* **Gaussian Blur (0~1.25)** 
   * A random size Gaussian kernel in the specified range is used to blur the image.
* **White Gaussian Noise (sigma = 0.08)**
   * A random white Gaussian noise is injected to the image in all color channels.
* **Translation (-3~3 pixels)**
   * A uniform random translation shifts the center of the image.
* **Cropping** 
   * The final step crops the image back to the default input size at 32x32. Due to the padding step above, the cropping is guaranteed to yield the same default neural net input size.


|Example #1 | Example#2 |
| :-------------------:|:-----------------:|
| ![Alt text][image9] | ![Alt text][image10] |
___
## Neural network model
* Why Lenet-5 
   * LeNet-5 is a well known model invented by Yann LeCun in the 90s for handwritten digit recognition. The traffic signs are simple graphics with mostly low freuency image contents in similar complexity to handwritten digits.  If equipped with  adqeuate numbers of kernels in the convolutional layers, I beleive LeNet-5 is more than capable of recognizing traffic signs with the benefit of fast training and simple architechture.
* LeNet-5 for this task is implemented with the following specifications. The implementation is encapsulated in the class [LeNet](./modules/Lenet.py).

  | Layers / Parameters   |      Descriptions	                          |
  |:---------------------:|:---------------------------------------------:|
  | Input           	     | 32x32x3 RGB image   			                 |
  | Convolution 5x5x64    | 1x1 stride, valid padding, outputs 28x28x64   |
  | RELU			           |						                             |
  | Max pooling	        | 2x2 stride, outputs 14x14x64	 		           |
  | Convolution 5x5x128   | 1x1 stride, valid padding, outputs 10x10x64   |
  | RELU			           |						                             |
  | Max pooling	        | 2x2 stride, outputs 5x5x128	 		           |
  | Dropout		           | Keep rate = 0.5				                    |
  | Fully connected	     | 512 						                          |
  | Dropout		           | Keep rate = 0.5                   				  |
  | Fully connected	     | 256						                          |
  | Dropout		           | Keep rate = 0.5				                    |
  | Softmax		           |        					                          |
  | Optimizer             | Adaptive moment estimation 			           |
  | Learning rate      	  | 0.001						                       |

___
## Training 
* Optimizer
  * Adaptive moment estimation (Adam) is chose as the optimizer as it is a well known adaptive learning rate scheme. It uses a running average to estimate the "flatness" in the cost function while approaching a minimum, thus lowers the learning rate as a necessary adaptation to avoid hopping around a minimum.
  * As running average takes multiple iterations to eatimate, the learning rate is often an under estaimate than over estimate. While it may be slow to adjust the learning rate but does it in a very steady fashion. 
  * The learning rate at 0.0001 is a well documented selection for LeNet-5 model using Adam optimizer.
* Dropouts
  * Dropouts are purposely installed prior to each fully connected layer. Dropout is a well known regularization technique to avoid over training. It can be deployed at any layer but is known to be especially effective for fully connected layers. The keep rate at 0.5 purposely disconnected half of the connections during and forces the desired memory recall to be distributed from multiple paths to achieve robustness.
* Number of kenerls in convolutional layers
  * 64 kernels in the first convolution layer is selected to be near 2x of the number of classes (43) while 128 kernels in the second convolutional layer is purposely selected as 2x of the number of kenerls in the previous conolutional layers. The transformation of input features from shallow to deep channels is known a key to successful feature extraction in Convolutional Neural Network.
* Batch size 
   * 512 images are selected as the batch size, which allows us to validate the state of training in the middle of an epoch. 
   * The selection of batch size is often meant for *batch normalization*, a technique to scale the intermediate features and enforce a layer to focus on the dense section of the data distribution, thus, achieves faster training convergence. Unfortonetely, batch normalization seems to have little or no effect in this task, due to well input normalization as not-so-deep architecture in this model.
* Number of epochs
  * Length of training is a guess work as long as the validation accurcay shows ample signs of fatigue in making new highs before the end of training. 
  * Training progression is monitored by validation accuracy as show below. In the old time before regularaization is introduced, overtraining is detected by the divergence between the validation and training errors.  As dropouts are heavily deployed in all fully connected layers in this model, overtraining is unlikely to happen. Thus, this curve is only used to tell if we have under train the model with insufficient epochs.  The plot confirms a well trained model.  
  
    ![Alt text][image11]
  
* Weight storage and restoration
   * The weights are saved in the "checkpoints" folder every time the validation accuracy makes a new high. This ensures the peak state of the model to be preserved for memory recalls once the training is done.  

* Recognition accuracy and model performance

  |  Data Set | Overall Weighted Accuracy |
  |:--------------:|:-----------------:|
  | Training set   | 0.9990804328867483 |
  | Validation set | 0.9886621236801147 |
  | Testing set    | 0.9725257158279419 |

* Recognition accuracy per class
  * The closer look at recognition accuracy on each class reveals uneven performance among them as shown below.

    ![Alt text][image12]

  * A quick check on correlation with data population shows that the poor accuracy are likely related to the number of variations in the training set, which is correlated to the sample size.
   
    ![Alt text][image13]

  * According to the correlation plot above, classes with low recognition rate are associated with lowest number of training data while some small classes are showing high recognition accuracy. This suggests that the testing set have some novelty that our augmentation scheme is deficient to counter with.
___
## Testing on novel data

 * To Test the training, German traffic sign images are found on [German Bicycle Laws website](http://bicyclegermany.com/german_bicycle_laws.html).

 * Novelty observed in the downloaded testing images
   * Perspective transformation to the left (test case #2).
   * Connected and confuse background (test case #3, and #5) 
   * Additional descriptive box in the lower half of image (test case #4).
   * Watermark (test cse #6).
   * Uneven lighting in once corner (test case #7)

 * Below are recognition results for the 7 signs belonged to the list of training data with all but one getting recognized as top-1 candidate. The only miss (general caution) is recognized within top-2 candidate.

  | Known Signs|Top 5 Predictions |probability	|
  |:----------:|-------------|:--------------:|
  |![alt text][image14]| Right-of-way at the next intersection | 0.9826 |
  ||Beware of ice/snow | 0.0173 |
  || Children crossing | 0.0000 |
  || Road work | 0.0000 |
  || Dangerous curve to the right | 0.0000 |
  |![alt text][image15]| Stop | 0.9999 |
  || Bicycles crossing | 0.0001 |
  || Speed limit (30km/h) | 0.0000 |
  || Road work | 0.0000 |
  || Speed limit (60km/h) | 0.0000 |
  |![alt text][image16]| Priority road | 1.0000 |
  || End of no passing by vehicles over 3.5 metric tons | 0.0000 |
  || No passing for vehicles over 3.5 metric tons | 0.0000 |
  || General caution | 0.0000 |
  || No entry | 0.0000 |
  |![alt text][image17]| No passing | 0.5149 |
  || General caution | 0.1520 |
  || Bicycles crossing | 0.1268 |
  || Children crossing | 0.0910 |
  || Dangerous curve to the right | 0.0449 |
  |![alt text][image18]| No entry | 1.0000 |
  || Stop | 0.0000 |
  || End of no passing by vehicles over 3.5 metric tons | 0.0000 |
  || No passing for vehicles over 3.5 metric tons | 0.0000 |
  || General caution | 0.0000 |
  |![alt text][image19]| Speed limit (30km/h) | 1.0000 |
  || Speed limit (100km/h) | 0.0000 |
  || Speed limit (50km/h) | 0.0000 |
  || Speed limit (80km/h) | 0.0000 |
  || Speed limit (20km/h) | 0.0000 |
  |![alt text][image20]| No passing | 1.0000 |
  || No passing for vehicles over 3.5 metric tons | 0.0000 |
  || Vehicles over 3.5 metric tons prohibited | 0.0000 |
  || No vehicles | 0.0000 |
  || Speed limit (60km/h) | 0.0000 |

 * Below are 2 signs that were not seen in the data sets for training. The top predictions below are mostly in the same category with round shape and blue color.

  | Unknown Signs | Top 5 Predictions |  probability	|
  |:-----------:|---------------|---------------------|
  |![alt text][image21]| Dangerous curve to the right | 0.8542 |
  || Go straight or left | 0.1315 |
  || Roundabout mandatory | 0.0086 |
  || Ahead only | 0.0027 |
  || Right-of-way at the next intersection | 0.0010 |
  |![alt text][image22]| Turn left ahead | 0.6002 |
  || Ahead only | 0.2458 |
  || Turn right ahead | 0.0622 |
  || Roundabout mandatory | 0.0454 |
  || Go straight or right | 0.0182 |

 * Feature map Visualization
  * 16 of the 64 kernel outputs by the first convolutional layer are dump below for two testing images.
 
  ![alt text][image23]
  ![alt text][image24]


___
## Takeaways on performance
 * The results show that traffic signs are in similar complexity as handwritten words that LeNet-5 architecture is adqeuate to give a decent performance.
 * More perturation could be used to improve the already high testing accuracy. The most notable one is projective transformation. 
 * The featuremaps at the first convolutional layer above shows simialr images suggesting that we may have allocated more than enough resouces to the layer. The dropout distributes the weights well into every kernel also contribute to the similar look in the featuremap.
 * The error in the 4th image is commited due to the extention of the bottom part, which is cropped out in the given data set. This image comes from a bicycle law website focusing on biker's information. Thus, its novelty is too much for our model to overcome.  Should we have complete look of the same lower part in the training set, 100% accuracy on novel data at this image quality should be very achievable.
 * 100% reacognition rate in the top-2 candidates and close output on the two unseen traffic signs suggest the model works well for this task and can be esily extended for more traffic signs.




