# **Traffic Sign Recognition** 

___

## Outlines ##

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

  * Major variations in data set are seen in size and lighting condition as shown below

  * ROI are annotated in a csv file

  | Traffic sign images  | Traffic sign name |
  | :-------------------:|:-----------------:|
  | ![Alt text][image2]  | Speed limit (100km/h)|
  | ![Alt text][image3]  |No passing        |
  | ![Alt text][image4]  | Ahead only       |
  | ![Alt text][image5]  | Stop             |
  | ![Alt text][image6]  | Children crossing|

* Uneven population from each class
  * To avoid training favoring populated classes, a list of image index are created as a proxy to equalize the number of training for each class (with random repetitions in less populated class).
  * Validation and testing sets are kept as-is, no equalization in testing/validation frequency per class

    ![example #1][image1]

---
## Image Processing Pipeline

This image processing module resides in the class "[TrafficSignData](./modules/dataLoader.py)" located in the modules folder. The module consists of three components to process each raw images.

* **Size normalization**

  * Images are scaled into 32x32 with the ROI scaled into the 28x28 center region.

* **Lighting equalization**

  * RGB to HLS scolor space conversion
  * L-Channel equalization
  * HLS to RGB conversion

* **Intensity noramalization**
  * Mean and standard deviation of the training set are computed and stored as the parameters.
  * All processed training/testing/validation images are normalized to be Normal distribution w.r.t. the training set. 
  * This step normalizes all image to be ready for neural net training.

* **Results** 
  * As shown below, the dark images are now *normalized* to look simiar to one another.

| After Size Normalization | After Lighting equalization |
| :-------------------:|:-----------------:|
| ![Alt text][image7] | ![Alt text][image8] |

___
## Data Augmentation
 Data augmentation is a stochastic step performed prior to each batch of training. Depending on the parameter (75% of the time), each batch of training images are injected with designed perturbation to create synthesized variations to the training set which promotes robustness. As the examples shown below, the augmentation module does the following steps to manufacture different looks of the same sign for the neural net.

* **Scaling (0.88~0.12)** 
   * The raw image is magnified/shrink within a small range such that the majority of ROI wills tay within the default image size 32x32.
* **Padding (32x32 -> 38x38)**
   * The image is padded by 3 pixels each side to provide wiggle room for expanded information due to scaling and rotation.
   * The padding size is selected by the range of translation, which moves the center of an image.
* **Rotation (-15~15 degree)**
   * The image is rotated within a small range to angles for the robustness to rotation variations.
* **Gaussian Blur (0~1.25)** 
   * A random size Gaussian kernel in the specified range is used to blur the image.
* **White Gaussian Noise (sigma = 0.08)**
   * A random white Gaussian noise is injected to the image in all color channels and resulting in random color speckles dropped on the images.
* **Translation (-3~3 pixels)**
   * A uniform random translation shifts the center of the image for the next stage.
* **Cropping** 
   * The final step crops the image back to the default input size at 32x32. Due to the padding, the cropping will be guranteed to match the default neural net input size.


|Example #1 | Example#2 |
| :-------------------:|:-----------------:|
| ![Alt text][image9] | ![Alt text][image10] |
___
## Neural network model
* LeNet-5 is selected for this task with the following specifications. The implementation is encapsulated in the class [LeNet](./modules/Lenet.py).

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
* Training progressed is monitored by validation accuracy as show below. The weights are saved into and overwrote the same checkpoint filename everytime the validation makes a new high. According to plot, the net is very well trainined at around 300 epochs.

    ![Alt text][image11]

* Recognition accuracy
   * The high training accuracy across all class suggests that the current design have been fully trained. Not much to gain should training continue.

  |  Data Set | Overall Weigthed Accuracy |
  |:--------------:|:-----------------:|
  | Training set   | 0.9990804328867483 |
  | Validation set | 0.9886621236801147 |
  | Testing set    | 0.9725257158279419 |

* Recognition accuracy per class
  * The closer look at recognition accuracy on each class reveals uneven performance among them as shown below.

    ![Alt text][image12]

  * A quick check on plot below shows that the poor accuracy are likely related to poor repreesentation of the training set in those classes.
   
    ![Alt text][image13]

  * Those classes performed poorly are for sure having fewer training data. But those mall classes are not necessarily perform poorly. This suggests the augmentation scheme may have missing some perturbation models to cover the different data variation in the test set.
___
## Testing on novel data

  * Gernman traffic sign images are found on [German Bicycle Laws website](http://bicyclegermany.com/german_bicycle_laws.html) from search engines.

  * Below are recognition results for the 7 signs belonged to the list of training data with all of them getting recognized correctly.

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

 * Below are 2 signs that were not seen in the data sets for training. The top predictions below are mostly in the same category of signs with round shape and blue color.

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

___
## Summary




