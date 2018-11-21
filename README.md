# **Traffic Sign Recognition** 

---

## Outlines ##

* Dataset overview
* Image processing pipeline
* Data augmentation pipeline
* Neural network model
* Training 
* Testing on novel data
* Summary


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
---

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
| Names of Traffic signs | [signnames.csv](./signnames.csv)|


* Samples of images from the data set

  * Major variations in data set are seen in size and lighting condition as shown below

  * ROI are annotated in a csv file

| Traffic sign images | Traffic sign name |
| :-------------------:|:-----------------:|
| ![Alt text][image2] | Speed limit (100km/h)|
| ![Alt text][image3] |No passing|
| ![Alt text][image4] | Ahead only|
| ![Alt text][image5] | Stop|
| ![Alt text][image6] | Children crossing|

* Uneven population from each class
  * To avoid training favoring populated classes, a list of image index are created as a proxy to equalize the number of training for each class (with random repetitions in less populated class).
  * Validation and testing sets are kept as-is, no equalization in testing/validation frequency per class

![example #1][image1]

---
## Image Processing Pipeline

This image processing module resides in the class "[TrafficSignData]([./modules/dataLoader.py)" located in the modules folder. The module consists of three components to process each raw images.

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
* LeNet-5 is selected for this task with the following specifications. The implementation is encapsulated in the class [LeNet](modules/Lenet.py).

| Layers / Parameters  	|     Descriptions	        		|
|:---------------------:|:---------------------------------------------:|
| Input        		| 32x32x3 RGB image   				|
| Convolution 5x5x64  	| 1x1 stride, valid padding, outputs 28x28x64 	|
| RELU			|						|
| Max pooling	      	| 2x2 stride, outputs 14x14x64	 		|
| Convolution 5x5x128  	| 1x1 stride, valid padding, outputs 10x10x64 	|
| RELU			|						|
| Max pooling	      	| 2x2 stride, outputs 5x5x128	 		|
| Dropout		| Keep rate = 0.5				|
| Fully connected	| 512 						|
| Dropout		| Keep rate = 0.5				|
| Fully connected	| 256						|
| Dropout		| Keep rate = 0.5				|
| Softmax		|        					|
| Optimizer             | Adaptive moment estimation 			|
| Learning rate 	| 0.001						|

___
## Training 


___
## Testing on novel data

___
## Summary








### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| Stop Sign      		| Stop sign   									|
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .60         			| Stop sign   									|
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


