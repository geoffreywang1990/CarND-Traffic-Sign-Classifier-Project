# **Traffic Sign Recognition** 

## Writeup
### YUJUN WANG
---

**Build a Traffic Sign Recognition

 Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/img1.jpg "Traffic Sign 1"
[image5]: ./examples/img2.jpg "Traffic Sign 2"
[image6]: ./examples/img3.jpg "Traffic Sign 3"
[image7]: ./examples/img4.jpg "Traffic Sign 4"
[image8]: ./examples/img5.jpg "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/geoffreywang1990/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the python to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799.
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is [32x32x3]
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![training data classes visualization][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because the color info not really matter in traffic sign, the shape is the most import infomation.

Here is an example of a traffic sign image before and after grayscaling.

![gray scale][image2]

As a last step, I normalized the image data because we want the mean of the input data centered to zero.

I decided to generate additional data because the data is inbalanced in the training dataset. It will overfitting to those classes with more training data if I don't balance the train data set. 

To add more data to the the data set, I used the following techniques: adding gaussian noise to existing training data to generate new training data.


The difference between the original data set and the augmented data set is the following :adding gaussian data to the original training data


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 30x30x12 	|
| TanH					|												|
| Max pooling	      	| 2x2 stride,  outputs 15x15x10 				|
| Convolution 4x4	    | 1x1 stride, valid padding, outputs 11x11x24	|
| TanH					|												|
| Max pooling	      	| 2x2 stride,  outputs 6x6x32 					|
| Flatten				|												|
| Fully connected		| outputs 160. 									|
| Drop out  			| rate 0.6. 									|
| Fully connected		| outputs 80. 									|
| Drop out  			| rate 0.6. 									|
| Fully connected		| outputs 60. 									|
| Drop out  			| rate 0.6. 									|
| Softmax				|												|
|						|												|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an BATCH_SIZE = 128, EPOCHS = 15 training. For initializing the weights, I used mean = 0 sigma =0.1 truncated normal distribuation. The drop off rate is 0.6 and learing rate is 0.001.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.995
* validation set accuracy of 0.952 
* test set accuracy of 0.931

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
   ** I have tried with 3 layers of conv2d and 3 layers of fully connected.
* What were some problems with the initial architecture?
    ** The result is bad. 
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
    ** I have taken away the third conv & max pooling layer and third fc layer. I thing it is because it's losing some important information at the third layer because of the pooling.
* Which parameters were tuned? How were they adjusted and why?
    ** the depth of conv layer, num_output of fully connect layer, and drop out rate have been tuned. 

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] 
![alt text][image5] 
![alt text][image6] 
![alt text][image7] 
![alt text][image8]


#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| Pedestrians  			| General caution 								|
| Speed limit (50km/h)	| Speed limit (50km/h)							|
| Yield					| Yield							 				|
| Speed limit (70km/h)	| Speed limit (70km/h)							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of 93.1%

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is vert sure that this is a stop sign (probability of 1.0), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0000000e+00			| Stop sign   									| 
| 2.0723953e-10			| Keep right 									|
| 4.2229813e-13			| Yield											|
| 7.4204138e-14			| Speed limit (60km/h)			 				|
| 6.9830365e-14			| No entry 										|

For the second image  the model is vert sure that this is a General caution (probability of 1.0), but the image does not contain a General caution, it is a Pedestrians. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0000000e+00			| General caution								| 
| 1.2706668e-08			| Traffic signals								|
| 1.3102033e-10			| Pedestrians									|
| 2.2210317e-11			| Right-of-way at the next intersection			|
| 7.9318496e-12			| Wild animals crossing							|

For the third image, the model is vert sure that this is a Speed limit (50km/h) (probability of 9.9999988e-01), and the image does contain a Speed limit (50km/h). The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 9.9999988e-01			| Speed limit (50km/h)							| 
| 1.3839883e-07			| Speed limit (30km/h)							|
| 1.4897499e-09			| Speed limit (60km/h)							|
| 1.0389795e-10			| Speed limit (80km/h)			 				|
| 5.6104491e-14			| Wild animals crossing							|

For the fourth image, the model is vert sure that this is a Yield (probability of 1.0), and the image does contain a Yield. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0000000e+00			| Yield											| 
| 9.3009490e-14			| No passing									|
| 8.4794467e-16			| Priority road									|
| 1.7289782e-16			| Keep right					 				|
| 3.2063873e-17			| No vehicles									|

For the fifth image, the model is vert sure that this is a Speed limit (70km/h) (probability of 1.0), and the image does contain a Speed limit (70km/h). The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0000000e+00			| Speed limit (70km/h)							| 
| 9.3009490e-14			| Speed limit (30km/h)							|
| 8.4794467e-16			| General caution								|
| 1.7289782e-16			| Speed limit (120km/h)					 		|
| 3.2063873e-17			| Speed limit (20km/h)							|



### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


