# **Traffic Sign Recognition** 

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"

---
### Writeup / README
This project is going to use a set of data to train neural network to acheive image recognization. The images used to train the neural network are traffic signs, and Tensor flow is used for the actural implementation.

### Data Set Summary & Exploration

#### 1. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799 samples
* The size of the validation set is 4410 samples
* The size of test set is 12630 samples
* The shape of a traffic sign image is 32,32
* The number of unique classes/labels in the data set is 43

Image Shape: (32, 32, 3)

Training Set:   34799 samples
Validation Set: 4410 samples
Test Set:       12630 samples

#### 2. Include an exploratory visualization of the dataset.

In total there are 43 signals in the training data sets.
Following shows the corresponding label to each signals which will be used later for varification and test. 

{'39': 'Keep left', '5': 'Speed limit (80km/h)', '3': 'Speed limit (60km/h)', '31': 'Wild animals crossing', '21': 'Double curve', '10': 'No passing for vehicles over 3.5 metric tons', '40': 'Roundabout mandatory', '20': 'Dangerous curve to the right', '25': 'Road work', '16': 'Vehicles over 3.5 metric tons prohibited', '37': 'Go straight or left', '14': 'Stop', '11': 'Right-of-way at the next intersection', '34': 'Turn left ahead', '26': 'Traffic signals', '33': 'Turn right ahead', '15': 'No vehicles', '36': 'Go straight or right', '9': 'No passing', '2': 'Speed limit (50km/h)', '29': 'Bicycles crossing', '1': 'Speed limit (30km/h)', '19': 'Dangerous curve to the left', '8': 'Speed limit (120km/h)', '38': 'Keep right', '7': 'Speed limit (100km/h)', '12': 'Priority road', '41': 'End of no passing', '24': 'Road narrows on the right', '35': 'Ahead only', '27': 'Pedestrians', '4': 'Speed limit (70km/h)', '42': 'End of no passing by vehicles over 3.5 metric tons', '32': 'End of all speed and passing limits', '6': 'End of speed limit (80km/h)', '22': 'Bumpy road', '30': 'Beware of ice/snow', '23': 'Slippery road', '17': 'No entry', '18': 'General caution', '28': 'Children crossing', '0': 'Speed limit (20km/h)', '13': 'Yield'}

Here is an exploratory visualization of the data set. It is a bar chart shows the traffic signals in the training data and number of data for each traffic signals in training data set.

![alt text](https://github.com/boweizhou/Traffic_Sign_Classifier/blob/master/images1/index_of_training_pics.png)


Here is all signals with label from the training set.
![alt text](https://github.com/boweizhou/Traffic_Sign_Classifier/blob/master/images1/Trafic_signal_with_lables.png)

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I convert the image from a 3 color RBG to a gray scale image which only include one color. Gray image seems to provide luminance which is by far more important in distinguishing visual features 

Here is an example of a traffic sign image before and after grayscaling.


![alt text](https://github.com/boweizhou/Traffic_Sign_Classifier/blob/master/images1/original_image.png)
![alt text](https://github.com/boweizhou/Traffic_Sign_Classifier/blob/master/images1/gray_image.png)

As next step, I normalized the image data and gain smaller numbers. 


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Gray image   							| 
| Convolution 5x5     	| 1x1 stride, VALID padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 5x5	    | 1x1 stride, VALID padding, outputs 10x10x16   									|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| Flatten  shape = 400
| Convolution 5x5	    | 1x1 stride, VALID padding, outputs 120  									|
| RELU					|												|
| Dropout  | keep 50%
| Fully connected		| input 120 output 84        									|
| RELU     |   									|
| Fully connected |  input 84 output 43
| Softmax				| etc.        									|
|						|												|
|						|												|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I have played with different number for epochs and learning rate. 
The more epochs I have, the better result I will have, and the accuracy improvement stopped at about 50 epochs with 95%
The smaller learning rate I have, the slower the accuracy can increase. 
the final paramater I used are Epochs = 50 and learning rate = 0.00097 , batch size = 156

To improve more accuracy, more data is necessary. 

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 1
* validation set accuracy of 0.95 
* test set accuracy of 0.930?

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
I am using the same architecture as Lenet, it was given good results on recognize handwritten numbers. 

* What were some problems with the initial architecture?
1. The data set was small to achieve better results
2. Need to add dropout to improve the training results
3. Need to transfer color from RGB to gray
4. Need to change Epochs and learning rate to achieve better results
5. Need to add Keep_prob = 0.5 for training and Keep_prob = 1 for predection.

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.

- Refer to privious answer.

* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

Other than ajusting the Epochs, batch size, and learing rate. I also changed dropout to 50%, this prevents the overfitting and improve the accuracy from 85% to 95%.


If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 Althought the leNet architecuture giving good results in this project, but this is an old type of architecture and many newer and better architectures are be developed. 
 If time allows, I would like to choose AlexNet. 
 It is based of Lenet but wider and deeper, this will allow to train network to recognize more complex object and provide much better accuracy.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:
![alt text](https://github.com/boweizhou/Traffic_Sign_Classifier/blob/master/images1/6_images.png)





Here are how image looks like when convert to gray scale.



![alt text](https://github.com/boweizhou/Traffic_Sign_Classifier/blob/master/images1/color_gray.png)

![alt text](https://github.com/boweizhou/Traffic_Sign_Classifier/blob/master/images1/color_to_gray1.png)
Image4, "no vechicles", seems like difficult to recognize, cause there is not a clear pattern. 

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit 30      		| Speed limit 30    									| 
| Bumpy road     			| Bumpy road 										|
| Ahead only					| Ahead only												|
| No Vehicles      		| No Vehicles					 				|
| Go straigt or left			| Go straigt or left     							|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 0.933.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 29th cell of the Ipython notebook.
Following are the results of the test. Accuracies for five selected signals are 100%.

*******************
1 real signal is Bumpy road
Bumpy road probability: 99.8
Bicycles crossing probability: 0.2
Go straight or right probability: 0.0
Dangerous curve to the right probability: 0.0
Children crossing probability: 0.0
Traffic signals probability: 0.0
*******************
2 real signal is Ahead only
Ahead only probability: 100.0
Yield probability: 0.0
Road work probability: 0.0
Speed limit (60km/h) probability: 0.0
Turn left ahead probability: 0.0
Go straight or right probability: 0.0
*******************
3 real signal is No vehicles
No vehicles probability: 64.13
Priority road probability: 34.5
Yield probability: 0.76
No passing probability: 0.6
End of all speed and passing limits probability: 0.0
Speed limit (60km/h) probability: 0.0
*******************
4 real signal is Go straight or left
Go straight or left probability: 100.0
Traffic signals probability: 0.0
Priority road probability: 0.0
Right-of-way at the next intersection probability: 0.0
Yield probability: 0.0
Roundabout mandatory probability: 0.0
*******************
5 real signal is General caution
General caution probability: 100.0
Pedestrians probability: 0.0
Traffic signals probability: 0.0
Right-of-way at the next intersection probability: 0.0
Speed limit (30km/h) probability: 0.0
Go straight or left probability: 0.0
*******************

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?



