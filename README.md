# Behavioral Cloning Project**

The goals/steps of this project are the following:
* Use the simulator to collect data of good driving behaviour
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarise the results with a written report


[//]: # (Image References)

[image1]: ./examples/center_driving.jpg "Center Driving"
[image2]: ./examples/recovery_1.jpg "Recovery"
[image3]: ./examples/recovery_1.jpg "Recovery"
[image4]: ./examples/recovery_1.jpg "Recovery"
[image5]: ./examples/mirror_2.jpg "Mirror"
[image6]: ./examples/mirror_1.jpg "Mirror "
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project contains the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

For my model, I used Nvidia's CNN architecture which is described [here](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf). My adaptation consists of eleven layers instead of the original nine. One normalization layer, one new cropping layer to remove the sky and the bonnet, five convolutional layers, and five fully connected layers. The first two convolutional layers have a filter size of five, whereas the last three have a filter size of three. The model includes RELU layers to introduce nonlinearity with every convolutional layer.

#### 2. Attempts to reduce overfitting in the model

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, data from driving the course the other way round and samples from particularly challenging sections of the course (the bridge). 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall approach for deriving a model architecture was to use a very simple model until preprocessing as well as the generator were functional. After everything else was in place, I implemented Nvidia's CNN architecture.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track, like the bridge. To improve the driving behavior in these cases, I generated new training data that especially focussed on these challenging spots.

At the end of the process, the vehicle can drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture consisted of a convolution neural network with the following layers and layer sizes 

**Nvidia Architecture**

* Normalization and cropping
* Convolution 24@31x98
* Convolution 36@14x47
* Convolution 48@5x22
* Convolution 64@3x20
* Convolution 64@1x18
* Flatten
* Fully connected 1164 neurons
* Fully connected 100 neurons
* Fully connected 50 neurons
* Fully connected 10 neurons
* output vehicle control



#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded six laps on track  four using center lane driving. Here is an example image of center lane driving:

![alt text][image1]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to resume their drive after nearly getting off track.
These images show what a recovery looks like starting from the right:

![alt text][image2]
![alt text][image3]
![alt text][image4]

To augment the data sat, I also flipped images and angles thinking that this would increase my dataset for free. For example, here is an image that has then been flipped:

![alt text][image5]
![alt text][image6]

After the collection process, I had 43713 number of data points, including the left and right cameras. I then preprocessed this data by normalizing it to be around -0.5 to 0.5 in value and cropped it to cut off the sky and the bonnet.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 10. I used an adam optimizer so that manually training the learning rate wasn't necessary.
