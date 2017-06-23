#**Traffic Sign Recognition** 

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

## Dataset Exploration

### Dataset Summary

| Dataset Features         		     | Values				| 
|:-------------------------------:|:---------:| 
| Number of training examples   		| 27839		   | 
| Number of testing examples     	| 12630 	   |
| Number of validation examples			|	6960			   | 
| Images Data Shape	      	       | (32,32,3) |
| Number of classes	              | 43							 |

The above table explains the data sets used in the project. Initially the total training samples was 34799 and I used 20 percent of the training set to create a larger validation set of 6960.

### Exploratory Visualization
There are 43 different classes in the training set and the images for all the 43 different classes are found in the images below.
![exploring the dataset](https://github.com/rakshithkeegadi/CarND-Traffic-Sign-Classifier-Project/blob/master/report_images/Image%20of%20all%20the%20classes.png)

#### Classes of signs vs Number of images per sign is as shown below
![classes_vs_images](https://github.com/rakshithkeegadi/CarND-Traffic-Sign-Classifier-Project/blob/master/report_images/classes_vs_Images.png)

The above graph shows the number of images of each type to train the model. Images of classes like Keep right, Yield and Priority road have around 1400 images whereas Speed Limit(20km/h), Dangerous curve to the left and Go straight or left have few images i.e around 200 or less.

## Design and Test a Model Architecture 

### Preprocessing
### Model Architecture
### Model Training
### Solution Approach


## Test a Model on New Images
### Acquiring New Images
### Performance on New Images
### Model Certaininty-Softmax Probabilites

## Further improvememts
