# **Traffic Sign Recognition** 
by [Rakshith Krishnamurthy](https://www.linkedin.com/in/rakshith-krishnamurthy-360682b/)

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

For preprocessing I chose to normalize the image by using (R-128)/128, (G-128)/128 and (B-128)/128 and then converted the image to gray scale to improve the classification of images.

#### The preprocessed keep right image is as shown below

![noramlized image](https://github.com/rakshithkeegadi/CarND-Traffic-Sign-Classifier-Project/blob/master/report_images/Normalized_image.png)

### Model Architecture

The architecture used is Lenet Architecture has
* Input of 32x32x3 which is convoluted to an output of 28x28x6 with valid padding.
* A relu is used as an activation function.
* The pooling layer converts 28x28x6 to an output of 14x14x6 with valid padding.
* The pooled output is then convoluted to get and output of 10x10x16.
* Another relu is used.
* The relu output is then using to pool and the imput 10x10x16 is converted to 5x5x16.
* To fully connect first it is flattened to form an output of 400.
* 400 is connverted to 120 by fully connected layer.
* 120 to 84 by fully connect.
* Finally 84 outputs 43 classes(logits) by fully connect.

To know more about Lenet architecture the [link](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf) can be used.

### Model Training
The model is trained initially with train dataset and then validated by the validation datat set. 
The hyper parameters like learning rate = 0.01, EPOCHS = 50, batch_size = 128, dropout= 0.5 for training and 1.0 for testing was used.
The EPOHS were 50 because the accuracy started flattening from epochs 48 onwards.
Logits computed from lenet are used to calculate cross entropy which gives us how different are from training labels.
An optimizer is used to average the cross entropy and then adam optimizer is used like SGD(Stocahistic Graident Descent)

After training the accurracy was achieved to be 99.7%

### Solution Approach
First the validation data set was run and the accuracy achieved was 98.5% and the testing accuracy was 99.7% .
To reach this efficiency the hyper parameters were chosen carefully and then image data sets were trained. Dropouts and normalization helped out to achieve better accurracy.

The epochs were for training and validation were run for 50 times and where as the testing was done only one.

## Test a Model on New Images

### Acquiring New Images

The images below 8 new images were chose to test on the trained model. The images were carefully chosen so that it can represent a good share of the 43 classes for example all the directional symbols can be represented by Turn right and also speed limits by 70km/h image and so on.

![new images](https://github.com/rakshithkeegadi/CarND-Traffic-Sign-Classifier-Project/blob/master/report_images/Web_images.png)

### Performance on New Images

The images were normalized and tested very much like the images used to train and validate.
The images were able to provide an accuracy of 75%. 
Out of the 8 images mentioned 2 were wrong predictions and 6 were good predicition.
Bumpy road and Speed Limit(70km/h) were the wrong predictions.


### Model Certaininty-Softmax Probabilites

The model Softmax probabilites are as shown below with prediction graph for each of the images.
0-1 on scale represent probabilities.

#### Wild animals crossing - Correct prediction

![Wild animals crossing](https://github.com/rakshithkeegadi/CarND-Traffic-Sign-Classifier-Project/blob/master/report_images/Wild_animals_crossing.png)

#### Turn righ ahead - Correct Prediction

![Turn right ahead](https://github.com/rakshithkeegadi/CarND-Traffic-Sign-Classifier-Project/blob/master/report_images/Turn_right_ahead.png)

#### Bumpy road - Incorrect prediction

![Bumpy road](https://github.com/rakshithkeegadi/CarND-Traffic-Sign-Classifier-Project/blob/master/report_images/Bumpy_road.png)

#### No entry - Correct prediction

![No entry](https://github.com/rakshithkeegadi/CarND-Traffic-Sign-Classifier-Project/blob/master/report_images/No_entry.png)

#### Speed limit (70km/h)- Incorrect prediction

![Speed limit (70km/h)](https://github.com/rakshithkeegadi/CarND-Traffic-Sign-Classifier-Project/blob/master/report_images/Speed_limit_70_km_h.png)

#### Roundabout mandatory - Correct prediction

![Roundabout mandatory](https://github.com/rakshithkeegadi/CarND-Traffic-Sign-Classifier-Project/blob/master/report_images/Roundabout_Mandatory.png)

#### End of all speed and passing limits - Correct prediction 

![End of all speed and passing limits](https://github.com/rakshithkeegadi/CarND-Traffic-Sign-Classifier-Project/blob/master/report_images/End_of_all_speeding_and_passing_limits.png)

#### No passing - Correct prediction

![No passing](https://github.com/rakshithkeegadi/CarND-Traffic-Sign-Classifier-Project/blob/master/report_images/No_passing.png)



## Further improvememts

* This success rate of the new images can be futher improved by providing more data sets to train.
* The limitation of my approach is that I have not used images that are rotated at various angles.
* The Lenet architecture can be better improved by using the approach as mentioned [here](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf). 
* [Inception model](https://hacktilldawn.com/2016/09/25/inception-modules-explained-and-implemented/) is another one that improves the efficiency and helps in better prediction and could be used to make my prediction better.
