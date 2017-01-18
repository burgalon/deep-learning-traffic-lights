# Nexar Challenge - Traffic Light Recognition

## The problem
* The challenge provided a dataset of ~18K images of dashboard car camera, with CSV labels of either green, red or no traffic light.
* Challenge rules required using a ConvNet solution.
* For the challenge score a test set of 500K images was provided, out of which only 10K were actually tested. The submission score took into account the model size according to these formulas:
<pre>
model_size_score = exp(-model_size_mb/100)
classification_accuracy = number of correctly labeled images/number of predictions
challenge_score = classification_accuracy*model_size_score
</pre>


## Used Tools
* [Keras](https://github.com/fchollet/keras/) with either TensorFlow backend or Theano
* Macbook Pro with GeForce 512MB :)

## Overview for the solution
For training the model, I had split the dataset into two segments: 90% for training, and 10% for validation. This was done in order to avoid dumping too much data.
I have come up with a network architecture which is composed of 3 convolutional layers with max pooling and one full connected layer.

## Things that helped improve the training
* To avoid overfitting, I used Dropout layers, as well as data augmentation such as horizontal flip, rotation and zoom
* After trying to fine-tune the hyper-parameters, I have settled on 3,3 filter sizes which seemed to have worked best
* As for working tools, after trying to use Tensorflow straight up, I switched to Keras together Python notebooks, which helped experiementation and visualization a lot.

## Things I tried which improve the training
* first batch normalization layer
* Cropping - I had a feeling that cropping the lower half of the picture might help the training, but it seemed like this actually decreased the learning.
* I have spent many hours working with Amazon GPU instances which helped me experiement, but in the end, since the model size had to remain small, experiementing on my own laptop was fast.

## Still on my TODO list
* Rotate all portrait images, and fix training dataset
* Clean ground truth labels of training set which was not clean
* Try resnet identity blocks as part of the network architecture


## Model summary
<pre>
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to
====================================================================================================
convolution2d_1 (Convolution2D)  (None, 112, 112, 16)  448         convolution2d_input_1[0][0]
____________________________________________________________________________________________________
maxpooling2d_1 (MaxPooling2D)    (None, 37, 37, 16)    0           convolution2d_1[0][0]
____________________________________________________________________________________________________
dropout_1 (Dropout)              (None, 37, 37, 16)    0           maxpooling2d_1[0][0]
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 37, 37, 32)    4640        dropout_1[0][0]
____________________________________________________________________________________________________
maxpooling2d_2 (MaxPooling2D)    (None, 12, 12, 32)    0           convolution2d_2[0][0]
____________________________________________________________________________________________________
dropout_2 (Dropout)              (None, 12, 12, 32)    0           maxpooling2d_2[0][0]
____________________________________________________________________________________________________
convolution2d_3 (Convolution2D)  (None, 12, 12, 64)    18496       dropout_2[0][0]
____________________________________________________________________________________________________
maxpooling2d_3 (MaxPooling2D)    (None, 6, 6, 64)      0           convolution2d_3[0][0]
____________________________________________________________________________________________________
dropout_3 (Dropout)              (None, 6, 6, 64)      0           maxpooling2d_3[0][0]
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 2304)          0           dropout_3[0][0]
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 128)           295040      flatten_1[0][0]
____________________________________________________________________________________________________
dropout_4 (Dropout)              (None, 128)           0           dense_1[0][0]
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 3)             387         dropout_4[0][0]
====================================================================================================
Total params: 319,011
Trainable params: 319,011
Non-trainable params: 0
____________________________________________________________________________________________________
Found 16767 images belonging to 3 classes.
</pre>


## Example images:
###
![No traffic light 1](/images/example-none-1.jpg)
![No traffic light 2](/images/example-none-2.jpg)
![Green light 1](/images/example-green-1.jpg)
![Green light 2](/images/example-green-2.jpg)
![Red light 1](/images/example-red-1.jpg)
![Red light 2](/images/example-red-2.jpg)
