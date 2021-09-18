## Problem 
The goal of this project is to classify images in order to identify sign language letters. The input size is 200x200x3 and the images are colorful. 3 represents the RGB values. Data includes a total of 29.000 images as Training and Test Datasets.

## Preprocessing
Firstly, we have done label conversion. Starting from ‘A’ we have mapped every letter to an integer value. For example, ‘A’ corresponds to 0 and ‘Z’ corresponds to  25. Similarly, ‘del’ is 26, ‘nothing’ is 27 and ‘space’ is 28. After mapping, we did one-hot encoding on labels to make them categorical.

In order to keep image and data size reasonable and efficient, we have resized images as 64*64*3. We normalized our pixel values by dividing them by 255.

We have tried many approaches. One of them was putting images into grayscale. However, when we have trained our model, test accuracy is reduced compared to colorful images. Hence, we eliminated this approach, namely, we decided to train our model with the images as they are given in terms of coloring. 

After reading the training set into a NumPy array, we have shuffled the data for our model to avoid learning using a label pattern. 

## Data Split 
We have splitted the training data as 80% training and 20% validation in order to fine-tune our hyperparameters with the validation data. The validation data includes an equal number of samples for each class. As a result, we have obtained 3 different datasets: train, validation, and test. These datasets do not overlap.

## Data Augmentation
Data augmentation is an approach for obtaining more data from the data we already have by rotating and randomly zooming them. Since using a large dataset is a way to prevent overfitting and underfitting, we followed this approach. We have applied the ImageDataGenerator function to our datasets. For example, we have rotated images by 10 degrees and zoomed with the value of 0.1, and obtained new images. In addition, we decided not to flip the images, because it would cause changing the meaning of the sign. Our batch size is 32, which means we gave 32 samples in each iteration. This is done due to memory limitations.

## Preprocessing Test Data 
Before doing predictions on the test data set, we have applied the preprocessing operation by normalizing the inputs and resizing the images as 64*64*3. Also, after predicting the labels, we have converted one-hot code correspondence of the letters back.


## Machine Learning Models
As we know, CNN (Convolutional Neural Network) performs very well on classifying an image dataset. Hence, we have used CNN. However, there were many approaches that we could follow.

## Hyperband Optimization 
Firstly, we have tried a 13 layer network. In order to find the best hyperparameters for this network, we have applied the HyperBand optimization approach. We trained our model with the training dataset and tested it with the validation dataset. 
There are many hyperparameters and we gave a valid range for each. The Hyperband found the best among them according to the validation accuracy score.
Hyperparameters:
Activation function
Dropout rate
Number of filters
Units (amount of neurons) in a layer
Learning rate
Epoch size

After tuning the hyperparameters, we have observed that the model with the best parameters includes learning rate = 0.000854 and ‘Relu’ activation function, etc. that has given the best validation accuracy. It uses 4,190,797 parameters. In order to prevent overfitting, we have also applied early stopping that monitors validation loss and stops before causing an overfit.
This model has given 99.59% accuracy on the validation data.
This model has given 99.57% accuracy on the first test data.

After this process, we trained the top 5 models that are given by hyperband and applied majority voting by giving equal weight to each of their predictions. As a result, we increased our accuracy score to 99.69%. This model allowed us to capture some mislabeled images that are missed out by the best model.


## Transfer Learning
There are many pre-trained models which are trained with large datasets. VGG19 is one of them and was created for large-scale image recognition specifically. 
We have applied the VGG19 model to the training dataset. This model has 47 layers consisting of convolution and MAX pooling layers. It uses 20,024,384 parameters. After generating the pre-trained model, we have implemented fine-tuning which is a step for the pre-trained model to learn our dataset. We have added a few more Dense layers and finally added the last layer that has 29 neurons, each representing a label. We have used ‘softmax’ activation in the last layer to retrieve a prediction result for multi-class labels. Since the model was trained before, it starts giving high accuracy scores on validation data. Initially, we tried early-stopping and we have obtained 99.75% accuracy on the first test data.
As a last improvement, we have tried different callback mechanisms to prevent overfitting. We have tried the ReduceLROnPlateau callback to our model. ReduceLROnPlateau reduces the learning rate at every step, therefore it prevents the model from oscillating while finding the minimum gradient descent. 
This model has given 99.83% accuracy on the validation data.
This model has given 99.954% accuracy on the first test data.


##  Discussion
Among various models, we observed that the pre-trained model gave the best result on the first test set. The reason is because the VGG19 model was trained with ‘imagenet’ dataset previously and it had obtained optimum weights with a great number of layers and filters. It started from those weights that it knew already (not starting with random values as was the case in our previous approaches) and improved gradually on our training data. As a result, we have decided to select this model to evaluate the second test data,  hoping it will predict the labels with a high accuracy.
