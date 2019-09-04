# EVA-DeepLearning
EVA-Visual Search and DeepLearning
Session 5(15Aug19):

Batch Normalization and regularization:-

Calculate the convolution layer by kernel is 3*3(3-2-2)=0, 5*5(5-2-2)=1,7*7(7-2-2)=3 so 7*7 will read each pixel 3 times and it will increase the accuracy for the network.
Global Average pooling (GAP) is used to get the 1 number ( it will get the average number from the kernel).
Global max pooling (GMP) also avail;able , but will not use this as by using this will lose max feature form the channel.
Overfitting is when training accuracy is greater than the validation accuracy then network is over fitting (Over fitting will be subjective).


Normalization:
To train the network in a better way we have to convert the image into higher variance then we have a large area and we can spread the learning spectrum

![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Session-5/S5_Images/hist-compare1.png)

Another example of normalization is just move the scale in that way where we can focus on only specific part of the image. Below is the example  where we have thermal image and in second image we are getting the specif (focusing the specif area of image)image which is important for us.

![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Session-5/S5_Images/0CAxx.jpg)

Equalization:


In equalization will divide the image in different salce and we have to make sure each scale have  equal number of pixel pixel.but this is bad for the network as we lost some of the feature form the image. 
So we have to use the normalization instead of equalization.(http://www.roborealm.com/forum/index.php?thread_id=4350)
In below image we can see the difference in all three images .  

![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Session-5/S5_Images/normalization.jpg)

How normalization will work, so we he below image which represent the number of rooms in the scale of 0 to 100 (in first image) which will be difficult to understand that the network as we have a very small area to train the network .
    Now in the second image we have applied the normalization where we are getting the same type of information but we have a large area where we can get the max feature form the image and train the network well. So it will properly distribute the image features, which is easy to learn for the kernel.

![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Session-5/S5_Images/outlier.png)

Again the same example where in second iumage picel is properly distributed between scale.

![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Session-5/S5_Images/normalization.png)

Calculation for the normalization is First we have to calculate the mean of the images then subtract the mean from the old dataset and divide the image by standard deviation.
For example we have the 3 channel (RGB) in image so we have mean of the image is 3
And 3 standard deviation as well
Sum of all the pixel / total number of pixels ( that will give us the one pixel value) will do this with each pixel and that will give us the mean of the one image.
Original value is 0 and 1
Mean will be around 0.5
Standard deviation is 0.2(mostly)
Then (-0.5/0.2)-(0.5/0.2) so -1or +1

![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Session-5/S5_Images/prepro1.jpeg)

Taking the mean (left) and standard deviation (right) of the batch, we get the following It will give the highlighted feature form the channel:

![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Session-5/S5_Images/HtRe8.png)

Our centered data resembles a face, but more importantly our standard deviation image shows great variance along the borders and everywhere except the face. From our z-score formula, we can predict that the standardized values near the border and irrelevant areas will be relatively squashed more than values around the face. The resultant normalization appears as such:(https://datascience.stackexchange.com/questions/26881/data-preprocessing-should-we-normalise-images-pixel-wise)

![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Session-5/S5_Images/oHtCK.png)

Let's think about our un-normalized kernels and how loss function would look like:

![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Session-5/S5_Images/UnNormalized.png)

If we had normalized our kernels (indirectly channels), this is how it would look like:

![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Session-5/S5_Images/Normalized.png)

Let's look at the top view to appreciate the trouble here:
Un-normalized:

![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Session-5/S5_Images/UnNormalizedTop.png)
‘
Normalized:

![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Session-5/S5_Images/NormalizedTop.png)

 If features are found at a similar scale, then weights would be in a similar scale, and then backprop would actually make sense.

Image Normalization:
# example of standardizing an image dataset
from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator
# load dataset
(trainX, trainy), (testX, testy) = mnist.load_data()
# reshape dataset to have a single channel
width, height, channels = trainX.shape[1], trainX.shape[2], 1
trainX = trainX.reshape((trainX.shape[0], width, height, channels))
testX = testX.reshape((testX.shape[0], width, height, channels))
# report pixel means and standard deviations
print('Statistics train=%.3f (%.3f), test=%.3f (%.3f)' % (trainX.mean(), trainX.std(), testX.mean(), testX.std()))
# create generator that centers pixel values
datagen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)
# calculate the mean on the training dataset
datagen.fit(trainX)
print('Data Generator mean=%.3f, std=%.3f' % (datagen.mean, datagen.std))
# demonstrate effect on a single batch of samples
iterator = datagen.flow(trainX, trainy, batch_size=64)
# get a batch
batchX, batchy = iterator.next()
# pixel stats in the batch
print(batchX.shape, batchX.mean(), batchX.std())
# demonstrate effect on entire training dataset
iterator = datagen.flow(trainX, trainy, batch_size=len(trainX), shuffle=False)
# get a batch
batchX, batchy = iterator.next()
# pixel stats in the batch
print(batchX.shape, batchX.mean(), batchX.std())



Statistics train=33.318 (78.567), test=33.791 (79.172)
Data Generator mean=33.318, std=78.567
(64, 28, 28, 1) 0.010656365 1.0107679
(60000, 28, 28, 1) -3.4560264e-07 0.9999998


Batch Normalization: (http://proceedings.mlr.press/v37/ioffe15.pdf)

 Batch normalization has the same features like normalization but the difference is we are normalizing the whole batch instead of one image .

 
Batch Normalization solves a problem called Internal Covariate shift. To understand BN we need to understand what is CS. 
 
 
 
 
 
Covariate means input features. Covariate shift means that the distribution of the features is different in different parts of the training/test data. 
 
 
 
Internal Covariate shift refers to changes within the neural network, between layers.  A kernel always giving out higher activation makes next layer kernels always expect this higher activation and so on. 
 
 
Imagine what would happen if One channel ranges between -1 to 1 and another between -10 to 10

![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Session-5/S5_Images/Batch_Normalization__performance_and_activations.png)


Very Deep nets can be trained faster and generalize better when the distribution of activations is kept normalized during BackProp.
 
 
We regularly see Ultra-Deep ConvNets like Inception, Highway Networks,  and ResNet.   And giant RNNs for speech recognition, machine translation (https://ai.googleblog.com/2016/09/a-neural-network-for-machine.html), etc.

![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Session-5/S5_Images/Batch-Normalization.png)

Detailed description of the formula:

![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Session-5/S5_Images/Batch-Normalization.png)

Some crucial points:
# t is the incoming tensor of shape [B, H, W, C]
# mean and stddev are computed along (0, 1, 2) axes and have just [C] shape
mean = mean(t, axis=(0, 1, 2))
stddev = stddev(t, axis=(0, 1, 2))
for i in 0..B-1, x in 0..H-1, y in 0..W-1:
  out[i,x,y,:] = norm(t[i,x,y,:], mean, stddev)

n total, there are only C means and standard deviations and each one of them is computed over B*H*W values.






BIAS GETS SUBTRACTED OUT IN BATCH NORMALIZATION(BIAS is not very useful as it will not make a difference in network for example for 64000 parameter we have 1 BIAS which will not make any change in network)


