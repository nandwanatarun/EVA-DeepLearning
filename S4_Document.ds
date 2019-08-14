
S4(8 Aug 2019) -
When we use the fully connected layer it will lose the spatial information from the channel.
Fully connected layer are heavy on process side as well. Here in below image we are converting the 2D image into the 1D and 
destroy the location information with FC layer.

![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Session-4/S4_Images/hqdefault.jpg)


The rows(it's a pixel with weight) are connected in FC layers has the weight. Below image has simple three 
layer(the input neuron layer, hidden neuron layer and output neuron layers) network, and each pixel is connected with 
second layer pixels with row on some weight.  

![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Session-4/S4_Images/3lnn.svg)

Before 2014 below is the network used to convert the image into the FC layer and this method use in the VGGA systems.

![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Session-4/S4_Images/image_0-8fa3b810.png)

Below is the background process for the image.  

![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Session-4/S4_Images/imagenet_vgg16.png)

If we see the below example on FC layer it will increase the parameter exponentially high and make process heavy.

![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Session-4/S4_Images/vgg16_keras.png)

Softmax:
Softmax algorithm used to make machine know the accuracy about the object in better or expressive, representative menner .
But it will not add any value for the image.(we not supposed to use this on critical scenarios like medical).


![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Session-4/S4_Images/softmax.png)


Softmax is not the probability as same numerator divided by same denominator and its sums of to one and thats why its not probability

For softmax formula:

![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Session-4/S4_Images/SoftmaxFormula.png)

Drop out will remove the Ovewrfitting problem. Example below how it will work. 

![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Session-4/S4_Images/dropout.gif)


