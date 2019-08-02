# EVA-DeepLearning
EVA-Visual Search and DeepLearning

<b>Session 3 (1 Aug 19)</b>

![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Session-3/S3_Images/no_padding_strides.gif)

In image processing, we will use the strip of 1(Kernel move 1 pixel from row or column) and because of that we lose 2 pixels per layer. 
If the strip is 2 then it will be 3 pixels loss per layer so image will get blurry and may lose the important feature form channel.

![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Session-3/S3_Images/Screen%20Shot%202018-10-05%20at%201.27.11%20PM.png)

Relu is used to remove the negative values from the channels it will be useful to get the highlighted features form the channel.
So we can decide with kernel we can choose that which values must highlights and which will be removed to the next layer.
Same feature we have used to get the vertical and horizontal features form the channel. 

![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Session-3/S3_Images/Screen%20Shot%202019-03-04%20at%207.08.46%20PM.png)

By using relu we can increase the resolution of the image but using this We will come across this checkerboard issue again 
in super-Resolution algorithms. Observe the last image in the sequence of images below.

![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Session-3/S3_Images/Screen%20Shot%202019-03-04%20at%207.08.01%20PM.png)

Please visit this link: https://distill.pub/2017/feature-visualization/ (Links to an external site.) to check out the other 
visualizations.


![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Session-3/S3_Images/1%20deVKbCzJs_7eL6p2ltkY0g.png)

Here mainly will discuss about the 1*1 kernel features . This is a powerful and process efficient kernel and used to mainly, 
when we are changing the kernel size high (512) to low(32).1*1 kernel will move pixel by pixel and provide the output in best 
matching colors(like a sketch pens)

![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Session-3/S3_Images/convolution.gif)

Above image is 3*3*3(Chennels)*4(kernel) is 4 kernel the size of 3x3 kernel for 3 different channels.


![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Session-3/S3_Images/V0iCy.jpg)

We really cannot find any pattern here. 3x3 is a very small area to actually gather any Perceivable information (for human eyes), 
especially when we are referring to an image of size 400x400.

5*5 

![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Session-3/S3_Images/i89Cq.jpg)

7*7 also will not get any features

![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Session-3/S3_Images/1%20Rrplue_mc9imvH19Q58BGA.png)

11x11

It is only at the receptive field of around 11x11 when we'd be able to make out thinks. Look what textures are extracting at 11x11.

![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Session-3/S3_Images/main-qimg-4bfdf63a4c5b24590f0deec9673eaee5.webp)

Below is an example where we are using the 400*400 channel and kernel would be 3*3 .
Kernel we are using here is 32-64-128-256-512 and out RF is 11*11 and where will get some meaningful feature to proceed. 
So after that we can apply the max pooling logic. We have to reduce the size of the kernel but it will be difficult to convolve 
the 512 kernel into the 32 kernel. So will use the 1*1 kernel to achieve the task. Here we can't use 3*3 kernel as it would be 
process consuming and will lose channel feature as well.


400x400x3     | (3x3x3)x32        | 398x398x32     RF of 3x3( LRF is for 398*398 is 1+2=3)
398x398x32   | (3x3x32)x64      | 396x396x64    RF of 5X5 ( LRF is for 396*396 is 3+2=5)
396x396x64   | (3x3x64)x128    | 394x394x128  RF of 7X7
394x394x128 | (3x3x128)x256 | 392x392x256  RF of 9X9
392x392x256 | (3x3x256)x512 | 390x390x512  RF of 11X11
MaxPooling
195x195x512 | (?(1)x?(1)x512)x32    | ?(195x?(195)x32 RF of 22x22 (After applying the maxpool GRF 11*2=22)

.. 3x3x32x64                    RF of 24x24
.. 3x3x64x128                  RF of 26x26
.. 3x3x128x256               RF of 38x28
.. 3x3x256x512                RF of 30x30


![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Session-3/S3_Images/1x1-1.gif)

What you see above is an input of size 32x32x10. We are using 4 1x1 kernels here. Since we have 10 channels in input, 
our 1x1 kernel also has 10 channels. 
32x32x10 | 1x1x10x4 | 32x32x4
We have reduced the number of channels from 10 to 4. Similarly, we will use 1x1 in our network to reduce the number of 
channels from 512 to 32. Let's look at the new network:
 
400x400x3     | (3x3x3)x32        | 398x398x32     RF of 3x3
CONVOLUTION BLOCK 1 BEGINS
         398x398x32   | (3x3x32)x64      | 396x396x64    RF of 5X5
         396x396x64   | (3x3x64)x128    | 394x394x128  RF of 7X7
         394x394x128 | (3x3x128)x256 | 392x392x256  RF of 9X9
         392x392x256 | (3x3x256)x512 | 390x390x512  RF of 11X11
CONVOLUTION BLOCK 1 ENDS
 
TRANSITION BLOCK 1 BEGINS
         MAXPOOLING(2x2)
         195x195x512 | (1x1x512)x32    | 195x195x32 RF of 22x22
TRANSITION BLOCK 1 ENDS
 
CONVOLUTION BLOCK 2 BEGINS
         195x195x32     |(3x3x32)x64        | 193x193x64      RF of 24x24
         193x193x64     |(3x3x64)x128      | 191x191x128    RF of 26x26
         191x191x128   |(3x3x128)x256   | 189x189x256    RF of 28x28
         189x189x256   |(3x3x256)x512   | 187x187x512    RF of 30x30
CONVOLUTION BLOCK 2 ENDS
 
TRANSITION BLOCK 2 BEGINS
         MAXPOOLING(2x2)
         93x93x512 | (1x1x512)x32   | 93x93x32 RF of 60x60
TRANSITION BLOCK 2 ENDS
 
CONVOLUTION BLOCK 3 BEGINS
         93x93x32 | (3x3x32)x64 | 91x91x64       RF of 62x62
...
Notice that we have kept the first convolution outside of our convolution block, as now we can create a functional block 
receiving 32 channels and then perform 4 convolutions, giving finally 512 channels, which can then be fed to transition 
block (hoping to receive 512 channels) which finally reduces channels to 32.


Activation Function

![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Session-3/S3_Images/12a55d89bef512adc2b29f9738df7375.jpg)

We have multiple functions Like Sigmoid,TanH,ReLU,LeakyReLU,SELU,ELU,SReLU,Swish.Below is the image which represent 
the different graf angles for all the functions.

![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Session-3/S3_Images/The-rectified-linear-unit-ReLU-the-leaky-ReLU-LReLU-a-01-the-shifted-ReLUs.png)

ReLU is the simple fast and more efficient function to use, But all other function is also will be in use and it depends on 
the required output. For information NVIDIA has acceleration for ReLU activation.


ReLU

![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Session-3/S3_Images/1%20DfMRHwxY1gyyDmrIAd-gjQ.png)

This is a very simple function. It allows all the positive numbers to pass on, as it is, and converts all the negatives to zero. It's message to backpropagation or kernels is simple. If you want some data to be passed on to the next layers, please make sure the values are positive. Negative values would be filtered out. This also means that if some value should not be passed on to the next layers, just convert them to negatives. 

ReLU  is fast, simple and efficient. 
ReLu- Rectified Linear units : It has become very popular in the past couple of years. It was recently proved that it had 6 times improvement in convergence from Tanh function. It’s just R(x) = max(0,x) i.e if x < 0 , R(x) = 0 and if x >= 0 , R(x) = x.
 
ReLU is linear (identity) for all positive values, and zero for all negative values. This means that:
It’s cheap to compute as there is no complicated math. The model can therefore take less time to train or run.


It converges faster. Linearity means that the slope doesn’t plateau, or “saturate,” when x gets large. It doesn’t have the vanishing gradient problem suffered by other activation functions like sigmoid or tanh.


It’s sparsely activated. Since ReLU is zero for all negative inputs, it’s likely for any given unit to not activate at all. 


Zoom on this image to see what ReLU does:

 ![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Session-3/S3_Images/0%20NxkHMcSKYSqRz1RU.jpeg)
 
Above image represent the how relu work and the accuracy using relu and maxpool.

