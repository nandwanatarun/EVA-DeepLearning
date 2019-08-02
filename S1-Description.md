# S1-EVA
<b>EVA-Visual Search and DeepLearning</b>

EVA is Extensive "Vision" AI program. Our focus is mostly on Vision, but you can use the learnings of EVA on other domains as well.

For example, today, Audio if first converted into a spectrogram:

![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Session-1/S1_Images/Spectrograms-and-Oscillograms-This-is-an-oscillogram-and-spectrogram-of-the-boatwhistle.png)

and the network learns to "see" how a word "looks" like. 

![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Session-1/S1_Images/CrossSection.JPG)

It is the rods and cones that capture the light intensities and colors respectively (dark and bright conditions) respectively. You might be surprised to know that we do not have RGB sensors (strictly) in our eyes.  
Look at where the sensitivities of our "cones" peak at:

![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Session-1/S1_Images/spectrum-absorption-light-eye.jpg)



This is about pixels and color combination used to create the different images . News paper is used 4 different colors 
types CMYK  (cyan, magenta, yellow and black inks)and monitors used 3 types RGB (Red, Green and Blue) Magazines used different 
color combinations and so on.  
You are seeing a combination of RGB on the LCD you are reading this. 

![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Session-1/S1_Images/science-of-cmyk-color-2.jpg)

CMYK are combined together to give us similar results:

![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Session-1/S1_Images/image-printing-spotprocess01.jpg)

Some magazines are printed in 6 colors. This means we can divide our image into any number of colors (or channels). 
Below you'd see an animation where the input image is converted into 4 channels. To divide into any number of channels is our decision. 

![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Session-1/S1_Images/Eye%20Anim.gif)


![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Session-1/S1_Images/gradients.jpg)

Above you see simple OpenCV edge detectors. 

THE CONNECTION TO REMEMBER

    KERNELS = FEATURE_EXTRACTOR

    CHANNELS = SIMILAR_FEATURE_BAGS

Back to Neural Computing
We need feature extraction methods and then combining methods

![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Session-1/S1_Images/Build%20Everything.png)

As we see above, same edges and gradients can be used to make different part of different objects, which can then be combined to make the final object. Spend some time on the image above. 
 
Convolution - Feature Extraction

 ![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Session-1/S1_Images/main-qimg-4bfdf63a4c5b24590f0deec9673eaee5-c.jpeg)

The features you see above are the basic building block for our "vision". Similar role is played by the alphabets in out speech. 
 
Can you imagine this happening?

![]( https://github.com/nandwanatarun/EVA-DeepLearning/blob/Session-1/S1_Images/screen-shot-2016-08-10-at-12-58-30-pm.png)

By now you should be able to imagine that you can start with simple edges and gradients and combine them to make complex parts, which can then be combined to make objects. 
Before we Proceed, play with this: http://scs.ryerson.ca/~aharley/vis/conv/flat.html (Links to an external site.)
 
CORE CONCEPTS: CONVOLUTION
 
![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Session-1/S1_Images/4-2ConvolutionSmall.gif)

In the image above, purple channel/image is 4x4 in size. Dark Purple patch of moving 3x3 pixels is our "kernel". Kernels are also known as filters, feature extractors or simply a 3x3 matrix. When we convolve on 4x4 image/channel with a kernel of size 3x3, the resulting (green) output is 2x2 in size. 

 3x3 on 5x5 

![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Session-1/S1_Images/5-3ConvolutionSmall.gif)

Similarly, when we convolve on 5x5 by a 3x3 we get a 3x3 sized output channel. 
If you have some DNN background you'd realize that out kernels are always 3x3 in size. There is strong reason behind this. 

Look at this image below:

![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Session-1/S1_Images/0.8y3e18blaxk.gif)

This is one of the most important image for us. 

On the left, a 3x3 kernel is convolving on 5x5. As expected result (yellow) is a 3x3 channel. If we convolve on 3x3, by a 3x3 kernel, we will get only 1 output (please spend some time on this). If we were to convolve on 5x5 by a 5x5 kernel, we should expect to get only 1x1 output. This means that convolving with 3x3 kernels twice is same as convolving with a 5x5 kernel 1. 
Now what sized kernel do you want, a 7x7? Just convolve 3 times. You get the picture. 
3x3 kernels repeated many times will give us any size we want (odd sized only). Reason we do not want an even sized is that there is no line of symmetry in even numbers. We need to know the central line to create concepts of left and right with a distinguishable central line. 
Now in the image above, every "pixel" in the yellow channel has "seen" 3x3 pixels. This is called Local Receptive Field. Receptive field is the number of pixels each output has seen. 
The Cyan final output is a result of 3x3 kernel. So local receptive field (LRF) there is also 3x3. 
But each pixel in yellow as already seen 3x3 pixel. This cumulatively increases the Global Receptive Feild of our Cyan pixel to 5x5. Global Receptive field is the total number of pixel seen in the original image. 
You can also imagine that we can open yellow pixels like a window, and that would allow our final output to see all the original 5x5 pixels. 
Concept of receptive field is important for us, and that would help us understand how many layers do we actually need to add to our network. 
 
 LAYERS 

Let us imaging an image of size 400x400. As seen above each time we add a layer (kernels), we increase our receptive field by 2 pixels. (Original image has RF of 1, each each pixel has seen itself). 
Let us imagine that we have added 50 layers. Our Global Receptive Field is 100 now. Each pixel in the output now (which is of size 300x300, please try and understand why output is 300x300) has a GRF of 100x100. 
This means that each pixel can see 100x100 pixels in the original image. Now if you were to ask this pixel what is it seeing, it would say "a part of the original image", which could be a part of the object or may be some sort of background. This one pixel cannot tell us what is there is the original image. In fact we'd need to "ask" a lot of pixels to understand what exactly is seen in the image. 
The point we are making here is, if our final layer has not seen the whole image, or to put words differently, if the GRF is not 400x400, why should we imaging that our network can tell us what is the object in the image. This is the answer to the question "how many layers should we add". Answer is, as many as required to reach GRF of 400x400. 
Now, to achieve a GRF of 400x400, we would need to add 200 layers (can you convince yourself?). 
200 layers is a lot many. Out GPU would not be happy and may actually run out of memory and show as a message known as OOM. 
We need to reduce the layers somehow to make sure we can actually train the network. 

In deep learning for image processing we have three categories like features( which has the horizontal and vertical lines),
kernel or filter(used to extract the features or property) and channel ( will be the full package of image). 
Each block will represent the pixel and once we go down the layer will lose two pixels for example 5*5 will user the kernel of 3*3 then will have the output in the next layer which will be 1. Bigger example is for cyan 400*400 and kernel is 3*3 then next layer will be 398*398 and again kernel is 3*3 then next layer will be 396*396 and so on. And for 400* 400 we will be needing 400/2=200 layers to get the single pixel. 
  
    To reduce the processing for big images like 400*400 will use the concept of max pooling which will reduce 
    the accuracy but make process bit fast.

One Example as given below-


400x400 | 3x3 > 398x398
398x398 | 3x3 > 396>396
396x396 | 3x3 > 394>394
394x394 | 3x3 > 392>392
392x392 | 3x3 > 390>390
390x390 | MaxPooling > 195x195(390/2=195)
195x195 | 3x3 > 193x193
193x193 | 3x3 > 191x191
191x191 | 3x3 > 189x189
189x189 | 3x3 > 187x187
187x187 | 3x3 > 185x185
185x185 | MaxPooling > 92x92(185/2=92.5)
92x92 | 3x3 > 90x90
90x90 | 3x3 > 88x88
88x88 | 3x3 > 86x86
86x86 | 3x3 > 84x84
84x84 | 3x3 > 82x82
82x82 | MaxPooling > 41x41(82/2=41)
41x41 | 3x3 > 39x39
39x39 | 3x3 > 37x37
37x37 | 3x3 > 35x35
35x35 | 3x3 > 33x33
33x33 | 3x3 > 31x31
31x31 | MaxPooling > 15x15(31/2=15.5)
15x15 | 3x3 > 13x13
13x13 | 3x3 > 11x11
11x11 | 3x3 > 9x9
9x9  | 3x3 > 7x7
7x7  | 3x3 > 5x5
5x5  | 3x3 > 3x3
3x3  | 3x3 > 1x1
