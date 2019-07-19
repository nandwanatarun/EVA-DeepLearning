# EVA-1
EVA-Visual Search and DeepLearning

This is about pixels and color combination used to create the different images . News paper is used 4 different colors 
types CMYK  (cyan, magenta, yellow and black inks)and monitors used 3 types RGB (Red, Green and Blue) Magazines used different 
color combinations and so on.  
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
