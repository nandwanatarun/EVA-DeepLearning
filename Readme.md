A3C & T3D :IMPLEMENTING T3D 

This is what you'll get post implementing T3D:https://www.youtube.com/watch?v=eYfIfJtS_Dk

This video above is trained using the code we will cover below and it is trained on Colab within few hours!

Let's look at these 15 steps through code:

 INITIALIZATION  

![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-9/Images/img1-2.png)

STEP 1 :
We initialize the Experience Replay Memory with a size of 1e6.
Then we populate it with new transitions

![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-9/Images/img2-1.png)

STEP 2 :
Build one DNN for the Actor model and one for Actor Target

![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-9/Images/img29.png)


![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-9/Images/img3-1.png)

STEP 3 :
Build two DNNs for the two Critic models and two DNNs for the two Critic Targets

![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-9/Images/img29.png)

![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-9/Images/img4-1.png)

STEP 4-15 :
Training process. Create a T3D class, initialize variables and get ready for step 4

![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-9/Images/img5-1.png)

STEP 4 :
Sample from a batch of transitions (s, s', a, r) from the memory

![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-9/Images/img6-1.png)

STEP 5 :
From the next state s', the actor target plays the next action a'

![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-9/Images/img7-1.png)

STEP 6 :
We add Gaussian noise to this next action a' and we clamp it in a range
of values supported by the environment

![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-9/Images/img8-1.png)

STEP 7 :
The two Critic targets take each the couple (s', a') as input and return two Q values,
Qt1(s', a') and Qt2(s', a') as outputs

![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-9/Images/img9-1.png)

STEP 8 :
Keep the minimum of these two Q-Values

![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-9/Images/img10-1.png)

STEP 9 :
We get the final target of the two Critic models, which is:
Qt = r + gamma * min(Qt1, Qt2)

We can define 
target_q or Qt as reward + discount  * torch.min(Qt1, Qt2)

but it won't work
First, we are only supposed to run this if the episode is over, which means we need to integrate Done

Second, target_q would create it's BP/computation graph, and without detaching Qt1/Qt2 from their own graph, we are complicating things, 
i.e. we need to use detach. Let's look below:

![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-9/Images/img11-1.png)

STEP 10 :
Two critic models take (s, a) and return two Q-Vales

![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-9/Images/img12-1.png)

STEP 11 :
Compute the Critic Loss

![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-9/Images/img13-1.png)

STEP 12 :
Backpropagate this critic loss and update the parameters of two
Critic models

![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-9/Images/img14-1.png)

STEP 13 :
Once every two iterations, we update our Actor model by performing
gradient ASCENT on the output of the first Critic model

![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-9/Images/img15-1.png)

STEP 14 :
Still, in once every two iterations, we update our Actor Target
by Polyak Averaging

![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-9/Images/img16.png)

STEP 15 :
Still, in once every two iterations, we update our Critic Target
by Polyak Averaging

![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-9/Images/img17-1.png)

T3D is DONE! 
