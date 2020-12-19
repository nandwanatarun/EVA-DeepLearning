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

## checks
15 steps in the T3D code:

# Step 1
Defining the experience replay buffer

This class needs to have 2 methods:
1. Append

    Given the replay buffer is of fixed size, after the reqd size is reached
2. Sample

![Step 1](https://drive.google.com/uc?export=view&id=1BaCmiR6zl23zcjlUJB24uIWxCWTZz5Sp)

# Step 2
Define the class for the Actor and Actor Target
    Define a simple DNN with 2 hidden layers.
    Both the Actor and Actor Target have the same DNN structure and hence dont require different classes. 

![Step 2](https://drive.google.com/uc?export=view&id=1TFKY4jzUnKywxD7_PUVP3YCwx99bVdTZ)

# Step 3
Define the class for the Critic and Critic Target
    There are 2 critic models in T3D. Hence we need to define 2 DNNs in this class
    The forward function returns a tuple of both DNNs outputs
    We also define a function to return the Q value from just the first critic DNN for updating the Actor weights.

![Step 3](https://drive.google.com/uc?export=view&id=1BS4_lneg7EwK10_2cC76aBPpx3TDLHzg)

# Intermediate step
Declare the T3D class
This houses 4 objects ( 6 DNN models )
    1. Actor
    2. Critic ( contains two critic DNNs internally )
    3. Actor target
    4. Critic target ( contains 2 critic DNNs internally )

![Step 3.5](https://drive.google.com/uc?export=view&id=1jQa0iMXgcPL7fyBfK-wR3ow98T05SEC3)

# Training begins
Do the following steps for each iteration ( num iterations = 100k )

# Step 4
Extract a sample from the replay memory
Convert each part of the sample into tensors

What we have now is a set of tuples of the form:

    <current state, current action, reward, next state>

![Step 4](https://drive.google.com/uc?export=view&id=1hOZ1ue7leRKRddoHOuM2ZnlHceT0ILLO)

# Step 5
Pass the next state through the actor target to obtain the right action to do at this point.

![Step 5](https://drive.google.com/uc?export=view&id=11tHKROsvyXLRLQ9P0_gcwNsDn9nppzYr)

# Step 6
Add gaussian noise to the action suggested by the actor target and clip the values to be within acceptable levels

![Step 6](https://drive.google.com/uc?export=view&id=1a-iiqCVMr4pI6HHkObAUIFPaeT6FFiF5)

# Step 7
Pass the next state and next action ( obtained in step 6 ) into the critic target to obtain two Q values from the two critic target DNNs

![Step 7](https://drive.google.com/uc?export=view&id=1cfz7l57qeSswhsq-qRXzww3LJMibZtgq)

# Step 8
Compute minimum of the 2 Q values obtained from the 2 critic target DNNs

![Step 8](https://drive.google.com/uc?export=view&id=1TX-yUF0_sZ9G4sEeiIKyiFey8Ws4LwXf)

# Step 9
Set target values for Q(current state, current action)

target Q := reward + ((1-done)*discount*target_Q).detach()

![Step 9](https://drive.google.com/uc?export=view&id=1mqFAXeuFVKIq7bOCIIQ3oTn_wMdnYySF)

# Step 10
Compute the outputs from the two critics for (current state, current action)

![Step 10](https://drive.google.com/uc?export=view&id=1CE5MyQYOdI94EdKVVg0QRXytT-nY0Jwp)

# Step 11
Compute critic loss and 

For each of the critic models it is the mean squared error between target Q value ( from step 9 ) and what that critic model provides

We sum the the losses across the two critic models

![Step 11](https://drive.google.com/uc?export=view&id=186vlMjSOtRet4J2Wk1xUkz8j42n7Ucuy)

# Step 12
Backpropogate the critic loss obtained in step 11

![Step 12](https://drive.google.com/uc?export=view&id=1CYolxou-lChAicsKDTZGeINbHmBvCCtr)

# Step 13
Every 2 iterations we compute loss on the actor and backpropogate the same

Actor loss is computed as the average ( across the batch sampled from replay memory ) of the critic 1's output value for (current state, current action)

![Step 13](https://drive.google.com/uc?export=view&id=1et19h7cAHZLeKO7TUgdWhOCLfXWlGuyj)

# Step 14 and 15
Every two iterations update the actor target and critic target with polyak averaging

![Step 14 15](https://drive.google.com/uc?export=view&id=1zOZAeZbRoMA8zxN_h5MO5NDsLFrPmDzJ)


T3D is DONE! 
