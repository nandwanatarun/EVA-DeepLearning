DEEP Q LEARNING  

Where we left the last few session:

Bellman Equation  



![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-7/Images/SVG1.png)



![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-7/Images/maze13.png)

MARKOV DECISION PROCESS - MDP  

![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-7/Images/maze16.png)



![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-7/Images/SVG2.png)

Our new Bellman Equation for MDP  

![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-7/Images/SVG3.png)


![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-7/Images/maze18.png)

LIVING PENALTY 

![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-7/Images/image1.png)



![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-7/Images/image5.png)


Q-LEARNING

![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-7/Images/SVG4.png)

![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-7/Images/image6.png)



![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-7/Images/image12.png)

TEMPORAL DIFFERENCE  

The most important concept! 

Let's look at our table. Let us assume in some way we have Q(s, a)
![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-7/Images/image16.png)



![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-7/Images/image17.png)

So after taking the step it again has of the same cell!

![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-7/Images/image18.png)

Let's look at the TD formula again:
![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-7/Images/SVG5.png)

We use this temporal difference to update our Q(s, a)!!
![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-7/Images/SVG6.png)

and LaTeX: \alpha α is our learning rate and this is how Q table is actually created!


Finally, our equation looks like this for Q:

![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-7/Images/SVG7.png)

 DEEP Q LEARNING  
 Let's re-look at our environment, but now with a spatial outlook. 

![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-7/Images/env1.png)


Now each state can be represented by a coordinate. Now we can feed in each cell to a DNN using a fixed representation:
![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-7/Images/env2.png)

We may have multiple hidden layers, but finally, it spits out 4 Q Values based on which we can decide on an action. 


Why?

Simple Q-Learning which we discussed till now is not scalable for complex environment,

and there is no limit to adding capacity to a DNN to handle any complex situation. 
 

Remember this image from the last session? 

![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-7/Images/image18.png)

TD should tend to zero as the training progress in the traditional Q-Learning approach. 

 

In DQN, the neural network will predict the 4 Q Values, but there is no direct before and after. 
 

This before and after was calculated for traditional QL when a step was being taken. 

 

In the case of DQN, the previous and current comparisons would be done for the time in the past when the agent was on the same block. 

![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-7/Images/env3.png)


![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-7/Images/env4.png)


These targets are the ones we had predicted, last time when we were in the same state. 

We are trying to adopt deep neural networks to work with Q-Learning here. 

This compared difference makes our loss function and that is what we use for back-prop:
![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-7/Images/env5.png)

 ACTION SELECTION 
 
 SoftMax! (& others)

![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-7/Images/env6.png)

So we have covered 2 concepts: Learning and Acting:
![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-7/Images/env7.png)

PLAN VS POLICY VS MODEL  

The Plan

![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-7/Images/maze17.png)


THE POLICY
![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-7/Images/maze20.png)

THE MODEL

![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-7/Images/env8.png)

Link:https://medium.com/@awjuliani/simple-reinforcement-learning-with-tensorflow-part-3-model-based-rl-9a6fe0cce99

A model is going to be a DNN that attempts to learn the dynamics of the real environment. 


For example, given a state, we'd like our DNN to predict the next action and state. 

 
By learning an accuracy model, we can train our agents using the model rather than requiring to use the real environment every time. 

 
This may seem less useful, but it can have huge advantages when attempting to learn policies for acting in the physical world. 

 
Unlike in computer simulations, physical environments take time to navigate, and the physical rules of the world prevent things as easy environment resets from being feasible. 


If we can build a "model" of the environment, an agent can "imagine" what it might be like to move around the real environment, and we can train a policy on this imagined environment in addition to real one. 


We are now using 2 networks:

First model learns to model the environment - given a state and action what happens

Second model learns to pick Q values or actions given the current state 

![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-7/Images/1%20AbbOXuPjEtNbFpifAXM7yg.png)

Our training procedure will involve switching between training our model using the real-environment and training our agent's policy using the model environment. 


By using this approach we will be able to learn a policy that allows our agent to solve the task without actually ever training the policy on the real environment!


We will touch MODELS in the future. Let's go back to basics. 


 
  EXPERIENCE REPLAY  

![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-7/Images/env9-1.png)

Please note that our state is not a simple X1X2 now. It is going to contain much more detail (via parameters) like speed, car angle, etc. 

 

As we see the car currently, every time the car has moved forward, nothing has changed, but it gets some rewards. 

 
That seems like a good approach, to just monotonously drive straight. And "overfits" our Policy model to learn, that's a good thing!


And the moment the curve comes in, it crashes!


So once we have consecutive correlated states, the model is overfitting to those states, and we don't want that!


And that is where Experience Replay comes in. 


We let the agent do its work, and do not backpropagate. 


Once the agent reaches a specific threshold, we decide to train it then. 


So the agent has tons of stored experiences.


When we do decide to train, we train it on randomly select, uniformly distributed sample experiences, and then train on it. 


Each experience is characterized by the state it was in, the action it took, the state it ended up in and the reward it achieved through that action. 

 Rolling Window of Experiences
 
 We also keep a rolling window of experiences, so we might have all the state histories, we pick a specific %age of the game and then randomly select from that.

These rolling windows have overlaps as well. New experiences come in, and older ones are kicked out. 

linK:https://arxiv.org/abs/1511.05952

Google DeepMind in 2015/16 released this paper which called for prioritizing experience replay. 

  

Prioritized sampling, as the name implies, will weigh the samples so that “important” ones are drawn more frequently for training.

 
In particular, we propose to more frequently replay transitions with high expected learning progress, as measured by the magnitude of their temporal-difference (TD) error. This prioritization can lead to a loss of diversity, which we alleviate with stochastic prioritization, and introduce bias, which we correct with importance sampling. Our resulting algorithms are robust and scalable, which we demonstrate on the Atari 2600 benchmark suite, where we obtain faster learning and state-of-the-art performance.

 

 
  ACTION SELECTION POLICIES  
 

or
 
  EXPLORATION VS EXPLOITATION 


![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-7/Images/env10.png)

We'd be using SoftMax. Benefits? 

![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-7/Images/env11.png)

Let's start some coding:

• Linux and Max users, please open your terminal. On Mac, the easiest way to open it is to press anywhere cmd + space, and then in the Spotlight Search you enter "terminal". On Linux, you will find it very easily, usually on the left side of your monitor. Then inside the terminal, copy paste and enter each of the following line commands separately:

• conda install pytorch==0.3.1 -c pytorch

• conda install -c conda-forge kivy


• And Windows users, please open the anaconda prompt, which you can find this way: Windows Button in the lower left corner -> List of programs -> anaconda -> anaconda prompt Then inside the anaconda prompt, copy paste and enter each of the following line commands separately:

• conda install -c peterjc123 pytorch-cpu

• conda install -c conda-forge kivy

• Download these files:https://drive.google.com/file/d/1H7iTOsv54AtafxPnL5zs2_0RoQ9J0sow/view



 
