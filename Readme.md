CONTINUOUS ACTION SPACES 
DQN cannot be applied to continuous action spaces. 

 In DQN we get the Max-Q value for our actions. When we have continuous action spaces, meaning action which can have infinite possible values (like how much to turn). 

So DQN is ruled out. 

We need to adapt DQN to continuous action space. 


 A3C 
Link[Asynchronous Methods for Deep Reinforcement Learning]:https://arxiv.org/pdf/1602.01783.pdf


![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-8/Images/img1.png)

The 3As of A3C 

  Asynchronous Advantage Actor-Critic  
   ACTOR-CRITIC  

 This is Convolutional DQN
![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-8/Images/img2.png)

Last RED Box is the Q Values resulting from the prediction. 

![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-8/Images/image9-1.png)

Remember this V vs Q value discussion we had? Let's discuss that again. What is Q-Value and what is V-Value?

In some literature, the Q values are also called Policy(s)

  This is Actor-Critic  (naive version)  
  
![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-8/Images/img3.png)

 ASYNCHRONOUS  
 
 Remember that till now we have 1 agent!
 
![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-8/Images/cycle-1.png)

But what if we have multiple agents initialized at different locations simultaneously?

![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-8/Images/img4.png)

Each one of them is learning from a broader range of experiences. This also reduces the chances of getting stuck in a local maximum. 
 

Sharing experience makes our RL agents learn faster. 

So this is what we are going to do:

![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-8/Images/img5.png)

But right now this is as good as running 3 programs on separate computers. 

We need to share their experiences somehow. 

What can we do?

![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-8/Images/img6.png)

We will further optimize this soon

ADVANTAGE  

Now we have 2 losses!

![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-8/Images/img7.png)

How good is the Q value selecting when compared to the V Value?

We are minimizing Value Loss and maximize The Advantage! Why?

+LSTM  

Not a direct part of the main algorithm, but added to improve the overall performance. 


![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-8/Images/img9.png)

![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-8/Images/img8.png)

ANOTHER VIEW OF ACTOR CRITIC  (Advanced)  

We need not keep the same model to predict the actions (Q-Values) and the Values (Max-Q values). We can build two separate models to do this like the one below:


![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-8/Images/img10.png)

This is a preferred model for modern RL algorithms. 

TAXONOMY OF AI MODELS  

Model-Free: directly taking data from the environment

Model-Based: create a model for the environment and use them for data

Value-Based: Q Leaning or DQN - a network which assigns a value to the state

Policy-Based: A network which learns to give a definite output by giving a particular input

Off-Policy: when you can learn on historical data (replay memory)

On-Policy: can only learn from new data.

Let's make it easier through an image:


![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-8/Images/img11.png)

  ACTOR-CRITIC MODELS    
  
![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-8/Images/img12.png)

This is what we are going to keep in our minds when we think of Actor-Critic models and we will use this architecture as well. 

So we have an actor, which is trying to predict an action based on the current state (policy network),

and we also have a critic which is trying to predict the V-Values (Max Q-Values) given the state and actions. 

Let us see how we train AC models  

First let's remind ourselves, that this is how we store our data in the replay buffer:


![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-8/Images/SVG1.png)

This is how things are structured right now. 


![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-8/Images/img30.png)

Training the ACTOR

We can connect Actor-Critic like this:


![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-8/Images/img31.png)

Now we can maximize Q, for which we can figure out what Actions were to be predicted and then train the Actor model. 

Training the CRITIC  
Critic Predicts Q.
So we need to figure out a way to calculate loss linked to Q. 
Remember this equation?

![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-8/Images/SVG2.png)

So here is what we can do!
![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-8/Images/img32.png)

And that's how we train Actor-Critic Models. 

T3D OR TWIN DELAYED DDPG  

The Twin stands for 2 Critics!

![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-8/Images/img13.png)

T3D 
TWIN DELAYED DEEP DETERMINISTIC POLICY GRADIENT MODELS

LET'S COVER THE T3D IMPLEMENTATION STEP BY STEP TO UNDERSTAND IT BETTER 

INITIALIZATION 

STEP 1  :

We initialize the Experience Replay Memory, with a size of 20000. We will populate it with each new transition

![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-8/Images/img14.png)

STEP 2  :

 We build TWO kind of actor models. One called the Actor Model and another called the Actor Target.
 
![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-8/Images/img15.png)

Actor Model and Actor Target models have exactly the same DNN definitions.

Why?

We'll see soon, but for all practical purposes (till we cover the reason, assume we have one).

STEP 3  :

We build TWO kinds of Critic Models. Once called Critic Model and another called Critic Target, BUT

We have 2 versions of the Critic Model and 2 Versions of Critic Target models!

![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-8/Images/img17.png)

All 4 have exactly the same DNN definition. 

 T3D MODEL ARCHITECTURE  
 
![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-8/Images/img18.png)

Actors are learning the policies and Critics are learning the Q-Values 

TRAINING PROCESS 

We run full episodes with the first 10,000 actions played randomly, and then with actions played by the Actor Model. This is required to fill up the Replay Memory. 

One Episode:

![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-8/Images/img19.png)

STEP 4  :

We sample a batch of transitions (s, s`, a, r) from the memory. 

![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-8/Images/img20.png)

STEP 5  :

Then from each element of the batch, From the next state s`, the Actor target plays the next action a`.

![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-8/Images/img21-1.png)

STEP 6  :

We add Gaussian noise to this next action a` and we clamp it in a range of values supported by the environment. This is same as exploration!

STEP 7  :

The two Critic Targets each take (s`, a`) as input and return two Q-values as output:

LaTeX: Critic1\:\Longrightarrow\:Q_{t1}\left(s',\:a'\right) C r i t i c 1 ⟹ Q t 1 ( s ′ , a ′ )

LaTeX: Critic2\:\Longrightarrow\:Q_{t2}\left(s',\:a'\right)

Basically:

![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-8/Images/img22.png)

STEP 8  :

We keep the minimum of these two Q-values: LaTeX: \min\left(Q_{t1},\:Q_{t2}\right) min ( Q t 1 , Q t 2 )

It represents the approximated values of the next state.

Taking a minimum of the 2 Q-values prevents too optimistic estimates of that value of the state! In classic Actor-Critic Method (with 1 Critic) we had overly optimistic estimates which prevented the training process from being stable, and taking the minimum of 2 Q-Values here adds that stability which was required. 

![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-8/Images/img23.png)

STEP 9  

We get the final target of the two Critic Models, which is:

LaTeX: Q_t\:=\:R\:+\:\gamma\ast\min\left(Q_{t1},\:Q_{t2}\right) Q t = R + γ ∗ min ( Q t 1 , Q t 2 )

where LaTeX: Q_t Q t is the target-Q

STEP 10  :

The two Critic Models each take the couple (s, a) as input and return two Q-values 

LaTeX: CriticModel1\:\Longrightarrow\:Q_1\left(s,\:a\right) C r i t i c M o d e l 1 ⟹ Q 1 ( s , a )

LaTeX: CriticModel2\:\Longrightarrow\:Q_2\left(s,\:a\right)

![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-8/Images/img24.png)

STEP 11  :

We compute the loss coming from the two Critic models: 

LaTeX: Critic\:Loss\:=\:MSELoss\left(Q_1\left(s,\:a\right),\:Q_t\right)\:+\:MSELoss\left(Q_2\left(s,\:a\right),\:Q_t\right)

![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-8/Images/img25.png)

STEP 12  :

We backpropagate this Critic Loss and update the parameters of the two Critic Models

![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-8/Images/img26.png)

STEP 13  :

Once every two iterations, we update our Actor Model by performing gradient ascent on the output of the first Critic Model

![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-8/Images/img27.png)

STEP 14  :

Now, why do we have Different Actor Target and Actor Models? 

Well, they can be the same, and in fact, in many naive RL models, they are the same.  

But we can improve overall performance by keeping two models and updating them from each other. 

Once every two iterations, we update the weights of the Actor target by Polyak averaging

![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-8/Images/SVG3.png)

This way our target comes closer to the model. 

![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-8/Images/img28.png)

STEP 15  :

Same steps for the Critics

Once every two iterations, we update the weights of the Critic target by Polyak averaging

LaTeX: \theta'\:\longleftarrow\:\tau\theta+\left(1+\tau\right)\theta' θ ′ ⟵ τ θ + ( 1 + τ ) θ ′

Where is the DELAYED part of the T3D?

We update our models at every step, but our target once every two steps. 

Complete Steps

![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-8/Images/img29.png)
