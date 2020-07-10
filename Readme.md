Reinforcement Learning

video Link: https://www.youtube.com/watch?v=gn4nRCC9TwQ
Reinforcement learning is simultaneously a problem, a class of solution methods that work well on the class of problems, and the field that studies these problems and their solution methods. 

Reinforcement Learning problems involve learning what to do - how to map situations to actions - so as to maximize a numerical reward signal. 

In an essential way, they are closed-loop problems because the learning system's actions influence their later inputs.

The learner is not told which actions to take but instead must discover which actions yield the most reward by trying them out. 

In the most interesting and challenging cases, actions may not affect not only the immediate reward but also the next situation and, through that, all subsequent rewards.

 
These three characteristics:

    being closed-loop in an essential way, 
    not having direct instructions as to what actions to take, and
    where the consequences of actions, including reward signals, play out over extended time periods

are the three most distinguishing features of reinforcement learning problems.

![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-5/Images/environment.png)

Here we've got a little maze. And this maze is our representation of our environment. And that is what we're going to be dealing with.

It is in these environments, in which our artificial agent is going to be performing or going to be taking actions to beat these environments. 

![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-5/Images/agent.png)

And this is our Agent! The agent is our AI. That's the mind which going to be navigating these environments and learning from the feedback which it will receive from these environments to perform it's actions. 

![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-5/Images/cycle.png)

So the way it works is that the agent performs certain actions in this environment. 

As a result of this action, the state in which it is, changes. This might mean that it is closer or farther away from the target. It might also have some certain other parameters that describe its state and those parameters might also change (like the gravitational force at that point). 

The state is going to change because of the action it takes, and it will also get rewards based on that action. 
 
Every time it takes action, the state will change and it'll get a reward. 


Bear in mind that sometime it might happen that its state won't change or it won't receive any reward for taking an action (in a certain state). 
 

Irrespective of this, our agent is going to be taking action. And it is this act of performing actions that allows it to explore the environment and learn what actions lead to good rewards, as well as, which states are favorable and what actions lead to bad rewards and unfavorable states.

Actions and their consequences. Driving has become routine for you and may sound mundane, but it involves so many steps in order. 


In driving as well, your certain actions take you to certain states, while others lead to other states. 

It is very important to understand that the actions should be taken at the correct point in time. 

Correct actions in the correct order in the correct states would ideally lead you to your final reward. 


In terms of AI, the simplest way to think of reinforcement learning is like training a dog. 

You give your dog certain commands, and if it obeys those commands, then you give it a treat, like a biscuit, etc., and when it doesn't, well depending on what kind of trainer you are, your penalty would differ. 

 

It is through this process it learns what certain commands desire certain actions in certain states. 

![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-5/Images/rewards.png)

In the world of AI, we have numbers or digital that act as rewards. 



There is no programmed algorithm!
video Link:(https://www.youtube.com/watch?v=ITfBKjBH46E)

BELLMAN EQUATION 

Concepts:

    LaTeX: s\:-\:state s − s t a t e
    LaTeX: a\:-\:action a − a c t i o n
    LaTeX: R\:-\:reward R − r e w a r d
    LaTeX: \gamma\:-\:discount\:factor γ − d i s c o u n t f a c t o r

Let's look at the Bellman Equation step-by-step. 

![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-5/Images/maze1.png)

There's our lovely agent in the bottom left corner and he is in a maze. 

We have got white blocks, in which the agent can step into. The grey blocks are not acceptable (maybe it is a huge pillar which you can't penetrate). 

In this maze, the green block is where the agent is aiming to end up at, that's the goal. 

And the red is a firepit, so if the agent falls into this firepit, he will lose the game. 

In the firepit, the reward is -1. It is a way of telling our agent, that's not something we want it to do. 

![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-5/Images/maze2.png)

On the other hand, if it ends up in the green square, it will get a +1 reward meaning that that is what we want it to do. 


How does the agent now figure out what to do? 
 

We're just going to let it know that it can take 4 possible actions, go up, down, left or right. We'll let it play around and see what it can come up with. 

It will roam around a lot!


![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-5/Images/maze3.png)

And nothing might happen for quite some time. But after a while:

![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-5/Images/maze4.png)


when it steps into the Green Square, it gets a +1 reward, and that triggers the algorithm. This makes it think, 

"I am rewarded for getting up in this square >> so I want to end up in this square >> but how did I end up in this square? >> what was the preceding state I was in and what action I took there which got me here?"

The preceding state:

![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-5/Images/maze5.png)

"I am just 1 step away from my dream score of +1". "If next time I reach this red arrow square, all I have to do is to move right"

How does it remember this red arrow square? 


Well if you think about it, there is no difference between this red arrow square and the green square if it can mark that as soon as it is in this req arrow square, all it needs to do is to move right. So we can mark this req arrow square with the final reward value = 1. 

![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-5/Images/maze6.png)


V = 1 is the perceived value of being in this square. 

If we repeat this process, we will mark the left square to it as V = 1, and so on. ..

After some time, this is how our value table would look like

![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-5/Images/maze7.png)

This is great, and we have a path now, but the problem comes when the agent is initialized in some other square, for example in the image below:


![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-5/Images/maze8.png)

Now the agent is confused! Whether it should go to the right, or go down, as the values on both the sides are identical, and it can only know the value of its next state (as it stands right now). 

And that is why this approach doesn't work. We cannot just carry the value backward naively. 

![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-5/Images/maze9.png)

This is where the Bellman Equation comes to our rescue. 

![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-5/Images/svg1.png)

LaTeX: s_t s t is the current state, and LaTeX: V\left(s_t\right) V ( s t ) is the value of the current state. 

This means LaTeX: V\left(s_{t+1}\right) V ( s t + 1 ) is the value of the state that you will end up after this state. 

LaTeX: R\left(s_t,\:a\right) R ( s t , a ) is the reward you will get if in the current state you'll take action LaTeX: a a .

We know that the agent can take many actions, and that is why we have that LaTeX: \max_a max a

 

Let's look at below and see what is the value at "?".

![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-5/Images/bellman_visualized.png)

LaTeX: R\left(s,\:a\right) R ( s , a ) right now for this state maybe 0, as there might not be any reward for taking action in this state. It is only after the action is taken that we actually get a reward. 

 "?" will be equal to LaTeX: \gamma\times1 γ × 1

 

![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-5/Images/svg2.png)

LaTeX: \gamma γ is there to solve the problem we got stuck with. Earlier we were not able to compare states as we were carrying back the values as is. 

 

But now with LaTeX: \gamma γ we can. It is called a discounting factor. 

![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-5/Images/svg3.png)

Let's fill in this block and write the values ourselves assuming LaTeX: \gamma=0.9 γ = 0.9 :


![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-5/Images/maze1.png)

The value of the block left to our green block is +1. Why?

![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-5/Images/maze10-1.png)

Now if we try and fill up the rest of the values:

![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-5/Images/maze11.png)

We will not start with the blue arrow above.  

We'll start from here, and we have 3 possible actions:

![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-5/Images/maze12.png)

After filling up all the values:
![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-5/Images/maze13.png)

Now, we can start at any block and still find the correct path!


THE PLAN
![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-5/Images/maze14.png)

his is what a plan is. Our agent now knows what to do in a specific state. It sort of has a map to refer to. 

 

Once you have a map, you're done!

A Plan is not a Policy.  And a plan is not really useful as the plan works if everything else works as expected. 

A policy is very similar to a plan, but with a difference, it works with stochasticity. 

MARKOV DECISION PROCESS - MDP

Before we look at MDP, let's learn about Deterministic and non-Deterministic (or stochastic) Search
![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-5/Images/maze15.png)

The idea here is to create a more realistic model of what could actually happen even in a real-world situation, where things aren't 100% guaranteed. 


MARKOV PROCESS  

A stochastic process has the Markov property if the conditional probability distribution of future states of the process (condition on both past and present states) depends only upon the present state, not on the sequence of events that preceded it. A process with this property is called Markov Process.

The results of your action in a particular state is independent of how you reached that state. 

What this basically means is, the number below does not depend on how our agent arrived at this block. 


![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-5/Images/maze16.png)

You might agree that many real-world events or processes are not Markov Processes, but many can be converted into a simpler problem if we conder them to be. 

MDP provides a mathematical framework for modeling decision making in situations where outcomes are partly random and under the control of a decision-maker. 

This is our Bellman Equation so far:

![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-5/Images/SVG4.png)
Now LaTeX: s_{t+1} s t + 1 isn't guaranteed!

![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-5/Images/SVG5.png)


![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-5/Images/SVG6.png)


![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-5/Images/SVG7.png)

Our new Bellman Equation for MDP
![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-5/Images/SVG8.png)

 PLAN vs POLICY  
 
 The Plan
![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-5/Images/maze17.png)
But our plan won't work now because it is not in our control because of uncertainties. 

Let's look at these values:

![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-5/Images/maze18.png)

There are some probabilities associated with the actions now, because of which the values have changed. We can't calculate these values manually as it requires recursion now, but our agent can. 

Look at the value next to the Firepit now!

And look at 0.46!

Let's look at the arrows now
![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-5/Images/maze19.png)
Look at the arrows next to the firepit now!


![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-5/Images/maze20.png)
This is our policy!


