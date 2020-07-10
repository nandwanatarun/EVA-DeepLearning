Q Learning  

This is what we covered in short in the last session:

![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-6/Images/maze1.png)



![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-6/Images/maze9.png)

This is where the Bellman Equation comes to our rescue. 
![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-6/Images/SVG1.png)


![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-6/Images/maze13.png)

 THE PLAN 
![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-6/Images/maze14.png)

A Plan is not a Policy.  And a plan is not really useful as the plan works if everything else works as expected.

MARKOV DECISION PROCESS - MDP  


![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-6/Images/maze16.png)


![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-6/Images/SVG2.png)

 Our new Bellman Equation for MDP  
 
![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-6/Images/SVG3.png)

![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-6/Images/maze18.png)

  THE POLICY  
  
![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-6/Images/maze20.png)

 LIVING PENALTY 
 
 Every time the agent moves, it gets a negative reward. Notice that he is on a tile with -0.04 negative reward, but he doesn't get that if he moves into this tile. 

He has to perform an action to get any reward. 
![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-6/Images/image1.png)

If we add this living penalty, let's see what happens:
![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-6/Images/image2.png)


![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-6/Images/image3.png)


![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-6/Images/image4.png)


![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-6/Images/image5.png)

 Q-LEARNING  
 
![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-6/Images/SVG4.png)


![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-6/Images/image6.png)

Till now we have values as in the left image above. Those are the ones that we calculated with the new Probabilistic MDP Bellman equation above. 

We're going to slightly modify that and now use Q values.  Q may stand for Quality.
![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-6/Images/image7.png)


![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-6/Images/image8.png)

Now, let us see how we calculate Q-values
![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-6/Images/image9.png)

Exactly!
![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-6/Images/image10.png)

Let's get rid of V!!
![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-6/Images/image11.png)


![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-6/Images/image12.png)

This is the formula that is actually used by the agent!

 TEMPORAL DIFFERENCE
 
 The most important concept! 
 
 Remember deterministic and non-deterministic search?
![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-6/Images/image13.png)

In case of deterministic search, calculating the value table was straight forward,

but for the non-deterministic search we said we'll learn how to calculate the values later on:
![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-6/Images/image14.png)

There is a lot of recursions which are happing up there and are subject to change! How do we calculate this then?


This is where Temporal Difference comes into play. 
 

Let's look at the equations again but use a simplified formula (so we can track it easily, but...):


![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-6/Images/SVG5.png)

Let's look at our table. Let us assume in some way we have Q(s, a)

![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-6/Images/image16.png)

![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-6/Images/image17.png)

So after taking the step it again has of the same cell!
![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-6/Images/image18.png)

Let's look at the TD formula again:
![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-6/Images/SVG6.png)

If the table is finalized, then this would be zero, but not when we are learning (why?). 

 
We don't just forget the old value when we have the new value (why?) 

And that is why we need the temporal (time) difference!

We use this temporal difference to update our Q(s, a)!!
![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-6/Images/SVG7.png)

and LaTeX: \alpha α is our learning rate and this is how Q table is actually created!


Finally, our equation looks like this for Q:
![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-6/Images/SVG8.png)

Look at how LaTeX: \alpha α values change how Q is updated, and it should be clear that it is no 0 or 1!

When TDs become 0, we say that our algorithm has converged (why?)

But if the environment is constantly changing, then TDs won't be 0,
and we would continue the online-training of our model (why?). 

AI IN ACTION!  

check out this:http://ai.berkeley.edu/home.html  (ZIP:http://ai.berkeley.edu/reinforcement.html)

Let's run what we've got (please remember to activate P2.7 environment for this). 

Let's start with python Gridworld.py -h and then -m and notice the stochastic nature of the environment. 


Now let's try python Gridworld.py -k 100 -a q -s 2000
