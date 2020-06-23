
 # LSTMs  

Let's revise RNNs again, needed before we jump in LSTMs.

![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-3/Images/1_h_cfQuMl30szUkDAi7wrCA.png)

![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-3/Images/RNN.jpg)

 # RNN BackProp  
 
![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-3/Images/RNN_BackProp.jpg)

TRANSITIONING FROM RNNs TO LSTM

 

Let us imagine that we are working on the next word prediction model, given some initial text.

Dog scares Cat.
Cat scares Mouse.
Mouse scares Dog.

 

RNN model will see <EMPTY> before Dog/Cat/Mouse

 

RNN model also sees each word followed by scared and LaTeX: \bullet ∙ after them as well "an equal number of times"

 

Assuming the RNN model has memory only of the last step (exaggerated example), it can very well start prediction:

 

<Empty> Dog scares Dog.

<Empty> Mouse scares Cat.

<Empty> Mouse scares Cat scares Dog.


Let's look at the RNN again:

![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-3/Images/rnn1-2.png)


If we start with <Empty> we might Predict Dog. After receiving Dog as the input, there are equal chances of prediction (*) or scares. (That's where the error crops in). 


 
 
# The TanH - Squashing Function

![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-3/Images/images.jpg)
 
 Helps maintain output between -1 & +1. If we don't have it, then after 500 iterations a big output may explode!

 
  LETS ADD MEMORY TO OUR NETWORK  



![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-3/Images/svg.png)


Elementwise multiplication acts like a router, controlling what goes out and what doesn't. 

 
We'd like to use a function which is between 0 and 1 for this. We use...

![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-3/Images/1%20JHWL_71qml0kP_Imyx4zBg.png)
 
![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-3/Images/rnn2.png)
  
  We are holding the current prediction for the next step. So next time the main block predicts (after receiving input <Empty>) Dog, it will be stored in the memory. 

The "dog" goes as the input for the next stage. But this time MEMORY remembers DOG. 

 
 # LSTM  
  
![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-3/Images/lstm.png)
   
   An LSTM has a similar control flow as a recurrent neural network. It processes data passing on information as it propagates forward. The differences are the operations within the LSTM’s cells.
   
![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-3/Images/1_0f8r3Vd-i4ueYND1CUrhMA.png)
  
  These operations are used to allow the LSTM to keep or forget information. Now looking at these operations can get a little overwhelming so we’ll go over this step by step.
  
 #   Core Concept  

 
The core concept of LSTM’s are the cell state, and it’s various gates. The cell state act as a transport highway that transfers relative information all the way down the sequence chain. You can think of it as the “memory” of the network. The cell state, in theory, can carry relevant information throughout the processing of the sequence. So even information from the earlier time steps can make it’s way to later time steps, reducing the effects of short-term memory. As the cell state goes on its journey, information get’s added or removed to the cell state via gates. The gates are different neural networks that decide which information is allowed on the cell state. The gates can learn what information is relevant to keep or forget during training.

 
#  Sigmoid  

Gates contains sigmoid activations. A sigmoid activation is similar to the tanh activation. Instead of squishing values between -1 and 1, it squishes values between 0 and 1. That is helpful to update or forget data because any number getting multiplied by 0 is 0, causing values to disappears or be “forgotten.” Any number multiplied by 1 is the same value therefore that value stay’s the same or is “kept.” The network can learn which data is not important therefore can be forgotten or which data is important to keep.

![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-3/Images/1_rOFozAke2DX5BmsX2ubovw.gif)

Let’s dig a little deeper into what the various gates are doing, shall we? So we have three different gates that regulate information flow in an LSTM cell. A forget gate, input gate, and output gate.

 

# Forget gate

First, we have the forget gate. This gate decides what information should be thrown away or kept. Information from the previous hidden state and information from the current input is passed through the sigmoid function. Values come out between 0 and 1. The closer to 0 means to forget, and the closer to 1 means to keep.


![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-3/Images/1_GjehOa513_BgpDDP6Vkw2Q.gif)

 # Input Gate  

To update the cell state, we have the input gate. First, we pass the previous hidden state and current input into a sigmoid function. That decides which values will be updated by transforming the values to be between 0 and 1. 0 means not important, and 1 means important. You also pass the hidden state and current input into the tanh function to squish values between -1 and 1 to help regulate the network. Then you multiply the tanh output with the sigmoid output. The sigmoid output will decide which information is important to keep from the tanh output.

![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-3/Images/1_TTmYy7Sy8uUXxUXfzmoKbA.gif)

 # Cell State  

Now we should have enough information to calculate the cell state. First, the cell state gets pointwise multiplied by the forget vector. This has the possibility of dropping values in the cell state if it gets multiplied by values near 0. Then we take the output from the input gate and do a pointwise addition which updates the cell state to new values that the neural network finds relevant. That gives us our new cell state.


![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-3/Images/1_S0rXIeO_VoUVOyrYHckUWg.gif)

# Output Gate  

Last we have the output gate. The output gate decides what the next hidden state should be. Remember that the hidden state contains information on previous inputs. The hidden state is also used for predictions. First, we pass the previous hidden state and the current input into a sigmoid function. Then we pass the newly modified cell state to the tanh function. We multiply the tanh output with the sigmoid output to decide what information the hidden state should carry. The output is the hidden state. The new cell state and the new hidden is then carried over to the next time step.

![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-3/Images/1_VOXRGhOShoWWks6ouoDN3Q.gif)

 # Code Demo  

For those of you who understand better through seeing the code, here is an example using python pseudo-code.

![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-3/Images/1_p2yXhtxmYflEUrTC1rCoUA.png)


1. First, the previous hidden state and the current input get concatenated. We’ll call it to combine.
2. Combine gets fed into the forget layer. This layer removes non-relevant data.
4. A candidate layer is created using combine. The candidate holds possible values to add to the cell state.
3. Combine also get’s fed into the input layer. This layer decides what data from the candidate should be added to the new cell state.
5. After computing the forget layer, candidate layer, and the input layer, the cell state is calculated using those vectors and the previous cell state.
6. The output is then computed.
7. Pointwise multiplying the output and the new cell state gives us the new hidden state.

That’s it! The control flow of an LSTM network is a few tensor operations and a for a loop. You can use the hidden states for predictions. Combining all those mechanisms, an LSTM can choose which information is relevant to remember or forget during sequence processing.
