GRU, ATTENTION MECHANISM & MEMORY NETWORKS
LSTM RECAP   
RNN  

Mango-Banana-Peanut-Apple  

![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-4/Images/rnn1-2.png)

The TanH - Squashing Function

![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-4/Images/images.jpeg)
Helps maintain output between -1 & +1. If we don't have it, then after 500 iterations a big output may explode!

SIGMOID SELECTION GATE  
We'd like to use a function which is between 0 and 1 for this. We use...
![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-4/Images/1%20JHWL_71qml0kP_Imyx4zBg.png)

![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-4/Images/SVG.png)

LETS ADD MEMORY TO OUR NETWORK  

![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-4/Images/rnn2.png)

LSTM 

![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-4/Images/lstm.png)

An LSTM has a similar control flow as a recurrent neural network. It processes data passing on information as it propagates forward. The differences are the operations within the LSTM’s cells.

![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-4/Images/1_0f8r3Vd-i4ueYND1CUrhMA.png)

The core idea behind LSTM:

![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-4/Images/LSTM3-C-line.png)

The key to LSTMs is the cell state or memory. It is shown as the horizontal line running through the top of the diagram (and as the circular loop in our image). 

The cell-state or the memory is maintained by the forgetting gate. Unlike in RNN where will forget because of over-writes, memory in LSTM is maintained by the forget gate removing the un-necessary edits. 
Source:https://colah.github.io/posts/2015-08-Understanding-LSTMs/

Let's go through LSTM's again step-by-step:

 

STEP 1: FORGET GATE

Forget the additional information which might have entered in the immediate last step, and maintain the long term information required. This is why it is called Forget Gate:

![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-4/Images/LSTM3-focus-f.png)

STEP 2: INPUT GATE

Not let's decide what information we want to add based on the new-input. For this, we will use 1 DNN to predict all possible values, and other re-scores or filter-outs the values, like a manager, would. 

![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-4/Images/LSTM3-focus-i.png)

STEP 3: UPDATE THE MEMORY/CELL-STATE

Now let's forget whatever we might have added in the last step, and add new information through the input gate. 


![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-4/Images/LSTM3-focus-C.png)

STEP 4: THE OUTPUT

Till now we have updated our Memory/Cell-state. It is time to now provide the output. Our Memory is not our output, we need to filter things out. 


![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-4/Images/LSTM3-focus-o.png)

VARIANTS

Peephole: Now since you have gone through simple LSTM, we can create many different variants. Look at this one called Peephole LSTM. 


![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-4/Images/LSTM3-var-peepholes.png)

Coupled Forget-Input LSTM: Instead of separately deciding what to forget and what we should add, we can make those decisions together. We only forget when we're going to input something in it's place. Forget and Input becomes Ying-Yang!


![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-4/Images/LSTM3-var-tied.png)

GRU 

![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-4/Images/LSTM3-var-GRU.png)

A slightly more dramatic variant on the LSTM is the Gater Recurrent Unit or GRU.

    It combines the forget and input gates into a single "update gate".
    It also merges the cell state and hidden state
    It updates the memory twice, the first time (using old state and new input, called Reset Gate) and the second time (as final output). 
    Old cell state or hidden state (with input) is used for its own update as well as for deciding 

GRU has been growing increasingly popular and is a default alternative for LSTMs. 

 
ENCODER-DECODER ARCHITECTURE IN RNN/LSTM

The RNN encoder-decoder architecture looks like this [https://medium.com/datadriveninvestor/attention-in-rnns-321fbcd64f05]

![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-4/Images/1 iK8Wel75Ri55rSZfwAKHCA.jpeg)

The RNN encoder has an input sequence x1, x2, x3, x4. We denote the encoder states by c1, c2, c3. The encoder outputs a single output vector c which is passed as input to the decoder. Like the encoder, the decoder is also a single-layered RNN, we denote the decoder states by s1, s2, s3 and the network’s output by y1, y2, y3, y4.

A problem with this architecture lies in the fact that the decoder needs to represent the entire input sequence x1, x2, x3, x4 as a single vector c, which can cause information loss. Moreover, the decoder needs to decipher the passed information from this single vector, a complex task in itself.

RNN/LSTM WITH AN ATTENTION MECHANISM

An attention RNN/LSTM looks like this:

![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-4/Images/1 wnXVyE8LXPfODvB_Z5vu8A.jpeg)

Our attention model has a single RNN encoder, again with 4-time steps. We denote the encoder's input vectors by LaTeX: x_1,\:x_2,\:x_3,\:x_4 x 1 , x 2 , x 3 , x 4 and the output vectors by LaTeX: h_1,\:h_2,\:h_3,\:h_4 h 1 , h 2 , h 3 , h 4 . 

The attention mechanism is located between the encoder and the decoder. 

Its output is composed of the encoder's input vectors LaTeX: h_1,\:h_2,\:h_3,\:h_4 h 1 , h 2 , h 3 , h 4 and the states of the decoder LaTeX: s_0,\:s_1,\:s_2,\:s_3 s 0 , s 1 , s 2 , s 3 , the attention's output is a sequence of vectors called context vectors denoted by LaTeX: c_1,\:c_2,c_3,c_4 c 1 , c 2 , c 3 , c 4 .

 
 
THE CONTEXT VECTOR


The context vectors enable the decoder to focus on certain parts of the input when predicting its output. 
 

Each context vector is a weighted sum of the encoder's output vector LaTeX: h_1,\:h_2,\:h_3,\:h_4 h 1 , h 2 , h 3 , h 4

 
Each vector LaTeX: h_i h i contains information about the whole input sequence up to that moment with a strong focus on the LaTeX: i^{th} i t h stage. 


The vectors LaTeX: h_1,\:h_2,\:h_3,\:h_4 h 1 , h 2 , h 3 , h 4 are scaled by weights LaTeX: \alpha_{ij} α i j capturing the degree of relevance of input.
 

The context vectors LaTeX: c_1,\:c_2,\:c_3,\:c_4 c 1 , c 2 , c 3 , c 4 are given by


![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-4/Images/SVG2.png)


The attention weights are learned using an additional fully connected shallow network, denoted by fc, this is where the LaTeX: s_0,\:s_1,\:s_2,\:s_3 s 0 , s 1 , s 2 , s 3 part of the attention mechanism's input comes into play. Computation of the attention weights are given by:

![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-4/Images/SVG3.png)

The attention weights are learned using the attention fully-connected network and a softmax function:


![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-4/Images/1 wxv56cPyJdrEFSkknrlP-A.jpeg)


As can be seen in the image above, the fc receives the concatenated vectors LaTeX: \left[s_{i-1},\:h_i\right] [ s i − 1 , h i ] as the input at time step i. The network has a single fc layer, the outputs of the layer are passed through a softmax function computing the attention weights

 
Notice that we are using the same fully-connected network for all the concatenated pairs [s0,h1],[s1,h2],[s2,h3],[s3,h4]

![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-4/Images/1 jRBjCcGSoVL-rDb_zBXyPQ.jpeg)


The fc network is trained along with the encoder and decoder using backpropagation, the RNN's prediction error terms are backpropagated backward through the decoder, then through the fc attention network and from there to the encoder.


By letting the decoder have an attention mechanism, we relieve the encoder from having to encode all information in the input sequence into a single vector. 


COMPUTING THE ATTENTION WEIGHTS AND CONTEXT VECTORS

Let's go over a detailed example:


The first act performed is the computation of vectors LaTeX: h_1,\:h_2,\:h_3,\:h_4 h 1 , h 2 , h 3 , h 4 by the encoder. 
 

These are then used as inputs of the attention mechanism. This is where the decoder is first involved by inputting its initial state vector LaTeX: s_0 s 0 and we have the first attention input sequence  [s0,h1],[s0,h2],[s0,h3],[s0,h4]

![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-4/Images/1 IT-_Z0arAHdRnbf4T-BUKw.jpeg)

The attention mechanism computes the first set of attention weights enabling the computation of the first context vector LaTeX: c_1 c 1 . The decoder now uses LaTeX: \left[s_0,\:c_1\right] [ s 0 , c 1 ] and computes the first RNN output LaTeX: y_1 y 1 .


![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-4/Images/1 52xHMRpOX_88hhrtQ70zPw.jpeg)

At the following step, the attention mechanism has as input the sequence  [s1,h1],[s1,h2],[s1,h3],[s1,h4]

![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-4/Images/1 tXchCn0hBSUau3WO0ViD7w.jpeg)


It computes a second set of weights enabling computation of the second context vector LaTeX: c_2 c 2 . The decoder uses LaTeX: \left[s_1,\:c_2\right] [ s 1 , c 2 ] and computes the second RNN output LaTeX: y_2 y 2 .

![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-4/Images/1 Tb0Rxi7IBsl1eJ0rl8700w.jpeg)

This next step should be clear:

![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-4/Images/1 dCe0Asb9p-kad84eHvvIeg.jpeg)

then this:

![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-4/Images/1 or93mZa0-8Wxd9hQLg_0oA.jpeg)

Then this: 

![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-4/Images/1 ygL5NkL3cTHy16Wmi7Sd6g.jpeg)

and finally this:

![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-4/Images/1 O3fVgfNEYGTLLFid2DsFhg.jpeg)

In the end we have consumed 4 vectors, and with attention focused on specific vector, we should and then made the prediction for each step. 


Below are two alignments found by the attention RNN. The x-axis and y-axis of each plot correspond to the words in the source sentence (English) and the generated translation (French), respectively.
Each pixel shows the weight αij of the j-th source word and the i-th target word, in grayscale (0: black, 1: white).

 

![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-4/Images/1 UGr73oPi6f0_OfUIHLsACA.jpeg)

Better illustrations can be found here: https://distill.pub/2016/augmented-rnns/

This was the end of DNN for us. Now we'd be moving to Reinforcement Learning. Early few lectures are on basics so we have time to learn PyTorch. 

![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-4/Images/1 Tb0Rxi7IBsl1eJ0rl8700w.jpeg)

![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-4/Images/1 Tb0Rxi7IBsl1eJ0rl8700w.jpeg)

![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-4/Images/1 Tb0Rxi7IBsl1eJ0rl8700w.jpeg)

![](https://github.com/nandwanatarun/EVA-DeepLearning/blob/Phase2_Session-4/Images/1 Tb0Rxi7IBsl1eJ0rl8700w.jpeg)
