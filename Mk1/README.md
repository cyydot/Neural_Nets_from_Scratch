# MK1

This contains my very first attempt at constructing neural network architecture from scratch.
Here's whats in this folder:

NN_1.py
  This Python script contains code for the network architecture. It is designed to compute 
  feed forward, fully connected neural networks. It's sections are as follows:

  1. This section contains the imports (numpy and random), code for the network layers, and
     code for computing the loss function. So far there are only three layer types: "linear",
     which performs matrix multiplcaton with no activation function, "relu", which performs a
     matrix multiplication followed by a ReLU activation, and "softmax", which has softmax as
     its activation function. Least Squares and Log-likelihood are used as loss functions.
     
  2. The normalizer function takes in the input data and restructures it so that every value
     is between 0 and 1. At the moment, this function isn't used within NN_1.py, only in the
     test files.
     
     * This is also the first appearance of the variable input_samples, which stores the test
       samples for training. It is assumed to be formatted [xlist, ylist], where xlist is a
       list of input values, and ylist is the list of output values.

  3. 







test_1.x.py - tests the architecture by building simple neural networks and calculating prediction

