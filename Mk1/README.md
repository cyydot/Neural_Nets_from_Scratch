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
     
  2. The "normalizer" function takes in the input data and restructures it so that every value
     is between 0 and 1. At the moment, this function isn't used within NN_1.py, only in the
     test files.
     
     * This is also the first appearance of the variable "input_samples", which stores the test
       samples for training. Each entry in input_samples is assumed to be formatted
       [xlist, ylist], where xlist is a list of input values, and ylist is the list of output
       values.

  3. The "sequential" function constructs the neural network. It has three input variables; the
     first is "input_layer", which takes in a list of values as the input layer. The second
     variable "weights" takes in a list of all weights used in the neural network. The third
     variable "net_structure" gives the structure of the network.

     * net structure is assumed to be of the format
             [["layer1", n1], ["layer2", n2], ..., ["layerN", nN]],
       where ["layer1", n1] is the hidden layer after the input layer, and ["layerN", nN] is the
       final layer. "layeri" will always be "linear", "relu", or "softmax".

  4. "loss" computes the loss function. Instead of input_layer, the loss function takes in
     input_samples.

  5. The "gradient" function computes the gradient of the loss function with respect to each
     weight. In addition to the same variables as the loss function, there are two new variables,
     "dx" and "stoc". Currently, the gradient is calculated by
             (f(x+dx) - f(x-dx)) / (2 * dx),
     so dx determines the precision of this calculation. stoc enables stochastic gradient descent,
     where setting stoc to an integer value less than or equal the number of input samples causes
     the gradient to be calculated only stoc many input samples, which are randomly sampled without
     replacement from input_samples.

  6. This section contains the "gradient_descent" function, which performs the gradient descent.
     The variables "w_start" and "w_stop" determine the range that weights will be initialized in.
     The variable "runs" determines how many times you will be performing a single gradient descent
     run, and "iters" determines the number of steps that will be taken in each gradient descent run.
     "learning_rate" is the learning rate of the gradient descent. The output of this function is
     a list of weights.

  7. The "NN" function takes in the network structure and the optimized weights returned from
     gradient_descent and returns a fitted function that will predict values.
     

test_1.x.py 
  These files contain two example neural networks. Both are predicting a binary qualitative output.
  test_1.1 creates a single layer network, and test_1.2 contains a network with 2 layers.

data_1.x.csv
  These csv files contain the data used in the test files. This data was retrieved from the following
  online source: https://cs.colby.edu/courses/F22/cs343/projects/p1adaline/p1adaline.html .



# Intended Use

Here is the intended process to use this architecture.

  1. Fit training data into a list of the form
       data = [[sampleinput1, sampleprediction1], ..., [sampleinputN, samplepredictionN]].

    * if data isn't normalized, run the normalizer:
      nordata = NN.normalizer(copy.deepcopy(data))
       
     
  2. Describe network topology with list of the form
       net = ["layer1", n1], ["layer2", n2], ..., ["layerN", nN]].
     
  3. find optimized weights by running gradient_descent:
       weights = gradient_descent(data, net, 10,50, -1, 1, 0.001, 0.001, len(data))

    *  weights = gradient_descent(nordata[0], net, 10,50, -1, 1, 0.001, 0.001, len(nordata[0]))

  5. create the neural network:
       model = NN.NN(weights, net)

  6. predict!
       model(x)

    *  model(NN.input_normalizer(x, nordata[1]))


# Future Updates

There are many aspects of this current architecture that can be improved or implemented. Here is what
I am planning on adding in the future.

  1. Analytic Gradient - Currently, the gradient is calculated numerically, requiring two computations
     of the loss function. Implementing an analytic gradient designed similarly to the sequential
     function would improve accuracy of the calculation, and may also improve the speed of the program.

  2. Gradient Descent with Momentum / Regularization

  3. Improved Weight Initialization - Currently, weights are all initialized within the same
     range, which isn't ideal for larger networks. Implementing more advanced initialization techniques,
     such as Glorot, He, or LeCun initialization, may improve learning and speed.

  4. Add Convolution and Pooling Layers
  

