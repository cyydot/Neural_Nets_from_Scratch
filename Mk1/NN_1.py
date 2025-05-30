import numpy as np
import random

def linear(input, weights, layer):
    weights = np.array(weights)
    xx = np.append([1],input)
    row_len = len(xx)
    y = np.matmul(weights.reshape(layer, row_len), xx.reshape(row_len, 1))
    return y.flatten()

def relu(input, weights, layer):
    return [x if x > 0 else 0 for x in linear(input, weights, layer)]   

def softmax(input, weights, layer):
    y = linear(input, weights, layer)
    return np.exp(y)/sum(np.exp(y))

layers = {"relu":relu, "linear":linear, "softmax":softmax}
    
def least_squares(x_1, x_2): return (x_1 - x_2) * (x_1 - x_2)

def log_like(x_1, x_2): return x_1 * np.log(x_2) * -1

###########################################################################################################
###########################################################################################################

def normalizer(input_samples):
    num_samples = len(input_samples)
    data = [x[0] for x in input_samples]
    mnds = []
    for i in range(len(input_samples[0][0])):
        vals = [data[j][i] for j in range(num_samples)]
        minn = min(vals)
        dif = max(vals) - minn
        mnds.append([minn, dif])
        for j in range(num_samples):
            input_samples[j][0][i] = (input_samples[j][0][i] - minn) / dif
    
    return [input_samples, mnds]

###########################################################################################################
###########################################################################################################

def sequential(input_layer, weights, net_structure):
    
    layer = len(net_structure)

    if layer == 0:
        return input_layer 
    elif layer == 1:
        nodes_prev = len(input_layer)
        nodes = net_structure[0][1]   
    else:
        nodes_prev = net_structure[-2][1]
        nodes = net_structure[-1][1]

    y_prev = sequential(input_layer, weights[:-(nodes_prev + 1) * nodes], net_structure[:-1])
    weights_here = weights[-(nodes_prev + 1) * nodes:]
    return layers[net_structure[-1][0]](y_prev, weights_here, nodes)

###########################################################################################################
###########################################################################################################

def loss(input_samples, weights, net_structure,):  
    if net_structure[-1][0] == 'softmax':
        out = log_like
    else:
        out = least_squares

    parts = []

    for i in input_samples:
        y = sequential(i[0], weights, net_structure)
        for j in range(len(y)): parts.append(out(i[1][j], y[j]))   
    
    return sum(parts)

###########################################################################################################
###########################################################################################################

def gradient(input_samples, weights, net_structure, dx, stoc):

    if stoc != len(input_samples):
        input_samples = random.sample(input_samples, stoc)

    grad = []
    num_weights = len(weights)

    for i in range(num_weights):
        y_b = loss(input_samples, [weights[j] if j != i else weights[j] - dx for j in range(num_weights)], net_structure)
        y_f = loss(input_samples, [weights[j] if j != i else weights[j] + dx for j in range(num_weights)], net_structure)
        main_val = (y_f - y_b) / (2*dx) 
        grad.append(main_val)

    return grad

###########################################################################################################
###########################################################################################################

def gradient_descent(input_samples, net_structure, runs, iters, w_start, w_stop, dx, learning_rate, stoc):
    nweights = 0
    prev = len(input_samples[0][0])
    for layer in net_structure:
        nweights = nweights + (prev + 1)*layer[1]
        prev = layer[1]


    values = (w_start - w_stop) * np.random.rand(nweights) + w_stop
    f_min = loss(input_samples, values, net_structure)

    print("fmin start = " + str(f_min))
    for i in range(runs):
        print("run = " + str(i + 1))
        
        x = (w_start - w_stop) * np.random.rand(nweights) + w_stop 

        y = loss(input_samples, x, net_structure)
        print("y = " + str(y))
        grad = gradient(input_samples, x, net_structure, dx, stoc)

        fail = 0
        for j in range(iters):
            print("iter = " + str(j + 1))
            
            x_1 = [x[l] - (grad[l] * learning_rate) for l in range(len(x))]
            y_1 = loss(input_samples, x_1, net_structure)
            
            print("y1 = " + str(y_1))
            
            if (y_1 >= y):
                fail = fail + 1
                if fail == 3: print("break"); break
            else: 
                fail = 0
                y = y_1
            
            x = x_1
            grad = gradient(input_samples, x, net_structure, dx, stoc)
            
        if y < f_min:
            f_min = y
            values = x
            print("new fmin = " + str(f_min))

    print("Complete.")
    print(values)
    return values

###########################################################################################################
###########################################################################################################

def NN(weights, net_structure):
    def actual(x): return sequential(x, weights, net_structure)
    return actual
