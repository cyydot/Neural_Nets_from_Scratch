import numpy as np
import NN_1 as NN
import copy.deepcopy as deepcopy


data = np.loadtxt('data_1.1.csv', delimiter=",", dtype=float)
org = []
for sample in data:
    x = [sample[1], sample[2]]
    y = [1,0] if sample[3] == -1.0 else [0,1]
    org.append([x,y])
orgnor = NN.normalizer(deepcopy(org))


net = [["relu", 2], ["softmax", 2]]
weights = NN.gradient_descent(orgnor[0][:-72], net, 10, 10, -10, 10, 0.001, 0.001, 200)
model = NN.NN(weights, net, orgnor[1])


perc = 0
for point in org[-72:]:
    
    probs = model(point[0])
    print("x = " + str(point[0]))
    print("y' = " + str(probs))
    
    if probs[0] < probs[1]: y = [0,1]
    else: y = [1,0]

    print("y' adjusted = " + str(y))
    print("y = " + str(point[1]))

    if y == point[1]: perc = perc + 1

print("accuracy = " + str(perc / 72))





"""Pretrained Models

accuracy = 0.9861111111111112
net = [["linear", 2], ["softmax", 2]]
weights = NN.gradient_descent(10, 10, -10, 10, org[0][:-72], net, 0.001, 0.001, 50)

[np.float64(5.8908009300209585), np.float64(-4.06201549227101), np.float64(-9.512581538031814), np.float64(-0.3056309762959583), 
np.float64(-9.330762905245557), np.float64(3.6184398326409086), np.float64(8.703348761707087), np.float64(2.3682762975934115), 
np.float64(4.00877634953945), np.float64(-9.495737341382618), np.float64(-4.145028210885875), np.float64(-0.6570601208916085)]




accuracy: 1.0
net = [["relu", 2], ["softmax", 2]]
weights = NN.gradient_descent(10, 10, -10, 10, org[0][:-72], net, 0.001, 0.001, 200)

[np.float64(3.3793836518328635), np.float64(-5.4338897218253885), np.float64(0.6803095156807648), np.float64(0.4063355828538171), 
np.float64(0.9854898521350033), np.float64(-6.025156110437971), np.float64(-1.6714019007858045), np.float64(7.681917648874414), 
np.float64(6.921714334577038), np.float64(9.19635370905565), np.float64(-1.3932191128023133), np.float64(8.672140795595949)]



"""
