""" prob1.py 
Ryan George
Deep Learning CPSC 663"

import mnist_loader as ml
import network

training_data, validation_data, test_data =  ml.load_data_wrapper()

nw = network.Network([784, 10])
net.SGD(training_data, 30, 10, 3.0, test_data=test_data)

