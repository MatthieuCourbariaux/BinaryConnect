
import sys
import os
import time

import numpy as np
import cPickle, gzip
from collections import OrderedDict

if __name__ == "__main__":
        
    # Loading the MNIST test set
    # You can get mnist.pkl.gz at http://deeplearning.net/data/mnist/mnist.pkl.gz
    f = gzip.open('mnist.pkl.gz', 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    x = test_set[0]
    t = test_set[1]
    # mean centering the dataset
    x = x - np.mean(x, axis=0)
    
    # Loading all parameters
    with np.load('parameters.npz') as f:
        parameters = [f['arr_%d' % i] for i in range(len(f.files))]

    # Computing the neural network output
    n_layers = len(parameters)
    for i in range(n_layers):
        
        # W is stored in a bool 2D array
        W = 2.* parameters[i][0] - 1.
        # a and b are stored as floats
        a = parameters[i][1]
        b = parameters[i][2]
        
        # scaled weighted sum
        y = a * np.dot(x,W) + b
        
        # ReLU activation function (except for the ouptut layer)
        if i < (n_layers - 1):
            y = np.maximum(0.,y)
        
        # the output of a layer is the input of the next layer
        x = y
            
    # Computing the test error rate
    y = np.argmax(y,axis=1)
    error_rate = np.mean(y!=t)*100
    print("Test set error rate: "+str(error_rate)+"%")
    