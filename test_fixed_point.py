
import sys
import os
import time

import numpy as np
import cPickle, gzip
from collections import OrderedDict

# Sign
SIGN = True
# NOB = total number of bits, sign included
NOB = 12
# NOIB = number of (integer) bits, positioned before the (fixed) point
NOIB = 6
    
def fixed_point(x):
    
    power = 2.**(NOB - SIGN - NOIB)
    max = (2.**(NOB - SIGN))-1
    value = x*power  

    # rounding or flooring
    value = np.round(value)
    # value = np.floor(value)
    
    # Saturation arithmetic
    # You may want to try out clockwork arithmetic instead
    if SIGN == True:
        value = np.clip(value, -max, max) 
    else:
        value = np.clip(value, 0, max)
        
    value = value/power
    return value 

if __name__ == "__main__":
        
    # Loading the MNIST test set
    # You can get mnist.pkl.gz at http://deeplearning.net/data/mnist/mnist.pkl.gz
    f = gzip.open('mnist.pkl.gz', 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    x = test_set[0]
    t = test_set[1]
    
    x = fixed_point(x)

    # Loading all parameters
    with np.load('parameters.npz') as f:
        parameters = [f['arr_%d' % i] for i in range(len(f.files))]

    # Computing the neural network output
    n_layers = len(parameters)
    for i in range(n_layers):
        
        # W is stored in a bool 2D array
        W = 2.* parameters[i][0] - 1.
        
        a = fixed_point(parameters[i][1])
        b = fixed_point(parameters[i][2]/(a +1e-9))
        
        # scaled weighted sum
        y = fixed_point(np.dot(x,W) + b) # We assume there is sufficient depth to this adder to ignore other clippings.
        
        # ReLU activation function (except for the ouptut layer)
        if i < (n_layers - 1):
            y = fixed_point(np.maximum(0.,y*a)) # I put the renormalization over here to avoid having a multiplier in each 'core'.
        
        # the output of a layer is the input of the next layer
        x = y
            
    # Computing the test error rate
    y = np.argmax(y,axis=1)
    error_rate = np.mean(y!=t)*100
    print("Test set error rate: "+str(error_rate)+"%")
    
