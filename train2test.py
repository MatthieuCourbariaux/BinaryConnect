
import sys
import os
import time

import numpy as np
np.random.seed(1234)  # for reproducibility

from collections import OrderedDict

if __name__ == "__main__":

    # Loading all parameters
    with np.load('best_model.npz') as f:
        train_parameters = [f['arr_%d' % i] for i in range(len(f.files))]
    
    # print(np.shape(train_parameters))
    
    # [W, b, mean, std, beta, gamma, W, b, mean, std, beta, gamma, W, b, mean, std, beta, gamma, W, b, mean, std, beta, gamma]
    test_parameters = []
    for i in range(0,len(train_parameters),6):
        W = train_parameters[i]
        mean = train_parameters[i+2]
        std = train_parameters[i+3]
        beta = train_parameters[i+4]
        gamma = train_parameters[i+5]
        
        Wb = np.bool_((np.sign(W)+1)/2)
        a = gamma / (std+ 1e-4)
        b = beta - a * mean
        test_parameters.append([Wb,a,b])

    # print(np.shape(test_parameters[0][0]))
    # print(np.dtype(test_parameters[0][0][0][0]))
    # print(test_parameters[0][0][0])
    # print(test_parameters[0][1][0])
    # print(test_parameters[0][2][0])
    
    np.savez('parameters.npz', *test_parameters)
    
    # with np.load('test_parameters.npz') as f:
        # test_parameters = [f['arr_%d' % i] for i in range(len(f.files))]
        
    # print(np.shape(test_parameters[0][0]))
    # print(np.dtype(test_parameters[0][0][0][0]))
    # print(test_parameters[0][0][0])
    # print(test_parameters[0][1][0])
    # print(test_parameters[0][2][0])
    