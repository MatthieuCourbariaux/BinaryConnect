# Copyright 2015 Matthieu Courbariaux

# This file is part of BinaryConnect.

# BinaryConnect is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# BinaryConnect is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with BinaryConnect.  If not, see <http://www.gnu.org/licenses/>.

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
    