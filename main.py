# Copyright 2014 Matthieu Courbariaux

# This file is part of deep-learning-storage.

# deep-learning-storage is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# deep-learning-storage is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with deep-learning-storage.  If not, see <http://www.gnu.org/licenses/>.

import gzip
import cPickle
import numpy as np
import os
import os.path
import sys
import time

from trainer import Trainer
from model import PI_MNIST_model

from pylearn2.datasets.mnist import MNIST
from pylearn2.utils import serial

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from filter_plot import tile_raster_images
          
def onehot(x,numclasses=None):

    if x.shape==():
        x = x[None]
    if numclasses is None:
        numclasses = np.max(x) + 1
    result = np.zeros(list(x.shape) + [numclasses], dtype="int")
    z = np.zeros(x.shape)
    for c in range(numclasses):
        z *= 0
        z[np.where(x==c)] = 1
        result[...,c] += z

    result = np.reshape(result,(np.shape(result)[0], np.shape(result)[result.ndim-1]))
    return result
       
# MAIN

# stochastic learning rate works with continuous weights
# multilayer does not work so far 
# discrete weights + continuous bias do work for one layer
# batch and activations -> only update the most wrong weight

if __name__ == "__main__":
       
    print 'Beginning of the program'
    start_time = time.clock()   
    
    print 'Loading the dataset' 
    
            
    train_set = MNIST(which_set= 'train', start=0, stop = 50000, center = True)
    valid_set = MNIST(which_set= 'train', start=50000, stop = 60000, center = True)
    test_set = MNIST(which_set= 'test', center = True)
    
    # for both datasets, onehot the target
    train_set.y = np.float32(onehot(train_set.y))
    valid_set.y = np.float32(onehot(valid_set.y))
    test_set.y = np.float32(onehot(test_set.y))
    
    train_set.y = 2* train_set.y - 1.
    valid_set.y = 2* valid_set.y - 1.
    test_set.y = 2* test_set.y - 1.
    
    # print train_set.X
    # print np.shape(train_set.X)
    # print np.max(train_set.X)
    # print np.min(train_set.X)
        
    print 'Creating the model'

    rng = np.random.RandomState(1234)
    batch_size = 1000
    LR = .3
    gpu_batches = 50000/batch_size
    n_epoch = 500
    monitor_step = 10
    
    model = PI_MNIST_model(rng = rng)
    
    trainer = Trainer(rng = rng,
        train_set = train_set, valid_set = valid_set, test_set = test_set,
        model = model,
        LR = LR, LR_decay = 0.99, LR_fin = LR/10000.,
        batch_size = batch_size, gpu_batches = gpu_batches,
        n_epoch = n_epoch, monitor_step = monitor_step,
        shuffle_batches = False, shuffle_examples = True)

    print 'Building'
    
    trainer.build()
    
    print 'Training'
    
    trainer.train()
    
    print 'Display weights'
    
    W = np.transpose(model.layer[0].W.get_value())
    print np.max((W==0.))
    W = tile_raster_images(W,(28,28),(10,10),(2, 2))
    plt.imshow(W, cmap = cm.Greys_r)
    plt.show()

    end_time = time.clock()
    print 'The code ran for %i seconds'%(end_time - start_time)
    