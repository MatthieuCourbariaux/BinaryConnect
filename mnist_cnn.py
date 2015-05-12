# Copyright 2014 Matthieu Courbariaux

# This file is part of deep-learning-discrete.

# deep-learning-discrete is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# deep-learning-discrete is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with deep-learning-discrete.  If not, see <http://www.gnu.org/licenses/>.

import gzip
import cPickle
import numpy as np
import os
import os.path
import sys
import time

from trainer import Trainer
from model import Network
from layer import linear_layer, ReLU_layer, ReLU_conv_layer  

from pylearn2.datasets.mnist import MNIST
from pylearn2.utils import serial
from pylearn2.train_extensions.window_flip import _zero_pad

# import matplotlib.pyplot as plt
# import matplotlib.cm as cm
# from filter_plot import tile_raster_images
          
def onehot(x,numclasses=None):

    if x.shape==():
        x = x[None]
    if numclasses is None:
        numclasses = np.max(x) + 1
    result = np.zeros(list(x.shape) + [numclasses], dtype="int")
    z = np.zeros(x.shape, dtype="int")
    for c in range(numclasses):
        z *= 0
        z[np.where(x==c)] = 1
        result[...,c] += z

    result = np.reshape(result,(np.shape(result)[0], np.shape(result)[result.ndim-1]))
    return result
       
# MAIN

if __name__ == "__main__":
          
    print 'Loading the dataset' 
    
    train_set = MNIST(which_set= 'train', start=0, stop = 50000, center = True)
    # train_set = MNIST(which_set= 'train', start=0, stop = 128, center = True) # for testing data augmentation
    valid_set = MNIST(which_set= 'train', start=50000, stop = 60000, center = True)
    test_set = MNIST(which_set= 'test', center = True)
    
    # bc01 format
    train_set.X = train_set.X.reshape(50000,1,28,28)
    valid_set.X = valid_set.X.reshape(10000,1,28,28)
    test_set.X = test_set.X.reshape(10000,1,28,28)
    
    # zero padding, cost little may help Data Augmentation
    # train_set.X = _zero_pad(array=train_set.X, amount=2, axes=(2, 3))
    # valid_set.X = _zero_pad(array=valid_set.X, amount=2, axes=(2, 3))
    # test_set.X = _zero_pad(array=test_set.X, amount=2, axes=(2, 3))
    
    # Onehot the targets
    train_set.y = np.float32(onehot(train_set.y))
    valid_set.y = np.float32(onehot(valid_set.y))
    test_set.y = np.float32(onehot(test_set.y))
    
    # for hinge loss
    train_set.y = 2* train_set.y - 1.
    valid_set.y = 2* valid_set.y - 1.
    test_set.y = 2* test_set.y - 1.
    
    # print train_set.X
    # print np.shape(train_set.X)
    # print np.max(train_set.X)
    # print np.min(train_set.X)
        
    print 'Creating the model'
    
    rng = np.random.RandomState(1234)
    batch_size = 64
    # batch_size = 4096
    
    class MNIST_model(Network):

        def __init__(self, rng):
            
            prop_bit_width = 1
            prop_stochastic_rounding = False # stochastic rounding does not seem to work at all for propagations
            update_bit_width = None
            # update_bit_width = 1
            update_stochastic_rounding = True
            # BN = True
            BN = False
            
            Network.__init__(self, n_hidden_layer = 4) 
            
            print "    Convolution layer 1:"
        
            self.layer.append(ReLU_conv_layer(
                rng,
                image_shape=(batch_size, 1, 28, 28),
                # image_shape=(batch_size, 1, 32, 32),
                zero_pad = 0, # add n zeros on both sides of the input
                # filter_shape=(32, 1, 12, 12),
                filter_shape=(32, 1, 8, 8),
                filter_stride = 1,
                pool_shape=(4, 4),
                pool_stride = 2,
                output_shape = (batch_size, 32, 10, 10),
                partial_sum=1, 
                BN=BN,
                prop_bit_width=prop_bit_width, 
                prop_stochastic_rounding=prop_stochastic_rounding,
                update_bit_width=update_bit_width, 
                update_stochastic_rounding=update_stochastic_rounding
            ))
            
            print "    NiN layer:"
        
            self.layer.append(ReLU_conv_layer(
                rng,
                image_shape=(batch_size, 32, 10, 10),
                zero_pad = 0,
                filter_shape=(16, 32, 1, 1),
                filter_stride = 1,
                pool_shape=(1, 1),
                pool_stride = 1,
                output_shape = (batch_size, 16, 10, 10),
                partial_sum=1, 
                BN=BN,
                prop_bit_width=prop_bit_width, 
                prop_stochastic_rounding=prop_stochastic_rounding,
                update_bit_width=update_bit_width, 
                update_stochastic_rounding=update_stochastic_rounding
            ))

            print "    Convolution layer 2:"
        
            self.layer.append(ReLU_conv_layer(
                rng,
                image_shape=(batch_size, 16, 10, 10),
                zero_pad = 3, 
                filter_shape=(16, 16, 8, 8),
                filter_stride = 1,
                pool_shape=(4, 4),
                pool_stride =2,
                output_shape = (batch_size, 16, 4, 4),
                partial_sum=1, 
                BN=BN,
                prop_bit_width=prop_bit_width, 
                prop_stochastic_rounding=prop_stochastic_rounding,
                update_bit_width=update_bit_width, 
                update_stochastic_rounding=update_stochastic_rounding
            ))
            
            print "    Convolution layer 3:"
        
            self.layer.append(ReLU_conv_layer(
                rng,
                image_shape=(batch_size, 16, 4, 4),
                zero_pad = 3,
                filter_shape=(16, 16, 5, 5),
                filter_stride = 1,
                pool_shape=(2, 2),
                pool_stride =2,
                output_shape = (batch_size, 16, 3, 3),
                partial_sum=1, 
                BN=BN,
                prop_bit_width=prop_bit_width, 
                prop_stochastic_rounding=prop_stochastic_rounding,
                update_bit_width=update_bit_width, 
                update_stochastic_rounding=update_stochastic_rounding
            ))
            
            print "    L2 SVM layer:"
            
            self.layer.append(linear_layer(
                rng = rng, 
                n_inputs= 16*3*3, 
                n_units = 10, 
                BN=BN,
                prop_bit_width=prop_bit_width, 
                prop_stochastic_rounding=prop_stochastic_rounding,
                update_bit_width=update_bit_width, 
                update_stochastic_rounding=update_stochastic_rounding
            ))
            
    model = MNIST_model(rng = rng)
    
    print 'Creating the trainer'
    
    LR = .01
    gpu_batches = 50000/batch_size
    
    n_epoch = 1000
    monitor_step = 2
    LR_decay = .99
    
    # n_epoch = 5000
    # monitor_step = 1000
    # LR_decay = .9995
    
    trainer = Trainer(rng = rng,
        train_set = train_set, valid_set = valid_set, test_set = test_set,
        model = model,
        LR = LR, LR_decay = LR_decay, LR_fin = LR/10000.,
        batch_size = batch_size, gpu_batches = gpu_batches,
        n_epoch = n_epoch, monitor_step = monitor_step,
        shuffle_batches = False, shuffle_examples = True)

    print 'Building'
    
    trainer.build()
    
    print 'Training'
    
    start_time = time.clock()  
    trainer.train()
    end_time = time.clock()
    print 'The training took %i seconds'%(end_time - start_time)
    
    # print 'Display weights'
    
    # W = 2.* (np.transpose(model.layer[0].W.get_value())>=0.) - 1.
    # W = tile_raster_images(W,(28,28),(5,5),(2, 2))
    # plt.imshow(W, cmap = cm.Greys_r)
    # plt.show()
