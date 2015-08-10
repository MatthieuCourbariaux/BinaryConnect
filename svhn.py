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
# from layer import linear_layer, Maxout_layer, Maxout_conv_layer  

# from pylearn2.datasets.mnist import MNIST
# from pylearn2.datasets.zca_dataset import ZCA_Dataset    
from pylearn2.datasets.svhn import SVHN
from pylearn2.utils import serial
       
# MAIN

if __name__ == "__main__":
    
    rng = np.random.RandomState(1234)
    
    # data augmentation
    zero_pad = 0
    affine_transform_a = 0
    affine_transform_b = 0
    horizontal_flip = False
    
    # batch
    # keep a factor of 10000 if possible
    # 10000 = (2*5)^4
    batch_size = 100
    number_of_batches_on_gpu = 50000/batch_size
    BN = True
    BN_epsilon=1e-4 # for numerical stability
    BN_fast_eval= True
    dropout_hidden = 1.
    shuffle_examples = False
    shuffle_batches = True

    # Termination criteria
    n_epoch = 60
    monitor_step = 1 
    # core_path = "cnn_exp/" + str(sys.argv)
    load_path = None    
    # load_path = core_path + ".pkl"
    save_path = None
    # save_path = core_path + ".pkl"
    
    # LR 
    LR = .3
    LR_fin = .001
    LR_decay = (LR_fin/LR)**(1./n_epoch)    
    M= 0.
    
    # BinaryConnect
    BinaryConnect = True
    stochastic = True
    
    # Old hyperparameters
    binary_training=False 
    stochastic_training=False
    binary_test=False
    stochastic_test=False
    if BinaryConnect == True:
        binary_training=True      
        if stochastic == True:   
            stochastic_training=True  
        else:
            binary_test=True
    
    print 'Loading the dataset' 
    
    train_set = SVHN(
            which_set= 'splitted_train',
            path= "${SVHN_LOCAL_PATH}",
            axes= ['b', 'c', 0, 1])
     
    valid_set = SVHN(
        which_set= 'valid',
        path= "${SVHN_LOCAL_PATH}",
        axes= ['b', 'c', 0, 1])
    
    test_set = SVHN(
        which_set= 'test',
        path= "${SVHN_LOCAL_PATH}",
        axes= ['b', 'c', 0, 1])
    
    # bc01 format
    # print train_set.X.shape
    train_set.X = np.reshape(train_set.X,(598388,3,32,32))
    valid_set.X = np.reshape(valid_set.X,(6000,3,32,32))
    test_set.X = np.reshape(test_set.X,(26032,3,32,32))
    
    # for hinge loss
    train_set.y = np.subtract(np.multiply(2,train_set.y),1.)
    valid_set.y = np.subtract(np.multiply(2,valid_set.y),1.)
    test_set.y = np.subtract(np.multiply(2,test_set.y),1.)
    
    print 'Creating the model'
    
    class DeepCNN(Network):

        def __init__(self, rng):

            Network.__init__(self, n_hidden_layer = 8, BN = BN)
            
            print "    C3 layer:"
                
            self.layer.append(ReLU_conv_layer(
                rng,
                filter_shape=(128, 3, 3, 3),
                pool_shape=(1,1),
                pool_stride=(1,1),
                BN = BN,                     
                BN_epsilon=BN_epsilon,
                binary_training=binary_training, 
                stochastic_training=stochastic_training,
                binary_test=binary_test, 
                stochastic_test=stochastic_test
            ))
            
            print "    C3 P2 layers:"
                
            self.layer.append(ReLU_conv_layer(
                rng,
                filter_shape=(128, 128, 3, 3),
                pool_shape=(2,2),
                pool_stride=(2,2),
                BN = BN,                     
                BN_epsilon=BN_epsilon,
                binary_training=binary_training, 
                stochastic_training=stochastic_training,
                binary_test=binary_test, 
                stochastic_test=stochastic_test
            ))
            
            print "    C2 layer:"
                
            self.layer.append(ReLU_conv_layer(
                rng,
                filter_shape=(256, 128, 2, 2),
                pool_shape=(1,1),
                pool_stride=(1,1),
                BN = BN,                     
                BN_epsilon=BN_epsilon,
                binary_training=binary_training, 
                stochastic_training=stochastic_training,
                binary_test=binary_test, 
                stochastic_test=stochastic_test
            ))
            
            print "    C2 P2 layers:"
            
            self.layer.append(ReLU_conv_layer(
                rng,
                filter_shape=(256, 256, 2, 2),
                pool_shape=(2,2),
                pool_stride=(2,2),
                BN = BN,                     
                BN_epsilon=BN_epsilon,
                binary_training=binary_training, 
                stochastic_training=stochastic_training,
                binary_test=binary_test, 
                stochastic_test=stochastic_test
            ))
            
            print "    C2 layer:"
                
            self.layer.append(ReLU_conv_layer(
                rng,
                filter_shape=(512, 256, 2, 2),
                pool_shape=(1,1),
                pool_stride=(1,1),
                BN = BN,                     
                BN_epsilon=BN_epsilon,
                binary_training=binary_training, 
                stochastic_training=stochastic_training,
                binary_test=binary_test, 
                stochastic_test=stochastic_test
            ))
            
            print "    C2 P2 layers:"
            
            self.layer.append(ReLU_conv_layer(
                rng,
                filter_shape=(512, 512, 2, 2),
                pool_shape=(2,2),
                pool_stride=(2,2),
                BN = BN,                     
                BN_epsilon=BN_epsilon,
                binary_training=binary_training, 
                stochastic_training=stochastic_training,
                binary_test=binary_test, 
                stochastic_test=stochastic_test
            ))
            
            print "    C2 layer:"
                
            self.layer.append(ReLU_conv_layer(
                rng,
                filter_shape=(1024, 512, 2, 2),
                pool_shape=(1,1),
                pool_stride=(1,1),
                BN = BN,                     
                BN_epsilon=BN_epsilon,
                binary_training=binary_training, 
                stochastic_training=stochastic_training,
                binary_test=binary_test, 
                stochastic_test=stochastic_test
            ))
            
            print "    FC layer:"
            
            self.layer.append(ReLU_layer(
                    rng = rng, 
                    n_inputs = 1024, 
                    n_units = 1024, 
                    BN = BN, 
                    BN_epsilon=BN_epsilon, 
                    dropout=dropout_hidden, 
                    binary_training=binary_training, 
                    stochastic_training=stochastic_training,
                    binary_test=binary_test, 
                    stochastic_test=stochastic_test
            ))
            
            print "    L2 SVM layer:"
            
            self.layer.append(linear_layer(
                rng = rng, 
                n_inputs= 1024, 
                n_units = 10, 
                BN = BN,
                BN_epsilon=BN_epsilon,
                dropout = dropout_hidden,
                binary_training=binary_training, 
                stochastic_training=stochastic_training,
                binary_test=binary_test, 
                stochastic_test=stochastic_test
            ))
            
    model = DeepCNN(rng = rng)
    
    print 'Creating the trainer'
    
    trainer = Trainer(rng = rng,
        train_set = train_set, valid_set = valid_set, test_set = test_set,
        model = model, load_path = load_path, save_path = save_path,
        zero_pad=zero_pad,
        affine_transform_a=affine_transform_a, # a is (more or less) the rotations
        affine_transform_b=affine_transform_b, # b is the translations
        horizontal_flip=horizontal_flip,
        LR = LR, LR_decay = LR_decay, LR_fin = LR_fin,
        M = M,
        BN = BN, BN_fast_eval=BN_fast_eval,
        batch_size = batch_size, number_of_batches_on_gpu = number_of_batches_on_gpu,
        n_epoch = n_epoch, monitor_step = monitor_step,
        shuffle_batches = shuffle_batches, shuffle_examples = shuffle_examples)
    
    print 'Building'
    
    trainer.build()
    
    print 'Training'
    
    start_time = time.clock()  
    trainer.train()
    end_time = time.clock()
    print 'The training took %i seconds'%(end_time - start_time)
