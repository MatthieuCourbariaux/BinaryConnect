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
from pylearn2.datasets.zca_dataset import ZCA_Dataset    
# from pylearn2.datasets.svhn import SVHN
from pylearn2.utils import serial
          
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
    
    print 'Hyperparameters' 
    
    rng = np.random.RandomState(1234)
    # rng = np.random.RandomState(int(sys.argv[1]))
    
    # data augmentation
    zero_pad = 2
    affine_transform_a = 0
    affine_transform_b = 0
    horizontal_flip = False
    
    # batch
    # keep a factor of 10000 if possible
    # 10000 = (2*5)^4
    batch_size = 100
    # batch_size = int(sys.argv[1])
    number_of_batches_on_gpu = 45000/batch_size
    BN = True
    BN_epsilon=1e-4 # for numerical stability
    BN_fast_eval= True
    # dropout_input = .8
    # dropout_hidden = 1.
    dropout_hidden = .7
    # dropout_hidden = float(sys.argv[2])
    shuffle_examples = True
    shuffle_batches = False

    # Termination criteria
    n_epoch = 100
    # n_epoch = int(sys.argv[3])
    monitor_step = 2 
    # core_path = "cnn_exp/" + str(sys.argv)
    load_path = None    
    # load_path = core_path + ".pkl"
    save_path = None
    # save_path = core_path + ".pkl"
    # print save_path
    
    # LR 
    LR = .3
    # LR = float(sys.argv[4])
    LR_fin = .01
    # LR_fin = float(sys.argv[5])
    # LR_decay = 1. 
    LR_decay = (LR_fin/LR)**(1./n_epoch)    
    M= 0.
    
    # BinaryConnect
    BinaryConnect = False
    # BinaryConnect = int(sys.argv[7])
    stochastic = False
    # stochastic = int(sys.argv[8])
    
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
    
    preprocessor = serial.load("${PYLEARN2_DATA_PATH}/cifar10/pylearn2_gcn_whitened/preprocessor.pkl")
    train_set = ZCA_Dataset(
        preprocessed_dataset=serial.load("${PYLEARN2_DATA_PATH}/cifar10/pylearn2_gcn_whitened/train.pkl"), 
        preprocessor = preprocessor,
        start=0, stop = 45000)
    valid_set = ZCA_Dataset(
        preprocessed_dataset= serial.load("${PYLEARN2_DATA_PATH}/cifar10/pylearn2_gcn_whitened/train.pkl"), 
        preprocessor = preprocessor,
        start=45000, stop = 50000)  
    test_set = ZCA_Dataset(
        preprocessed_dataset= serial.load("${PYLEARN2_DATA_PATH}/cifar10/pylearn2_gcn_whitened/test.pkl"), 
        preprocessor = preprocessor)
    
    # bc01 format
    # print train_set.X.shape
    train_set.X = train_set.X.reshape(45000,3,32,32)
    valid_set.X = valid_set.X.reshape(5000,3,32,32)
    test_set.X = test_set.X.reshape(10000,3,32,32)
    
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
    
    class DeepCNN(Network):

        def __init__(self, rng):

            Network.__init__(self, n_hidden_layer = 4, BN = BN)
            
            print "    C4 P3S2 layer:"
                
            self.layer.append(ReLU_conv_layer(
                rng,
                filter_shape=(128, 3, 4, 4),
                pool_shape=(3,3),
                pool_stride=(2,2),
                BN = BN,                     
                BN_epsilon=BN_epsilon,
                binary_training=binary_training, 
                stochastic_training=stochastic_training,
                binary_test=binary_test, 
                stochastic_test=stochastic_test
            ))
            
            print "    C4 P3S2 layer:"
                
            self.layer.append(ReLU_conv_layer(
                rng,
                filter_shape=(256, 128, 4, 4),
                pool_shape=(3,3),
                pool_stride=(2,2),
                BN = BN,                     
                BN_epsilon=BN_epsilon,
                binary_training=binary_training, 
                stochastic_training=stochastic_training,
                binary_test=binary_test, 
                stochastic_test=stochastic_test
            ))
            
            print "    C4 layer:"
                
            self.layer.append(ReLU_conv_layer(
                rng,
                filter_shape=(256, 256, 4, 4),
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
                    n_inputs = 256*3*3, 
                    n_units = 2048, 
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
                n_inputs= 2048, 
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
    
    # print 'Save first hidden layer weights'
    
    # W = model.layer[1].W.get_value()
    # import pickle
    # pickle.dump( W, open( "W.pkl", "wb" ) )
    
    # print 'Display weights'
    
    # import matplotlib.pyplot as plt
    # import matplotlib.cm as cm
    # from filter_plot import tile_raster_images
    
    # W = np.transpose(model.layer[0].W.get_value())
    
    # print "min(W) = " + str(np.min(W))
    # print "max(W) = " + str(np.max(W))
    # print "mean(W) = " + str(np.mean(W))
    # print "mean(abs(W)) = " + str(np.mean(abs(W)))
    # print "var(W) = " + str(np.var(W))
    
    # plt.hist(W,bins=100)
    # plt.show()
    
    # W = tile_raster_images(W,(28,28),(5,5),(2, 2))
    # plt.imshow(W, cmap = cm.Greys_r)
    # plt.show()

