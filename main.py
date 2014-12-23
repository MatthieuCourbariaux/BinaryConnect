# Copyright 2014 Matthieu Courbariaux

# This file is part of Deep learning arithmetic simulator.

# Deep learning arithmetic simulator is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# Deep learning arithmetic simulator is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with Deep learning arithmetic simulator.  If not, see <http://www.gnu.org/licenses/>.

import gzip
import cPickle
import numpy as np
import os
import os.path
import sys
import time

from trainer import Trainer
from model import PI_MNIST_model, MNIST_model, CIFAR10_SVHN_model

from pylearn2.datasets.mnist import MNIST   
from pylearn2.datasets.zca_dataset import ZCA_Dataset    
from pylearn2.datasets.svhn import SVHN
          
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

if __name__ == "__main__":
       
    print 'Beginning of the program'
    start_time = time.clock()   
    
    print 'Loading the dataset' 
    
    dataset = sys.argv[1]
    
    if dataset == "PI_MNIST" or dataset == "MNIST":
            
        train_set = MNIST(which_set= 'train',start=0, stop = 50000)#, center = True)
        valid_set = MNIST(which_set= 'train',start=50000, stop = 60000)#, center = True)
        test_set = MNIST(which_set= 'test')#, center = True)
        
        # for both datasets, onehot the target
        train_set.y = np.float32(onehot(train_set.y))
        valid_set.y = np.float32(onehot(valid_set.y))
        test_set.y = np.float32(onehot(test_set.y))
        
    elif dataset == "CIFAR10":
            
        preprocessor = cPickle.load(open("/data/lisa/data/cifar10/pylearn2_gcn_whitened/preprocessor.pkl",'rb'))
        train_set = ZCA_Dataset(
            preprocessed_dataset=cPickle.load(open("/data/lisa/data/cifar10/pylearn2_gcn_whitened/train.pkl",'rb')), 
            preprocessor = preprocessor,
            start=0, stop = 45000)
        valid_set = ZCA_Dataset(
            preprocessed_dataset= cPickle.load(open("/data/lisa/data/cifar10/pylearn2_gcn_whitened/train.pkl",'rb')), 
            preprocessor = preprocessor,
            start=45000, stop = 50000)  
        test_set = ZCA_Dataset(
            preprocessed_dataset= cPickle.load(open("/data/lisa/data/cifar10/pylearn2_gcn_whitened/test.pkl",'rb')), 
            preprocessor = preprocessor) 
        
        # for both datasets, onehot the target
        train_set.y = np.float32(onehot(train_set.y))
        valid_set.y = np.float32(onehot(valid_set.y))
        test_set.y = np.float32(onehot(test_set.y))
        
    elif dataset == "SVHN":
        
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
        
    print 'Creating the model'
    
    # arithmetic hyperparameters
    format = sys.argv[2]
    
    initial_range = 0
    comp_precision = 0
    update_precision = 0
    range_update_frequency = 0
    max_overflow = 0
    range_init_epoch = 0
    
    if format == "FXP" or format == "DFXP":
        initial_range = int(sys.argv[3])
        comp_precision = int(sys.argv[4])
        update_precision = int(sys.argv[5])
        
        if format == "DFXP":
            range_update_frequency = int(sys.argv[6])
            max_overflow = float(sys.argv[7])
            range_init_epoch = int(sys.argv[8])   
        
    if dataset == "PI_MNIST":
        
        rng = np.random.RandomState(1234)
        LR_start = 0.05
        batch_size = 100
        gpu_batches = 500
        n_epoch = 1000 
        
        model = PI_MNIST_model(rng = rng, batch_size = batch_size,
            n_input = 784, n_output = 10, n_hidden = 240, n_pieces = 5, n_hidden_layers = 2, 
            p_input = 0.8, scale_input = 1., p_hidden = 0.5, scale_hidden = 0.5, 
            max_col_norm = 1.9365, format = format,
            comp_precision = comp_precision, update_precision = update_precision, 
            initial_range = initial_range, max_overflow = max_overflow)
        
        trainer = Trainer(rng = rng, load_path = None, save_path = None,
            train_set = train_set, valid_set = valid_set, test_set = test_set,
            model = model,
            LR_start = LR_start, LR_sat = n_epoch/2, LR_fin = LR_start/10, M_start = 0.5, M_sat = n_epoch/4, M_fin = 0.7, 
            batch_size = batch_size, gpu_batches = gpu_batches,
            n_epoch = n_epoch,
            shuffle_batches = False, shuffle_examples = True,
            format = format, range_update_frequency = range_update_frequency,range_init_epoch=range_init_epoch)
    
    elif dataset == "MNIST":
    
        rng = np.random.RandomState(1234)
        LR_start = 0.02
        batch_size = 128
        gpu_batches = 391 # 391 -> 50000, 196 -> 25000, 79 -> 10000
        n_epoch = 800
        
        model = MNIST_model(rng = rng, batch_size = batch_size, format = format,
            comp_precision = comp_precision, update_precision = update_precision, 
            initial_range = initial_range, max_overflow = max_overflow)
        
        trainer = Trainer(rng = rng, load_path = None, save_path = None,
            train_set = train_set, valid_set = valid_set, test_set = test_set,
            model = model,
            LR_start = LR_start, LR_sat = n_epoch/2, LR_fin = LR_start/10, M_start = 0.5, M_sat = n_epoch/4, M_fin = 0.7, 
            batch_size = batch_size, gpu_batches = gpu_batches,
            n_epoch = n_epoch,
            shuffle_batches = False, shuffle_examples = True,
            format = format, range_update_frequency = range_update_frequency,range_init_epoch=range_init_epoch)
    
    elif dataset == "CIFAR10":
    
        rng = np.random.RandomState(1234)
        LR_start = 0.02
        batch_size = 128
        gpu_batches = 391 # 391 -> 50000, 196 -> 25000, 79 -> 10000
        n_epoch = 400
        
        model = CIFAR10_SVHN_model(rng = rng, batch_size = batch_size, format = format,
            comp_precision = comp_precision, update_precision = update_precision, 
            initial_range = initial_range, max_overflow = max_overflow)
        
        trainer = Trainer(rng = rng, load_path = None, save_path = None,
            train_set = train_set, valid_set = valid_set, test_set = test_set,
            model = model,
            LR_start = LR_start, LR_sat = n_epoch/2, LR_fin = LR_start/10, M_start = 0.5, M_sat = n_epoch/2, M_fin = 0.7, 
            batch_size = batch_size, gpu_batches = gpu_batches,
            n_epoch = n_epoch,
            shuffle_batches = False, shuffle_examples = True,
            format = format, range_update_frequency = range_update_frequency,range_init_epoch=range_init_epoch)
    
    elif dataset == "SVHN":
        
        rng = np.random.RandomState(1234)
        LR_start = 0.05
        batch_size = 128
        gpu_batches = 391 # 391 -> 50000, 196 -> 25000, 79 -> 10000
        n_epoch = 160
        
        model = CIFAR10_SVHN_model(rng = rng, batch_size = batch_size, format = format,
            comp_precision = comp_precision, update_precision = update_precision, 
            initial_range = initial_range, max_overflow = max_overflow)
        
        trainer = Trainer(rng = rng, load_path = None, save_path = None,
            train_set = train_set, valid_set = valid_set, test_set = test_set,
            model = model,
            LR_start = LR_start, LR_sat = n_epoch, LR_fin = LR_start/10, M_start = 0.5, M_sat = n_epoch, M_fin = 0.7, 
            batch_size = batch_size, gpu_batches = gpu_batches,
            n_epoch = n_epoch,
            shuffle_batches = True, shuffle_examples = False,
            format = format, range_update_frequency = range_update_frequency,range_init_epoch=range_init_epoch)

    print 'Building'
    
    trainer.build()
    
    print 'Training'
    
    trainer.train()

    end_time = time.clock()
    print 'The code ran for %i seconds'%(end_time - start_time)
    