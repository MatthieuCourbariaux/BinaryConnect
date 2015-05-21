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
import theano 
import theano.tensor as T
import time

# for data augmentation
from scipy.ndimage.interpolation import rotate, affine_transform
from pylearn2.train_extensions.window_flip import _zero_pad

class dataset(object):
    def __init__(self,set):
        self.X = np.copy(set.X)
        self.y = np.copy(set.y)

# TRAINING

class Trainer(object):
    
    def __init__(self,
            rng,
            train_set, valid_set, test_set,
            zero_pad,
            affine_transform_a,
            affine_transform_b,
            horizontal_flip,
            model, save_path, load_path,
            LR, LR_decay, LR_decay_patience,
            M, 
            BN,
            batch_size, number_of_batches_on_gpu,
            n_epoch, monitor_step,
            shuffle_batches, shuffle_examples):
        
        self.zero_pad = zero_pad
        print "    zero_pad = "+str(zero_pad)   
        self.affine_transform_a = affine_transform_a
        print "    affine_transform_a = "+str(affine_transform_a)   
        self.affine_transform_b = affine_transform_b
        print "    affine_transform_b = "+str(affine_transform_b)   
        self.horizontal_flip = horizontal_flip
        print "    horizontal_flip = "+str(horizontal_flip)   
        print '    shuffle_batches = %i' %(shuffle_batches)
        print '    shuffle_examples = %i' %(shuffle_examples)
        
        print '    Learning rate = %f' %(LR)
        print '    Learning rate decay = %f' %(LR_decay)
        print '    LR_decay_patience = %f' %(LR_decay_patience)
        print '    Momentum = %f' %(M)
        
        self.BN = BN
        print "    BN = "+str(BN)        
        
        self.batch_size = batch_size
        print '    batch_size = %i' %(batch_size)
        self.number_of_batches_on_gpu = number_of_batches_on_gpu
        print '    number_of_batches_on_gpu = %i' %(number_of_batches_on_gpu)
        
        print '    Number of epochs = %i' %(n_epoch)
        print '    Monitor step = %i' %(monitor_step)
        
        # zero padding, may help Data Augmentation
        train_set.X = _zero_pad(array=train_set.X, amount=self.zero_pad, axes=(2, 3))
        valid_set.X = _zero_pad(array=valid_set.X, amount=self.zero_pad, axes=(2, 3))
        test_set.X = _zero_pad(array=test_set.X, amount=self.zero_pad, axes=(2, 3))
        
        # save the dataset
        self.train_set = train_set
        self.valid_set = valid_set
        self.test_set = test_set
        self.rng = rng
        self.shuffle_batches = shuffle_batches
        self.shuffle_examples = shuffle_examples
        
        # in order to avoid augmenting already augmented data
        self.DA_train_set = dataset(train_set)
        
        # save the model
        self.model = model
        self.load_path = load_path
        self.save_path = save_path
        
        # save the parameters
        self.LR = LR
        self.M = M
        self.LR_decay = LR_decay
        self.LR_decay_patience = LR_decay_patience
        self.n_epoch = n_epoch
        self.step = monitor_step
        
        # put a part of the dataset on gpu
        shared_size = self.batch_size*self.number_of_batches_on_gpu
        self.shared_x = theano.shared(
            np.asarray(self.train_set.X[0:shared_size], dtype=theano.config.floatX))
        self.shared_y = theano.shared(
            np.asarray(self.train_set.y[0:shared_size], dtype=theano.config.floatX))
    
    def shuffle(self, set):
        
        shuffled_set = dataset(set)
                
        shuffled_index = range(set.X.shape[0])
        self.rng.shuffle(shuffled_index)
        
        for i in range(set.X.shape[0]):
            
            shuffled_set.X[i] = set.X[shuffled_index[i]]
            shuffled_set.y[i] = set.y[shuffled_index[i]]
            
        return shuffled_set
            
    def affine_transformations(self,set):
        
        DA_set = dataset(set)

        # for every samples in the training set
        for i in range(set.X.shape[0]):
            
            # making an affine transformation of the coordinate of the points of the image
            # (x',y') = A(x,y) + B
            # result is rotation, translation, scaling on each axis
            # to adjust a and b, limit the size of the dataset
            
            # a = .1 # best for CNN MNIST, 128 samples
            A = np.identity(n=2)+self.rng.uniform(low=-self.affine_transform_a,high=self.affine_transform_a,size=(2, 2))
            # b = .5 # best for CNN MNIST, 128 samples
            B = self.rng.uniform(low=-self.affine_transform_b,high=self.affine_transform_b,size=(2))
            
            # for every channels
            for j in range(set.X.shape[1]):
            
                DA_set.X[i][j]=affine_transform(set.X[i][j],A,offset=B,order=2)
                
                # max_rot = 15
                # angle = self.rng.random_integers(-max_rot,max_rot)
                # DA_set.X[i] = rotate(DA_set.X[i].reshape(28,28),angle, reshape=False).reshape(784)
        
        return DA_set
    
    def set_BN_mean_var(self):
            
        # reset cumulative mean and var
        self.reset_mean_var()
        
        # not on the DA training set
        # because no DA on valid and test
        self.set_mean_var(self.train_set)
        self.set_mean_var(self.valid_set)
        
        return
    
    def window_flip(self,set):
        
        DA_set = dataset(set)

        # for every samples in the training set
        for i in range(set.X.shape[0]):
            
            # for every channels
            for j in range(set.X.shape[1]):
                
                if bool(self.rng.random_integers(0,1)) == True:
                    DA_set.X[i][j]=np.fliplr(set.X[i][j])
        
        return DA_set
    
    def init(self):
        
        if self.load_path != None:
            self.model.load_params_file(self.load_path)
        
        # self.go_on = True
        self.no_improvement = 0
        self.epoch = 0
        self.best_epoch = self.epoch
        
        # set the mean and variance for BN
        if self.BN == True: 
            self.set_BN_mean_var()
        
        # test it on the validation set
        self.validation_ER = self.test_epoch(self.valid_set)
        # test it on the test set
        self.test_ER = self.test_epoch(self.test_set)
        
        self.best_validation_ER = self.validation_ER
        self.best_test_ER = self.test_ER
            
    
    def train(self):        
        
        self.init()
        self.monitor()
        
        # while (self.go_on):
        while (self.epoch<self.n_epoch):
            
            self.update()   
            self.monitor()
    
    def update(self):
        
        # start by shuffling train set
        if self.shuffle_examples == True:
            self.train_set = self.shuffle(self.train_set)
        
        # data augmentation
        self.DA_train_set = self.train_set
        if (self.affine_transform_a != 0) or (self.affine_transform_b !=0):
            self.DA_train_set = self.affine_transformations(self.DA_train_set)
        if self.horizontal_flip==True:
            self.DA_train_set = self.window_flip(self.DA_train_set)
        
        self.epoch += self.step
        
        for k in range(self.step):
        
            # train the model on all training examples
            self.train_epoch(self.DA_train_set)
            
        # set the mean and variance for BN
        # not on the DA training set
        # because no DA on valid and test
        if self.BN == True: 
            self.set_BN_mean_var()
        
        # test it on the validation set
        self.validation_ER = self.test_epoch(self.valid_set)
        
        # test it on the test set
        self.test_ER = self.test_epoch(self.test_set) 
        
        # save the best parameters
        if self.validation_ER < self.best_validation_ER:
            
            self.best_validation_ER = self.validation_ER
            self.best_test_ER = self.test_ER
            self.best_epoch = self.epoch
            self.no_improvement=0
            if self.save_path != None:
                self.model.save_params_file(self.save_path)
        
        # reduce the LR if the performance does not improve for too long
        # question = should I load the previous best ?
        else:
            self.no_improvement += self.step
            if self.no_improvement >= self.LR_decay_patience:
                self.LR *= self.LR_decay
                self.no_improvement=0
    
    def load_shared_dataset(self, set, start,size):
        
        self.shared_x.set_value(
            set.X[start:(size+start)])
        self.shared_y.set_value(
            set.y[start:(size+start)])
    
    def train_epoch(self, set):
        
        # number of batch in the dataset
        n_batches = np.int(np.floor(set.X.shape[0]/self.batch_size))
        # number of group of batches (in the memory of the GPU)
        n_number_of_batches_on_gpu = np.int(np.floor(n_batches/self.number_of_batches_on_gpu))
        
        # number of batches in the last group
        if self.number_of_batches_on_gpu<=n_batches:
            n_remaining_batches = n_batches%self.number_of_batches_on_gpu
        else:
            n_remaining_batches = n_batches
        
        shuffled_range_i = range(n_number_of_batches_on_gpu)
        
        if self.shuffle_batches==True:
            self.rng.shuffle(shuffled_range_i)
        
        for i in shuffled_range_i:
        
            self.load_shared_dataset(set,
                start=i*self.number_of_batches_on_gpu*self.batch_size,
                size=self.number_of_batches_on_gpu*self.batch_size)
            
            shuffled_range_j = range(self.number_of_batches_on_gpu)
            
            if self.shuffle_batches==True:
                self.rng.shuffle(shuffled_range_j)
            
            for j in shuffled_range_j:  

                self.train_batch(j, self.LR, self.M)
        
        # load the last incomplete gpu batch of batches
        if n_remaining_batches > 0:
        
            self.load_shared_dataset(set,
                    start=n_number_of_batches_on_gpu*self.number_of_batches_on_gpu*self.batch_size,
                    size=n_remaining_batches*self.batch_size)
            
            shuffled_range_j = range(n_remaining_batches)
            if self.shuffle_batches==True:
                self.rng.shuffle(shuffled_range_j)
            
            for j in shuffled_range_j: 

                self.train_batch(j, self.LR, self.M)
    
    # batch normalization function
    # not exactly True, but seems to do the job well enough.
    # the problem is that I only compute the true mean and var for the first layer.
    def set_mean_var(self, set):
        
        n_batches = np.int(np.floor(set.X.shape[0]/self.batch_size))
        n_number_of_batches_on_gpu = np.int(np.floor(n_batches/self.number_of_batches_on_gpu))
        
        if self.number_of_batches_on_gpu<=n_batches:
            n_remaining_batches = n_batches%self.number_of_batches_on_gpu
        else:
            n_remaining_batches = n_batches
        
        # have to compute mean and var for each layer
        # cannot do all at the same time because of memory
        for k in range(self.model.n_hidden_layers+1):
            for i in range(n_number_of_batches_on_gpu):
            
                self.load_shared_dataset(set,
                    start=i*self.number_of_batches_on_gpu*self.batch_size,
                    size=self.number_of_batches_on_gpu*self.batch_size)
                
                for j in range(self.number_of_batches_on_gpu): 
                    self.BN_updates[k](j)
            
            # load the last incomplete gpu batch of batches
            if n_remaining_batches > 0:
                
                self.load_shared_dataset(set,
                        start=n_number_of_batches_on_gpu*self.number_of_batches_on_gpu*self.batch_size,
                        size=n_remaining_batches*self.batch_size)
                
                for j in range(n_remaining_batches): 
                    self.BN_updates[k](j)
        
        return
    
    def test_epoch(self, set):
        
        n_batches = np.int(np.floor(set.X.shape[0]/self.batch_size))
        n_number_of_batches_on_gpu = np.int(np.floor(n_batches/self.number_of_batches_on_gpu))
        
        if self.number_of_batches_on_gpu<=n_batches:
            n_remaining_batches = n_batches%self.number_of_batches_on_gpu
        else:
            n_remaining_batches = n_batches
        
        error_rate = 0.
        
        for i in range(n_number_of_batches_on_gpu):
        
            self.load_shared_dataset(set,
                start=i*self.number_of_batches_on_gpu*self.batch_size,
                size=self.number_of_batches_on_gpu*self.batch_size)
            
            for j in range(self.number_of_batches_on_gpu): 
                
                error_rate += self.test_batch(j)
        
        # load the last incomplete gpu batch of batches
        if n_remaining_batches > 0:
        
            self.load_shared_dataset(set,
                    start=n_number_of_batches_on_gpu*self.number_of_batches_on_gpu*self.batch_size,
                    size=n_remaining_batches*self.batch_size)
            
            for j in range(n_remaining_batches): 

                error_rate += self.test_batch(j)
        
        error_rate /= (n_batches*self.batch_size)
        error_rate *= 100.
        
        return error_rate
    
    def monitor(self):
    
        print '    epoch %i:' %(self.epoch)
        print '        learning rate %f' %(self.LR)
        print '        momentum %f' %(self.M)
        print '        validation error rate %f%%' %(self.validation_ER)
        print '        test error rate %f%%' %(self.test_ER)
        print '        epoch associated to best validation error %i' %(self.best_epoch)
        print '        best validation error rate %f%%' %(self.best_validation_ER)
        print '        test error rate associated to best validation error %f%%' %(self.best_test_ER)
        self.model.monitor()
    
    def build(self):
        
        # input and output variables
        x = T.tensor4('x')
        y = T.matrix('y')
        index = T.scalar('index', dtype='int64') 
        LR = T.scalar('LR', dtype=theano.config.floatX)
        M = T.scalar('M', dtype=theano.config.floatX)

        # before the build, you work with symbolic variables
        # after the build, you work with numeric variables
        
        self.train_batch = theano.function(inputs=[index,LR,M], updates=self.model.parameters_updates(x,y,LR,M),givens={ 
                x: self.shared_x[index * self.batch_size:(index + 1) * self.batch_size], 
                y: self.shared_y[index * self.batch_size:(index + 1) * self.batch_size]},
                name = "train_batch", on_unused_input='warn')
        
        self.test_batch = theano.function(inputs = [index], outputs=self.model.errors(x,y), givens={
                x: self.shared_x[index * self.batch_size:(index + 1) * self.batch_size], 
                y: self.shared_y[index * self.batch_size:(index + 1) * self.batch_size]},
                name = "test_batch", on_unused_input='warn')
        
        # batch normalization specific functions
        if self.BN == True: 
        
            # batch normalization specific functions
            # I am forced to compute mean and var incrementally because of memory constraints.
            self.BN_updates = []
            for k in range(self.model.n_hidden_layers+1):
                self.BN_updates.append(theano.function(inputs = [index], updates=self.model.BN_updates(k,x), givens={
                        x: self.shared_x[index * self.batch_size:(index + 1) * self.batch_size]},
                        name = "BN_updates", on_unused_input='ignore'))
                    
            self.reset_mean_var = theano.function(inputs = [], updates=self.model.BN_reset(),
                    name = "reset_mean_var")            
               