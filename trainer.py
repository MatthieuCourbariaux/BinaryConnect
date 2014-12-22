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
import theano 
import theano.tensor as T
import time

# TRAINING

class Trainer(object):
    
    def __init__(self,
            rng, save_path, load_path,
            train_set, valid_set, test_set,
            model,
            LR_start, LR_sat, LR_fin, M_start, M_sat, M_fin, 
            batch_size, gpu_batches,
            n_epoch,
            dynamic_range, range_update_frequency,
            shuffle_batches, shuffle_examples):
        
        print '    Training algorithm:'
        print '        Learning rate = %f' %(LR_start)
        print '        Learning rate saturation = %i' %(LR_sat)
        print '        Final learning rate = %f' %(LR_fin)
        print '        Momentum = %f' %(M_start)
        print '        Momentum saturation = %i' %(M_sat)
        print '        Final momentum = %f' %(M_fin)
        print '        Batch size = %i' %(batch_size)
        print '        gpu_batches = %i' %(gpu_batches)
        print '        Number of epochs = %i' %(n_epoch)
        print '        shuffle_batches = %i' %(shuffle_batches)
        print '        shuffle_examples = %i' %(shuffle_examples)
        print '        Dynamic range = %i' %(dynamic_range)
        print '        Range update frequency = %i' %(range_update_frequency)

        # save the dataset
        self.rng = rng
        self.shuffle_batches = shuffle_batches
        self.shuffle_examples = shuffle_examples
        self.load_path = load_path
        self.save_path = save_path
        self.train_set = train_set
        self.valid_set = valid_set
        self.test_set = test_set
        
        # save the model
        self.model = model
        
        # save the parameters
        self.LR_start = LR_start
        self.LR_sat = LR_sat
        self.LR_fin = LR_fin
        self.M_start = M_start
        self.M_sat = M_sat
        self.M_fin = M_fin
        self.batch_size = batch_size
        self.gpu_batches = gpu_batches
        self.n_epoch = n_epoch
        self.dynamic_range = dynamic_range
        self.range_update_frequency = range_update_frequency
        
        # put a part of the dataset on gpu
        self.shared_x = theano.shared(
            np.asarray(self.train_set.X[0:self.batch_size*self.gpu_batches], dtype=theano.config.floatX))
        self.shared_y = theano.shared(
            np.asarray(self.train_set.y[0:self.batch_size*self.gpu_batches], dtype=theano.config.floatX))
    
    def shuffle(self, set):
        
        # on the CPU for the moment.
        X = np.copy(set.X)
        y = np.copy(set.y)
                
        shuffled_index = range(set.X.shape[0])
        self.rng.shuffle(shuffled_index)
        
        for i in range(set.X.shape[0]):
            set.X[i] = X[shuffled_index[i]]
            set.y[i] = y[shuffled_index[i]]
    
    def init_range(self):
        
        # save the precisions and the random parameters of the model
        comp_precision = self.model.get_comp_precision()
        update_precision = self.model.get_update_precision()
        self.model.save_params()
        
        # set a good precision 
        self.model.set_comp_precision(31)
        self.model.set_update_precision(31)
        
        self.train_epoch(self.train_set)
        
        # for k in range(15):
        
            # train the model on a the valid set a few times (because valid is small)
            # self.train_epoch(self.valid_set)
            
            # updating the range
            # self.update_range()
        
        # set back the precision and the random parameters
        self.model.set_comp_precision(comp_precision)
        self.model.set_update_precision(update_precision)
        self.model.load_params()
    
    def init(self):
        
        if self.load_path != None:
            self.model.load_params_file(self.load_path)

        self.LR = self.LR_start
        self.LR_step = (self.LR_fin-self.LR_start)/self.LR_sat 
        self.M = self.M_start 
        self.M_step = (self.M_fin-self.M_start)/self.M_sat 
        
        self.epoch = 0
        self.best_epoch = self.epoch
        
        # test it on the validation set
        self.validation_ER = self.test_epoch(self.valid_set)
        # test it on the test set
        self.test_ER = self.test_epoch(self.test_set)
        
        self.best_validation_ER = self.validation_ER
        self.best_test_ER = self.test_ER
        
        if self.dynamic_range == True : 
            self.init_range()
 
    def update(self):
        
        # start by shuffling train set
        if self.shuffle_examples == True:
            self.shuffle(self.train_set)
        
        self.epoch += 1
        
        # train the model on all training examples
        self.train_epoch(self.train_set)
        
        # test it on the validation set
        self.validation_ER = self.test_epoch(self.valid_set)
        
        # test it on the test set
        self.test_ER = self.test_epoch(self.test_set) 
        
        # update LR and M as well during the first phase
        self.update_LR()
        self.update_M()
        
        # if self.dynamic_range == True : 
            # self.update_range()
        
        # save the best parameters
        if self.validation_ER < self.best_validation_ER:
            self.best_validation_ER = self.validation_ER
            self.best_test_ER = self.test_ER
            self.best_epoch = self.epoch
            if self.save_path != None:
                self.model.save_params_file(self.save_path)
    
    def load_shared_dataset(self, set, start,size):
        
        self.shared_x.set_value(
            set.X[self.batch_size*start:self.batch_size*(size+start)])
        self.shared_y.set_value(
            set.y[self.batch_size*start:self.batch_size*(size+start)])
    
    def train_epoch(self, set):
        
        # number of batch in the dataset
        n_batches = np.int(np.floor(set.X.shape[0]/self.batch_size))
        # number of group of batches (in the memory of the GPU)
        n_gpu_batches = np.int(np.floor(n_batches/self.gpu_batches))
        
        # number of batches in the last group
        if self.gpu_batches<=n_batches:
            n_remaining_batches = n_batches%self.gpu_batches
        else:
            n_remaining_batches = n_batches
        
        # batch counter
        k = 0
        
        shuffled_range_i = range(n_gpu_batches)
        
        if self.shuffle_batches==True:
            self.rng.shuffle(shuffled_range_i)
        
        for i in shuffled_range_i:
        
            self.load_shared_dataset(set,
                start=i*self.gpu_batches,
                size=self.gpu_batches)
            
            shuffled_range_j = range(self.gpu_batches)
            
            if self.shuffle_batches==True:
                self.rng.shuffle(shuffled_range_j)
            
            for j in shuffled_range_j: 

                self.train_batch(j, self.LR, self.M)
                
                if self.dynamic_range==True:
                    k+=1
                    if k==self.range_update_frequency:
                        self.update_range(k)
                        k=0
        
        # load the last inclompete gpu batch of batches
        if n_remaining_batches > 0:
        
            self.load_shared_dataset(set,
                    start=n_gpu_batches*self.gpu_batches,
                    size=n_remaining_batches)
            
            shuffled_range_j = range(n_remaining_batches)
            if self.shuffle_batches==True:
                    self.rng.shuffle(shuffled_range_j)
            
            for j in shuffled_range_j: 

                self.train_batch(j, self.LR, self.M)
    
    def test_epoch(self, set):
        
        n_batches = np.int(np.floor(set.X.shape[0]/self.batch_size))
        n_gpu_batches = np.int(np.floor(n_batches/self.gpu_batches))
        
        if self.gpu_batches<=n_batches:
            # print "here"
            n_remaining_batches = n_batches%self.gpu_batches
        else:
            # print "there"
            n_remaining_batches = n_batches
            
        # print n_batches     
        # print n_gpu_batches        
        # print n_remaining_batches
        
        # input("wait")
        
        error_rate = 0.
        
        for i in range(n_gpu_batches):
        
            self.load_shared_dataset(set,
                start=i*self.gpu_batches,
                size=self.gpu_batches)
            
            for j in range(self.gpu_batches): 

                error_rate += self.test_batch(j)
        
        # load the last inclompete gpu batch of batches
        if n_remaining_batches > 0:
        
            self.load_shared_dataset(set,
                    start=n_gpu_batches*self.gpu_batches,
                    size=n_remaining_batches)
            
            for j in range(n_remaining_batches): 

                error_rate += self.test_batch(j)
        
        error_rate /= (n_batches*self.batch_size)
        error_rate *= 100.
        
        return error_rate
    
    def update_LR(self):

        if self.LR > self.LR_fin:
            self.LR += self.LR_step
        else:
            self.LR = self.LR_fin
    
    def update_M(self):
    
        if self.M < self.M_fin: 
            self.M += self.M_step
        else:
            self.M = self.M_fin
    
    def monitor(self):
    
        print '    epoch %i:' %(self.epoch)
        print '        learning rate %f' %(self.LR)
        print '        momentum %f' %(self.M)
        print '        validation error rate %f%%' %(self.validation_ER)
        print '        test error rate %f%%' %(self.test_ER)
        print '        epoch associated to best validation error %i' %(self.best_epoch)
        print '        best validation error rate %f%%' %(self.best_validation_ER)
        print '        test error rate associated to best validation error %f%%' %(self.best_test_ER)
        
        if self.dynamic_range == True : 
            self.model.print_range()
    
    def train(self):        
        
        self.init()
        self.monitor()
        
        for epoch in range(self.n_epoch):
            
            self.update()   
            self.monitor()
    
    def build(self):
        
        # input and output variables
        x = T.matrix('x')
        y = T.matrix('y')
        index = T.lscalar() 
        batch_count = T.lscalar() 
        LR = T.scalar('LR', dtype=theano.config.floatX)
        M = T.scalar('M', dtype=theano.config.floatX)

        # before the build, you work with symbolic variables
        # after the build, you work with numeric variables
        
        self.train_batch = theano.function(inputs=[index,LR,M], updates=self.model.updates(x,y,LR,M,self.dynamic_range),givens={ 
                x: self.shared_x[index * self.batch_size:(index + 1) * self.batch_size], 
                y: self.shared_y[index * self.batch_size:(index + 1) * self.batch_size]},
                name = "train_batch", on_unused_input='warn')
        
        self.test_batch = theano.function(inputs=[index],outputs=self.model.errors(x,y),givens={
                x: self.shared_x[index * self.batch_size:(index + 1) * self.batch_size], 
                y: self.shared_y[index * self.batch_size:(index + 1) * self.batch_size]},
                name = "test_batch")
                
        if self.dynamic_range == True : 
            self.update_range = theano.function(inputs=[batch_count],updates=self.model.range_updates(batch_count), name = "update_range")
