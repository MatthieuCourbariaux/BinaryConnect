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
import theano 
import theano.tensor as T
import theano.printing as P 
from theano import pp
import time
import scipy.stats
from pylearn2.sandbox.cuda_convnet.filter_acts import FilterActs
from theano.sandbox.cuda.basic_ops import gpu_contiguous
from pylearn2.sandbox.cuda_convnet.pool import MaxPool

from format import apply_format
        
class layer(object):
    
    def __init__(self, rng, n_inputs, n_units, d):
        
        self.rng = rng
        
        self.n_units = n_units
        self.n_inputs = n_inputs
        # self.threshold = 0.1* n_inputs
        self.d = d

        W_values = 2.* np.asarray(self.rng.binomial(n=1, p=.5, size=(n_inputs, n_units)),dtype=theano.config.floatX) - 1.
        # W_values = np.asarray(self.rng.binomial(n=1, p=.5, size=(n_inputs, n_units)),dtype=theano.config.floatX)-0.5
        # W1_values = np.asarray(self.rng.binomial(n=1, p=.5, size=(n_units, n_inputs)),dtype=theano.config.floatX)-0.5
        # W1_values = W_values.T
        
        # self.high= np.sqrt(6. / (n_inputs + n_units))
        # W_values = self.high * np.asarray(self.rng.binomial(n=1, p=.5, size=(n_inputs, n_units)),dtype=theano.config.floatX) - self.high/2.
        
        high = np.float(np.sqrt(6. / (n_inputs + n_units)))
        self.W_LR_scale = 1. / ((high/2.)**2.)
        # print self.high

        # W_values = np.asarray(self.rng.uniform(low=-self.high,high=self.high,size=(n_inputs, n_units)),dtype=theano.config.floatX)
        
        # W1_values = np.asarray(self.rng.uniform(low=low,high=high,size=(n_units, n_inputs)),dtype=theano.config.floatX)
        
        # W_values = np.zeros((n_inputs, n_units),dtype=theano.config.floatX)
        # W1_values = np.zeros((n_units, n_inputs),dtype=theano.config.floatX)

        # b_values = np.zeros((n_units), dtype=theano.config.floatX) - n_units/2. # to have 1/2 neurons firing
        # b1_values = np.zeros((n_inputs), dtype=theano.config.floatX) - n_inputs /2. # to have 1/2 neurons firing
        b_values = np.zeros((n_units), dtype=theano.config.floatX)        
        a_values = np.ones((n_units), dtype=theano.config.floatX)
        
        # creation of shared symbolic variables
        # shared variables are the state of the built function
        # in practice, we put them in the GPU memory
        self.W = theano.shared(value=W_values, name='W')
        # self.W1 = theano.shared(value=W1_values, name='W1')
        self.b = theano.shared(value=b_values, name='b')
        self.a = theano.shared(value=a_values, name='a')
        # self.b1 = theano.shared(value=b1_values, name='b1')
        
        self.mean = theano.shared(value=b_values, name='mean')
        self.var = theano.shared(value=b_values, name='var')
    
    def activation(self, z):
        return z
    
    def fprop(self, x, can_fit, binary):
        
        # self.W_prop = T.cast(self.high * (T.ge(self.W,0.)-T.lt(self.W,0.)), theano.config.floatX)
        
        # weights are either 1, either -1 -> propagating = sum and additions
        self.W_prop = 2.* T.cast(T.ge(self.W,0.), theano.config.floatX) - 1.
        
        # weights are either 1, either 0 -> propagating = sums
        # self.W_prop = T.cast(T.ge(self.W,.5), theano.config.floatX)
        
        self.x = x
        
        # weighted sum
        z = T.dot(self.x, self.W_prop)
        
        # batch normalization
        self.new_mean = T.switch(can_fit, T.mean(z,axis=0), self.mean)
        self.new_var = T.switch(can_fit, T.var(z,axis=0), self.var)
        z = (z - self.new_mean)/(T.sqrt(self.new_var+1e-9))
        z = self.a * z + self.b
        
        # activation function
        y = self.activation(z)
        
        return y
        
    def BN_updates(self):
        
        updates = []
        updates.append((self.mean, self.new_mean))
        updates.append((self.var, self.new_var))
        
        return updates
        
    def bprop(self, cost):
        
        self.dEdb = T.grad(cost=cost, wrt=self.b)
        self.dEda = T.grad(cost=cost, wrt=self.a)
        self.dEdW_prop = T.grad(cost=cost, wrt=self.W_prop)
        
    def parameters_updates(self, LR):    
        
        # srng = T.shared_randomstreams.RandomStreams(self.rng.randint(999999))
        # LR_mask = T.cast(srng.binomial(n=1, p=LR, size=T.shape(self.W)), theano.config.floatX)
        # new_W = self.W + LR_mask * self.dEdW
        
        # new_W = T.cast(T.clip(self.W - (T.ge(self.dEdW*LR,self.d) - T.ge(-self.dEdW*LR,self.d)),-.5,.5), theano.config.floatX)
        
        # max = T.max(self.dEdW)
        # min = T.min(self.dEdW)
        # new_W = T.cast(T.clip(self.W - (T.ge(self.dEdW,max) - T.ge(-self.dEdW,-min)),-.5,.5), theano.config.floatX)
        
        # max = T.max(abs(self.dEdW))
        # comp = (1.-2.*LR) * max
        # comp = T.exp(-5 * LR) * max
        # new_W = T.cast(T.clip(self.W - (T.ge(self.dEdW,comp) - T.ge(-self.dEdW,comp)),-.5,.5), theano.config.floatX)
        
        new_W = self.W - self.W_LR_scale * LR * self.dEdW_prop
        
        # 8 bits representation with 1 bit of sign, -2 bits for integer part, and 7 bits for fraction
        # round to nearest -> try stochastic rounding ?
        new_W = apply_format("FXP", new_W, 7, -2) 
        
        # new_W = self.W - LR * self.dEdW
        
        new_b = self.b - LR * self.dEdb
        new_a = self.a - LR * self.dEda
        # new_b1 = self.b1 - LR * self.dEdb1

        # return the updates of shared variables
        updates = []
        updates.append((self.W, new_W))
        # updates.append((self.W1, new_W1))
        updates.append((self.b, new_b))
        updates.append((self.a, new_a))
        # updates.append((self.b1, new_b1))
        
        return updates

class ReLU_layer(layer):

    def __init__(self, rng, n_inputs, n_units, d):
        
        # call mother class constructor
        layer.__init__(self, rng, n_inputs, n_units, d)   

    def activation(self,z):
    
        # return T.maximum(0.,z)
        return T.maximum(z/3.,z)
        # return z
        
