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
import theano.printing as P 
from theano import pp
import time
import scipy.stats

# for convolution layers
# from pylearn2.sandbox.cuda_convnet.filter_acts import FilterActs
# from theano.sandbox.cuda.basic_ops import gpu_contiguous
# from pylearn2.sandbox.cuda_convnet.pool import MaxPool

from format import stochastic_rounding

class linear_layer(object):
    
    def __init__(self, rng, n_inputs, n_units,
        BN=False, BN_epsilon=1e-4,
        dropout=1.,
        binary_training=False, stochastic_training=False,
        binary_test=False, stochastic_test=0):
        
        self.rng = rng
        
        self.n_units = n_units
        print "        n_units = "+str(n_units)
        self.n_inputs = n_inputs
        print "        n_inputs = "+str(n_inputs)
        self.BN = BN
        print "        BN = "+str(BN)
        self.BN_epsilon = BN_epsilon
        print "        BN_epsilon = "+str(BN_epsilon)
        self.dropout = dropout
        print "        dropout = "+str(dropout)
        
        self.binary_training = binary_training
        print "        binary_training = "+str(binary_training)
        self.stochastic_training = stochastic_training
        print "        stochastic_training = "+str(stochastic_training)     
        self.binary_test = binary_test
        print "        binary_test = "+str(binary_test)
        self.stochastic_test = stochastic_test
        print "        stochastic_test = "+str(stochastic_test)     
        
        self.high = np.float32(np.sqrt(6. / (n_inputs + n_units)))
        self.W0 = np.float32(self.high/2)
        
        W_values = np.asarray(self.rng.uniform(low=-self.high,high=self.high,size=(n_inputs, n_units)),dtype=theano.config.floatX)
        b_values = np.zeros((n_units), dtype=theano.config.floatX)        
        a_values = np.ones((n_units), dtype=theano.config.floatX)
        
        # creation of shared symbolic variables
        # shared variables are the state of the built function
        # in practice, we put them in the GPU memory
        self.W = theano.shared(value=W_values, name='W')
        self.b = theano.shared(value=b_values, name='b')
        self.a = theano.shared(value=a_values, name='a')
        
        self.mean = theano.shared(value=b_values, name='mean')
        self.var = theano.shared(value=b_values, name='var')
        self.n_samples = theano.shared(value=np.float32(0),name='n_samples')
        
        # momentum
        self.update_W = theano.shared(value=np.zeros((n_inputs, n_units), dtype=theano.config.floatX), name='update_W')
        self.update_b = theano.shared(value=b_values, name='update_b')
    
    def activation(self, z):
        return z
    
    def binarize_weights(self,W,eval):
        
        binary_deterministic_training = (self.binary_training == True) and (self.stochastic_training == False)
        binary_stochastic_training = (self.binary_training == True) and (self.stochastic_training == True)
        binary_deterministic_test = (self.binary_test == True) and (self.stochastic_test == False)
        binary_stochastic_test = (self.binary_test == True) and (self.stochastic_test == True)       
        binary_deterministic = ((binary_deterministic_training == True) and (eval==False)) or ((binary_deterministic_test==True) and (eval==True))
        binary_stochastic = ((binary_stochastic_training == True) and (eval==False)) or ((binary_stochastic_test==True) and (eval==True))
        
        # print "        binary_training = "+str(self.binary_training)
        # print "        stochastic_training = "+str(self.stochastic_training)
        # print "        binary_test = "+str(self.binary_test)
        # print "        stochastic_test = "+str(self.stochastic_test)
        # print "        eval = "+str(eval)
        # print "        binary_deterministic = "+str(binary_deterministic)
        # print "        binary_stochastic = "+str(binary_stochastic)
        
        # Binary weights
        # I could scale x or z instead of W 
        # and the dot product would become an accumulation
        # I am not doing it to keep the code simple
        if binary_deterministic == True:
                
            # in the round to nearest case, we use binary weights during eval and training
            # [?,?] -> -W0 or W0
            Wb = self.W0 * T.sgn(W)            
        
        elif binary_stochastic == True:
            
            # clip is a kind of piecewise linear tanh
            # BTW, if I clip W directly, I do not need to clip Wb.
            # [?,?] -> [-W0,W0]
            Wb = T.clip(W, -self.W0, self.W0)
            
            # [-W0,W0] -> [-1,1]
            Wb = Wb/self.W0
            
            # [-1,1] -> [0,1]
            Wb = (Wb + 1.)*.5
            
            # rounding
            # [0,1] -> 0 or 1
            Wb = stochastic_rounding(Wb,self.rng)                
            
            # 0 or 1 -> -1 or 1
            Wb = 2. * Wb -1.
            
            # -1 or 1 -> -W0 or W0
            Wb = self.W0 * Wb
            
        # continuous weights
        else:
            Wb = W
            
        return Wb
    
    def fprop(self, x, can_fit, eval):
        
        # shape the input as a matrix (batch_size, n_inputs)
        self.x = x.flatten(2)
        
        # apply dropout mask
        if self.dropout < 1.:
            
            if eval == False:
                # The cast is important because
                # int * float32 = float64 which pulls things off the gpu
                srng = T.shared_randomstreams.RandomStreams(self.rng.randint(999999))
                mask = T.cast(srng.binomial(n=1, p=self.dropout, size=T.shape(self.x)), theano.config.floatX)
                
                # apply the mask
                self.x = self.x * mask
            else:
                self.x = self.x * self.dropout     
        
        # binarize the weights
        self.Wb = self.binarize_weights(self.W,eval)
        
        z = T.dot(self.x, self.Wb)
        
        # for BN updates
        self.z = z
        
        # batch normalization
        if self.BN == True:
            
            self.batch_mean = T.mean(z,axis=0)
            self.batch_var = T.var(z,axis=0)
            
            if can_fit == True:
                mean = self.batch_mean
                var = self.batch_var

            else:
                mean = self.mean
                var = self.var
        
            z = (z - mean)/(T.sqrt(var+self.BN_epsilon))
            z = self.a * z
        
        z = z + self.b
        
        # activation function
        y = self.activation(z)
        
        return y
    
    def bprop(self, cost):
        
        if self.binary_training == True:
            dEdWb = T.grad(cost=cost, wrt=self.Wb) 
            self.dEdW = dEdWb
            
        else:
            self.dEdW = T.grad(cost=cost, wrt=self.W)
        
        self.dEdb = T.grad(cost=cost, wrt=self.b)
        
        if self.BN == True:
            self.dEda = T.grad(cost=cost, wrt=self.a)
        
    def parameters_updates(self, LR, M):    
        
        updates = []

        # compute updates
        new_update_W = M * self.update_W - LR * self.dEdW
        new_update_b = M * self.update_b - LR * self.dEdb
        
        # compute new parameters. Note that we use a better precision than the other operations
        new_W = self.W + new_update_W
        new_b = self.b + new_update_b
        
        # clip the new weights when using binary weights
        # it is equivalent to doing 2 things:
        # 1) clip the weights during propagations
        # 2) backprop the clip function with a rule for the boundaries:
        # if W is equal to W0, then I can only reduce W
        # if W is equal to -W0, then I can only augment W
        if self.binary_training==True:
            new_W = T.clip(new_W, -self.W0, self.W0)
        
        updates.append((self.W, new_W))
        updates.append((self.b, new_b))
        updates.append((self.update_W, new_update_W))
        updates.append((self.update_b, new_update_b)) 
        
        if self.BN == True:
            new_a = self.a - LR * self.dEda
            updates.append((self.a, new_a))

        return updates
    
    def BN_updates(self):
        
        updates = []
        
        # batch_size = T.shape(self.z)[0]
        new_n_samples = self.n_samples + 1
        
        new_mean = (self.n_samples/new_n_samples) * self.mean + (1/new_n_samples) * self.batch_mean
        # very sligthly biased variance estimation
        new_var = (self.n_samples/new_n_samples) * self.var + (1/new_n_samples) * self.batch_var
        
        updates.append((self.n_samples, new_n_samples)) 
        updates.append((self.mean, new_mean))
        updates.append((self.var, new_var))
        
        return updates
        
    def BN_reset(self):
    
        updates = []
        
        updates.append((self.mean, self.mean*0.)) 
        updates.append((self.var, self.var*0.))
        updates.append((self.n_samples, self.n_samples*0.))
        
        return updates

class ReLU_layer(linear_layer):
    
    def __init__(self, ReLU_slope, **kwargs):
        
        self.ReLU_slope = ReLU_slope
        print "        ReLU_slope = "+str(ReLU_slope)
        
        linear_layer.__init__(self,**kwargs)
    
    def activation(self,z):
    
        # return T.maximum(0.,z)
        return T.maximum(z*self.ReLU_slope,z)
        
        # Roland activation function
        # return T.ge(z,1.)*z
        
class ReLU_conv_layer(linear_layer): 
    
    def __init__(self, rng, image_shape, filter_shape, pool_shape, ReLU_slope,
        BN, BN_epsilon=1e-4,
        binary_training=False, stochastic_training=False,
        binary_test=False, stochastic_test=0):
        
        self.rng = rng
        
        self.image_shape = image_shape
        print "        image_shape = "+str(image_shape)
        self.filter_shape = filter_shape
        print "        filter_shape = "+str(filter_shape)
        self.pool_shape = pool_shape
        print "        pool_shape = "+str(pool_shape)
        self.ReLU_slope = ReLU_slope
        print "        ReLU_slope = "+str(ReLU_slope)
        self.BN = BN
        print "        BN = "+str(BN)
        self.BN_epsilon = BN_epsilon
        print "        BN_epsilon = "+str(BN_epsilon)
        # self.W_lr_scale = W_lr_scale
        # print "    W_lr_scale = "+str(W_lr_scale)
        
        self.binary_training = binary_training
        print "        binary_training = "+str(binary_training)
        self.stochastic_training = stochastic_training
        print "        stochastic_training = "+str(stochastic_training)     
        self.binary_test = binary_test
        print "        binary_test = "+str(binary_test)
        self.stochastic_test = stochastic_test
        print "        stochastic_test = "+str(stochastic_test)     

        # range of init
        n_inputs = np.prod(filter_shape[1:])
        n_units = (filter_shape[0] * np.prod(filter_shape[2:])/ np.prod(pool_shape)) 

        # initialize weights with random weights
        self.high = np.float32(np.sqrt(6. / (n_inputs + n_units)))
        self.W0 = np.float32(self.high/2)
        
        # filters parameters
        W_values = np.asarray(rng.uniform(low=-self.high, high=self.high, size=self.filter_shape),dtype=theano.config.floatX)
        self.W = theano.shared(W_values)
                
         # the bias is a 1D tensor -- one bias per output feature map
        b_values = np.zeros((self.filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values)
        
        # BN stuff
        a_values = np.ones((self.filter_shape[0],), dtype=theano.config.floatX)
        self.a = theano.shared(value=a_values, name='a')        
        
        self.mean = theano.shared(value=b_values, name='mean')
        self.var = theano.shared(value=b_values, name='var')
        self.n_samples = theano.shared(value=np.float32(0),name='n_samples')
        
        # momentum
        self.update_W = theano.shared(value=np.zeros(self.filter_shape, dtype=theano.config.floatX), name='update_W')
        self.update_b = theano.shared(value=b_values, name='update_b')

    def fprop(self, x, can_fit, eval):
        
        # shape the input as it should be (not necessary)
        # x = x.reshape(self.image_shape)
        
        # binarize the weights
        self.Wb = self.binarize_weights(self.W,eval)
        
        # convolution
        z = T.nnet.conv.conv2d(x, self.Wb, border_mode='valid')

        # Maxpooling
        if self.pool_shape != (1,1):
            z = T.signal.downsample.max_pool_2d(input=z, ds=self.pool_shape)
        
        # for BN
        self.z = z
        
        # batch normalization
        if self.BN == True:
            
            # in the convolutional case, there is only a mean per feature map and not per location
            # http://arxiv.org/pdf/1502.03167v3.pdf   
            self.batch_mean = T.mean(z,axis=(0,2,3))
            self.batch_var = T.var(z,axis=(0,2,3))
                    
            if can_fit == True:
                mean = self.batch_mean
                var = self.batch_var

            else:
                mean = self.mean
                var = self.var
        
            z = (z - mean.dimshuffle('x', 0, 'x', 'x'))/(T.sqrt(var.dimshuffle('x', 0, 'x', 'x')+self.BN_epsilon))
            z = self.a.dimshuffle('x', 0, 'x', 'x') * z
        
        # bias
        z = z + self.b.dimshuffle('x', 0, 'x', 'x')
        
        # activation
        y = self.activation(z)
        
        return y
        
def activation(self,z):

        return T.maximum(z*self.ReLU_slope,z)
        
