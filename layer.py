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
import theano 
import theano.tensor as T
import theano.printing as P 
from theano import pp
import time
import scipy.stats

# for convolution layers
from pylearn2.sandbox.cuda_convnet.filter_acts import FilterActs
from theano.sandbox.cuda.basic_ops import gpu_contiguous
from pylearn2.sandbox.cuda_convnet.pool import MaxPool

from format import linear_quantization, binarize, tanh

class linear_layer(object):
    
    def __init__(self, rng, n_inputs, n_units, BN=False,
        max_col_norm = None, saturation = None,
        prop_bit_width=None, prop_stochastic_rounding=False,
        update_bit_width=None, update_stochastic_rounding=False):
        
        self.rng = rng
        
        self.n_units = n_units
        print "        n_units = "+str(n_units)
        self.n_inputs = n_inputs
        print "        n_inputs = "+str(n_inputs)
        self.BN = BN
        print "        BN = "+str(BN)
        # self.W_lr_scale = W_lr_scale
        # print "    W_lr_scale = "+str(W_lr_scale)
        
        self.max_col_norm = max_col_norm
        print "        max_col_norm = "+str(max_col_norm)
        self.saturation = saturation
        print "        saturation = "+str(saturation)
        
        self.prop_bit_width = prop_bit_width
        print "        prop_bit_width = "+str(prop_bit_width)
        self.prop_stochastic_rounding = prop_stochastic_rounding
        print "        prop_stochastic_rounding = "+str(prop_stochastic_rounding)     
        self.update_bit_width = update_bit_width
        print "        update_bit_width = "+str(update_bit_width)
        self.update_stochastic_rounding = update_stochastic_rounding
        print "        update_stochastic_rounding = "+str(update_stochastic_rounding)   
        
        # self.threshold = 0.1* n_inputs
        
        # W_values = 2.* np.asarray(self.rng.binomial(n=1, p=.5, size=(n_inputs, n_units)),dtype=theano.config.floatX) - 1.
        # W_values = np.asarray(self.rng.binomial(n=1, p=.33, size=(n_inputs, n_units)),dtype=theano.config.floatX) - np.asarray(self.rng.binomial(n=1, p=.33, size=(n_inputs, n_units)),dtype=theano.config.floatX)
        
        # W_values = np.asarray(self.rng.binomial(n=1, p=.5, size=(n_inputs, n_units)),dtype=theano.config.floatX)-0.5
        # W1_values = np.asarray(self.rng.binomial(n=1, p=.5, size=(n_units, n_inputs)),dtype=theano.config.floatX)-0.5
        # W1_values = W_values.T
        
        # self.high= np.sqrt(6. / (n_inputs + n_units))
        # W_values = self.high * np.asarray(self.rng.binomial(n=1, p=.5, size=(n_inputs, n_units)),dtype=theano.config.floatX) - self.high/2.
        
        self.high = np.float32(np.sqrt(6. / (n_inputs + n_units)))
        self.W0 = np.float32(self.high/2)
        # print self.high
        
        W_values = np.asarray(self.rng.uniform(low=-self.high,high=self.high,size=(n_inputs, n_units)),dtype=theano.config.floatX)
        # W_values = np.zeros((n_inputs, n_units),dtype=theano.config.floatX)
        
        # W1_values = np.asarray(self.rng.uniform(low=low,high=high,size=(n_units, n_inputs)),dtype=theano.config.floatX)
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
        self.sum = theano.shared(value=b_values, name='sum')
        self.sum2 = theano.shared(value=b_values, name='sum2')
        
        # momentum
        self.update_W = theano.shared(value=np.zeros((n_inputs, n_units), dtype=theano.config.floatX), name='update_W')
        self.update_b = theano.shared(value=b_values, name='update_b')
    
    def activation(self, z):
        return z
    
    def fprop(self, x, can_fit, eval):
        
        # self.Wb = T.cast(self.high * (T.ge(self.W,0.)-T.lt(self.W,0.)), theano.config.floatX)
        
        # weights are either 1, either -1 -> propagating = sum and sub
        # And scaling down weights to normal values
        # self.Wb = self.W_scale * (2.* T.cast(T.ge(self.W,0.), theano.config.floatX) - 1.)
        # self.Wb = self.W_scale * discretize(self.W)
        
        # weights are either 1, either -1, either 0 -> propagating = sum and sub
        # results are not better, maybe worse (it requires more bits for integer part of fxp)
        # self.Wb = T.cast(T.ge(self.W,.5)-T.le(self.W,-.5), theano.config.floatX)
        
        # 2.* T.cast(T.ge(self.W,0.), theano.config.floatX) - 1.
        
        # weights are either 1, either 0 -> propagating = sums
        # self.Wb = T.cast(T.ge(self.W,.5), theano.config.floatX)
        
        # shape the input as a matrix (batch_size, n_inputs)
        self.x = x.flatten(2)
        
        # weighted sum
        # z = T.dot(self.x, self.Wb)
        
        if eval == False:
            self.Wb = self.W0 * tanh(self.W/self.W0,binary=self.prop_bit_width,stochastic=self.prop_stochastic_rounding,rng=self.rng)
        else:
            self.Wb = self.W0 * tanh(self.W/self.W0)
            
        z =  T.dot(self.x, self.Wb)
        
        # discrete weights
        # I could scale x or z instead of W 
        # and the dot product would become an accumulation
        # I am not doing it because it would mess up with Theano automatic differentiation.
        # if self.prop_bit_width is not None:
            
            # self.Wb = linear_quantization(x=self.W,bit_width=self.prop_bit_width,min=-self.high,max=self.high,
                # stochastic=self.prop_stochastic_rounding,rng=self.rng)  
                
            # self.Wb = linear_quantization(x=self.W,bit_width=self.prop_bit_width,
                # stochastic=self.prop_stochastic_rounding,rng=self.rng)
                
            # self.Wb = binarize(self.W)
            # self.Wb = binary(self.W,stochastic=self.prop_stochastic_rounding,rng=self.rng)
            
            # self.Wb = binary(self.W,np.float32(self.high/2),stochastic=self.prop_stochastic_rounding,rng=self.rng)
            # z =  T.dot(self.x, self.Wb)
            
            # self.Wb = self.high * (T.ge(self.W,0.)-.5)
            
        # continuous weights
        # else:
            # z =  T.dot(self.x, self.W)
            
        # self.Wb = self.high * (T.ge(self.W,0.)-.5)               
        
        # for BN updates
        self.z = z
        
        # batch normalization
        if can_fit == True:
            mean = T.mean(z,axis=0)
            var = T.var(z,axis=0)
            
        else:
            mean = self.mean
            var = self.var
        
        if self.BN == True:
            z = (z - mean)/(T.sqrt(var+1e-9))
            z = self.a * z
            
        z = z + self.b
        
        # activation function
        y = self.activation(z)
        
        return y
        
    def BN_updates_1(self):
        
        updates = []
        
        if self.BN == True:
        
            updates.append((self.sum, self.sum + T.sum(self.z,axis=0))) 
            updates.append((self.sum2, self.sum2 + T.sum(self.z**2,axis=0)))
        
        return updates
        
    def BN_updates_2(self,n_samples):
        
        updates = []
        
        if self.BN == True:
        
            # reset the sums
            updates.append((self.sum, 0.* self.sum))
            updates.append((self.sum2, 0.* self.sum2))
            
            # for the GPU
            n_samples = T.cast(n_samples,dtype=theano.config.floatX)
            
            # compute the mean and variance
            mean = self.sum/n_samples
            mean2 = self.sum2/n_samples
            
            updates.append((self.mean, mean))
            
            # variance = mean(x^2) - mean(x)^2
            updates.append((self.var, mean2 - mean**2))
        
        return updates
        
    def bprop(self, cost):
        
        # tanh derivative...
        # I have to do that manually because of the rounding
        dEdWb = T.grad(cost=cost, wrt=self.Wb)
        dWbdW = 1 - self.Wb**2
        self.dEdW = dWbdW*dEdWb
        
        # self.dEdW = T.grad(cost=cost, wrt=self.Wb)
        
        self.dEdb = T.grad(cost=cost, wrt=self.b)
        
        if self.BN == True:
            self.dEda = T.grad(cost=cost, wrt=self.a)
        
    def parameters_updates(self, LR, M):    
        
        updates = []
        
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
        
        # updating and scaling up gradient to 0 and ones
        # new_W = self.W - LR * self.dEdWb / self.W_scale
        # new_W = self.W - LR / (self.W_scale ** 2) * self.dEdW 
        # new_W = self.W - LR / self.W_scale * self.dEdW 

        # compute updates
        new_update_W = M * self.update_W - LR * self.dEdW
        new_update_b = M * self.update_b - LR * self.dEdb
        
        # compute new parameters. Note that we use a better precision than the other operations
        new_W = self.W + new_update_W
        neWb = self.b + new_update_b
        
        # classic update
        # new_W = self.W - LR * self.W_lr_scale * self.dEdW 
        # new_W = self.W - LR * self.dEdW 
        
        # discretization + stochastic rounding
        # new_W = T.clip(new_W,-self.w0,self.w0)
        
        # new_W = (new_W+self.w0)/(2*self.w0) 
        # new_W is in [0,1]
        
        # new_W = stochastic_rounding(new_W,self.rng)
        # new_W = T.round(new_W)
        
        # BTW, as we are in [0,1], no need for the sign bit
        # new_W = fixed_point(X=new_W,NOB=8,NOIB=0,saturation=True)
        # new_W = fixed_point(X=new_W,NOB=7,NOIB=-8,saturation=True)
        # new_W = fixed_point(X=new_W,NOB=7,NOIB=-8,saturation=True,stochastic=True , rng=self.rng)
        
        # new_W is either 0 or 1
        # new_W = self.w0*(2*new_W-1)
        # new_W is either -w0 or w0        
        
        # saturation learning rule
        if self.saturation is not None:
            
            high = np.float32(self.saturation*self.high)
            new_W = T.clip(new_W,-high,high) 
            
            # new_W = T.clip(new_W,-self.saturation,self.saturation)
            
            # new_W = T.clip(new_W,-self.high,self.high)
        
        # TODO: does it work with CNN ?
        # L2 column constraint on W
        if self.max_col_norm is not None:
            col_norms = T.sqrt(T.sum(T.sqr(new_W), axis=0))
            # col_norms = T.max(new_W, axis=0)
            desired_norms = T.clip(col_norms, 0, self.max_col_norm) # clip = saturate below min and beyond max
            new_W = new_W * (desired_norms / (1e-7 + col_norms))
        
        # if self.saturation == True :
            # print "OK"
            # new_W = T.clip(new_W,-self.high,self.high)
        
        # linear quantization
        if self.update_bit_width is not None:
            
            new_W = linear_quantization(x=new_W,bit_width=self.update_bit_width,min=-self.high,max=self.high,
                stochastic=self.update_stochastic_rounding,rng=self.rng)
        
        updates.append((self.W, new_W))
        updates.append((self.b, neWb))
        updates.append((self.update_W, new_update_W))
        updates.append((self.update_b, new_update_b)) 
        
        if self.BN == True:
            new_a = self.a - LR * self.dEda
            updates.append((self.a, new_a))

        return updates

class ReLU_layer(linear_layer):
        
    def activation(self,z):
    
        # return T.maximum(0.,z)
        return T.maximum(z*.01,z)
        
        # Roland activation function
        # return T.ge(z,1.)*z
        
class ReLU_conv_layer(linear_layer): 
    
    def __init__(self, rng, image_shape, zero_pad, filter_shape, filter_stride, pool_shape, pool_stride, output_shape, partial_sum, BN,
        prop_bit_width=None, prop_stochastic_rounding=False,
        update_bit_width=None, update_stochastic_rounding=False):
        
        self.rng = rng
        
        self.image_shape = image_shape
        print "        image_shape = "+str(image_shape)
        self.zero_pad = zero_pad
        print "        zero_pad = "+str(zero_pad)
        self.filter_shape = filter_shape
        print "        filter_shape = "+str(filter_shape)
        self.filter_stride = filter_stride
        print "        filter_stride = "+str(filter_stride)
        self.pool_shape = pool_shape
        print "        pool_shape = "+str(pool_shape)
        self.pool_stride = pool_stride
        print "        pool_stride = "+str(pool_stride)
        self.output_shape = output_shape
        print "        output_shape = "+str(output_shape)         
        self.partial_sum = partial_sum
        print "        partial_sum = "+str(partial_sum)
        self.BN = BN
        print "        BN = "+str(BN)
        # self.W_lr_scale = W_lr_scale
        # print "    W_lr_scale = "+str(W_lr_scale)
        
        self.prop_bit_width = prop_bit_width
        print "        prop_bit_width = "+str(prop_bit_width)
        self.prop_stochastic_rounding = prop_stochastic_rounding
        print "        prop_stochastic_rounding = "+str(prop_stochastic_rounding)     
        self.update_bit_width = update_bit_width
        print "        update_bit_width = "+str(update_bit_width)
        self.update_stochastic_rounding = update_stochastic_rounding
        print "        update_stochastic_rounding = "+str(update_stochastic_rounding)   

        # range of init
        n_inputs = np.prod(filter_shape[1:])
        n_units = (filter_shape[0] * np.prod(filter_shape[2:])/ np.prod(pool_shape)) 

        # initialize weights with random weights
        self.high = np.float32(np.sqrt(6. / (n_inputs + n_units)))
        
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
        self.sum = theano.shared(value=b_values, name='sum')
        self.sum2 = theano.shared(value=b_values, name='sum2')
        
    def fprop(self, x, can_fit):
        
        # shape the input as it should be (not necessary)
        # x = x.reshape(self.image_shape)
        
        # discrete weights
        if self.prop_bit_width is not None:
            self.Wb = linear_quantization(x=self.W,bit_width=self.prop_bit_width,min=-self.high,max=self.high,
                stochastic=self.prop_stochastic_rounding,rng=self.rng)
            
        # continuous weights
        else:
            self.Wb = self.W
        
        # convolution
        x = x.dimshuffle(1, 2, 3, 0) # bc01 to c01b
        W = self.Wb.dimshuffle(1, 2, 3, 0) # bc01 to c01b
        conv_op = FilterActs(stride=self.filter_stride, partial_sum=self.partial_sum,pad = self.zero_pad)
        x = gpu_contiguous(x)
        W = gpu_contiguous(W)
        z = conv_op(x, W)
        
        # Maxpooling
        pool_op = MaxPool(ds=self.pool_shape, stride=self.pool_stride)
        z = pool_op(z)
        z = z.dimshuffle(3, 0, 1, 2) # c01b to bc01
        
        # for BN
        self.z = z
        
        # batch normalization
        if can_fit == True:
            
            # in the convolutional case, there is only a mean per feature map and not per location
            # http://arxiv.org/pdf/1502.03167v3.pdf
            mean = T.mean(z,axis=(0,2,3))
            var = T.var(z,axis=(0,2,3))
            
        else:
            mean = self.mean
            var = self.var
        
        if self.BN == True:
            z = (z - mean.dimshuffle('x', 0, 'x', 'x'))/(T.sqrt(var.dimshuffle('x', 0, 'x', 'x')+1e-9))
            z = self.a.dimshuffle('x', 0, 'x', 'x') * z
        
        # bias
        z = z + self.b.dimshuffle('x', 0, 'x', 'x')
        
        # activation
        y = self.activation(z)
        
        return y
        
    def BN_updates_1(self):
        
        updates = []
        
        if self.BN == True:
        
            updates.append((self.sum, self.sum + T.sum(self.z,axis=(0,2,3)))) 
            updates.append((self.sum2, self.sum2 + T.sum(self.z**2,axis=(0,2,3))))
        
        return updates
        
    def BN_updates_2(self,n_samples):
        
        updates = []
        
        if self.BN == True:
        
            # reset the sums
            updates.append((self.sum, 0.* self.sum))
            updates.append((self.sum2, 0.* self.sum2))
            
            # for the GPU
            n_samples = T.cast(n_samples*self.output_shape[2]*self.output_shape[3],dtype=theano.config.floatX)
            
            # compute the mean and variance
            mean = self.sum/n_samples
            mean2 = self.sum2/n_samples
            
            updates.append((self.mean, mean))
            
            # variance = mean(x^2) - mean(x)^2
            updates.append((self.var, mean2 - mean**2))
        
        return updates
        
    def activation(self,z):
    
        # return T.maximum(0.,z)
        return T.maximum(z*.01,z)
        
