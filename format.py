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
import time

# For the Discretization
from theano.scalar.basic import UnaryScalarOp, BinaryScalarOp, upcast_out, same_out_nocomplex
from theano.tensor.elemwise import Elemwise

# The only reason I am implementing my own op is for the identity gradient
class Discretize(BinaryScalarOp):
    
    # Is bprop discrete ?
    
    # In theory yes:
    # z = W_d * x
    # According to the chain rule, 
    # Bprop is discrete for the inputs gradient:
    # dEdx = dEdz * dzdx = dEdz * W_d
    # However, it is not for the parameters gradient:
    # dEdw = dEdz * dzd(W_d) * d(W_d)dW 
    # The last term is identity, so:
    # dEdw = dEdz * dzd((W_d)) = dEdz * x

    # did experiments to check:
    # multiply both W_lr_scale and W initialization by 10 -> no change on the results
    # manual gradient instead of T.grad -> same results
    
    def c_code(self, node, name, (x, y), (z, ), sub):
        return "%(z)s = %(y)s * (2*(%(x)s >= 0)-1);" % locals()

    def grad(self, (x, y), (gz, )):
        return gz, y.zeros_like().astype(theano.config.floatX)

discretize = Elemwise(Discretize(upcast_out, name='discretize'))

def stochastic_rounding(x, rng):
    
    p = x-T.floor(x)
    
    theano.sandbox.rng_mrg.MRG_RandomStreams
    
    # much slower :(
    # srng = T.shared_randomstreams.RandomStreams(rng.randint(999999))
    
    # much faster :)
    # https://github.com/Theano/Theano/issues/1233#event-262671708
    # does it work though ?? It seems so :)
    srng = theano.sandbox.rng_mrg.MRG_RandomStreams(rng.randint(999999))
    
    p_mask = T.cast(srng.binomial(n=1, p=p, size=T.shape(x)), theano.config.floatX)
    
    x = T.floor(x) + p_mask
    
    return x

# unlike fixed point:
# min and max are not the same power of 2,
# sign is counted in the bit_width
def linear_quantization(x,bit_width,min=None,max=None,stochastic=False,rng=None):
    
    # n is the number of possible values other than 0 (hence the -1)
    n = -1 + 2**bit_width
    
    # unfortunately, this part requires to have the whole matrix with high precision
    # As a result, I am not using it
    if min == None:
        min = T.min(x)
    if max == None:
        max = T.max(x)
    
    # should not be necessary but one never knows
    x = T.clip(x,min,max)
    # x is in [min,max]
    x = (x-min)/(max-min)
    # x is in [0,1]
    x = x*n
    # x is in [0,n]
    
    if stochastic == True:
        x = stochastic_rounding(x,rng)
    else: 
        x = T.round(x)
        
    # x is an integer in [0,n]
    x = x/n
    # x is in [0,1]
    x = x*(max-min) + min
    # x is in [min,max]
    
    return x

# merge those with linear quantization ...

# PB: is it exponential or log quantization ??
# I get a better resolution for the values near 0
def exponential_quantization(x,bit_width,min=None,max=None,stochastic=False,rng=None):
    
    # n is the number of possible values other than 0 (hence the -1)
    n = -1 + 2**bit_width
    
    # unfortunately, this part requires to have the whole matrix with high precision
    # As a result, I am not using it
    if min == None:
        min = T.min(x)
    if max == None:
        max = T.max(x)
    
    # should not be necessary but one never knows
    x = T.clip(x,min,max)
    # x is in [min,max]
    x = (x-min)/(max-min)
    # x is in [0,1]
    
    e = np.exp(1)
    x = x*np.float32(e-1)+np.float32(1)
    # x is in [1,e]
    
    x = T.log(x)
    # x is in [0,1]
    
    x = x*n
    # x is in [0,n]
    
    if stochastic == True:
        x = stochastic_rounding(x,rng)
    else: 
        x = T.round(x)
        
    # x is an integer in [0,n]
    x = x/n
    # x is in [0,1]
    
    x = T.exp(x)
    # x is in [1,e]
    
    x = (x-np.float32(1))/np.float32(e-1)
    # x is in [0,1]
    
    x = x*(max-min) + min
    # x is in [min,max]
    
    return x

# PB: is it quadratic or root quantization ???
def quadratic_quantization(x,bit_width,min=None,max=None,stochastic=False,rng=None):
    
    # n is the number of possible values other than 0 (hence the -1)
    n = -1 + 2**bit_width
    
    # unfortunately, this part requires to have the whole matrix with high precision
    # As a result, I am not using it
    if min == None:
        min = T.min(x)
    if max == None:
        max = T.max(x)
    
    # should not be necessary but one never knows
    x = T.clip(x,min,max)
    # x is in [min,max]
    x = (x-min)/(max-min)
    # x is in [0,1]
    
    x = T.sqrt(x)
    # x = T.pow(x,1./1.5)
    # x is in [0,1]
    
    x = x*n
    # x is in [0,n]
    
    if stochastic == True:
        x = stochastic_rounding(x,rng)
    else: 
        x = T.round(x)
        
    # x is an integer in [0,n]
    x = x/n
    # x is in [0,1]
    
    x = T.sqr(x)
    # x = x = T.pow(x,1.5)
    # x is in [0,1]
    
    x = x*(max-min) + min
    # x is in [min,max]
    
    return x
    
def root_quantization(x,bit_width,min=None,max=None,stochastic=False,rng=None):
    
    # n is the number of possible values other than 0 (hence the -1)
    n = -1 + 2**bit_width
    
    # unfortunately, this part requires to have the whole matrix with high precision
    # As a result, I am not using it
    if min == None:
        min = T.min(x)
    if max == None:
        max = T.max(x)
    
    # should not be necessary but one never knows
    x = T.clip(x,min,max)
    # x is in [min,max]
    x = (x-min)/(max-min)
    # x is in [0,1]
    
    x = T.sqr(x)
    # x is in [0,1]
    
    x = x*n
    # x is in [0,n]
    
    if stochastic == True:
        x = stochastic_rounding(x,rng)
    else: 
        x = T.round(x)
        
    # x is an integer in [0,n]
    x = x/n
    # x is in [0,1]
    
    x = T.sqrt(x)
    # x is in [0,1]
    
    x = x*(max-min) + min
    # x is in [min,max]
    
    return x
    