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

# float16 function
# we are using the nvidia cuda function (only works on GPU)
class Float16(UnaryScalarOp):

    def impl(self, x):
        return numpy.float32(numpy.float16(x))
    
    def c_code(self, node, name, (x,), (z,), sub):
        return "%(z)s = __half2float(__float2half_rn(%(x)s));" % locals()  
float16_scalar = Float16(same_out_nocomplex, name='float16')
float16 = Elemwise(float16_scalar)

def stochastic_rounding(x, rng):
    
    p = x-T.floor(x)
    
    srng = T.shared_randomstreams.RandomStreams(rng.randint(999999))
    p_mask = T.cast(srng.binomial(n=1, p=p, size=T.shape(x)), theano.config.floatX)
    
    x = T.floor(x) + p_mask
    
    return x
    
# this function simulate the precision and the range of a fixed point 
# while working with floats
# NOB = Number Of Bits = bit-width
# NOIB = Number Of Integer Bits = position of the radix point = range
def fixed_point(X,NOB, NOIB, saturation=True, stochastic=False, rng=None):
    
    power = T.cast(2.**(NOB - NOIB), theano.config.floatX) # float !
    max = T.cast((2.**NOB)-1, theano.config.floatX)
    value = X*power 
    
    if stochastic == True:
        value = stochastic_rounding(value, rng)
    
    else:
        value = T.round(value) # round to nearest
    
    if saturation == True:
        value = T.clip(value, -max, max) # saturation arithmetic
    
    else:
        # TODO http://en.wikipedia.org/wiki/Modular_arithmetic
        raise NotImplementedError("Modular arithmetic not implemented yet")
    
    value = value/power
    return value
    