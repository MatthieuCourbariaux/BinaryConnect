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

# this function simulate the precision and the range of a fixed point while working with floats
# saturation arithmetic
# rounding
# NOB = Number Of Bits = bit-width
# NOIB = Number Of Integer Bits = position of the radix point = range
def to_fixed(X,NOB, NOIB):
    
    power = T.cast(2.**(NOB - NOIB), theano.config.floatX) # float !
    max = T.cast((2.**NOB)-1, theano.config.floatX)
    value = X*power    
    value = T.round(value) #value = T.floor(value)
    value = T.clip(value, -max, max)
    value = value/power
    return value
    
    # return X 
        
    # T.float16(X)
        
# compute the new range of the fixed point representation
def new_range(overflow, overflow_1, max_overflow):
    
    # the goal is to have a new NOIB with  (overflow_rate(NOIB) <= max_overflow_rate) and (overflow_rate(NOIB+1) > max_overflow_rate)
    # the incremental solution is to :
        # compute the overflow_rate of NOIB and NOIB-1
        # increment NOIB if overflow_rate(NOIB)>max_overflow_rate
        # decrement NOIB if overflow_rate(NOIB-1)<= max_overflow_rate
        # keep NOIB if overflow_rate(NOIB) <= max_overflow_rate and overflow_rate(NOIB-1)> max_overflow_rate
    return T.switch(T.gt(overflow, max_overflow), 1, 
        T.switch(T.gt(overflow_1, max_overflow), 0, - 1))

# Overflow counter of a vector knowing its NOIB and NOB
def overflow(vector, NOB, NOIB):
    
    # compute the max value of the fixed point representation (i.e. the overflow value)
    max = ((2.**NOB)-1)/(2.**(NOB - NOIB))
    
    # compute the overflow rate of the vector
    overflow = T.mean(T.switch(T.ge(T.abs_(vector), max), 1., 0.))
    
    return overflow
    