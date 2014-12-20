import gzip
import cPickle
import numpy as np
import os
import os.path
import sys
import theano 
import theano.tensor as T
import time
import scipy.stats
from pylearn2.sandbox.cuda_convnet.filter_acts import FilterActs
from theano.sandbox.cuda.basic_ops import gpu_contiguous
from pylearn2.sandbox.cuda_convnet.pool import MaxPool

# this function simulate the precision and the range of a fixed point while working with floats
# saturation arithmetic
# rounding
# NOB = Number Of Bits = bit-width
# NOIB = Number Of Integer Bits = position of the radix point
def to_fixed(X,NOB, NOIB):
    
    power = T.cast(2.**(NOB - NOIB), theano.config.floatX) # float !
    max = T.cast((2.**NOB)-1, theano.config.floatX)
    value = X*power    
    value = T.round(value) #value = T.floor(value)
    value = T.clip(value, -max, max)
    value = value/power
    return value
    
    # return X 
    
    # return T.float16(X)
        
# compute the new range of the fixed point representation
def new_NOIB(vector, NOB, NOIB, max_sat):
    
    # compute the new range of the fixed point representation
    
    # the goal is to have a new NOIB with sat_ratio(NOIB) <= max ratio and sat_ratio(NOIB-1)> max ratio
    
    # the incremental solution is to :
    # compute the sat ratio of NOIB and NOIB-1
    # increment NOIB if sat_ratio(NOIB)>max_ratio
    # keep NOIB if sat_ratio(NOIB) <= max ratio and sat_ratio(NOIB-1)> max ratio
    # decrement NOIB if sat_ratio(NOIB-1)<= max_ratio
    
    return T.switch(T.gt(sat_ratio(vector,NOB,NOIB), max_sat), NOIB + 1, T.switch(T.gt(sat_ratio(vector,NOB,NOIB-1), max_sat), NOIB, NOIB - 1))
    
    # test of more secure way
    # return T.switch(T.gt(sat_ratio(vector,NOB,NOIB-1), max_sat), NOIB + 1, T.switch(T.gt(sat_ratio(vector,NOB,NOIB-2), max_sat), NOIB, NOIB - 1))

# compute the sat ratio of a vector knowing only its NOIB and NOB
def sat_ratio(vector, NOB, NOIB):
    
    # compute the max value of the fixed point representation (i.e. the saturation value)
    max = ((2.**NOB)-1)/(2.**(NOB - NOIB))
    
    # compute the saturation ratio of the vector
    sat_ratio = T.mean(T.switch(T.ge(T.abs_(vector), max), 1., 0.))
    
    return sat_ratio