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
        
class layer(object):
    
    def __init__(self, rng, n_inputs, n_units):
        
        self.rng = rng
        
        self.n_units = n_units
        self.n_inputs = n_inputs
        # self.threshold = 0.1* n_inputs
        
        W_values = np.asarray(self.rng.binomial(n=1, p=.5, size=(n_inputs, n_units)),dtype=theano.config.floatX)-0.5
        W1_values = np.asarray(self.rng.binomial(n=1, p=.5, size=(n_units, n_inputs)),dtype=theano.config.floatX)-0.5
        # W1_values = W_values.T
        
        # low=-np.sqrt(6. / (n_inputs + n_units))
        # high=np.sqrt(6. / (n_inputs + n_units))
        # W_values = np.asarray(self.rng.uniform(low=low,high=high,size=(n_inputs, n_units)),dtype=theano.config.floatX)
        # W1_values = np.asarray(self.rng.uniform(low=low,high=high,size=(n_units, n_inputs)),dtype=theano.config.floatX)
        
        # W_values = np.zeros((n_inputs, n_units),dtype=theano.config.floatX)
        # W1_values = np.zeros((n_units, n_inputs),dtype=theano.config.floatX)

        # b_values = np.zeros((n_units), dtype=theano.config.floatX) - n_units/2. # to have 1/2 neurons firing
        # b1_values = np.zeros((n_inputs), dtype=theano.config.floatX) - n_inputs /2. # to have 1/2 neurons firing
        b_values = np.zeros((n_units), dtype=theano.config.floatX)
        # b1_values = np.zeros((n_inputs), dtype=theano.config.floatX)
        
        # creation of shared symbolic variables
        # shared variables are the state of the built function
        # in practice, we put them in the GPU memory
        self.W = theano.shared(value=W_values, name='W')
        self.W1 = theano.shared(value=W1_values, name='W1')
        self.b = theano.shared(value=b_values, name='b')
        # self.b1 = theano.shared(value=b1_values, name='b1')
        
    def fprop(self, x):
        
        # for bprop
        self.x = x
        
        # scaled weighted sum
        self.z = T.dot(x, self.W) + self.b
        # self.z = T.dot(x, self.W)
        
        # activation (-.5 because tanh > sigmoid ?)
        self.y = T.cast(T.ge(self.z, 0.), theano.config.floatX)-0.5
        # self.y = T.maximum(self.z, 0.)
        # self.y = T.cast(T.ge(self.z, 0.), theano.config.floatX)
        # self.y = T.cast(T.eq(self.z, T.max(self.z)), theano.config.floatX)
        # y = T.ge(z, self.threshold)
        # self.y = T.ge(z, 12.)
        # y = z
        
        # return the output
        return self.y
        
    def bprop(self, dEdy):
        
        # t = self.y - dEdy
        # t = T.clip(self.y - dEdy,-.5,.5)
        
        # self.dEdW = self.W
        self.dEdb = T.sum(dEdy,axis=0)

        # case where xi = 1, yj = 0, tj = 1 and Wij = -1 -> update Wij to +1
        # case where xi = 1, yj = 1, tj = 0 and Wij = +1 -> update Wij to -1
        # self.dEdW = T.dot(self.x.T,(1.-self.y)*t)*0.5*(1.-self.W) - T.dot(self.x.T,self.y*(1.-t))*0.5*(self.W+1.)
        self.dEdW = T.dot(self.x.T,dEdy)
        # self.dEdW = T.dot((self.x.T+.5),(.5-y)*(t+.5))*(.5-self.W) - T.dot((self.x.T+.5),(y+.5)*(.5-t))*(self.W+.5)
        # self.dEdW = T.dot(self.x.T,(1.-self.y)*t)*(.5-self.W) - T.dot(self.x.T,self.y*(1.-t))*(self.W+.5)
        
        # case where yj = 0, tj = 1 and Wij = 0 -> update Wij to 1
        # self.dEdW1 = T.dot(T.ones_like(self.x.T),(1.-self.y)*t)*(1-self.W)
        

        # we only update the most wrong weight
        # dEdW1 = T.eq(dEdW1,T.maximum(T.max(dEdW1),1.))
        # dEdW0 = T.eq(dEdW0,T.maximum(T.max(dEdW0),1.))
        # self.dEdW = dEdW0-dEdW1
        
        # case where tj = 1, yj = 0 and xi = 0 -> update xi to 1
        # feedback alignment: take a random matrix instead of weight transpose
        # B = self.rng.binomial(n=1, p=0.5, size=(self.n_units,self.n_inputs))
        # B = self.W.get_value().T
        # dEdx1 = T.dot(t*(1.-self.y),self.W1)*(1.-self.x)
        # dEdx0 = T.dot((1.-t)*self.y,self.W1)*self.x 
        
        # dEdx0 = T.dot((1.-t)*self.y,self.W.T)*self.x
        # dEdx1 = T.eq(dEdx1,T.maximum(T.max(dEdx1),1.))
        # dEdx0 = T.eq(dEdx0,T.maximum(T.max(dEdx0),1.))
        
        # dEdx1 = T.ge(dEdx1,1.)
        # dEdx0 = T.ge(dEdx0,1.)    
        # dEdx = dEdx0-dEdx1
        # new_x = self.x - dEdx
        
        # T.cast(, theano.config.floatX)
        # return self.x, out
        
        # backpropagate the target
        
        # difference target prop
        # t1 = T.cast(T.ge(T.dot(t,self.W1), - self.b1), theano.config.floatX)
        # y1 = T.cast(T.ge(T.dot(y,self.W1), - self.b1), theano.config.floatX)
        
        # vanilla target prop
        # y1 = T.cast(T.ge(T.dot(self.y,self.W1), - self.b1), theano.config.floatX)
        
        # dEdy1 = y1 - self.x
        
        # self.dEdW = self.W
        # self.dEdb1 = T.sum(dEdy1,axis=0)

        # case where xi = 1, yj = 0, tj = 1 and Wij = 0 -> update Wij to 1
        # case where xi = 1, yj = 1, tj = 0 and Wij = 1 -> update Wij to 0
        # self.dEdW1 = T.dot(t.T,(1.-y1)*self.x)*(1-self.W1) - T.dot(t.T,y1*(1.-self.x))*self.W1
        # self.dEdW1 = T.dot(t.T,dEdy1)
        
        # t1 = T.cast(T.ge(T.dot(t,self.W1), - self.b1), theano.config.floatX)
        
        # return y1, t1
        # return t1
        
        # dEdx = T.cast(T.ge(T.dot(dEdy,self.W.T), 0.), theano.config.floatX)
        # dEdx = T.cast(T.ge(T.dot(dEdy,self.W1), 0.), theano.config.floatX)
        
        # Here is my current problem
        # different solutions: bprop, feedback alignment, difference target prop, vanilla target prop
        # self.dEdx = T.dot((.5-y)*(.5+t),(.5+self.W.T))*(.5-self.x) - T.dot((.5+y)*(.5-t),(.5+self.W.T))*(.5+self.x)
        # dEdx = T.dot(dEdy,self.W1)
        dEdx = T.dot(dEdy,self.W.T)
        # dEdx = T.clip(T.dot(dEdy,self.W.T),-.5,.5)
        # t = T.clip(self.x - (T.ge(dEdx,1.) - T.ge(-dEdx,1.))
        
        return dEdx
        # return dEdx/self.n_inputs
        
    def updates(self, LR):    
        
        # srng = T.shared_randomstreams.RandomStreams(self.rng.randint(999999))
        # LR_mask = T.cast(srng.binomial(n=1, p=LR, size=T.shape(self.W)), theano.config.floatX)
        # new_W = self.W + LR_mask * self.dEdW
        
        # batch_size = T.shape(self.x)[0]
        # self.dEdW1 = T.ge(self.dEdW1,batch_size*LR)
        # self.dEdW0 = T.ge(self.dEdW0,batch_size*LR)
        # dEdW = self.dEdW0-self.dEdW1
        # new_W = self.W - dEdW
        
        new_W = T.clip(self.W - (T.ge(self.dEdW*LR,1.) - T.ge(-self.dEdW*LR,1.)),-.5,.5)
        # new_W1 = self.W1 + T.ge(self.dEdW1*LR,1.) - T.ge(-self.dEdW1*LR,1.)
        # new_W = self.W - LR * self.dEdW
        # new_W1 = self.W1 - LR * self.dEdW1
        
        # new_W = T.cast(T.ge(self.W - T.ge(self.dEdW*LR,1.),1.), theano.config.floatX)
        
        # b_LR_mask = T.cast(srng.binomial(n=1, p=LR, size=T.shape(self.b)), theano.config.floatX)
        # self.dEdb = T.round(self.dEdb / 100)
        # new_b = self.b - b_LR_mask * self.dEdb
        new_b = self.b - LR * self.dEdb
        # new_b1 = self.b1 - LR * self.dEdb1

        # return the updates of shared variables
        updates = []
        updates.append((self.W, new_W))
        # updates.append((self.W1, new_W1))
        updates.append((self.b, new_b))
        # updates.append((self.b1, new_b1))
        
        return updates
        
