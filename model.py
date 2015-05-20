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
        
class Network(object):
    
    layer = []                
    
    def __init__(self, n_hidden_layer, BN):
        
   
        self.n_hidden_layers = n_hidden_layer
        print "    n_hidden_layers = "+str(n_hidden_layer)    
        self.BN = BN
        print "    BN = "+str(BN)   
    
    # Monte Carlo averaging
    # So far, I change the weigths at each minibatch, which makes no sense hardware wise.
    # the way it would make sense = sample x set of weights, and use them from A to Z, then monte carlo.
    # how would I do it ?
    # I should use a shared variable for the binarized weights
    # I should use a shared variable for the output average.
    # I should compute BN mean and var for each of the sampled set of weights
    
    def fprop(self, x, eval):
    
        for k in range(self.n_hidden_layers+1):
            x = self.layer[k].fprop(x, eval)
        
        return x

    # when you use fixed point, you cannot use T.grad directly -> bprop modifications.
    def bprop(self, y, t):
        
        batch_size = T.cast(T.shape(y)[0], dtype=theano.config.floatX)
        
        # MSE
        # cost = T.sum(T.sqr(y-t))/batch_size
        
        # squared hinge loss
        cost = T.sum(T.sqr(T.maximum(0.,1.-t*y)))/batch_size
        
        # multi class squared hinge loss
        # cost = T.mean(T.sqr(T.maximum(0.,T.max(1.-t*y,axis=1))))
        
        # hinge loss
        # cost = T.sum(T.maximum(0.,1.-t*y))/batch_size
        
        # bprop
        for k in range(self.n_hidden_layers,-1,-1):
            self.layer[k].bprop(cost)
    
    def BN_updates(self,x):
        
        y = self.fprop(x=x,eval=False) 
        
        updates = self.layer[0].BN_updates()
        for k in range(1,self.n_hidden_layers+1):
            updates = updates + self.layer[k].BN_updates()
        
        return updates
        
    def BN_reset(self):
        
        updates = self.layer[0].BN_reset()
        for k in range(1,self.n_hidden_layers+1):
            updates = updates + self.layer[k].BN_reset()
        
        return updates
    
    # you give it the input and the target and it gives you the updates
    def parameters_updates(self, x, t, LR, M):
        
        y = self.fprop(x=x,eval=False)        
        self.bprop(y, t)
        
        # updates
        updates = self.layer[0].parameters_updates(LR, M)
        for k in range(1,self.n_hidden_layers+1):
            updates = updates + self.layer[k].parameters_updates(LR, M)
        
        return updates
    
    def errors(self, x, t):
        
        y = self.fprop(x=x,eval=True)
        
        # error function
        errors = T.sum(T.neq(T.argmax(y, axis=1), T.argmax(t, axis=1)))
        
        return errors
        
    def monitor(self):
        
        for k in range(0,self.n_hidden_layers+1):
        
            W = self.layer[k].W.get_value()
            # W1 = self.layer[k].W1.get_value()
            b = self.layer[k].b.get_value()
            # b1 = self.layer[k].b1.get_value()

            # print "        layer "+str(k)+" weights max abs = "+str(np.max(np.abs(W)))   
            print "        layer "+str(k)+" weights mean abs = "+str(np.mean(np.abs(W)))   
            # print "        layer "+str(k)+" weights = "+str(W)  
            
            # print "         Layer "+str(k)+":"
            # print "             Weights abs max = "+str(np.max(np.abs(W)))
            # print "             Weights 1 max = "+str(np.max(W1))
            # print "             Weights 1 min = "+str(np.min(W1)) 
            # print "             Weights 1 mean = "+str(np.mean(W1)) 
            # print "             Bias max 1 = "+str(np.max(b1)) 
            # print "             Bias min 1 = "+str(np.min(b1)) 
            # print "             Bias mean 1 = "+str(np.mean(b1))

    def save_params_file(self, path):        
        
        # Open the file and overwrite current contents
        save_file = open(path, 'wb')
        
        # write all the parameters in the file
        for k in xrange(self.n_hidden_layers+1):
            cPickle.dump(self.layer[k].W.get_value(), save_file, -1)
            cPickle.dump(self.layer[k].b.get_value(), save_file, -1)
            
            # BN stuff  
            if self.BN == True:
                cPickle.dump(self.layer[k].a.get_value(), save_file, -1)
                cPickle.dump(self.layer[k].mean.get_value(), save_file, -1)
                cPickle.dump(self.layer[k].var.get_value(), save_file, -1)
            
        # close the file
        save_file.close()
        
    def load_params_file(self, path): 
        
        # Open the file
        save_file = open(path)
        
        # read an load all the parameters
        for k in xrange(self.n_hidden_layers+1):
            self.layer[k].W.set_value(cPickle.load(save_file))
            self.layer[k].b.set_value(cPickle.load(save_file))
            
            # BN stuff  
            if self.BN == True:
                self.layer[k].a.set_value(cPickle.load(save_file))
                self.layer[k].mean.set_value(cPickle.load(save_file))
                self.layer[k].var.set_value(cPickle.load(save_file))

        # close the file
        save_file.close()
        