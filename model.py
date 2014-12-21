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
import scipy.stats
from pylearn2.sandbox.cuda_convnet.filter_acts import FilterActs
from theano.sandbox.cuda.basic_ops import gpu_contiguous
from pylearn2.sandbox.cuda_convnet.pool import MaxPool

from layer import Maxout_conv_layer, SoftmaxLayer, MaxoutLayer            
        
        
class deep_dropout_network(object):
    
    layer = []                
    
    def __init__(self, rng, batch_size, n_hidden_layers, comp_precision, update_precision, initial_range, max_sat):
        
        print '    Overall description:'
        print '        Batch size = %i' %(batch_size)
        print '        Number of layers = %i' %(n_hidden_layers)
        print '        Computation precision = %i bits' %(comp_precision)
        print '        Update precision = %i bits' %(update_precision)
        print '        Initial range = %i bits' %(initial_range)
        print '        Maximum ratio of saturated variables = %f %%' %(max_sat*100)   
        
        self.rng = rng
        self.batch_size = batch_size
        self.n_hidden_layers = n_hidden_layers
        self.comp_precision = comp_precision
        self.update_precision = update_precision
        self.initial_range = initial_range
        self.max_sat = max_sat
    
    def fprop(self, x):
    
        y = self.layer[0].fprop(x)
        
        for k in range(1,self.n_hidden_layers+1):

            y = self.layer[k].fprop(y)
        
        return y
    
    def dropout_fprop(self, x):
        
        y = self.layer[0].dropout_fprop(x)
        
        for k in range(1,self.n_hidden_layers+1):

            y = self.layer[k].dropout_fprop(y)
        
        return y
        
    # when you use fixed point, you cannot use T.grad directly -> bprop modifications.
    def bprop(self, y, t):
        
        # there is a simplification between softmax derivative and nll derivative        
        dEdy = (y-t)/T.cast(T.shape(y)[1],dtype=theano.config.floatX) # /2. # actually, it is dEdz and not dEdy
        
        # bprop
        for k in range(self.n_hidden_layers,-1,-1):
            dEdy = self.layer[k].bprop(dEdy)
            
    # you give it the input and the target and it gives you the updates
    def updates(self, LR, M):
            
        # updates
        updates = self.layer[0].updates(LR, M)
        for k in range(1,self.n_hidden_layers+1):
            updates = updates + self.layer[k].updates(LR, M)
        
        return updates

    # train function
    def bprop_updates(self, x, t, LR, M):
    
        y = self.dropout_fprop(x)
        self.bprop(y,t)
        updates = self.updates(LR,M)
        
        return updates   
    
    def errors(self, x, t):
        
        y = self.fprop(x) 
        
        # error function
        errors = T.sum(T.neq(T.argmax(y, axis=1), T.argmax(t, axis=1)))
        
        return errors
    
    def save_params(self):        
        
        self.W_save = []
        self.b_save = []
        
        for k in xrange(self.n_hidden_layers+1):
            self.W_save.append(self.layer[k].W.get_value())
            self.b_save.append(self.layer[k].b.get_value())
        
    def load_params(self): 
        
        # read an load all the parameters
        for k in xrange(self.n_hidden_layers+1):
            self.layer[k].W.set_value(self.W_save[k])
            self.layer[k].b.set_value(self.b_save[k])
    
    def save_params_file(self, path):        
        
        # Open the file and overwrite current contents
        save_file = open(path, 'wb')
        
        # write all the parameters in the file
        for k in xrange(self.n_hidden_layers+1):
            cPickle.dump(self.layer[k].W.get_value(), save_file, -1)
            cPickle.dump(self.layer[k].b.get_value(), save_file, -1)
        
        # close the file
        save_file.close()
        
    def load_params_file(self, path): 
        
        # Open the file
        save_file = open(path)
        
        # read an load all the parameters
        for k in xrange(self.n_hidden_layers+1):
            self.layer[k].W.set_value(cPickle.load(save_file))
            self.layer[k].b.set_value(cPickle.load(save_file))

        # close the file
        save_file.close()
    
    # function that updates the ranges of all fixed point vectors
    def range_updates(self,x,t):
            
        y = self.dropout_fprop(x)
        self.bprop(y,t)
        
        range_updates = self.layer[0].range_updates()
        for k in range(1,self.n_hidden_layers+1):
            range_updates = range_updates + self.layer[k].range_updates()
        
        return range_updates
        
    def print_range(self):
        
        for k in xrange(self.n_hidden_layers+1):
            print '        Layer %i range:'%(k)
            self.layer[k].print_range()
            
    def set_comp_precision(self, comp_precision):
        
        for k in xrange(self.n_hidden_layers+1):
            self.layer[k].comp_precision.set_value(comp_precision)
    
    def get_comp_precision(self):
        
        return self.layer[0].comp_precision.get_value()
            
    def set_update_precision(self, update_precision):
        
        for k in xrange(self.n_hidden_layers+1):
            self.layer[k].update_precision.set_value(update_precision)
    
    def get_update_precision(self):
        
        return self.layer[0].update_precision.get_value()
    
    def set_max_ratio(self, max_ratio):
        
        for k in xrange(self.n_hidden_layers+1):
            self.layer[k].max_ratio.set_value(max_ratio)
            
    def get_max_ratio(self):
        
        return self.layer[0].max_ratio.get_value()

class PI_MNIST_model(deep_dropout_network):

    def __init__(self, rng, batch_size, n_input, n_output, n_hidden, n_pieces, n_hidden_layers, 
        p_input,  scale_input, p_hidden, scale_hidden, max_col_norm, 
        comp_precision, update_precision, initial_range, max_sat):
        
        deep_dropout_network.__init__(self, rng, batch_size, n_hidden_layers, comp_precision, update_precision, initial_range, max_sat)
        
        print '        n_input = %i' %(n_input)
        print '        n_output = %i' %(n_output)
        print '        n_hidden = %i' %(n_hidden)
        print '        n_pieces = %i' %(n_pieces)
        print '        p_input = %f' %(p_input)
        print '        scale_input = %f' %(scale_input)
        print '        p_hidden = %f' %(p_hidden)
        print '        scale_hidden = %f' %(scale_hidden)
        print '        max_col_norm = %f' %(max_col_norm)
        
        # save the parameters
        self.n_input = n_input
        self.n_output = n_output
        self.n_hidden = n_hidden
        self.n_pieces = n_pieces
        self.p_input = p_input
        self.scale_input = scale_input
        self.p_hidden = p_hidden
        self.scale_hidden = scale_hidden
        self.max_col_norm = max_col_norm
        
        # Create MLP layers    
        if self.n_hidden_layers == 0 :
            
            print "    Softmax layer:"
            
            self.layer.append(SoftmaxLayer(rng = self.rng, n_inputs=self.n_input, n_units=self.n_output, 
                p = self.p_input, scale = self.scale_input, max_col_norm = self.max_col_norm,
                comp_precision = self.comp_precision, update_precision = self.update_precision, initial_range = self.initial_range, max_sat = self.max_sat))

        else :
            
            print "    Maxout layer 1:"
            
            self.layer.append(MaxoutLayer(rng = self.rng, n_inputs = self.n_input, n_units = self.n_hidden, n_pieces = self.n_pieces, 
                p = self.p_input, scale = self.scale_input, max_col_norm = self.max_col_norm,
                comp_precision = self.comp_precision, update_precision = self.update_precision, initial_range = self.initial_range, max_sat = self.max_sat))

            for k in range(1,self.n_hidden_layers):
                
                print "    Maxout layer "+str(k+1)+":"
                self.layer.append(MaxoutLayer(rng = self.rng, n_inputs = self.n_hidden, n_units = self.n_hidden, n_pieces = self.n_pieces, 
                p = self.p_hidden, scale = self.scale_hidden, max_col_norm = self.max_col_norm,
                comp_precision = self.comp_precision, update_precision = self.update_precision, initial_range = self.initial_range, max_sat = self.max_sat))
            
            print "    Softmax layer:"
            
            self.layer.append(SoftmaxLayer(rng = self.rng, n_inputs= self.n_hidden, n_units= self.n_output, 
                p = self.p_hidden, scale = self.scale_hidden, max_col_norm = self.max_col_norm,
                comp_precision = self.comp_precision, update_precision = self.update_precision, initial_range = self.initial_range, max_sat = self.max_sat))     
        
class MNIST_model(deep_dropout_network):
    
    def __init__(self, rng, batch_size, comp_precision, update_precision, initial_range, max_sat):
        
        deep_dropout_network.__init__(self, rng, batch_size, 3, comp_precision, update_precision, initial_range, max_sat)
        
        print "    Convolution layer 1:"
        
        self.layer.append(Maxout_conv_layer(
            rng,
            image_shape=(batch_size, 1, 28, 28),
            zero_pad = 0, 
            output_shape=(batch_size, 48, 10, 10),
            filter_shape=(48, 1, 8, 8),
            filter_stride = 1,
            n_pieces = 2,
            pool_shape=(4, 4),
            pool_stride = 2,
            p = 0.8, 
            scale = 1., 
            max_col_norm = 0.9,
            comp_precision = comp_precision, 
            update_precision = update_precision, 
            initial_range = initial_range, 
            max_sat = max_sat
        ))
        
        
        print "    Convolution layer 2:"
        
        self.layer.append(Maxout_conv_layer(
            rng,
            image_shape=(batch_size, 48, 10, 10),
            zero_pad = 3, # add n zero on both side of the input
            output_shape=(batch_size, 48, 4, 4),
            filter_shape=(48, 48, 8, 8),
            filter_stride = 1,
            n_pieces = 2,
            pool_shape=(4, 4),
            pool_stride =2,
            p = 0.5, 
            scale = 0.5, 
            max_col_norm = 1.9365,
            comp_precision = comp_precision, 
            update_precision = update_precision, 
            initial_range = initial_range, 
            max_sat = max_sat
        ))   
               
        
        print "    Convolution layer 3:"
        
        self.layer.append(Maxout_conv_layer(
            rng,
            image_shape=(batch_size, 48, 4, 4),
            zero_pad = 3, # add n zero on both side of the input
            output_shape=(batch_size, 24, 3, 3),
            filter_shape=(24, 48, 5, 5),
            filter_stride = 1,
            n_pieces = 4,
            pool_shape=(2, 2),
            pool_stride =2,
            p = 0.5, 
            scale = 0.5, 
            max_col_norm = 1.9365,
            comp_precision = comp_precision, 
            update_precision = update_precision, 
            initial_range = initial_range, 
            max_sat = max_sat
        )) 
        
        print "    Softmax layer:"
        
        self.layer.append(SoftmaxLayer(
            rng = rng, 
            n_inputs= 24*3*3, 
            n_units = 10, 
            p = 0.5, 
            scale = 0.5, 
            max_col_norm =1.9365,
            comp_precision = comp_precision, 
            update_precision = update_precision, 
            initial_range = initial_range, 
            max_sat = max_sat
        ))
        
class CIFAR10_SVHN_model(deep_dropout_network):
    
    def __init__(self, rng, batch_size, comp_precision, update_precision, initial_range, max_sat):
        
        deep_dropout_network.__init__(self, rng, batch_size, 4, comp_precision, update_precision, initial_range, max_sat)
        
        print "    Convolution layer 1:"
        
        self.layer.append(fixed_Maxout_conv_layer(
            rng,
            image_shape=(batch_size, 3, 32, 32),
            zero_pad = 2, 
            output_shape=(batch_size, 64, 16, 16), # 64 does fit in memory
            filter_shape=(64, 3, 5, 5),
            filter_stride = 1,
            n_pieces = 2,
            pool_shape=(3, 3),
            pool_stride = 2,
            p = 0.8, 
            scale = 1., 
            max_col_norm = 0.9,
            comp_precision = comp_precision, 
            update_precision = update_precision, 
            initial_range = initial_range, 
            max_sat = max_sat,
            w_LR_scale = 0.2,
            b_LR_scale = 0.2,
            # partial_sum = 32 # total number = 33*33
        ))
        
        
        print "    Convolution layer 2:"
        
        self.layer.append(fixed_Maxout_conv_layer(
            rng,
            image_shape=(batch_size, 64, 16, 16),
            zero_pad = 2, # add n zero on both side of the input
            output_shape=(batch_size, 128, 8, 8),
            filter_shape=(128, 64, 5, 5),
            filter_stride = 1,
            n_pieces = 2,
            pool_shape=(3, 3),
            pool_stride =2,
            p = 0.5, 
            scale = 0.5, 
            max_col_norm = 1.9365,
            comp_precision = comp_precision, 
            update_precision = update_precision, 
            initial_range = initial_range, 
            max_sat = max_sat,
            w_LR_scale = 0.2,
            b_LR_scale = 0.2,
            # partial_sum = 16 # total number = 15*15
        ))   
               
        
        print "    Convolution layer 3:"
        
        self.layer.append(fixed_Maxout_conv_layer(
            rng,
            image_shape=(batch_size, 128, 8, 8),
            zero_pad = 2, # add n zero on both side of the input
            output_shape=(batch_size, 128, 4, 4),
            filter_shape=(128, 128, 5, 5),
            filter_stride = 1,
            n_pieces = 2,
            pool_shape=(3, 3),
            pool_stride =2,
            p = 0.5, 
            scale = 0.5, 
            max_col_norm = 1.9365,
            comp_precision = comp_precision, 
            update_precision = update_precision, 
            initial_range = initial_range, 
            max_sat = max_sat,
            w_LR_scale = 0.2,
            b_LR_scale = 0.2,
            # partial_sum = 8 # total number = 9*9
        )) 
        
        print "    Maxout layer:"
        
        self.layer.append(fixed_MaxoutLayer(
            rng = rng, 
            n_inputs= 128*4*4, 
            n_units = 400,
            n_pieces = 5,
            p = 0.5, 
            scale = 0.5, 
            max_col_norm = 1.9365,
            comp_precision = comp_precision, 
            update_precision = update_precision, 
            initial_range = initial_range, 
            max_sat = max_sat
        ))
        
        print "    Softmax layer:"
        
        self.layer.append(fixed_SoftmaxLayer(
            rng = rng, 
            n_inputs= 400, 
            n_units = 10, 
            p = 0.5, 
            scale = 0.5, 
            max_col_norm = 1.9365,
            comp_precision = comp_precision, 
            update_precision = update_precision, 
            initial_range = initial_range, 
            max_sat = max_sat
        ))
          