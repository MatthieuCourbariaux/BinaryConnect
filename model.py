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

from layer import Maxout_conv_layer, SoftmaxLayer, fixed_SoftmaxLayer, MaxoutLayer, fixed_MaxoutLayer, Sigmoid_layer, fixed_Maxout_conv_layer

from filter_plot import tile_raster_images
import Image

# def one_hot(t, r=None):

    # if r is None:
        # r = T.max(t) + 1
        
    # ranges = T.shape_padleft(T.arange(r), t.ndim)
    # rvalue = T.eq(ranges, T.shape_padright(t, 1))
    # rvalue = T.reshape(rvalue,(T.shape(t)[0],T.cast(r,dtype='int64')))
    
    # return rvalue

class deep_dropout_network(object):
    
    layer = []        
    
    def __init__(self, n_hidden_layers):
        
        print '    Number of layers = %i' %(n_hidden_layers)
        
        # save the parameters
        self.n_hidden_layers = n_hidden_layers
        
    def fprop(self, x):
    
        #x.reshape((100, 1, 28, 28))
        y = self.layer[0].fprop(x)
        
        for k in range(1,self.n_hidden_layers+1):
            #y.flatten(2)
            y = self.layer[k].fprop(y)
        
        return y
    
    def dropout_fprop(self, x):
        
        #x.reshape((100, 1, 28, 28))
        y = self.layer[0].dropout_fprop(x)
        
        for k in range(1,self.n_hidden_layers+1):
            #y.flatten(2)
            y = self.layer[k].dropout_fprop(y)
        
        return y

    def bprop(self, y, t):
        
        # is that the way they do it in pylearn 2 ?
        # in deeplearning.net they say mean to make it less dependant on the batch size
        # MSE = T.sum((y-t)*(y-t))/2./T.shape(y)[0] 
        NLL = T.sum(-T.log(y)*t)/T.shape(y)[1]
        
        # bprop
        for k in range(0,self.n_hidden_layers+1):
            self.layer[k].bprop(NLL)
            
    def target_prop(self, t):
        
        target = t
        
        for k in range(self.n_hidden_layers,-1,-1):
            target = self.layer[k].target_prop(target)
        
    # train function
    def bprop_updates(self, x, t, LR, M):
    
        y = self.dropout_fprop(x)
        self.bprop(y,t)
        updates = self.updates(LR,M)
        
        return updates 
        
    # train function
    def target_prop_updates(self, x, t, LR, M):
    
        y = self.dropout_fprop(x)
        self.target_prop(t-y)
        # self.target_prop(t)
        updates = self.updates(LR,M)
        
        return updates
    
    # you give it the input and the target and it gives you the updates
    def updates(self, LR, M):
            
        # updates
        updates = self.layer[0].updates(LR, M)
        for k in range(1,self.n_hidden_layers+1):
            updates = updates + self.layer[k].updates(LR, M)
        
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
 
class Sigmoid_MLP(deep_dropout_network):
    
    def __init__(self, rng,  n_hidden_layers, 
        n_input, p_input, scale_input, max_col_norm_input,
        n_hidden, p_hidden, scale_hidden, max_col_norm_hidden,
        n_output, p_output, scale_output, max_col_norm_output):
        
        deep_dropout_network.__init__(self, n_hidden_layers)
        
        # save the parameters
        self.rng = rng

        self.n_input = n_input
        self.p_input = p_input
        self.scale_input = scale_input   
        self.max_col_norm_input = max_col_norm_input         
        
        self.n_output = n_output
        self.p_output = p_output
        self.scale_output = scale_output
        self.max_col_norm_output = max_col_norm_output 

        self.n_hidden = n_hidden
        self.p_hidden = p_hidden
        self.scale_hidden = scale_hidden
        self.max_col_norm_hidden = max_col_norm_hidden 
        
        
        self.layer.append(Sigmoid_layer(rng = self.rng, n_inputs = self.n_input, n_units = self.n_hidden,
            p = self.p_input, scale = self.scale_input, max_col_norm = self.max_col_norm_input))

        for k in range(1,self.n_hidden_layers):
            self.layer.append(Sigmoid_layer(rng = self.rng, n_inputs = self.n_hidden, n_units = self.n_hidden,
            p = self.p_hidden, scale = self.scale_hidden, max_col_norm = self.max_col_norm_hidden))
        
        self.layer.append(Sigmoid_layer(rng = self.rng, n_inputs= self.n_hidden, n_units= self.n_output,
            p = self.p_output, scale = self.scale_output, max_col_norm = self.max_col_norm_output)) 
 
class My_CNN(deep_dropout_network):
    
    def __init__(self, rng, batch_size):
        
        deep_dropout_network.__init__(self, 3)
        
        self.rng = rng
        self.batch_size = batch_size
        
        print "    Convolution layer 1:"
        
        
        self.layer.append(Maxout_conv_layer(
            rng,
            image_shape=(batch_size, 1, 28, 28),
            zero_pad = 0, 
            filter_shape=(32, 1, 5, 5),
            filter_stride = 1,
            n_pieces = 4,
            pool_shape=(2, 2),
            pool_stride = 2,
            p = 0.8, 
            scale = 1., 
            max_col_norm = 1.9365
        ))
        
        print "    Convolution layer 2:"
        
        self.layer.append(Maxout_conv_layer(
            rng,
            image_shape=(batch_size, 32, 12, 12),
            zero_pad = 0, # add n zero on both side of the input
            filter_shape=(96, 32, 5, 5),
            filter_stride = 1,
            n_pieces = 4,
            pool_shape=(2, 2),
            pool_stride = 2,
            p = 0.5, 
            scale = 0.5, 
            max_col_norm = 1.9365
        ))
        
        print "    Convolution layer 3:"
        
        self.layer.append(Maxout_conv_layer(
            rng,
            image_shape=(batch_size, 96, 4, 4),
            zero_pad = 0, # add n zero on both side of the input
            filter_shape=(384, 96, 4, 4),
            filter_stride = 1,
            n_pieces = 4,
            pool_shape=(1, 1),
            pool_stride = 1,
            p = 0.5, 
            scale = 0.5, 
            max_col_norm = 1.9365
        ))
        
        print "    Fully connected layer:"
        
        self.layer.append(MaxoutLayer(rng = rng, n_inputs= 384, n_units = 10, n_pieces = 4, p = 0.5, scale = 0.5, max_col_norm =1.9365))
        
class Ian_CNN(deep_dropout_network):
    
    def __init__(self, rng, batch_size):
        
        deep_dropout_network.__init__(self, 3)
        
        self.rng = rng
        self.batch_size = batch_size
        
        print "    Convolution layer 1:"
        
        self.layer.append(Maxout_conv_layer(
            rng,
            image_shape=(batch_size, 1, 28, 28),
            zero_pad = 0, 
            filter_shape=(48, 1, 8, 8),
            filter_stride = 1,
            n_pieces = 2,
            pool_shape=(4, 4),
            pool_stride = 2,
            p = 0.8, 
            scale = 1., 
            max_col_norm = 0.9
        ))
        
        print "    Convolution layer 2:"
        
        self.layer.append(Maxout_conv_layer(
            rng,
            image_shape=(batch_size, 48, 10, 10),
            zero_pad = 3, # add n zero on both side of the input
            filter_shape=(48, 48, 8, 8),
            filter_stride = 1,
            n_pieces = 2,
            pool_shape=(4, 4),
            pool_stride = 2,
            p = 0.5, 
            scale = 0.5, 
            max_col_norm = 1.9365
        ))
        
        print "    Convolution layer 3:"
        
        self.layer.append(Maxout_conv_layer(
            rng,
            image_shape=(batch_size, 48, 4, 4),
            zero_pad = 3, # add n zero on both side of the input
            filter_shape=(24, 48, 5, 5),
            filter_stride = 1,
            n_pieces = 4,
            pool_shape=(2, 2),
            pool_stride = 2,
            p = 0.5, 
            scale = 0.5, 
            max_col_norm = 1.9365
        ))
        
        print "    Softmax layer:"
        
        self.layer.append(SoftmaxLayer(rng = rng, n_inputs= 24 * 3 * 3, n_units = 10, p = 0.5, scale = 0.5, max_col_norm =1.9365))        
        
        
class maxout_MLP(deep_dropout_network):
    
    def __init__(self, rng,  n_pieces, n_hidden_layers, 
        n_input, p_input, scale_input, max_col_norm_input,
        n_hidden, p_hidden, scale_hidden, max_col_norm_hidden,
        n_output, p_output, scale_output, max_col_norm_output):
        
        deep_dropout_network.__init__(self, n_hidden_layers)
        

        print '    n_pieces = %i' %(n_pieces)
        
        print '    n_input = %i' %(n_input)
        print '    p_input = %f' %(p_input)
        print '    scale_input = %f' %(scale_input)
        print '    max_col_norm_input = %f' %(max_col_norm_input)
        
        print '    n_hidden = %i' %(n_hidden)
        print '    p_hidden = %f' %(p_hidden)
        print '    scale_hidden = %f' %(scale_hidden)
        print '    max_col_norm_hidden = %f' %(max_col_norm_hidden)
        
        print '    n_output = %i' %(n_output)
        print '    p_output = %f' %(p_output)
        print '    scale_output = %f' %(scale_output)
        print '    max_col_norm_output = %f' %(max_col_norm_output)
        
        # save the parameters
        self.rng = rng
        self.n_pieces = n_pieces
        
        self.n_input = n_input
        self.p_input = p_input
        self.scale_input = scale_input   
        self.max_col_norm_input = max_col_norm_input         
        
        self.n_output = n_output
        self.p_output = p_output
        self.scale_output = scale_output
        self.max_col_norm_output = max_col_norm_output 

        self.n_hidden = n_hidden
        self.p_hidden = p_hidden
        self.scale_hidden = scale_hidden
        self.max_col_norm_hidden = max_col_norm_hidden 
        
        # Create MLP layers    
        if self.n_hidden_layers == 0 :
        
            self.layer.append(SoftmaxLayer(rng = self.rng, n_inputs=self.n_input, n_units=self.n_output, 
                p = self.p_input, scale = self.scale_input, max_col_norm = self.max_col_norm_input))

        else :
        
            self.layer.append(MaxoutLayer(rng = self.rng, n_inputs = self.n_input, n_units = self.n_hidden, n_pieces = self.n_pieces, 
                p = self.p_input, scale = self.scale_input, max_col_norm = self.max_col_norm_input))

            for k in range(1,self.n_hidden_layers):
                self.layer.append(MaxoutLayer(rng = self.rng, n_inputs = self.n_hidden, n_units = self.n_hidden, n_pieces = self.n_pieces, 
                p = self.p_hidden, scale = self.scale_hidden, max_col_norm = self.max_col_norm_hidden))

            self.layer.append(SoftmaxLayer(rng = self.rng, n_inputs= self.n_hidden, n_units= self.n_output, 
                p = self.p_output, scale = self.scale_output, max_col_norm = self.max_col_norm_output))    
            
            # self.layer.append(MaxoutLayer(rng = self.rng, n_inputs= self.n_hidden, n_units= self.n_output, n_pieces = self.n_pieces,
                # p = self.p_output, scale = self.scale_output, max_col_norm = self.max_col_norm_output))    

class fixed_deep_dropout_network(deep_dropout_network):
    
    def __init__(self, n_hidden_layers, comp_precision, update_precision, initial_range, max_sat):
        
        print '    Computation precision = %i bits' %(comp_precision)
        print '    Update precision = %i bits' %(update_precision)
        print '    Initial range = %i bits' %(initial_range)
        print '    Maximum ratio of saturated variables = %f %%' %(max_sat*100)   
        
        self.comp_precision = comp_precision
        self.update_precision = update_precision
        self.initial_range = initial_range
        self.max_sat = max_sat
        
        deep_dropout_network.__init__(self, n_hidden_layers)
    
    # when you use fixed point, you cannot use T.grad directly -> bprop modifications.
    def bprop(self, y, t):
        
        # there is a simplification between softmax derivative and nll derivative
        
        # print "y "+str(y.type)
        # print "t "+str(t.type)
        # print "one_hot(t) "+str(one_hot(t).type)
        
        dEdy = (y-t)/T.cast(T.shape(y)[1],dtype=theano.config.floatX) # /2. # actually, it is dEdz and not dEdy
        # MSE = T.sum((y-t)*(y-t))/2./T.shape(y)[1]
        # NLL = T.sum(-T.log(y)*t)/T.shape(y)[1]
        # dEdy = T.grad(NLL,y)
        
        # dEdy = T.addbroadcast(dEdy,1)
        
        # print "dEdy "+str(dEdy)
        # print "dEdy "+str(dEdy.type)
        
        # bprop
        for k in range(self.n_hidden_layers,-1,-1):
            dEdy = self.layer[k].bprop(dEdy)
    
    # function that updates the range of all fixed point vectors
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

class fixed_Ian_CNN(fixed_deep_dropout_network):
    
    def __init__(self, rng, batch_size, comp_precision, update_precision, initial_range, max_sat):
        
        fixed_deep_dropout_network.__init__(self, 3, comp_precision, update_precision, initial_range, max_sat)
        
        self.rng = rng
        self.batch_size = batch_size
        
        print "    Convolution layer 1:"
        
        self.layer.append(fixed_Maxout_conv_layer(
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
        
        self.layer.append(fixed_Maxout_conv_layer(
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
        
        self.layer.append(fixed_Maxout_conv_layer(
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
        
        self.layer.append(fixed_SoftmaxLayer(
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
        
class fixed_Ian_CNN_CIFAR10(fixed_deep_dropout_network):
    
    def __init__(self, rng, batch_size, comp_precision, update_precision, initial_range, max_sat):
        
        fixed_deep_dropout_network.__init__(self, 4, comp_precision, update_precision, initial_range, max_sat)
        
        self.rng = rng
        self.batch_size = batch_size
        
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
            # w_LR_scale = 0.2,
            # b_LR_scale = 0.2,
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
            max_col_norm = 0.9,
            comp_precision = comp_precision, 
            update_precision = update_precision, 
            initial_range = initial_range, 
            max_sat = max_sat,
            # w_LR_scale = 0.2,
            # b_LR_scale = 0.2,
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
            max_col_norm = 0.9,
            comp_precision = comp_precision, 
            update_precision = update_precision, 
            initial_range = initial_range, 
            max_sat = max_sat,
            # w_LR_scale = 0.2,
            # b_LR_scale = 0.2,
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
            max_col_norm = 0.9,
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
            max_col_norm = 0.9,
            comp_precision = comp_precision, 
            update_precision = update_precision, 
            initial_range = initial_range, 
            max_sat = max_sat
        ))
        
class fixed_maxout_MLP(maxout_MLP, fixed_deep_dropout_network):

    def __init__(self, rng, n_input, n_output, n_hidden, n_pieces, n_hidden_layers, 
        p_input,  scale_input, p_hidden, scale_hidden, max_col_norm, 
        comp_precision, update_precision, initial_range, max_sat):
        
        fixed_deep_dropout_network.__init__(self, n_hidden_layers, comp_precision, update_precision, initial_range, max_sat)
        
        print '    n_input = %i' %(n_input)
        print '    n_output = %i' %(n_output)
        print '    n_hidden = %i' %(n_hidden)
        print '    n_pieces = %i' %(n_pieces)
        print '    p_input = %f' %(p_input)
        print '    scale_input = %f' %(scale_input)
        print '    p_hidden = %f' %(p_hidden)
        print '    scale_hidden = %f' %(scale_hidden)
        print '    max_col_norm = %f' %(max_col_norm)
        
        # save the parameters
        self.rng = rng
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
        
            self.layer.append(fixed_SoftmaxLayer(rng = self.rng, n_inputs=self.n_input, n_units=self.n_output, 
                p = self.p_input, scale = self.scale_input, max_col_norm = self.max_col_norm,
                comp_precision = self.comp_precision, update_precision = self.update_precision, initial_range = self.initial_range, max_sat = self.max_sat))

        else :
        
            self.layer.append(fixed_MaxoutLayer(rng = self.rng, n_inputs = self.n_input, n_units = self.n_hidden, n_pieces = self.n_pieces, 
                p = self.p_input, scale = self.scale_input, max_col_norm = self.max_col_norm,
                comp_precision = self.comp_precision, update_precision = self.update_precision, initial_range = self.initial_range, max_sat = self.max_sat))

            for k in range(1,self.n_hidden_layers):
                self.layer.append(fixed_MaxoutLayer(rng = self.rng, n_inputs = self.n_hidden, n_units = self.n_hidden, n_pieces = self.n_pieces, 
                p = self.p_hidden, scale = self.scale_hidden, max_col_norm = self.max_col_norm,
                comp_precision = self.comp_precision, update_precision = self.update_precision, initial_range = self.initial_range, max_sat = self.max_sat))

            self.layer.append(fixed_SoftmaxLayer(rng = self.rng, n_inputs= self.n_hidden, n_units= self.n_output, 
                p = self.p_hidden, scale = self.scale_hidden, max_col_norm = self.max_col_norm,
                comp_precision = self.comp_precision, update_precision = self.update_precision, initial_range = self.initial_range, max_sat = self.max_sat))        