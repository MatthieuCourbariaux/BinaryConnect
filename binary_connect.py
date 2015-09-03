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

import time

from collections import OrderedDict

import numpy as np

# specifying the gpu to use
# import theano.sandbox.cuda
# theano.sandbox.cuda.use('gpu1') 
import theano
import theano.tensor as T

import lasagne

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

def compute_grads(loss,network):
        
    layers = lasagne.layers.get_all_layers(network)
    grads = []
    
    for layer in layers:
    
        params = layer.get_params(trainable=True)
        
        for param in params:
            if param.name == "W":
                # print(param.name)
                grads.append(theano.grad(loss, wrt=layer.Wb))
            else:
                # print("here")
                grads.append(theano.grad(loss, wrt=param))
                
    return grads

def weights_clipping(updates,network):
    
    layers = lasagne.layers.get_all_layers(network)
    updates = OrderedDict(updates)
    
    for layer in layers:
    
        params = layer.get_params(trainable=True)
        
        for param in params:
            if param.name == "W":
                # print("K")
                updates[param] = T.clip(updates[param], -layer.H, layer.H)           

    return updates
    
# def weights_clipping(updates, H):
    
    # params = updates.keys()
    # updates = OrderedDict(updates)

    # for param in params:        
        # if param.name == "W":            
            # updates[param] = T.clip(updates[param], -H, H)

    # return updates

def hard_sigmoid(x):
    return T.clip((x+1.)/2.,0,1)
    
class DenseLayer(lasagne.layers.DenseLayer):
    
    def __init__(self, incoming, num_units, 
        # binary = True, stochastic = True, H=1., **kwargs):
        binary = True, stochastic = True, **kwargs):
        
        self.binary = binary
        self.stochastic = stochastic
        
        # self.H = H
        num_inputs = int(np.prod(incoming.output_shape[1:]))
        self.H = np.float32(np.sqrt(1.5/ (num_inputs + num_units)))
        # print("H = "+str(self.H))
        
        if self.stochastic:
            self._srng = RandomStreams(lasagne.random.get_rng().randint(1, 2147462579))
            
        if self.binary:
            super(DenseLayer, self).__init__(incoming, num_units, W=lasagne.init.Uniform((-self.H,self.H)), **kwargs)   
        
        else:
            super(DenseLayer, self).__init__(incoming, num_units, **kwargs)
        
    def get_output_for(self, input, deterministic=False, **kwargs):
    
        if input.ndim > 2:
            # if the input has more than two dimensions, flatten it into a
            # batch of feature vectors.
            input = input.flatten(2)        
        
        # (deterministic == True) <-> test-time
        if not self.binary or (deterministic and self.stochastic):
            
            self.Wb = self.W
        
        else:
            # [-1,1] -> [0,1]
            self.Wb = hard_sigmoid(self.W/self.H)
            
            # Stochastic BinaryConnect
            if self.stochastic:
                self.Wb = T.cast(self._srng.binomial(n=1, p=self.Wb, size=T.shape(self.Wb)), theano.config.floatX)

            # Deterministic BinaryConnect (round to nearest)
            else:
                self.Wb = T.round(self.Wb)
            
            # 0 or 1 -> -1 or 1
            self.Wb = T.cast(T.switch(self.Wb,self.H,-self.H), theano.config.floatX)
        
        activation = T.dot(input,self.Wb)

        if self.b is not None:
            activation = activation + self.b.dimshuffle('x', 0)
        return self.nonlinearity(activation)

class Conv2DLayer(lasagne.layers.Conv2DLayer):
    
    def __init__(self, incoming, num_filters, filter_size,
        binary = True, stochastic = True, **kwargs):
        
        self.binary = binary
        self.stochastic = stochastic

        num_inputs = int(np.prod(filter_size)*incoming.output_shape[1])
        # theoretically, I should divide num_units by the pool_shape
        num_units = int(np.prod(filter_size)*num_filters)
        self.H = np.float32(np.sqrt(1.5 / (num_inputs + num_units)))
        # print("H = "+str(self.H))
        # self.H = .05

        if self.stochastic:
            self._srng = RandomStreams(lasagne.random.get_rng().randint(1, 2147462579))
            
        if self.binary:
            super(Conv2DLayer, self).__init__(incoming, num_filters, filter_size, W=lasagne.init.Uniform((-self.H,self.H)), **kwargs)   
        
        else:
            super(Conv2DLayer, self).__init__(incoming, num_filters, filter_size, **kwargs)    

    def get_output_for(self, input, input_shape=None, deterministic=False, **kwargs):
        # The optional input_shape argument is for when get_output_for is
        # called directly with a different shape than self.input_shape.
        
        # print("deterministic = "+str(deterministic))
        
        if input_shape is None:
            input_shape = self.input_shape
        
        # (deterministic == True) <-> test-time
        if not self.binary or (deterministic and self.stochastic):
            
            self.Wb = self.W
        
        else:
            # [-1,1] -> [0,1]
            self.Wb = hard_sigmoid(self.W/self.H)
            
            # Stochastic BinaryConnect
            if self.stochastic:
                self.Wb = T.cast(self._srng.binomial(n=1, p=self.Wb, size=T.shape(self.Wb)), theano.config.floatX)

            # Deterministic BinaryConnect (round to nearest)
            else:
                self.Wb = T.round(self.Wb)
            
            # 0 or 1 -> -1 or 1
            self.Wb = T.cast(T.switch(self.Wb,self.H,-self.H), theano.config.floatX)
        
        if self.stride == (1, 1) and self.pad == 'same':
            # simulate same convolution by cropping a full convolution
            conved = self.convolution(input, self.Wb, subsample=self.stride,
                                      image_shape=input_shape,
                                      filter_shape=self.get_W_shape(),
                                      border_mode='full')
            shift_x = (self.filter_size[0] - 1) // 2
            shift_y = (self.filter_size[1] - 1) // 2
            conved = conved[:, :, shift_x:input.shape[2] + shift_x,
                            shift_y:input.shape[3] + shift_y]
        else:
            # no padding needed, or explicit padding of input needed
            if self.pad == 'full':
                border_mode = 'full'
                pad = [(0, 0), (0, 0)]
            elif self.pad == 'same':
                border_mode = 'valid'
                pad = [(self.filter_size[0] // 2,
                        (self.filter_size[0] - 1) // 2),
                       (self.filter_size[1] // 2,
                        (self.filter_size[1] - 1) // 2)]
            else:
                border_mode = 'valid'
                pad = [(self.pad[0], self.pad[0]), (self.pad[1], self.pad[1])]
            if pad != [(0, 0), (0, 0)]:
                input = padding.pad(input, pad, batch_ndim=2)
                input_shape = (input_shape[0], input_shape[1],
                               None if input_shape[2] is None else
                               input_shape[2] + pad[0][0] + pad[0][1],
                               None if input_shape[3] is None else
                               input_shape[3] + pad[1][0] + pad[1][1])
            conved = self.convolution(input, self.Wb, subsample=self.stride,
                                      image_shape=input_shape,
                                      filter_shape=self.get_W_shape(),
                                      border_mode=border_mode)

        if self.b is None:
            activation = conved
        elif self.untie_biases:
            activation = conved + self.b.dimshuffle('x', 0, 1, 2)
        else:
            activation = conved + self.b.dimshuffle('x', 0, 'x', 'x')

        return self.nonlinearity(activation)
        
def train(train_fn,val_fn,
            batch_size,
            LR_start,LR_decay,
            num_epochs,
            X_train,y_train,
            X_val,y_val,
            X_test,y_test):
            
    def shuffle(X,y):
    
        shuffled_range = range(len(X))
        np.random.shuffle(shuffled_range)
        # print(shuffled_range[0:10])
        
        new_X = np.copy(X)
        new_y = np.copy(y)
        
        for i in range(len(X)):
            
            new_X[i] = X[shuffled_range[i]]
            new_y[i] = y[shuffled_range[i]]
            
        return new_X,new_y
        
    def train_epoch(X,y,LR):
        
        loss = 0
        batches = len(X)/batch_size
        
        for i in range(batches):
            loss += train_fn(X[i*batch_size:(i+1)*batch_size],y[i*batch_size:(i+1)*batch_size],LR)
        
        loss/=batches
        
        return loss
        
    def val_epoch(X,y):
        
        err = 0
        loss = 0
        batches = len(X)/batch_size
        
        for i in range(batches):
            new_loss, new_err = val_fn(X[i*batch_size:(i+1)*batch_size], y[i*batch_size:(i+1)*batch_size])
            err += new_err
            loss += new_loss
        
        err = err / batches * 100
        loss /= batches

        return err, loss
    
    # shuffle the train set
    X_train,y_train = shuffle(X_train,y_train)
    best_val_err = 100
    best_epoch = 1
    LR = LR_start
    
    # We iterate over epochs:
    for epoch in range(num_epochs):
        
        start_time = time.time()
        
        train_loss = train_epoch(X_train,y_train,LR)
        X_train,y_train = shuffle(X_train,y_train)
        LR *= LR_decay
        
        val_err, val_loss = val_epoch(X_val,y_val)
        
        # test if validation error went down
        if val_err <= best_val_err:
            
            best_val_err = val_err
            best_epoch = epoch+1
            
            test_err, test_loss = val_epoch(X_test,y_test)

        epoch_duration = time.time() - start_time
        
        # Then we print the results for this epoch:
        print("Epoch "+str(epoch + 1)+" of "+str(num_epochs)+" took "+str(epoch_duration)+"s")
        print("  LR:                            "+str(LR))
        print("  training loss:                 "+str(train_loss))
        print("  validation loss:               "+str(val_loss))
        print("  validation error rate:         "+str(val_err)+"%")
        print("  best epoch:                    "+str(best_epoch))
        print("  best validation error rate:    "+str(best_val_err)+"%")
        print("  test loss:                     "+str(test_loss))
        print("  test error rate:               "+str(test_err)+"%") 