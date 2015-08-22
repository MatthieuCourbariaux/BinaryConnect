
from collections import OrderedDict

import numpy as np

import theano
import theano.tensor as T

import lasagne

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

def compute_grads(loss,network):
        
    layers = lasagne.layers.get_all_layers(network)
    grads = []
    
    for layer in layers:
    
        # params = lasagne.layers.get_all_params(layer, trainable=True)
        params = layer.get_params(trainable=True)
        
        for param in params:
            if param.name == "W":
                # print(param.name)
                grads.append(theano.grad(loss, wrt=layer.Wb))
            else:
                # print("here")
                grads.append(theano.grad(loss, wrt=param))
                
    return grads

def weights_clipping(updates):
    
    params = updates.keys()
    updates = OrderedDict(updates)

    for param in params:        
        if param.name == "W":
            # print("ok")
            updates[param] = T.clip(updates[param], -1, 1)

    return updates

def hard_sigmoid(x):
    return T.clip((x+1.)/2.,0,1)
    
class DenseLayer(lasagne.layers.DenseLayer):
    
    def __init__(self, incoming, num_units, stochastic_rounding = True, W=lasagne.init.Uniform((-1,1)), **kwargs):
        
        super(BinaryDenseLayer, self).__init__(incoming, num_units, W, **kwargs)
        
        self.stochastic_rounding = stochastic_rounding
        if self.stochastic_rounding == True:
            self._srng = RandomStreams(lasagne.random.get_rng().randint(1, 2147462579))
        
    def get_output_for(self, input, deterministic=False, **kwargs):
    
        if input.ndim > 2:
            # if the input has more than two dimensions, flatten it into a
            # batch of feature vectors.
            input = input.flatten(2)        
        
        # (deterministic == True) <-> test-time
        if deterministic == True and self.stochastic_rounding == True:
            self.Wb = self.W
        
        else:
        
            # [-1,1] -> [0,1]
            self.Wb = hard_sigmoid(self.W)
            
            # Stochastic BinaryConnect
            if self.stochastic_rounding == True:
                self.Wb = T.cast(self._srng.binomial(n=1, p=self.Wb, size=T.shape(self.Wb)), theano.config.floatX)

            # Deterministic BinaryConnect (round to nearest)
            else:
                self.Wb = T.round(self.Wb)
            
            # 0 or 1 -> -1 or 1
            self.Wb = T.cast(T.switch(self.Wb,1,-1), theano.config.floatX)
        
        activation = T.dot(input,self.Wb)
        if self.b is not None:
            activation = activation + self.b.dimshuffle('x', 0)
        return self.nonlinearity(activation)