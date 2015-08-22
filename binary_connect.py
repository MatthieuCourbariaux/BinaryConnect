
from collections import OrderedDict

import numpy as np

import theano
import theano.tensor as T

def weights_clipping(updates):
    
    params = updates.keys()
    updates = OrderedDict(updates)

    for param in params:        
        if param.name is not None:
            if "W" in param.name:
                # print("ok")
                updates[param] = T.clip(updates[param], -1, 1)

    return updates

from theano.scalar.basic import UnaryScalarOp, same_out_nocomplex
from theano.tensor.elemwise import Elemwise
    
class Binarize(UnaryScalarOp):
    
    def c_code(self, node, name, (x,), (z,), sub):
        return "%(z)s = 2*(%(x)s >= 0)-1;" % locals()
    
    def grad(self, (x, ), (gz, )):
        return [gz]
        
binarize = Elemwise(Binarize(same_out_nocomplex, name='binarize'))

import lasagne

class BinaryDenseLayer(lasagne.layers.DenseLayer):
    
    def __init__(self, incoming, num_units, W=lasagne.init.Uniform((-1,1)), **kwargs):
        
        super(BinaryDenseLayer, self).__init__(incoming, num_units, W, **kwargs)
        # self._srng = RandomStreams(lasagne.random.get_rng().randint(1, 2147462579))
        
    # def get_output_for(self, input, deterministic=False, **kwargs):
    def get_output_for(self, input, **kwargs):
    
        if input.ndim > 2:
            # if the input has more than two dimensions, flatten it into a
            # batch of feature vectors.
            input = input.flatten(2)        
        
        # deterministic BinaryConnect
        # Wb = T.cast(T.switch(T.ge(self.W,0),1,-1), theano.config.floatX)
        Wb = binarize(self.W)
        
        activation = T.dot(input,Wb)
        if self.b is not None:
            activation = activation + self.b.dimshuffle('x', 0)
        return self.nonlinearity(activation)