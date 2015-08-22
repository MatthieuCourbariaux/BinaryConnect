
from collections import OrderedDict

import numpy as np

import theano
import theano.tensor as T

# def compute_grads(binaryNetwork):
    
    # params = get_all_params(binaryNetwork)

def weights_clipping(updates):
    
    params = updates.keys()
    updates = OrderedDict(updates)

    for param in params:        
        if param.name is not None:
            if "W" in param.name:
                # print("ok")
                updates[param] = T.clip(updates[param], -1, 1)

    return updates

# I redefine Clip and Round with identity gradient
# (otherwise, the gradient of W would be 0)
# from theano.scalar.basic import UnaryScalarOp, same_out_nocomplex
# from theano.tensor.elemwise import Elemwise
    
# class Clip(UnaryScalarOp):
    
    # def c_code(self, node, name, (x,), (z,), sub):
        # return "%(z)s = (%(x)s <= 0) ? 0 : (%(x)s >= 1) ? 1 : %(x)s;" % locals()
    
    # def grad(self, (x, ), (gz, )):
        # return [gz]
        
# clip = Elemwise(Clip(same_out_nocomplex, name='clip'))

# class Round(UnaryScalarOp):
    
    # def c_code(self, node, name, (x,), (z,), sub):
        # return "%(z)s = floor(%(x)s + 0.5);" % locals()
    
    # def grad(self, (x, ), (gz, )):
        # return [gz]
        
# round = Elemwise(Round(same_out_nocomplex, name='round'))

import lasagne
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

class BinaryDenseLayer(lasagne.layers.DenseLayer):
    
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
        
        # deterministic = test-time
        if deterministic == True and self.stochastic_rounding == True:
            self.Wb = self.W
        
        else:
            # Hard sigmoid of W
            # [-1,1] -> [0,1]
            self.Wb = T.clip((self.W+1.)/2.,0,1)
            
            # Stochastic BinaryConnect
            if self.stochastic_rounding == True:
                self.Wb = T.cast(self._srng.binomial(n=1, p=self.Wb, size=T.shape(self.Wb)), theano.config.floatX)

            # Deterministic BinaryConnect (round to nearest)
            else:
                self.Wb = T.round(self.Wb)
            
            # 0 or 1 -> -1 or 1
            self.Wb = self.Wb*2.-1.
        
        activation = T.dot(input,self.Wb)
        if self.b is not None:
            activation = activation + self.b.dimshuffle('x', 0)
        return self.nonlinearity(activation)