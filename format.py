
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

def stochastic_rounding(x, rng):
    
    p = x-T.floor(x)
    
    # much slower :(
    # srng = T.shared_randomstreams.RandomStreams(rng.randint(999999))
    
    # much faster :)
    # https://github.com/Theano/Theano/issues/1233#event-262671708
    # does it work though ?? It seems so :)
    srng = theano.sandbox.rng_mrg.MRG_RandomStreams(rng.randint(999999))
    
    p_mask = T.cast(srng.binomial(n=1, p=p, size=T.shape(x)), theano.config.floatX)
    
    x = T.floor(x) + p_mask
    
    return x