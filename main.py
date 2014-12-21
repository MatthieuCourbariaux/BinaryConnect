import gzip
import cPickle
import numpy as np
import os
import os.path
import sys
import theano
import theano.tensor as T
from theano import pp
import time
import scipy.stats
from pylearn2.sandbox.cuda_convnet.filter_acts import FilterActs
from theano.sandbox.cuda.basic_ops import gpu_contiguous
from pylearn2.sandbox.cuda_convnet.pool import MaxPool

from trainer import Trainer
from model import PI_MNIST_model, MNIST_model, CIFAR10_SVHN_model

from pylearn2.datasets.mnist import MNIST   
from pylearn2.datasets.zca_dataset import ZCA_Dataset    
from pylearn2.datasets.svhn import SVHN
          
def onehot(x,numclasses=None):

    if x.shape==():
        x = x[None]
    if numclasses is None:
        numclasses = np.max(x) + 1
    result = np.zeros(list(x.shape) + [numclasses], dtype="int")
    z = np.zeros(x.shape)
    for c in range(numclasses):
        z *= 0
        z[np.where(x==c)] = 1
        result[...,c] += z

    result = np.reshape(result,(np.shape(result)[0], np.shape(result)[result.ndim-1]))
    return result
       
# MAIN

if __name__ == "__main__":
       
    print 'Beginning of the program'
    start_time = time.clock()
    
    print 'Loading the dataset'
    
    # MNIST
    
    train_set = MNIST(which_set= 'train',start=0, stop = 50000)#, center = True)
    valid_set = MNIST(which_set= 'train',start=50000, stop = 60000)#, center = True)
    test_set = MNIST(which_set= 'test')#, center = True)
    
    train_set.y = np.float32(onehot(train_set.y))
    valid_set.y = np.float32(onehot(valid_set.y))
    test_set.y = np.float32(onehot(test_set.y))

    # CIFAR10
    # preprocessor = cPickle.load(open("/data/lisa/data/cifar10/pylearn2_gcn_whitened/preprocessor.pkl",'rb'))
    # train_set = ZCA_Dataset(
        # preprocessed_dataset=cPickle.load(open("/data/lisa/data/cifar10/pylearn2_gcn_whitened/train.pkl",'rb')), 
        # preprocessor = preprocessor,
        # start=0, stop = 45000)
    # valid_set = ZCA_Dataset(
        # preprocessed_dataset= cPickle.load(open("/data/lisa/data/cifar10/pylearn2_gcn_whitened/train.pkl",'rb')), 
        # preprocessor = preprocessor,
        # start=45000, stop = 50000)  
    # test_set = ZCA_Dataset(
        # preprocessed_dataset= cPickle.load(open("/data/lisa/data/cifar10/pylearn2_gcn_whitened/test.pkl",'rb')), 
        # preprocessor = preprocessor) 
    
    # train_set.y = np.float32(onehot(train_set.y))
    # valid_set.y = np.float32(onehot(valid_set.y))
    # test_set.y = np.float32(onehot(test_set.y))
    
    # SVHN
    # train_set = SVHN(
        # which_set= 'splitted_train',
        # path= "${SVHN_LOCAL_PATH}",
        # axes= ['b', 'c', 0, 1])
 
    # valid_set = SVHN(
        # which_set= 'valid',
        # path= "${SVHN_LOCAL_PATH}",
        # axes= ['b', 'c', 0, 1])
    
    # test_set = SVHN(
        # which_set= 'test',
        # path= "${SVHN_LOCAL_PATH}",
        # axes= ['b', 'c', 0, 1])
    
    print 'Creating the model'

    # PI MNIST
    
    rng = np.random.RandomState(1234)
    LR_start = 0.2
    batch_size = 100
    gpu_batches = 500
    
    model = PI_MNIST_model(rng = rng, batch_size = batch_size,
        n_input = 784, n_output = 10, n_hidden = 240, n_pieces = 5, n_hidden_layers = 2, 
        p_input = 0.8, scale_input = 1., p_hidden = 0.5, scale_hidden = 0.5, 
        max_col_norm = 1.9365, 
        comp_precision = int(sys.argv[1]), update_precision = int(sys.argv[2]), initial_range = int(sys.argv[3]), max_sat = float(sys.argv[4]))
    
    core_path = sys.argv[1]+"_"+sys.argv[2]+"_"+sys.argv[3]+"_"+sys.argv[4]+"_"+sys.argv[5]
    load_path = None # "best_params_" + core_path+".pkl"
    save_path = None # "best_params_" + core_path+".pkl"
    
    trainer = Trainer(rng = rng, load_path = load_path, save_path = save_path,
        train_set = train_set, valid_set = valid_set, test_set = test_set,
        model = model,
        LR_start = LR_start, LR_sat = 250, LR_fin = LR_start*0.1, M_start = 0.5, M_sat = 250, M_fin = 0.7, 
        batch_size = batch_size, gpu_batches = gpu_batches,
        n_epoch = 1000,
        shuffle_batches = False, shuffle_examples = True,
        dynamic_range = int(sys.argv[5]))
    
    # MNIST
    
    # rng = np.random.RandomState(1234)
    # LR_start = 0.1
    # batch_size = 128
    # gpu_batches = 500
    
    # model = MNIST_model(rng = rng, batch_size = batch_size,
        # comp_precision = int(sys.argv[1]), update_precision = int(sys.argv[2]), 
        # initial_range = int(sys.argv[3]), max_sat = float(sys.argv[4]))
    
    # core_path = sys.argv[1]+"_"+sys.argv[2]+"_"+sys.argv[3]+"_"+sys.argv[4]+"_"+sys.argv[5]
    # load_path = None # "best_params_" + core_path+".pkl"
    # save_path = None # "best_params_" + core_path+".pkl"
    
    # trainer = Trainer(rng = rng, load_path = load_path, save_path = save_path,
        # train_set = train_set, valid_set = valid_set, test_set = test_set,
        # model = model,
        # LR_start = LR_start, LR_sat = 250, LR_fin = LR_start*0.1, M_start = 0.5, M_sat = 250, M_fin = 0.7, 
        # batch_size = batch_size, gpu_batches = gpu_batches,
        # n_epoch = 1000,
        # shuffle_batches = False, shuffle_examples = True,
        # dynamic_range = int(sys.argv[5]))
    
    # CIFAR10 and SVHN
    
    # rng = np.random.RandomState(1234)
    # LR_start = 0.05
    # batch_size = 128
    # gpu_batches = 391 # 391 -> 50000, 196 -> 25000, 79 -> 10000
    
    # model = CIFAR10_SVHN_model(rng = rng, batch_size = batch_size,
        # comp_precision = int(sys.argv[1]), update_precision = int(sys.argv[2]), 
        # initial_range = int(sys.argv[3]), max_sat = float(sys.argv[4]))
    
    # core_path = sys.argv[1]+"_"+sys.argv[2]+"_"+sys.argv[3]+"_"+sys.argv[4]+"_"+sys.argv[5]
    # load_path = None # "best_params_" + core_path+".pkl"
    # save_path = None # "best_params_" + core_path+".pkl" 3 careful, not in data/lisa/exp
    
    # trainer = Trainer(rng = rng, load_path = load_path, save_path = save_path,
        # train_set = train_set, valid_set = valid_set, test_set = test_set,
        # model = model,
        # LR_start = LR_start, LR_sat = 160, LR_fin = LR_start*0.1, M_start = 0.5, M_sat = 160, M_fin = 0.7, 
        # batch_size = batch_size, gpu_batches = gpu_batches,
        # n_epoch = 160,
        # shuffle_batches = True, shuffle_examples = False,
        # dynamic_range = int(sys.argv[5]))
        
    # if I don't shuffle examples each epochs, I should at least do it at the beginning ?
    # trainer.shuffle(train_set) # do it once on SVHN

    print 'Building'
    
    trainer.build()
    
    print 'Training'
    
    trainer.train()

    end_time = time.clock()
    print 'The code ran for %i seconds'%(end_time - start_time)
    
    