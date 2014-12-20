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

from trainer import Trainer, fixed_Trainer
from model import Ian_CNN, My_CNN, maxout_MLP, Sigmoid_MLP, fixed_maxout_MLP, fixed_Ian_CNN, fixed_Ian_CNN_CIFAR10
from dataset import onehot

from pylearn2.datasets.mnist import MNIST   
from pylearn2.datasets.cifar10 import CIFAR10 
from pylearn2.datasets.zca_dataset import ZCA_Dataset    
from pylearn2.datasets.svhn import SVHN   

from filter_plot import tile_raster_images
import Image
# import matplotlib.pyplot as plt
       
# MAIN

if __name__ == "__main__":
       
    print 'Beginning of the program'
    start_time = time.clock()
    
    print 'Loading the dataset'
    
    # MNIST
    
    # train_set = MNIST(which_set= 'train',start=0, stop = 50000)#, center = True)
    # valid_set = MNIST(which_set= 'train',start=50000, stop = 60000)#, center = True)
    # test_set = MNIST(which_set= 'test')#, center = True)
    
    # train_set.y = np.float32(onehot(train_set.y))
    # valid_set.y = np.float32(onehot(valid_set.y))
    # test_set.y = np.float32(onehot(test_set.y))

    # CIFAR10 gcn and zca
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
    
    # preprocessed SVHN
    train_set = SVHN(
        which_set= 'splitted_train',
        path= "${SVHN_LOCAL_PATH}",
        axes= ['b', 'c', 0, 1])
 
    valid_set = SVHN(
        which_set= 'valid',
        path= "${SVHN_LOCAL_PATH}",
        axes= ['b', 'c', 0, 1])
    
    test_set = SVHN(
        which_set= 'test',
        path= "${SVHN_LOCAL_PATH}",
        axes= ['b', 'c', 0, 1])
        
    # print train_set.X.shape
    # print valid_set.X.shape
    # print test_set.X.shape
    # print train_set.y.shape
    # print valid_set.y.shape
    # print test_set.y.shape
    
    # print train_set.y[0:5]
    
    # input("batman")
    
    print 'Creating the model'

    # PI sigmoid
    
    # batch_size = 100
    # LR =0.02
    
    # model = Sigmoid_MLP(rng = np.random.RandomState(1234),
        # n_hidden_layers = 1, 
        # n_input = 784, p_input = 1., scale_input = 1., max_col_norm_input =50,
        # n_hidden = 512, p_hidden = 1., scale_hidden = 1., max_col_norm_hidden = 50,
        # n_output = 10, p_output =  1., scale_output = 1., max_col_norm_output = 50)
    
    # trainer = Trainer(train_set_x = train_set_x, valid_set_x = valid_set_x, test_set_x = test_set_x,
        # train_set_y = train_set_y, valid_set_y = valid_set_y, test_set_y = test_set_y,
        # model = model,
        # LR_start = LR, LR_decay = 0.998, LR_fin = LR/10, M_start = 0, M_sat = 250, M_fin = 0, 
        # batch_size = batch_size, patience = 100) 
    
    # PI maxout
        
    # batch_size = 100
    # LR =0.1
    
    # model = maxout_MLP(rng = np.random.RandomState(1234),
        # n_pieces = 5, n_hidden_layers = 2, 
        # n_input = 784, p_input = 0.8, scale_input = 1., max_col_norm_input =1.9365,
        # n_hidden = 240, p_hidden = 0.5, scale_hidden = 0.5, max_col_norm_hidden = 1.9365,
        # n_output = 10, p_output =  0.5, scale_output = 0.5, max_col_norm_output = 1.9365)
    
    # trainer = Trainer(train_set_x = train_set_x, valid_set_x = valid_set_x, test_set_x = test_set_x,
        # train_set_y = train_set_y, valid_set_y = valid_set_y, test_set_y = test_set_y,
        # model = model,
        # LR_start = LR, LR_decay = 0.998, LR_fin = LR/10, M_start = 0.95, M_sat = 250, M_fin = 0.95, 
        # batch_size = batch_size, n_epoch = 500) 

    # CNN, MNIST, float, Moi
    
    # batch_size = 128
    # LR = 0.05
    # model = My_CNN(rng = np.random.RandomState(1234), batch_size = batch_size)
     
    # trainer = Trainer(train_set_x = train_set_x, valid_set_x = valid_set_x, test_set_x = test_set_x,
        # train_set_y = train_set_y, valid_set_y = valid_set_y, test_set_y = test_set_y,
        # model = model,
        # LR_start = LR, LR_decay = 0.985, LR_fin = LR/10, M_start = 0, M_sat = 250, M_fin = 0, 
        # batch_size = batch_size, patience = 100) 
    
    # PI, MNIST, Softmax, NLL, the same as Ian.
    
    # model = maxout_MLP(rng = np.random.RandomState(1234),
        # n_pieces = 5, n_hidden_layers = 2, 
        # n_input = 784, p_input = 0.8, scale_input = 1., max_col_norm_input =1.9365,
        # n_hidden = 240, p_hidden = 0.5, scale_hidden = 0.5, max_col_norm_hidden = 1.9365,
        # n_output = 10, p_output =  0.5, scale_output = 0.5, max_col_norm_output = 1.9365)
     
    # trainer = Trainer(train_set_x = train_set_x, valid_set_x = valid_set_x, test_set_x = test_set_x,
        # train_set_y = train_set_y, valid_set_y = valid_set_y, test_set_y = test_set_y,
        # model = model,
        # LR_start = .1, LR_decay = 0.998, LR_fin = 0.001, M_start = 0.5, M_sat = 250, M_fin = 0.7, 
        # batch_size = 100, n_epoch = 5) 
    
    # fixed point, PI, MNIST
    
    # rng = np.random.RandomState(1234)
    # LR_start = 0.1
    # batch_size = 100
    # gpu_batches = 500
    
    # model = fixed_maxout_MLP(rng = rng,
        # n_input = 784, n_output = 10, n_hidden = 240, n_pieces = 5, n_hidden_layers = 2, 
        # p_input = 0.8, scale_input = 1., p_hidden = 0.5, scale_hidden = 0.5, 
        # max_col_norm = 1.9365, 
        # comp_precision = int(sys.argv[1]), update_precision = int(sys.argv[2]), initial_range = int(sys.argv[3]), max_sat = 0.)
    
    # core_path = sys.argv[1]+"_"+sys.argv[2]+"_"+sys.argv[3]+"_"+sys.argv[4]
    # load_path = None
    # save_path = None # "best_params_" + core_path+".pkl"
    
    # trainer = fixed_Trainer(rng = rng, load_path = load_path, save_path = save_path,
        # train_set = train_set, valid_set = valid_set, test_set = test_set,
        # model = model,
        # LR_start = LR_start, LR_sat = 250, LR_fin = LR_start*0.1, M_start = 0.5, M_sat = 250, M_fin = 0.7, 
        # batch_size = batch_size, gpu_batches = gpu_batches,
        # n_epoch = 1000,
        # shuffle_batches = False, shuffle_examples = True,
        # dynamic_range = int(sys.argv[4])) 
    
    # fixed point, PI, CIFAR10
     
    # rng = np.random.RandomState(1234)
     
    # model = fixed_maxout_MLP(rng = np.random.RandomState(1234),
        # n_input = 32*32*3, n_output = 10, n_hidden = 240, n_pieces = 5, n_hidden_layers = 2, 
        # p_input = 0.8, scale_input = 1., p_hidden = 0.5, scale_hidden = 0.5, 
        # max_col_norm = 1.9365, 
        # comp_precision = int(sys.argv[1]), update_precision = int(sys.argv[2]), initial_range = int(sys.argv[3]), max_sat = 0.)
    
    # LR_start = 0.1
    # trainer = fixed_Trainer(rng = rng, train_set_x = train_set_x, valid_set_x = valid_set_x, test_set_x = test_set_x,
        # train_set_y = train_set_y, valid_set_y = valid_set_y, test_set_y = test_set_y,
        # model = model,
        # LR_start = LR_start, LR_sat = 500, LR_fin = LR_start*0.01, M_start = 0.5, M_sat = 250, M_fin = 0.7, 
        # batch_size = 100, n_epoch = 800,
        # dynamic_range = int(sys.argv[4]))  
     
    # CNN, MNIST, float, Ian
    
    # batch_size = 128
    # model = Ian_CNN(rng = np.random.RandomState(1234), batch_size = batch_size)
     
    # trainer = Trainer(train_set_x = train_set_x, valid_set_x = valid_set_x, test_set_x = test_set_x,
        # train_set_y = train_set_y, valid_set_y = valid_set_y, test_set_y = test_set_y,
        # model = model,
        # LR_start =0.05, LR_decay = 0.985, LR_fin = .000001, M_start = 0.5, M_sat = 250, M_fin = 0.7, 
        # batch_size = batch_size, n_epoch = 5)
     
    # CNN fixed point, MNIST
    
    # batch_size = 128
    # model = fixed_Ian_CNN(rng = np.random.RandomState(1234), batch_size = batch_size,
        # comp_precision = int(sys.argv[1]), update_precision = int(sys.argv[2]), initial_range = int(sys.argv[3]), max_sat = 0.)
     
    # trainer = fixed_Trainer(train_set_x = train_set_x, valid_set_x = valid_set_x, test_set_x = test_set_x,
        # train_set_y = train_set_y, valid_set_y = valid_set_y, test_set_y = test_set_y,
        # model = model,
        # LR_start =0.05, LR_decay = 0.985, LR_fin = .000001, M_start = 0.5, M_sat = 250, M_fin = 0.7, 
        # batch_size = batch_size, n_epoch = 250,
        # dynamic_range = int(sys.argv[4])) 
    
    # fixed point, PI, CIFAR10, SVHN
        
    # rng = np.random.RandomState(1234)
    # LR_start = 0.1
    # batch_size = 128
    # gpu_batches = 100
    
    # model = fixed_maxout_MLP(rng = rng,
        # n_input = 32*32*3, n_output = 10, n_hidden = 240, n_pieces = 5, n_hidden_layers = 2, 
        # p_input = 0.8, scale_input = 1., p_hidden = 0.5, scale_hidden = 0.5, 
        # max_col_norm = 1.9365, 
        # comp_precision = int(sys.argv[1]), update_precision = int(sys.argv[2]), initial_range = int(sys.argv[3]), max_sat = 0.)
    
    # core_path = sys.argv[1]+"_"+sys.argv[2]+"_"+sys.argv[3]+"_"+sys.argv[4]
    # load_path = None
    # save_path = None #"best_params_" + core_path+".pkl"
    
    # trainer = fixed_Trainer(rng = rng, load_path = load_path, save_path = save_path,
        # train_set = train_set, valid_set = valid_set, test_set = test_set,
        # model = model,
        # LR_start = LR_start, LR_sat = 200, LR_fin = LR_start*0.1, M_start = 0., M_sat = 200, M_fin = 0., 
        # batch_size = batch_size, gpu_batches = gpu_batches,
        # n_epoch = 5,
        # shuffle_batches = True, shuffle_examples = False,
        # dynamic_range = int(sys.argv[4])) 
    
    
    # if I don't shuffle examples each epochs, I should at least do it at the beginning ?
    # trainer.shuffle(train_set) # do it once on SVHN
    
    # CNN fixed point, CIFAR10, SVHN
    
    rng = np.random.RandomState(1234)
    LR_start = 0.01
    batch_size = 128
    gpu_batches = 391 # 391 -> 50000, 196 -> 25000, 79 -> 10000
    
    model = fixed_Ian_CNN_CIFAR10(rng = rng, batch_size = batch_size,
        comp_precision = int(sys.argv[1]), update_precision = int(sys.argv[2]), initial_range = int(sys.argv[3]), max_sat = 0.)
    
    core_path = sys.argv[1]+"_"+sys.argv[2]+"_"+sys.argv[3]+"_"+sys.argv[4]
    load_path = None # "best_params_" + core_path+".pkl"
    save_path = "best_params_" + core_path+".pkl"
    
    trainer = fixed_Trainer(rng = rng, load_path = load_path, save_path = save_path,
        train_set = train_set, valid_set = valid_set, test_set = test_set,
        model = model,
        LR_start = LR_start, LR_sat = 160, LR_fin = LR_start*0.1, M_start = 0.5, M_sat = 160, M_fin = 0.7, 
        batch_size = batch_size, gpu_batches = gpu_batches,
        n_epoch = 160,
        shuffle_batches = True, shuffle_examples = False,
        dynamic_range = int(sys.argv[4])) 
    

    print 'Building'
    
    trainer.build()
    
    print 'Training'
    
    trainer.train()
    
    # print 'Plotting parameters'
    
    # image_path = "best_params_image_" + core_path +".png"
    # model.layer[0].parameters_image(image_path)
    # histogram_path = "best_params_histogram_" + core_path +".png"
    # model.layer[0].parameters_histogram(histogram_path)
    
    end_time = time.clock()
    print 'The code ran for %i seconds'%(end_time - start_time)
    
    