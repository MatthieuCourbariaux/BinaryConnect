from __future__ import print_function

import sys
import os
import time

import numpy as np
np.random.seed(1234)  # for reproducibility

# specifying the gpu to use
# import theano.sandbox.cuda
# theano.sandbox.cuda.use('gpu1') 
import theano
import theano.tensor as T

import lasagne

import cPickle as pickle
import gzip

import batch_norm
import binary_connect

if __name__ == "__main__":
    
    # BN parameters
    batch_size = 200
    # alpha is the exponential moving average factor
    # alpha = .1 # for a minibatch of size 50
    # alpha = .2 # for a minibatch of size 100
    alpha = .33 # for a minibatch of size 200
    epsilon = 1e-4
    
    # MLP parameters
    num_units = 1024
    n_hidden_layers = 3
    
    # Training parameters
    num_epochs = 1000
    
    # Dropout parameters
    dropout_in = 0.
    dropout_hidden = 0.
    
    # BinaryConnect
    binary = True
    stochastic = True
    # H = (1./(1<<4))/10
    # H = 1./(1<<4)
    # H = .316
    # H = 1.
    
    # LR decay
    LR_start = 3.
    LR_fin = .1
    LR_decay = (LR_fin/LR_start)**(1./num_epochs) 
    # BTW, LR decay is good for the moving average...
    
    print('Loading MNIST dataset...')

    # We'll then load and unpickle the file.
    filename = "/data/lisa/data/mnist/mnist.pkl.gz"
    with gzip.open(filename, 'rb') as f:
        data = pickle.load(f)

    # The MNIST dataset we have here consists of six numpy arrays:
    # Inputs and targets for the training set, validation set and test set.
    X_train, y_train = data[0]
    X_val, y_val = data[1]
    X_test, y_test = data[2]

    # The inputs come as vectors, we reshape them to monochrome 2D images,
    # according to the shape convention: (examples, channels, rows, columns)
    X_train = X_train.reshape((-1, 1, 28, 28))
    X_val = X_val.reshape((-1, 1, 28, 28))
    X_test = X_test.reshape((-1, 1, 28, 28))
    
    # without standardization .97%, 1.11%
    # with standardization 1.06%, 1.34%
    # standardize the dataset
    # def standardize(X):
        # X -= X.mean(axis=0)
        # X /= (X.std(axis=0)+epsilon)
        # return X
    # X_train = standardize(X_train)
    # X_val = standardize(X_val)
    # X_test = standardize(X_test)

    # flatten the targets
    y_train = np.hstack(y_train)
    y_val = np.hstack(y_val)
    y_test = np.hstack(y_test)

    # Onehot the targets
    y_train = np.float32(np.eye(10)[y_train])    
    y_val = np.float32(np.eye(10)[y_val])
    y_test = np.float32(np.eye(10)[y_test])

    # prepare targets for hinge loss
    y_train = 2* y_train - 1.
    y_val = 2* y_val - 1.
    y_test = 2* y_test - 1.

    print('Building the MLP...') 
    
    # Prepare Theano variables for inputs and targets
    input = T.tensor4('inputs')
    target = T.matrix('targets')
    LR = T.scalar('LR', dtype=theano.config.floatX)

    mlp = lasagne.layers.InputLayer(
            shape=(None, 1, 28, 28),
            input_var=input)

    mlp = lasagne.layers.DropoutLayer(
            mlp, 
            p=dropout_in)
    
    for k in range(n_hidden_layers):

        mlp = binary_connect.DenseLayer(
                mlp, 
                binary=binary,
                stochastic=stochastic,
                # H=H,
                nonlinearity=lasagne.nonlinearities.identity,
                num_units=num_units)                  
        
        mlp = batch_norm.BatchNormLayer(
                mlp,
                epsilon=epsilon, 
                alpha=alpha,
                nonlinearity=lasagne.nonlinearities.rectify)
                
        mlp = lasagne.layers.DropoutLayer(
                mlp, 
                p=dropout_hidden)
    
    mlp = binary_connect.DenseLayer(
                mlp, 
                binary=binary,
                stochastic=stochastic,
                # H=H,
                nonlinearity=lasagne.nonlinearities.identity,
                num_units=10)      
                  
    mlp = batch_norm.BatchNormLayer(
            mlp,
            epsilon=epsilon, 
            alpha=alpha,
            nonlinearity=lasagne.nonlinearities.identity)

    train_output = lasagne.layers.get_output(mlp, deterministic=False)
    
    # squared hinge loss
    loss = T.mean(T.sqr(T.maximum(0.,1.-target*train_output)))
    
    params = lasagne.layers.get_all_params(mlp, trainable=True)
    
    if binary:
        grads = binary_connect.compute_grads(loss,mlp)
        # updates = lasagne.updates.adam(loss_or_grads=grads, params=params, learning_rate=LR)
        updates = lasagne.updates.sgd(loss_or_grads=grads, params=params, learning_rate=LR)
        # updates = binary_connect.weights_clipping(updates,H) 
        updates = binary_connect.weights_clipping(updates,mlp) 
        # using 2H instead of H with stochastic yields about 20% relative worse results
        
    else:
        # updates = lasagne.updates.adam(loss_or_grads=loss, params=params, learning_rate=LR)
        updates = lasagne.updates.sgd(loss_or_grads=loss, params=params, learning_rate=LR)
        # updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=0.01, momentum=0.9)

    test_output = lasagne.layers.get_output(mlp, deterministic=True)
    test_loss = T.mean(T.sqr(T.maximum(0.,1.-target*test_output)))
    test_err = T.mean(T.neq(T.argmax(test_output, axis=1), T.argmax(target, axis=1)),dtype=theano.config.floatX)
    
    # Compile a function performing a training step on a mini-batch (by giving the updates dictionary) 
    # and returning the corresponding training loss:
    train_fn = theano.function([input, target, LR], loss, updates=updates)
    # train_fn = theano.function([input, target], loss, updates=updates)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input, target], [test_loss, test_err])

    print('Training...')
    
    binary_connect.train(
            train_fn,val_fn,
            batch_size,
            LR_start,LR_decay,
            num_epochs,
            X_train,y_train,
            X_val,y_val,
            X_test,y_test)
    
    # print("display histogram")
    
    # W = lasagne.layers.get_all_layers(mlp)[2].W.get_value()
    # print(W.shape)
    
    # histogram = np.histogram(W,bins=1000,range=(-1.1,1.1))
    # np.savetxt(str(dropout_hidden)+str(binary)+str(stochastic)+str(H)+"_hist0.csv", histogram[0], delimiter=",")
    # np.savetxt(str(dropout_hidden)+str(binary)+str(stochastic)+str(H)+"_hist1.csv", histogram[1], delimiter=",")
    
    # Optionally, you could now dump the network weights to a file like this:
    # np.savez('model.npz', lasagne.layers.get_all_param_values(network))