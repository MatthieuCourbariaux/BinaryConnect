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
# def MNIST_exp(dropout_in,dropout_hidden,binary,stochastic,H):
    
    # BN parameters
    batch_size = 200
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
    H = 1.
    
    # LR decay
    # LR_start = .3
    # LR_fin = .01
    # LR_decay = (LR_fin/LR_start)**(1./num_epochs) 
    
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
    # LR = T.scalar('LR', dtype=theano.config.floatX)

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
                H=H,
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
                H=H,
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
        updates = lasagne.updates.adam(loss_or_grads=grads, params=params, learning_rate=0.001)
        # updates = lasagne.updates.sgd(grads, params, learning_rate=.3) 
        updates = binary_connect.weights_clipping(updates,H) 
        # using 2H instead of H with stochastic yields about 20% worse results
        
    else:
        updates = lasagne.updates.adam(loss_or_grads=loss, params=params)
        # updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=0.01, momentum=0.9)

    test_output = lasagne.layers.get_output(mlp, deterministic=True)
    test_loss = T.mean(T.sqr(T.maximum(0.,1.-target*test_output)))
    test_err = T.mean(T.neq(T.argmax(test_output, axis=1), T.argmax(target, axis=1)),dtype=theano.config.floatX)
    
    # Compile a function performing a training step on a mini-batch (by giving the updates dictionary) 
    # and returning the corresponding training loss:
    train_fn = theano.function([input, target], loss, updates=updates)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input, target], [test_loss, test_err])

    print('Training...')
    
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
    
    # shuffle the train set
    X_train,y_train = shuffle(X_train,y_train)
    best_val_err = 100
    best_epoch = 1
    
    # We iterate over epochs:
    for epoch in range(num_epochs):

        # In each epoch, we do a full pass over the training data:
        train_loss = 0
        train_batches = len(X_train)/batch_size
        start_time = time.time()
        
        for i in range(train_batches):
            train_loss += train_fn(X_train[i*batch_size:(i+1)*batch_size],y_train[i*batch_size:(i+1)*batch_size])
        
        train_loss/=train_batches
        
        # And a full pass over the validation data:
        val_err = 0
        val_loss = 0
        val_batches = len(X_val)/batch_size
        
        for i in range(val_batches):
            loss, err = val_fn(X_val[i*batch_size:(i+1)*batch_size], y_val[i*batch_size:(i+1)*batch_size])
            val_err += err
            val_loss += loss
        
        val_err = val_err / val_batches * 100
        val_loss /= val_batches
        
        # test if validation error went down
        if val_err <= best_val_err:
            
            best_val_err = val_err
            best_epoch = epoch+1
            
            test_err = 0
            test_loss = 0
            test_batches = len(X_test)/batch_size
            
            for i in range(test_batches):
                loss, err = val_fn(X_test[i*batch_size:(i+1)*batch_size], y_test[i*batch_size:(i+1)*batch_size])
                test_err += err
                test_loss += loss
                
            test_err = test_err / test_batches * 100
            test_loss /= test_batches
        
        # shuffle the train set
        X_train,y_train = shuffle(X_train,y_train)
        
        epoch_duration = time.time() - start_time
        
        # Then we print the results for this epoch:
        print("Epoch "+str(epoch + 1)+" of "+str(num_epochs)+" took "+str(epoch_duration)+"s")
        print("  training loss:                 "+str(train_loss))
        print("  validation loss:               "+str(val_loss))
        print("  validation error rate:         "+str(val_err)+"%")
        print("  best epoch:                    "+str(best_epoch))
        print("  best validation error rate:    "+str(best_val_err)+"%")
        print("  test loss:                     "+str(test_loss))
        print("  test error rate:               "+str(test_err)+"%") 
    
    # print("display histogram")
    
    # W = lasagne.layers.get_all_layers(mlp)[2].W.get_value()
    # print(W.shape)
    
    # histogram = np.histogram(W,bins=1000,range=(-1.1,1.1))
    # np.savetxt(str(dropout_hidden)+str(binary)+str(stochastic)+str(H)+"_hist0.csv", histogram[0], delimiter=",")
    # np.savetxt(str(dropout_hidden)+str(binary)+str(stochastic)+str(H)+"_hist1.csv", histogram[1], delimiter=",")
    
    # Optionally, you could now dump the network weights to a file like this:
    # np.savez('model.npz', lasagne.layers.get_all_param_values(network))
    
# if __name__ == "__main__":
if False:
    
    # baselines
    MNIST_exp(dropout_in=0.,dropout_hidden=0.,binary=False,stochastic=False,H=1.)
    MNIST_exp(dropout_in=0.2,dropout_hidden=0.5,binary=False,stochastic=False,H=1.)
    
    # stochastic BC
    MNIST_exp(dropout_in=0.,dropout_hidden=0.,binary=True,stochastic=True,H=1.)
    MNIST_exp(dropout_in=0.,dropout_hidden=0.,binary=True,stochastic=True,H=1./(1<<2))
    MNIST_exp(dropout_in=0.,dropout_hidden=0.,binary=True,stochastic=True,H=1./(1<<4))
    MNIST_exp(dropout_in=0.,dropout_hidden=0.,binary=True,stochastic=True,H=1./(1<<6))
    MNIST_exp(dropout_in=0.,dropout_hidden=0.,binary=True,stochastic=True,H=1./(1<<8))
    
    # deterministic BC
    MNIST_exp(dropout_in=0.,dropout_hidden=0.,binary=True,stochastic=False,H=1.)
    MNIST_exp(dropout_in=0.,dropout_hidden=0.,binary=True,stochastic=False,H=1./(1<<2))
    MNIST_exp(dropout_in=0.,dropout_hidden=0.,binary=True,stochastic=False,H=1./(1<<4))
    MNIST_exp(dropout_in=0.,dropout_hidden=0.,binary=True,stochastic=False,H=1./(1<<6))
    MNIST_exp(dropout_in=0.,dropout_hidden=0.,binary=True,stochastic=False,H=1./(1<<8))
    