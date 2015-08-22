from __future__ import print_function

import sys
import os
import time

import numpy as np
np.random.seed(1234)  # for reproducibility

import theano
import theano.tensor as T

import lasagne

import cPickle as pickle
import gzip

from batch_norm import BatchNormLayer

from binary_connect import weights_clipping, BinaryDenseLayer

if __name__ == "__main__":
    
    # BN parameters
    # alpha = .1 # for a minibatch of size 50
    # alpha = .2 # for a minibatch of size 100
    alpha = .33 # for a minibatch of size 200
    epsilon = 1e-4
    
    # MLP parameters
    num_units = 1024
    n_hidden_layers = 3
    
    # Training parameters
    num_epochs = 500
    batch_size = 200
    
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

    mlp = lasagne.layers.InputLayer(
            shape=(None, 1, 28, 28),
            input_var=input)

    # mlp = lasagne.layers.DropoutLayer(
            # mlp, 
            # p=0.2)
    
    for k in range(n_hidden_layers):
    
        # mlp = lasagne.layers.DenseLayer(
                # mlp, 
                # nonlinearity=lasagne.nonlinearities.identity,
                # num_units=num_units)  

        mlp = BinaryDenseLayer(
                mlp, 
                nonlinearity=lasagne.nonlinearities.identity,
                num_units=num_units)                  
        
        mlp = BatchNormLayer(
                mlp,
                epsilon=epsilon, 
                alpha=alpha,
                nonlinearity=lasagne.nonlinearities.rectify)
                
        # mlp = lasagne.layers.DropoutLayer(
                # mlp, 
                # p=0.5)
    
    mlp = BinaryDenseLayer(
                mlp, 
                nonlinearity=lasagne.nonlinearities.identity,
                num_units=10) 
                
    # mlp = lasagne.layers.DenseLayer(
            # mlp, 
            # nonlinearity=lasagne.nonlinearities.identity,
            # num_units=10) 
                  
    mlp = BatchNormLayer(
            mlp,
            epsilon=epsilon, 
            alpha=alpha,
            nonlinearity=lasagne.nonlinearities.identity)

    train_output = lasagne.layers.get_output(mlp)
    # squared hinge loss
    loss = T.mean(T.sqr(T.maximum(0.,1.-target*train_output)))
    params = lasagne.layers.get_all_params(mlp, trainable=True)
    # updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=0.1, momentum=0.9)
    updates = lasagne.updates.adam(loss, params)
    updates = weights_clipping(updates)

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

    # Optionally, you could now dump the network weights to a file like this:
    # np.savez('model.npz', lasagne.layers.get_all_param_values(network))