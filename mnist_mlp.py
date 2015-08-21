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
    
    # standardize the dataset
    def standardize(X):
        X -= X.mean(axis=0)
        X /= (X.std(axis=0)+epsilon)
        return X
    X_train = standardize(X_train)
    X_val = standardize(X_val)
    X_test = standardize(X_test)

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

    mlp = lasagne.layers.DropoutLayer(
            mlp, 
            p=0.2)
    
    for k in range(n_hidden_layers):
    
        mlp = lasagne.layers.DenseLayer(
                mlp, 
                nonlinearity=lasagne.nonlinearities.identity,
                num_units=num_units)    
        
        mlp = BatchNormLayer(
                mlp,
                epsilon=epsilon, 
                alpha=alpha,
                nonlinearity=lasagne.nonlinearities.rectify)
                
        mlp = lasagne.layers.DropoutLayer(
                mlp, 
                p=0.5)

    mlp = lasagne.layers.DenseLayer(
            mlp, 
            nonlinearity=lasagne.nonlinearities.identity,
            num_units=10)
            
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

    test_output = lasagne.layers.get_output(mlp, deterministic=True)
    test_loss = T.mean(T.sqr(T.maximum(0.,1.-target*test_output)))
    test_err = T.mean(T.neq(T.argmax(test_output, axis=1), T.argmax(target, axis=1)),dtype=theano.config.floatX)

    # Compile a function performing a training step on a mini-batch (by giving the updates dictionary) 
    # and returning the corresponding training loss:
    train_fn = theano.function([input, target], loss, updates=updates)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input, target], [test_loss, test_err])

    print('Training...')
    
    def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
        assert len(inputs) == len(targets)
        if shuffle:
            indices = np.arange(len(inputs))
            np.random.shuffle(indices)
        for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batchsize]
            else:
                excerpt = slice(start_idx, start_idx + batchsize)
            yield inputs[excerpt], targets[excerpt]

    # We iterate over epochs:
    for epoch in range(num_epochs):

        # In each epoch, we do a full pass over the training data:
        train_loss = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(X_train, y_train, batch_size, shuffle=True):
            inputs, targets = batch
            train_loss += train_fn(inputs, targets)
            train_batches += 1

        # And a full pass over the validation data:
        val_err = 0
        val_loss = 0
        val_batches = 0
        for batch in iterate_minibatches(X_val, y_val, batch_size, shuffle=False):
            inputs, targets = batch
            loss, err = val_fn(inputs, targets)
            val_err += err
            val_loss += loss
            val_batches += 1

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_loss / train_batches))
        print("  validation loss:\t\t{:.6f}".format(val_loss / val_batches))
        print("  validation error rate:\t{:.2f} %".format(val_err / val_batches * 100))

    # After training, we compute and print the test error:
    test_err = 0
    test_loss = 0
    test_batches = 0
    for batch in iterate_minibatches(X_test, y_test, batch_size, shuffle=False):
        inputs, targets = batch
        loss, err = val_fn(inputs, targets)
        test_err += err
        test_loss += loss
        test_batches += 1
    print("Final results:")
    print("  test loss:\t\t\t{:.6f}".format(test_loss / test_batches))
    print("  test error rate:\t\t{:.2f} %".format(test_err / test_batches * 100))

    # Optionally, you could now dump the network weights to a file like this:
    # np.savez('model.npz', lasagne.layers.get_all_param_values(network))