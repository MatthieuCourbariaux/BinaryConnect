# BinaryConnect

## Requirements

* Python, Numpy, Scipy
* Theano 0.6 (Bleeding edge version)
* Pylearn2 0.1
* PyTables (only for the SVHN dataset)
* a fast GPU or a large amount of patience

## Motivations

This repository allows to easily reproduce the experimental results reported in:
"BinaryConnect: Training Deep Neural Networks with binary weights during propagations".

## MNIST

    python mnist.py
    
This python script trains a MLP on MNIST with the stochastic version of BinaryConnect.
It should run for less than 1 hour on a Tesla M2050 GPU.
The final test error should be around 1.18%.

## CIFAR-10

    python cifar10.py
    
This python script trains a CNN on CIFAR-10 with the stochastic version of BinaryConnect.
It should run for about 5 hours on a Titan X GPU.
The final test error should be around 12.20%.

## SVHN
    
    export SVHN_LOCAL_PATH=/tmp/SVHN/
    svhn_preprocessing.py
    
This python script (taken from Pylearn2) computes a preprocessed version of the SVHN dataset in a temporary folder.

    python svhn.py
    
This python script trains a CNN on SVHN with the stochastic version of BinaryConnect.
It should run for about 15 hours on a Titan X GPU.
The final test error should be around 2.66%.

## How to play with it

The python scripts mnist.py, cifar10.py and svhn.py contain all the relevant hyperparameters.
It is very straightforward to modify them.
layer.py contains the binarization function (binarize_weights).