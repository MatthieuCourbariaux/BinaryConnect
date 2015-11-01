# BinaryConnect

## Requirements

* Python, Numpy, Scipy
* Theano 0.6 (Bleeding edge version)
* Pylearn2 0.1
* Lasagne 0.1
* a fast GPU or a large amount of patience

## Motivations

This repository allows to easily reproduce the experimental results reported in the paper:
"BinaryConnect: Training Deep Neural Networks with binary weights during propagations".

## MNIST

    python mnist.py
    
This python script trains an MLP on MNIST with the stochastic version of BinaryConnect.
It should run for about 30 minutes on a GTX 680 GPU.
The final test error should be around 1.15% (not exactly the same setup as in the paper).

## CIFAR-10

    python cifar10.py
    
This python script trains a CNN on CIFAR-10 with the stochastic version of BinaryConnect.
It should run for about 20 hours on a Titan Black GPU.
The final test error should be around 8.27%.

## SVHN

    export SVHN_LOCAL_PATH=/Tmp/SVHN/
    python svhn_preprocessing.py

This python script (taken from Pylearn2) computes a preprocessed (GCN and ZCA whitening) version of the SVHN dataset in a temporary folder (SVHN_LOCAL_PATH).

    python svhn.py
    
This python script trains a CNN on SVHN with the stochastic version of BinaryConnect.
It should run for about 2 days on a Titan Black GPU.
The final test error should be around 2.15%.

## How to play with it

The python scripts mnist.py, cifar10.py and svhn.py contain all the relevant hyperparameters.
It is very straightforward to modify them.
binary_connect.py contains the binarization function (called binarization).
