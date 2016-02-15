**Please checkout our latest work,  
[BinaryNet: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1](http://arxiv.org/abs/1602.02830),  
and the associated [github repository](https://github.com/MatthieuCourbariaux/BinaryNet).**

# BinaryConnect

## Motivations

The goal of this repository is to enable the reproduction of the experiments described in  
[BinaryConnect: Training Deep Neural Networks with binary weights during propagations](http://arxiv.org/abs/1511.00363).  
You may want to checkout our subsequent work:
* [Neural Networks with Few Multiplications](http://arxiv.org/abs/1510.03009)
* [BinaryNet: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1](http://arxiv.org/abs/1602.02830)

## Requirements

* Python, Numpy, Scipy
* [Theano](http://deeplearning.net/software/theano/install.html) (Bleeding edge version)
* [Pylearn2](http://deeplearning.net/software/pylearn2/)
* [Lasagne](http://lasagne.readthedocs.org/en/latest/user/installation.html)
* [PyTables](http://www.pytables.org/usersguide/installation.html) (only for the SVHN dataset)
* a fast Nvidia GPU or a large amount of patience

## MNIST

    python mnist.py
    
This python script trains an MLP on MNIST with the stochastic version of BinaryConnect.
It should run for about 30 minutes on a GTX 680 GPU.
The final test error should be around **1.15%**.
Please note that this is NOT the experiment reported in the article (which is in the "master" branch of the repository).

## CIFAR-10

    python cifar10.py
    
This python script trains a CNN on CIFAR-10 with the stochastic version of BinaryConnect.
It should run for about 20 hours on a Titan Black GPU.
The final test error should be around **8.27%**.

## SVHN

    export SVHN_LOCAL_PATH=/Tmp/SVHN/
    python svhn_preprocessing.py

This python script (taken from Pylearn2) computes a preprocessed (GCN and LCN) version of the SVHN dataset in a temporary folder (SVHN_LOCAL_PATH).

    python svhn.py
    
This python script trains a CNN on SVHN with the stochastic version of BinaryConnect.
It should run for about 2 days on a Titan Black GPU.
The final test error should be around **2.15%**.

## How to play with it

The python scripts mnist.py, cifar10.py and svhn.py contain all the relevant hyperparameters.
It is very straightforward to modify them.
binary_connect.py contains the binarization function (called binarization).

Have fun!
