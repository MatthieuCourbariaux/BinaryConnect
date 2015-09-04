# BinaryConnect

## Requirements

* Python, Numpy, Scipy
* Theano 0.6 (Bleeding edge version)
* Pylearn2 0.1
* Lasagne 0.1
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

## How to play with it

The python scripts mnist.py and cifar10.py contain all the relevant hyperparameters.
It is very straightforward to modify them.
binary_connect.py contains the binarization function (see DenseLayer and Conv2DLayer classes).
