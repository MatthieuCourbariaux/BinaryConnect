# BinaryConnect

## Requirements

* Python, Numpy, Scipy
* Theano 0.6 (Bleeding edge version)
* Pylearn2 0.1

## Motivations

Easily reproduce the results of:  
"BinaryConnect: Training Deep Neural Networks with binary weights during propagations".

## How to run it

    python mnist_mlp.py
    
This python script will train a MLP on MNIST with the stochastic version of BinaryConnect.
It should run for about an hour on an old GPU (Tesla M2050).
The final test error should be around 1.2%.

## How to play with it

mnist_mlp.py contains all the relevant hyperparameters.
It is very straightforward to modify it.
layer.py contains the binarization function (binarize_weights).