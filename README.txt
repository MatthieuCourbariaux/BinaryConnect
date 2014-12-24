Deep learning arithmetic simulator
==================================

Requirements:
    - Theano 0.6
    - Pylearn2 0.1 
    - PyTables (for the SVHN dataset)
    - a CUDA capable GPU

This code was written to allow anyone to easily reproduce the results 
presented in the arxiv paper http://arxiv.org/abs/1412.7024 .
However, the results might slightly vary depending on the environment.

The simulator is relatively easy to use:
    python simulator.py task format [initial range] [computations bit-width] 
        [parameters updates bit-width] [range update frequency]
        [maximum overflow rate] [number of epochs of range initialization]

Task:
    There are 4 different tasks: the permutation invariant MNIST (PI_MNIST), 
    MNIST, CIFAR10 and SVHN.
    A predetermined model is associated with each of those tasks 
    (The models are stored in model.py).
    For the SVHN dataset, 
    you need to set an environment variable: 
    SVHN_LOCAL_PATH=/tmp/SVHN/ 
    You then need to pre-process it with the script 
    utilities/svhn_preprocessing.py (courtesy of pylearn2)

Format:
    There are 4 different formats: floating point (FLP), 
    half floating point (HFLP), 
    fixed point (FXP) and dynamic fixed point (DFXP).

Initial range:
    Initial range is only useful for FXP and DFXP. 
    It is the initial position of the radix point 
    for the fixed point formats.
    5 works most of the time.

Computations bit-width and parameters updates bit-widths:
    Only useful for FXP and DFXP.
    Those are the bit-widths of respectively the 
    computations and the parameters updates.
    The sign is not counted in the bit-width.

Range update frequency:
    Range update frequency is only useful for DFXP.
    It is the number of batches between two range updates.

Maximum overflow rate:
    Only useful for DFXP.
    It is the amount of overflow tolerated before augmenting the range.
    
Number of epochs of range initialization:
    Only useful for DFXP.
    This is the number of epochs we train with high precision 
    to position the radix point.
    Once the radix point is positioned, 
    the parameters are reinitialized, and the DFXP training can begin.    
        
Examples:
    python simulator.py PI_MNIST FLP
    python simulator.py SVHN FXP 5 19 19
    python simulator.py CIFAR10 DFXP 5 9 11 100 0.0001 2
        