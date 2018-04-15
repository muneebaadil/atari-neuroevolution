import numpy as np 
from array import array 
from itertools import izip 

def Ind2Network(ind, opts):
    """Returns a neural network representation of the chromosome""" 

    ind_ = np.array(ind)
    weights = []
    biases = []

    curr_idx = 0 
    for new_dim, prev_dim in izip(opts.dims[1:], opts.dims[:-1]): 
        inc = int(new_dim * prev_dim)

        weight = np.array(ind[curr_idx:curr_idx+inc]).reshape((new_dim, prev_dim))
        weights.append(weight)

        curr_idx += inc 

    for new_dim, prev_dim in izip(opts.dims[1:], opts.dims[:-1]): 
        inc = int(new_dim)

        bias = np.array(ind[curr_idx:curr_idx+inc]).reshape((new_dim, 1))
        biases.append(bias)

        curr_idx += inc

    out = {'weights': weights, 'biases': biases}
    return out 


def ForwardPass(net, input): 
    """Translates the individual/chromosome into network and forward-pass the input"""
    
    return None 