import numpy as np 
from array import array 
from itertools import izip 
import matplotlib.pyplot as plt

import pdb

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
    """Forward pass the input through network (represented in dictionary)"""
    
    weights, biases = net['weights'], net['biases']

    num_layers = len(weights)

    for i, (w,b) in enumerate(izip(weights, biases)): 
        input = w.dot(input) + b

        if i != num_layers -1: #if not last layer..
            input[input < 0] = 0 #relu activation

    #softmax layer 
    output_ = input[:,0] 
    output_ -= output_.max()
    output_ = np.exp(output_)
    output = output_ / np.sum(output_)
    
    return output 

def WriteConfigToFile(fpath, optsDict): 
    fobj = open(fpath, 'w')

    for k,v in optsDict.items(): 
        fobj.write('{} >> {}\n'.format(str(k), str(v)))
    fobj.close()

def PlotLog(log, game_name, filename): 
    plt.style.use('ggplot')

    gen = log.select('gen')
    fig,ax=plt.subplots()
    
    for att in log.header: 
        if att not in ['gen','evals']: 
            att_ = log.select(att)
            ax.plot(gen,att_,label=att)
    
    ax.legend()
    plt.xlabel('Generations')
    plt.ylabel('Fitness Score')
    plt.title(game_name)

    plt.savefig(filename)