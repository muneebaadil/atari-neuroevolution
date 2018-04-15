from array import array 
import argparse
from deap import base, creator, tools, algorithms
from operator import mul, add
import random

from utils import * 

import pdb 

def GetParser():
    parser = argparse.ArgumentParser(description='Atari using Neurovolution')
    
    #neural network parameters
    parser.add_argument('--input_dim',action='store', type=str, default='80x80', dest='input_dim')
    parser.add_argument('--num_hidden',action='store', type=int, default=200, dest='num_hidden')
    parser.add_argument('--num_actions',action='store', type=int, default=2, dest='num_actions')

    #optimization parameters
    parser.add_argument('--population_size',action='store',type=int,default=10, dest='population_size')
    parser.add_argument('--num_gens',action='store',type=int,default=10, dest='num_gens')
    
    parser.add_argument('--crossover_type',action='store',type=str,default='cxTwoPoint', dest='crossover_type')
    parser.add_argument('--crossover_prob', action='store', type=float, default=.9, dest='crossover_prob')

    parser.add_argument('--mutate_type', action='store', type=str, default='mutGaussian', dest='mutate_type')
    parser.add_argument('--mutate_args',action='store',type=str,default='mu=1,sigma=1', dest='mutate_args')
    parser.add_argument('--mutate_prob',action='store',type=float,default=.1, dest='mutate_prob')
    parser.add_argument('--mutate_indpb',action='store',type=float,default=.5, dest='mutate_indpb')

    parser.add_argument('--select_type',action='store', type=str, default='selTournament', dest='select_type')
    parser.add_argument('--num_select',action='store', type=int, default=2, dest='num_select')
    parser.add_argument('--select_args',action='store', type=str, default='tournsize=2', dest='select_args')
    
    #misc
    # parser.add_argument('--resume',action='store', type=bool, default=False, dest='resume')
    # parser.add_argument('--render',action='store', type=bool, default=False, dest='render')

    return parser

def PostprocessOpts(opts): 

    opts.input_dim = [int(x) for x in opts.input_dim.split('x')]
    opts.dims = [opts.input_dim[0]*opts.input_dim[1], opts.num_hidden, opts.num_actions]
    opts.num_weights = reduce(mul, opts.dims, 1)
    opts.num_biases = reduce(add, opts.dims[1:], 0)
    opts.num_params = opts.num_weights + opts.num_biases

    opts.mutate_args = {x.split('=')[0]: x.split('=')[1] for x in opts.mutate_args.split(',')}
    opts.select_args = {x.split('=')[0]: x.split('=')[1] for x in opts.select_args.split(',')}
    return 

def Evolve(opts): 

    #base class for individuals and fitness function
    creator.create("FitnessMax", base.Fitness, weights=(1.,))
    creator.create("Network", array, fitness=creator.FitnessMax, typecode='f')

    #registering initialization functions..
    toolbox = base.Toolbox()
    toolbox.register("neuron_init", random.random)
    toolbox.register("network_init", tools.initRepeat, creator.Network, 
                                toolbox.neuron_init, n=opts.num_params)
    toolbox.register("population_init", tools.initRepeat, list, toolbox.network_init, n=opts.population_size)

    #cross-over, mutation and selection strategy.. 
    #toolbox.register("evaluate", EvaluateNetwork) fill this line..
    toolbox.register("mate", getattr(tools, opts.crossover_type))
    toolbox.register("mutate", getattr(tools, opts.mutate_type), indpb=opts.mutate_indpb, **opts.mutate_args)
    toolbox.register("select", getattr(tools, opts.select_type), **opts.select_args)

    #initial population generation and evolving.. 
    pop = toolbox.population_init()

    #testing code here
    final_pop, log = algorithms.eaSimple(pop, toolbox, cxpb=opts.crossover_prob, mutpb=opts.mutate_prob, 
                                        ngen=opts.num_gens)

if __name__=='__main__': 
    parser = GetParser()
    opts = parser.parse_args()
    PostprocessOpts(opts)

    Evolve(opts)