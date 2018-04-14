from array import array 
import gym
import argparse
from deap import base, creator, tools 
from operator import mul 
import random

def GetParser():
    parser = argparse.ArgumentParser(description='Atari using Neurovolution')
    
    #neural network parameters
    parser.add_argument('--input_dim',action='store', type=str, default='80x80', dest='input_dim')
    parser.add_argument('--num_hidden',action='store', type=int, default=200, dest='num_hidden')
    parser.add_argument('--num_actions',action='store', type=int, default=200, dest='num_actions')

    #optimization parameters
    parser.add_argument('--population_size',action='store',type=int,default=10, dest='population_size')
    parser.add_argument('--resume',action='store', type=bool, default=False, dest='resume')
    parser.add_argument('--render',action='store', type=bool, default=False, dest='render')

    return parser

def PostprocessOpts(opts): 
    opts.input_dim = [int(x) for x in opts.input_dim.split('x')]
    opts.num_neurons = reduce(mul, opts.input_dim, 1)
    return 

def Evolve(opts): 

    #base class for individuals and fitness function
    creator.create("FitnessMax", base.Fitness, weights=(1.,))
    creator.create("Network", array, fitness=creator.FitnessMax)

    #registering initialization functions..
    toolbox = base.Toolbox()
    toolbox.register("neuron_init", random.random)
    toolbox.register("network_init", tools.initRepeat, creator.Network, 
                                toolbox.neuron_init, n=opts.num_neurons)
    toolbox.register("population_init", tools.initRepeat, list, toolbox.network_init, n=opts.population_size)

    #cross-over, mutation and selection strategy.. 
    #toolbox.register("evaluate", EvaluateNetwork) fill this line..
    # toolbox.register("mate", )
    # toolbox.register("mutate")
    # toolbox.register("select")

if __name__=='__main__': 
    parser = GetParser()
    opts = parser.parse_args()
    PostprocessOpts(opts)

    Evolve(opts)