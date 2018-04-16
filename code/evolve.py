from array import array 
import argparse
from deap import base, creator, tools, algorithms
from operator import mul, add
import random
import os
from time import gmtime, strftime

from utils import * 
from evaluate import Evaluate

import pdb 

def GetParser():
    parser = argparse.ArgumentParser(description='Atari using Neuroevolution')
    
    #neural network parameters
    parser.add_argument('--input_dim',action='store', type=str, default='210x160x3', dest='input_dim')
    parser.add_argument('--num_hidden',action='store', type=int, default=6, dest='num_hidden')
    parser.add_argument('--num_actions',action='store', type=int, default=6, dest='num_actions')
    parser.add_argument('--init_func',action='store', type=str, default='gauss', dest='init_func')
    parser.add_argument('--init_args',action='store', type=str, default='mu=0,sigma=1', dest='init_args')

    #input/preprocess arguments
    parser.add_argument('--preprocess',action='store', type=str, default='FramesDiff', dest='preprocess')
    parser.add_argument('--preprocess_args',action='store', type=str, default='scale=255.', dest='preprocess_args')

    #optimization parameters
    parser.add_argument('--population_size',action='store',type=int,default=1, dest='population_size')
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
    
    #evaluation parameters
    parser.add_argument('--game_name',action='store', type=str, default='Pong-v0', dest='game_name')
    parser.add_argument('--num_episodes',action='store', type=int, default=1, dest='num_episodes')

    # logging/verbosity parameters.. 
    parser.add_argument('--hof_maxsize',action='store', type=int, default=1, dest='hof_maxsize')

    parser.add_argument('--render',action='store', type=bool, default=False, dest='render')
    parser.add_argument('--verbose',action='store', type=bool, default=False, dest='verbose')
    parser.add_argument('--exp_root_dir',action='store', type=str, default='../experiments', dest='exp_root_dir')
    parser.add_argument('--exp_name',action='store', type=str, default=strftime("%Y-%m-%d__%H-%M-%S",gmtime()),
                         dest='exp_name')

    # parser.add_argument('--resume',action='store', type=bool, default=False, dest='resume')

    return parser

def PostprocessOpts(opts): 

    opts.input_dim = [int(x) for x in opts.input_dim.split('x')]
    opts.dims = [reduce(mul, opts.input_dim, 1), opts.num_hidden, opts.num_actions]
    opts.num_weights = reduce(mul, opts.dims, 1)
    opts.num_biases = reduce(add, opts.dims[1:], 0)
    opts.num_params = opts.num_weights + opts.num_biases

    opts.mutate_args = {x.split('=')[0]: int(x.split('=')[1]) for x in opts.mutate_args.split(',')}
    opts.select_args = {x.split('=')[0]: int(x.split('=')[1]) for x in opts.select_args.split(',')}
    opts.init_args = {x.split('=')[0]: float(x.split('=')[1]) for x in opts.init_args.split(',')}
    opts.preprocess_args = {x.split('=')[0]: float(x.split('=')[1]) for x in opts.preprocess_args.split(',')}

    opts.exp_dir = os.path.join(opts.exp_root_dir, opts.exp_name)
    return 

def InitSetup(opts):
    #base class for individuals and fitness function
    creator.create("FitnessMax", base.Fitness, weights=(1.,))
    creator.create("Network", array, fitness=creator.FitnessMax, typecode='f')

    #registering initialization functions and game environment..
    toolbox = base.Toolbox()
    toolbox.register("neuron_init", getattr(random, opts.init_func), **opts.init_args)
    toolbox.register("network_init", tools.initRepeat, creator.Network, 
                                toolbox.neuron_init, n=opts.num_params)
    toolbox.register("population_init", tools.initRepeat, list, toolbox.network_init, n=opts.population_size)

    #cross-over, mutation and selection strategy.. 
    toolbox.register("evaluate", Evaluate, opts=opts)
    toolbox.register("mate", getattr(tools, opts.crossover_type))
    toolbox.register("mutate", getattr(tools, opts.mutate_type), indpb = opts.mutate_indpb, 
                    **opts.mutate_args)
    toolbox.register("select", getattr(tools, opts.select_type))

    #configuring logging
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)

    logbook, hof = tools.Logbook(), tools.HallOfFame(maxsize=opts.hof_maxsize)
    logbook.header = ['gen','evals','avg','min','max']

    #directory for experiments/checkpoints
    if not os.path.exists(opts.exp_dir):
        os.makedirs(opts.exp_dir)
    WriteConfigToFile(os.path.join(opts.exp_dir,'config.txt'), vars(opts))

    return creator, toolbox, stats, logbook, hof

def Evolve(opts): 
    creator, toolbox, stats, logbook, hof = InitSetup(opts)

    #initial population..
    pop = toolbox.population_init()    
    fitnesses = toolbox.map(toolbox.evaluate, pop)
    for ind, fit in izip(pop, fitnesses): 
        ind.fitness.values = fit 
    
    #evolving..
    for curr_gen in xrange(opts.num_gens): 

        #updating statistics..
        hof.update(pop)
        record = stats.compile(pop)
        logbook.record(gen=curr_gen, evals=len(fitnesses), **record)

        #verbosity 
        print_str = 'gen = {}, evals = {}'.format(curr_gen, len(fitnesses))
        for k in ['avg','min','max']: 
            print_str += ', {} = {}'.format(k, record[k])
        print print_str

        #actual evolution..
        parents = toolbox.select(pop, k=opts.num_select, **opts.select_args)
        offsprings = algorithms.varAnd(parents, toolbox, cxpb=opts.crossover_prob, 
                                        mutpb=opts.mutate_prob)
        fitnesses = toolbox.map(toolbox.evaluate, offsprings)
        for ind, fit in izip(offsprings, fitnesses): 
            ind.fitness.values = fit 

        pop = offsprings

    curr_gen += 1 
    #updating statistics for final generation..
    hof.update(pop)
    record = stats.compile(pop)
    logbook.record(gen=curr_gen, evals=len(fitnesses), **record)

    #verbosity 
    print_str = 'gen = {}, evals = {}'.format(curr_gen, len(fitnesses))
    for k in ['avg','min','max']: 
        print_str += ', {} = {}'.format(k, record[k])
    print print_str

    PlotLog(logbook, opts.game_name, os.path.join(opts.exp_dir, 'plot.png'))
    
if __name__=='__main__': 
    parser = GetParser()
    opts = parser.parse_args()
    PostprocessOpts(opts)

    Evolve(opts)