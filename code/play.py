import numpy as np
import gym
import argparse
from operator import mul
import pickle
from time import sleep 

from utils import * 
import preprocess

def GetParser(): 
    parser = argparse.ArgumentParser()

    #Atari game parameters 
    parser.add_argument('--game',action='store', type=str, default='Asterix-v0',
                         dest='game')
    parser.add_argument('--num_episodes',action='store', type=int, default=-1, 
                        dest='num_episodes')
    
    #neural net parameters
    parser.add_argument('--input_dim',action='store', type=str, default='210x160x3',
                         dest='input_dim')
    parser.add_argument('--num_hidden',action='store', type=int, default=6, 
                        dest='num_hidden')
    parser.add_argument('--num_actions',action='store', type=int, default=9,
                        dest='num_actions')
    parser.add_argument('--pretrained',action='store', type=str,dest='pretrained',
                         default='../experiments/asterix/network.pkl')

    #input/preprocess arguments
    parser.add_argument('--preprocess',action='store', type=str, default='FramesDiff',
                         dest='preprocess')
    parser.add_argument('--preprocess_args',action='store', type=str, 
                        default='scale=255.', dest='preprocess_args')

    return parser

def PostprocessOpts(opts): 
    opts.input_dim = [int(x) for x in opts.input_dim.split('x')]
    opts.dims = [reduce(mul, opts.input_dim, 1), opts.num_hidden, opts.num_actions]
    opts.num_episodes = float("inf") if (opts.num_episodes < 0) else opts.num_episodes
    opts.preprocess_args = {x.split('=')[0]: float(x.split('=')[1])\
                             for x in opts.preprocess_args.split(',')} \
                             if opts.preprocess_args != '' else {}
    return 

def Play(opts): 
    ind = np.load(opts.pretrained)
    net_dict = Ind2Network(ind, opts)

    game_env = gym.make(opts.game)
    pre_func = getattr(preprocess, opts.preprocess)(opts.input_dim, 
                                                    **opts.preprocess_args)

    num_done = 0
    
    while num_done < opts.num_episodes: 

        curr_input = game_env.reset()
        pre_func.Reset()
        net_input = pre_func(curr_input)
        done, total_reward = False, 0

        while not done:     
            game_env.render()

            out = ForwardPass(net_dict, net_input)
            action = out.argmax(axis=0)
        
            curr_input, reward, done, info = game_env.step(action)
            net_input = pre_func(curr_input)
        
            total_reward += reward
            sleep(0.05)
        num_done += 1 
        print 'Episode {}: Reward = {}'.format(num_done, total_reward)
    
    game_env.close()
    return 

if __name__=='__main__': 
    parser=GetParser()
    opts=parser.parse_args()
    PostprocessOpts(opts)

    Play(opts)
