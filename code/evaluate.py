import numpy as np 
import gym 
from utils import *
import preprocess

import pdb

def Evaluate(ind, opts): 

    net_dict = Ind2Network(ind, opts)

    game_env = gym.make(opts.game_name)
    pre_func = getattr(preprocess, opts.preprocess)(**opts.preprocess_args)

    curr_input = game_env.reset()
    net_input = pre_func(curr_input)

    total_reward, num_done = 0, 0
    
    while num_done < opts.num_episodes: 
        if opts.render: 
            game_env.render()

        out = ForwardPass(net_dict, net_input)
        action = out.argmax(axis=0)
        
        curr_input, reward, done, info = game_env.step(action)
        net_input = pre_func(curr_input)
        
        if reward !=0: #if a single game is ended..
            total_reward += reward

        if done: #if whole episode is ended..
            num_done += 1 
            curr_input = game_env.reset()
            pre_func.Reset()
            net_input = pre_func(curr_input)
    
    game_env.close()

    return total_reward,