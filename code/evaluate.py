import numpy as np 
import gym 
from utils import *
import preprocess

import pdb

def Evaluate(ind, opts): 

    net_dict = Ind2Network(ind, opts)

    game_env = gym.make(opts.game_name)
    pre_func = getattr(preprocess, opts.preprocess)(opts.input_dim, 
                                                    **opts.preprocess_args)

    total_reward, num_done = 0, 0
    
    while num_done < opts.num_episodes: 

        curr_input = game_env.reset()
        pre_func.Reset()
        net_input = pre_func(curr_input)
        done = False

        while not done:     
            if opts.render: 
                game_env.render()

            out = ForwardPass(net_dict, net_input)
            action = out.argmax(axis=0)
        
            curr_input, reward, done, info = game_env.step(action)
            net_input = pre_func(curr_input)
        
            total_reward += reward
        
        num_done += 1 
    
    game_env.close()
    return total_reward,