import numpy as np 
import gym 
from utils import *

import pdb

def Evaluate(ind, opts): 

    net_dict = Ind2Network(ind, opts)

    game_env = gym.make(opts.game_name)
    curr_input = game_env.reset().flatten()[:,np.newaxis]

    total_reward, num_done = 0, 0
    
    while num_done < opts.num_episodes: 

        out = ForwardPass(net_dict, curr_input)
        action = out.argmax(axis=0)
        
        curr_input, reward, done, info = game_env.step(action)
        curr_input = curr_input.flatten()[:,np.newaxis]

        if reward !=0: #if a single game is ended..
            total_reward += reward

        if done: 
            num_done += 1 
            game_env.reset().flatten()[:,np.newaxis]
            
    print 'total reward = {}'.format(total_reward)
    return total_reward,