import numpy as np 
import gym 
from utils import *

import pdb

def Evaluate(ind, opts): 

    net_dict = Ind2Network(ind, opts)

    game_env = gym.make(opts.game_name)
    curr_input = game_env.reset().flatten()[:,np.newaxis]

    total_reward, done = 0, None 
    
    while done!=False: 

        out = ForwardPass(net_dict, curr_input)
        action = out.argmax(axis=0)
        
        curr_input, reward, done, info = game_env.step(action)
        curr_input = curr_input.flatten()[:,np.newaxis]

        print 'reward = {}'.format(reward)
        if reward !=0: #if a single game is ended..
            total_reward += reward

    return total_reward