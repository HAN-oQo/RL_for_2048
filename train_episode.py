import numpy as np
import collections, random
from datetime import datetime
import os
import wandb

import torch 
import torch.nn as nn
import torch.nn.functional as F
from dqn import ReplayBuffer

def train_episode(n_epi, env, memory, agent, config, device):
    assert memory is not None
    assert agent is not None
    assert config is not None
    
    if config["algorithm"] == 'dqn':
        s, info = env.reset()
        action_mask = info["action_mask"]
        done = False
        trunc = False
        score = 0.
        loss =0.
        eps = config["eps_low"]+(config["eps_high"]-config["eps_low"]) * (np.exp(-1.0 * n_epi/config["eps_decay"]))
        while not (done or trunc):
            a = agent.get_action(
                torch.tensor(s).float().to(device).unsqueeze(0),
                eps = eps,
                action_mask = torch.tensor(action_mask).bool().to(device))
            s_prime, r, done, trunc, info = env.step(a)
            action_mask = info["action_mask"]
            if config["reward_scale"] == "log":
                r = np.log2(r) if r else 0.
            elif config["reward_scale"] == "divide_10":
                r = r/10
            elif config["reward_scale"] == "divide_100":
                r = r/100
            
            memory.put(s, a, r, s_prime, done)
            score += r
            s = s_prime
        
        if memory.size() > config["start_size"]:
            mini_batch = memory.sample(config["batch_size"])
            mini_batch = ReplayBuffer.batch_to_device(mini_batch, device)
            loss = agent.train_net(mini_batch)
        
        return score, loss, eps
        
    
    else:
        raise NotImplmentedError

    