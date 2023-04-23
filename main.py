import numpy as np
import collections, random
from datetime import datetime
import os
import wandb

import torch 
import torch.nn as nn
import torch.nn.functional as F

import gymnasium as gym
from gym_2048 import *
from dqn import DQN, ReplayBuffer
from utils import *
from train_episode import train_episode

def main(config):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    now = datetime.now()

    # Format the date and time as an integer in the format of 20230753
    formatted = now.strftime("%Y%m%d%H%M%S")
    model_save_dir = os.path.join(config["model_basedir"], config["env"], config["run_name"], formatted)
    os.makedirs(model_save_dir, exist_ok = True)

    env = gym.make(config["env"])
    env = gym.wrappers.TimeLimit(env, max_episode_steps = config["max_episode_steps"])

    n_observations = env.observation_space.shape[0]
    n_actions = env.action_space.n

    memory = ReplayBuffer(max_size = config["buffer_size"], input_shape= n_observations, n_actions=1)
    if config["algorithm"] == "dqn":
        agent = DQN(obs_dim = n_observations,
                    action_dim = n_actions,
                    config = config).to(device)
    else:
        raise NotImplmentedError

    best_score = 0.
    score_history = []
    average10, loss = 0., 0.
    with wandb.init(project="{}_{}".format(config["algorithm"],config["env"]), name="{}_{}".format(now, config["run_name"]), config=config):
        for n_epi in range(config["n_episodes"]):
            score, loss, eps = train_episode(n_epi, env, memory, agent, config, device)
            score_history.append(score)

            if n_epi > 10:
                    average10 = np.mean(score_history[-10:])

            if n_epi > 100:
                if  average10 > best_score:
                    best_score = average10
                    torch.save({
                        "q_action": agent.q_action.state_dict(),
                        "q_eval": agent.q_eval.state_dict(),
                        "optim": agent.optimizer.state_dict(),
                    }, os.path.join(model_save_dir, "best_score.ckpt"))
                    wandb.save(os.path.join(model_save_dir, "best_score.ckpt"))
                    
            if n_epi%config["log_every"]==0 and n_epi > 0:
                print("# of episode :{}, score1: {:.1f}, score10 : {:.1f}, buffer_size: {}, loss: {}.".format(n_epi, score, average10, memory.size(), loss))
                wandb.log({"Score_1": score,
                        "Score_10": average10,
                        "Loss":loss,
                        "Episode": n_epi ,
                        "Eps": eps,
                        "Buffer size": memory.size()})
            
            if n_epi%config["save_every"]==0:
                torch.save({
                        "q_action": agent.q_action.state_dict(),
                        "q_eval": agent.q_eval.state_dict(),
                        "optim": agent.optimizer.state_dict(),
                    }, os.path.join(model_save_dir, f"{n_epi}.ckpt"))
                wandb.save(os.path.join(model_save_dir, f"{n_epi}.ckpt"))

if __name__ == "__main__":
    config = get_config()
    main(config)