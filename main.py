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
from sac import SAC
from utils import *
from train_episode import train_episode
import shutil

def main(config):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    now = datetime.now()

    # Format the date and time as an integer in the format of 20230753
    formatted = now.strftime("%Y%m%d%H%M%S")
    model_save_dir = os.path.join(config["model_basedir"], config["env"], config["run_name"], formatted)
    os.makedirs(model_save_dir, exist_ok = True)
    shutil.copyfile(config["config_path"], f"{model_save_dir}/config.yaml")
    env = gym.make(config["env"])
    env = gym.wrappers.TimeLimit(env, max_episode_steps = config["max_episode_steps"])

    n_observations = env.observation_space.shape[0]
    n_actions = env.action_space.n

    memory = ReplayBuffer(max_size = config["buffer_size"], input_shape= n_observations, n_actions=1)
    if config["algorithm"] == "dqn":
        agent = DQN(obs_dim = n_observations,
                    action_dim = n_actions,
                    config = config).to(device)
    elif config["algorithm"] == "sac":
        agent = SAC(obs_dim = n_observations,
                    action_dim = n_actions,
                    config = config).to(device)
    else:
        raise NotImplmentedError

    best_score = 0.
    score_history = []
    average10, loss = 0., 0.
    with wandb.init(project="{}_{}".format(config["algorithm"],config["env"]), name="{}_{}".format(now, config["run_name"]), config=config):
        for n_epi in range(config["n_episodes"]):
            if config["algorithm"] == "dqn":
                score, loss, eps = train_episode(n_epi, env, memory, agent, config, device)
            elif config["algorithm"] == "sac":
                score, loss, loginfo = train_episode(n_epi, env, memory, agent, config, device)
            else:
                raise NotImplementedError
            score_history.append(score)

            if n_epi > 10:
                    average10 = np.mean(score_history[-10:])

            if n_epi > 100:
                if  average10 > best_score:
                    best_score = average10
                    agent.save_ckpt(os.path.join(model_save_dir, "best_score.ckpt"))
                    wandb.save(os.path.join(model_save_dir, "best_score.ckpt"))
                    
            if n_epi%config["log_every"]==0 and n_epi > 0:
                if config["algorithm"] == "dqn":
                    print("# of episode :{}, score1: {:.1f}, score10 : {:.1f}, buffer_size: {}, loss: {}.".format(n_epi, score, average10, memory.size(), loss))
                    wandb.log({"Score_1": score,
                            "Score_10": average10,
                            "Loss":loss,
                            "Episode": n_epi ,
                            "Eps": eps,
                            "Buffer size": memory.size()})

                elif  config["algorithm"] == "sac":
                    print("# of episode :{}, score1: {:.1f}, score10 : {:.1f}, buffer_size: {}, critic1_loss: {:4f}, critic2_loss: {:4f}, actor_loss: {:4f}, alpha_loss: {:4f}, alpha: {:4f}, entropy: {:4f}".format(n_epi, score, average10, memory.size(), loss["critic1_loss"], loss["critic2_loss"], loss["actor_loss"], loss["alpha_loss"], loginfo["alpha"], loginfo["entropy"]))

                    wandb.log({"Score_1": score,
                            "Score_10": average10,
                            "critic1_loss":loss["critic1_loss"],
                            "critic2_loss":loss["critic2_loss"],
                            "actor_loss":loss["actor_loss"],
                            "alpha_loss":loss["alpha_loss"],
                            "alpha":loginfo["alpha"],
                            "entropy":loginfo["entropy"],
                            "Episode": n_epi ,
                            "Buffer size": memory.size()})
                
                else:
                    NotImplementedError
            
            if n_epi%config["save_every"]==0:
                agent.save_ckpt(os.path.join(model_save_dir, f"{n_epi}.ckpt"))
                wandb.save(os.path.join(model_save_dir, f"{n_epi}.ckpt"))

if __name__ == "__main__":
    config = get_config()
    main(config)
