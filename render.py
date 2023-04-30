import numpy as np
from gym_2048.env import *
import gymnasium as gym
import sys
import pygame
from utils import *
import torch
from dqn import *
from termcolor import cprint

def render(config):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_ckpt = config["model_ckpt"]
    assert model_ckpt != ""
    model_name = model_ckpt.split(".")[1].split("/")[-2]

    env = gym.make("2048-v1", render_mode="rgb_array") 
    # env = gym.wrappers.TimeLimit(env, max_episode_steps=config["max_episode_steps"]) 
    # TODO: TimeLimit wrapper doesn't work for custom gym. Need to find way to make it work.

    n_observations = env.observation_space.shape[0]
    n_actions = env.action_space.n

    if config["algorithm"] == 'dqn':
        agent = DQN(obs_dim = n_observations,
                    action_dim = n_actions,
                    config = config).to(device)
        agent.load_ckpt(model_ckpt)
    else:
        raise NotImplmentedError
    
    animation_dir = f"./animation/{model_name}"
    os.makedirs(animation_dir, exist_ok=True)
    with torch.no_grad():
        for x in range(1):
            epi_dir = os.path.join(animation_dir, str(x))
            os.makedirs(epi_dir, exist_ok=True)
            obs, info = env.reset(seed=config["seed"]) 
            action_mask = info["action_mask"]
            trunc = False
            done = False
            i = 0
            score = 0
            while not (done or trunc):
                action = agent.get_action(torch.tensor(obs).float().unsqueeze(0).to(device), 
                                            eps=0., 
                                            action_mask = torch.tensor(action_mask).bool().to(device))
                obs, r, done, trunc, info = env.step(action)
                action_mask = info["action_mask"]
                score += r
                env.render()
                pygame.image.save(env.screen , os.path.join(epi_dir, f"screenshot_epi{x}_{i}.jpg"))
                i += 1
            cprint(f"Score of episode {x}: {score}", "green")
        make_gif(epi_dir, length=i)

if __name__ == "__main__":
    config = get_config()
    render(config)