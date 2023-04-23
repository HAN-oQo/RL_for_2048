import numpy as np
from gym_2048.env import *
import gymnasium as gym
import sys
import pygame

env = gym.make("2048-v1", render_mode="rgb_array")
env = gym.wrappers.TimeLimit(env, max_episode_steps=10)
obs, _ = env.reset(seed=0) 
trunc = False
done = False
i = 0
while not (done or trunc):
    action = np.random.choice(range(4), 1).item()
    obs, r, done, trunc, info = env.step(action)
    env.render()
    pygame.image.save(env.screen , f"screenshot{i}.jpg")
    i += 1