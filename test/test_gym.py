import pytest
from gym_2048.env import *
import gymnasium as gym
import sys

class TestGym():

    def init(self):
        env = gym.make("2048-v0")
        env = gym.wrappers.TimeLimit(env, max_episode_step=10)
        
        assert env.observation_space.shape == (4,4)
        assert env.action_space.n == 4


        observation, _ = env.reset(seed=0)
        # for i in range(10):
            
