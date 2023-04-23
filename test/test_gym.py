import pytest
import numpy as np
from gym_2048.env import *
import gymnasium as gym
import sys
import pygame

class TestGym():

    @pytest.fixture(autouse=True)
    def init(self):
        self.env = gym.make("2048-v1")
        self.env = gym.wrappers.TimeLimit(self.env, max_episode_steps=10)
    
    def test_init(self):
        assert self.env.observation_space.shape[0] == 16
        assert self.env.action_space.n == 4

    def test_seed(self):
        obs, _ = self.env.reset(seed=0)
        obs0, r, done, trunc, info = self.env.step(0)

        obs, _ = self.env.reset(seed=0)
        obs1, r, done, trunc, info = self.env.step(0)

        assert (obs0 == obs1).all()

    def test_trunc(self):
        obs, _ = self.env.reset(seed=0)
        for i in range(10):
            action = np.random.choice(range(4), 1).item()
            obs, r, done, trunc, info = self.env.step(action)
            assert obs.shape[0] == 16
        assert trunc == True
    
    def test_render(self):
        obs, _ = self.env.reset(seed=0) 
        trunc = False
        done = False
        while not (done or trunc):
            action = np.random.choice(range(4), 1).item()
            obs, r, done, trunc, info = self.env.step(action)
            # print() inside test code will be printed when the test failed
            print(r)
            self.env.render()
        assert (obs == self.env.board.flatten()).all()


    def test_render_rgb(self):
        self.env = gym.make("2048-v1", render_mode="rgb_array")
        self.env = gym.wrappers.TimeLimit(self.env, max_episode_steps=10)
        obs, _ = self.env.reset(seed=0) 
        trunc = False
        done = False
        i = 0
        while not (done or trunc):
            action = np.random.choice(range(4), 1).item()
            obs, r, done, trunc, info = self.env.step(action)
            self.env.render()
            pygame.image.save(self.env.screen , f"screenshot{i}.jpg")
            i += 1
        
            
