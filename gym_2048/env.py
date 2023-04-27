'''
https://github.com/activatedgeek/gym-2048/blob/master/gym_2048/env.py
TODO:
1. convert it into gymnasium  code
2. implement redering code according to https://github.com/rajitbanerjee/2048-pygame
'''
import numpy as np
import pygame
import json
import sys
import time
from copy import deepcopy
from pygame.locals import *
import os

import gymnasium as gym
import gymnasium.spaces as spaces
from gymnasium.utils import seeding


class Base2048Env(gym.Env):
  metadata = {
      'render_modes': ['human', "rgb_array"],
      'render_fps': 4
  }

  ##
  # NOTE: Don't modify these numbers as
  # they define the number of
  # anti-clockwise rotations before
  # applying the left action on a grid
  #
  LEFT = 0
  UP = 1
  RIGHT = 2
  DOWN = 3

  ACTION_STRING = {
      LEFT: 'left',
      UP: 'up',
      RIGHT: 'right',
      DOWN: 'down',
  }

  def __init__(self, render_mode=None, width=4, height=4):
    self.width = width
    self.height = height

    self.observation_space = spaces.Box(low=0,
                                        high=2**32,
                                        shape=(self.width*self.height,),
                                        dtype=np.int64)
    self.action_space = spaces.Discrete(4)

    # Internal Variables
    self.board = None
    self.np_random = None
    assert render_mode is None or render_mode in self.metadata["render_modes"]
    self.render_mode = render_mode
    if self.render_mode == "rgb_array":
      # set up pygame for main gameplay
      pygame.init()
      try:
        os.environ["DISPLAY"]
      except:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
      self.c = json.load(open("./gym_2048/constants.json", "r"))
      self.screen = pygame.display.set_mode(
          (self.c["size"], self.c["size"]))
      self.my_font = pygame.font.SysFont(self.c["font"], self.c["font_size"], bold=True)
      WHITE = (255, 255, 255)

  def _get_obs(self):
    return self.board.flatten()

  def reset(self, seed=None, options=None):
    """Place 2 tiles on empty board."""
    super().reset(seed=seed)
    self.board = np.zeros((self.width, self.height), dtype=np.int64)
    self._place_random_tiles(self.board, count=2)
    obs = self._get_obs()
    
    self.trunc_count = 0
    
    return obs, {"action_mask": [1, 1, 1, 1]}

  def step(self, action: int):
    """Rotate board aligned with left action"""

    # Align board action with left action
    prev_board = self.board.copy()

    rotated_obs = np.rot90(self.board, k=action)
    reward, updated_obs = self._slide_left_and_merge(rotated_obs)
    self.board = np.rot90(updated_obs, k=4 - action)
    
    # Place one random tile on empty location
    self._place_random_tiles(self.board, count=1)
    
    done, action_mask = self.is_done()
    obs = self._get_obs()

    if (prev_board == self.board).all() and prev_board.all():
      self.trunc_count += 1
    elif (prev_board == self.board).all() and prev_board.all():
      self.trunc_count = 0
    
    if self.trunc_count == 5:
      return obs, reward, done, True, {"action_mask": action_mask}
    
    return obs, reward, done, False, {"action_mask": action_mask}

  def is_done(self):
    copy_board = self.board.copy()

    action_mask = [1, 1, 1, 1]
    if not copy_board.all():
      return False, action_mask

    for action in [0, 1, 2, 3]:
      rotated_obs = np.rot90(copy_board, k=action)
      score, updated_obs = self._slide_left_and_merge(rotated_obs)
      # if not updated_obs.all():
      #     return False
      '''
      Why action mask?
      - Eventhough the game is not done, which means the agent still can move the board,
        agent with deterministic policy may take same action in wrong direction. 
      - To prevent such situation, I trying to use action mask.
      - Also, It can helps agent can explore better.
      '''
      if score == 0: 
        action_mask[action] = 0

    if sum(action_mask):
      return False, action_mask

    return True, action_mask

  def render(self):
    
    if self.render_mode == 'human':
      for row in self.board.tolist():
        print(' \t'.join(map(str, row)))
      print("=============================")
    elif self.render_mode == "rgb_array":
      
      self.display()
    else:
      raise NotImplmentedError

  def _sample_tiles(self, count=1):
    """Sample tile 2 or 4."""

    choices = [2, 4]
    probs = [0.9, 0.1]

    tiles = self.np_random.choice(choices,
                                  size=count,
                                  p=probs)
    return tiles.tolist()

  def _sample_tile_locations(self, board, count=1):
    """Sample grid locations with no tile."""

    zero_locs = np.argwhere(board == 0)
    zero_indices = self.np_random.choice(len(zero_locs), size=count)

    zero_pos = zero_locs[zero_indices]
    zero_pos = zero_pos.tolist()
    return zero_pos

  def _place_random_tiles(self, board, count=1):
    if not board.all():
      tiles = self._sample_tiles(count)
      tile_locs = self._sample_tile_locations(board, count)
      for tile,tile_loc in zip(tiles, tile_locs):
        
        board[tile_loc[0], tile_loc[1]] = tile
        

  def _slide_left_and_merge(self, board):
    """Slide tiles on a grid to the left and merge."""

    result = []

    score = 0
    for row in board:
      row = np.extract(row > 0, row)
      score_, result_row = self._try_merge(row)
      score += score_
      row = np.pad(np.array(result_row), (0, self.width - len(result_row)),
                   'constant', constant_values=(0,))
      result.append(row)

    return score, np.array(result, dtype=np.int64)

  @staticmethod
  def _try_merge(row):
    score = 0
    result_row = []

    i = 1
    while i < len(row):
      if row[i] == row[i - 1]:
        score += row[i] + row[i - 1]
        result_row.append(row[i] + row[i - 1])
        i += 2
      else:
        result_row.append(row[i - 1])
        i += 1

    if i == len(row):
      result_row.append(row[i - 1])

    return score, result_row

  def display(self, theme="light"):
    """
    Display the board 'matrix' on the game window.
    Parameters:
        board (list): game board
        theme (str): game interface theme
    """
    self.screen.fill(tuple(self.c["colour"][theme]["background"]))
    
    box = self.c["size"] // 4
    padding = self.c["padding"]
    for i in range(4):
        for j in range(4):
            colour = tuple(self.c["colour"][theme][str(self.board[i][j])])
            pygame.draw.rect(self.screen, colour, (j * box + padding,
                                              i * box + padding,
                                              box - 2 * padding,
                                              box - 2 * padding), 0)
            if self.board[i][j] != 0:
                if self.board[i][j] in (2, 4):
                    text_colour = tuple(self.c["colour"][theme]["dark"])
                else:
                    text_colour = tuple(self.c["colour"][theme]["light"])
                # display the number at the centre of the tile
                self.screen.blit(self.my_font.render("{:>4}".format(
                    self.board[i][j]), 1, text_colour),
                    # 2.5 and 7 were obtained by trial and error
                    (j * box + 2.5 * padding, i * box + 7 * padding))
    pygame.display.update()
