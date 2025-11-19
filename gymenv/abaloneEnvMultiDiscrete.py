import copy
import pygame
import torch
import random
import numpy as np
import helper as h
import gymnasium as gym
from typing import Tuple
from gymnasium import spaces
import gymenv.moveGenerator as mg
from gymenv.abaloneEnv import evaluate_grid, Abalone
from stable_baselines3.ppo.policies import MlpPolicy, CnnPolicy
from stable_baselines3.common.distributions import CategoricalDistribution
from torch.distributions import Categorical

class AbaloneEnvMultiDiscrete(gym.Env):
    def __init__(self, env: Abalone, agent_player, gamma=0.99, one_hot_encode=True, enable_swapping=False):
        super().__init__()
        self.env: Abalone = env
        self.env.parent = self
        self.gamma = gamma
        self.enable_swapping = enable_swapping
        self.agent_player = h.Cells.PlayerTwo if enable_swapping else agent_player
        self.enable_rotation = agent_player == h.Cells.PlayerTwo
        self.one_hot_encode = one_hot_encode

        # Start cell (int), selection_direction * magnitude, move_direction
        self.action_space = gym.spaces.MultiDiscrete([pow(env.grid_size, 2), 6 * 3, 6])

        if self.one_hot_encode:
            self.observation_space = spaces.Box(
                low=0, high=1, shape=(4, self.env.grid_size, self.env.grid_size), dtype=np.int8
            )
        else:
            # Keep original observation space
            self.observation_space = env.observation_space
 
    # Going to use concept called Potential Based Reward Shaping
    def step(self, action, render_opp_move=False, print_info=False):
        # h.print_hex_grid(self.env.state)
        current_eval = evaluate_grid(self.env.state, self.env.agent_player)
        
        next_state, reward, terminated, truncated, info = self.env.step(
            h.multi_discrete_to_action(action, grid_size=self.observation_space.shape[-1]), 
            render_opp_move=render_opp_move, print_info=print_info, 
            act_on_rotated_grid=self.enable_rotation
        )
        
        next_eval = evaluate_grid(next_state, self.env.agent_player)
        potential_reward = self.gamma * next_eval - current_eval
        potential_reward *= 5
        
        if self.one_hot_encode:
            if self.enable_rotation:
                return self.env.encoded_rotated_state, reward + potential_reward, terminated, truncated, info
            else:
                return self.env.encoded_state, reward + potential_reward, terminated, truncated, info
        else:
            if self.enable_rotation:
                return self.env.rotated_state, reward + potential_reward, terminated, truncated, info
            else:
                return self.env.state, reward + potential_reward, terminated, truncated, info

    def reset(self, seed=42, options=None, render_opp_move=False, print_info=False):
        if self.enable_swapping:
            self.swap_agent_player()
        
        _, info = self.env.reset(
            seed=seed, options=options, 
            render_opp_move=render_opp_move, print_info=print_info
        )
        
        if self.one_hot_encode:
            if self.enable_rotation:
                return self.env.encoded_rotated_state, info
            else:
                return self.env.encoded_state, info
        else:
            if self.enable_rotation:
                return self.env.rotated_state, info
            else:
                return self.env.state, info
    
    def render(self, print_info=False):
        return self.env.render(print_info=print_info)

    def generate_valid_moves(self):
        return self.env.generate_valid_moves()
    
    def swap_agent_player(self):
        self.agent_player = h.Cells.PlayerOne if self.agent_player == h.Cells.PlayerTwo else h.Cells.PlayerTwo
        self.enable_rotation = self.agent_player == h.Cells.PlayerTwo
        
        self.env.agent_player = self.agent_player
        self.env.enemy_player = h.Cells.PlayerOne if self.agent_player == h.Cells.PlayerTwo else h.Cells.PlayerTwo
        