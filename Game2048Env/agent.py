"""
define agent class
"""

import gymnasium as gym


class Agent:
    def __init__(self):
        self.action_space = gym.spaces.Discrete(4)

    def strat(self, observation, reward, terminated, truncated, info) -> int:
        return 0
