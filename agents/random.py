"""
define random agents
"""

import numpy as np
from Game2048Env.agent import Agent


class AgentGreedy(Agent):
    def __init__(self):
        super().__init__()

    def strat(self, observation, reward, terminated, truncated, info):
        action = np.random.randint(0, self.action_space.n)
        return int(action)
