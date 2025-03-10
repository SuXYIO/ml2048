"""
define neural agent
"""

import torch
from Game2048Env.agent import Agent


class AgentNN(Agent):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.model.eval()

    def strat(self, observation, reward, terminated, truncated, info):
        state = torch.tensor(observation.flatten(), dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action = self.model(state).max(1).indices.view(1, 1)
        return action.item()
