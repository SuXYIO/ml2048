"""
define greedy agents
"""

import gymnasium as gym
from Game2048Env.agent import Agent


class AgentGreedy(Agent):
    def __init__(self, stratagy=None):
        super().__init__()
        if stratagy is None:
            self.stratagy = {
                "vaction_episodes": 2,
                "vaction_steps": 2,
            }
        else:
            self.stratagy = stratagy

    def strat(self, observation, reward, terminated, truncated, info):
        # simulation result dict {virtual_action: virtual_reward}
        vactions = {}
        for _ in range(self.stratagy["vaction_episodes"]):
            for vaction in range(self.action_space.n):
                venv = gym.make("Game2048Env/Game2048-v0")
                venv.reset(options={"board": observation})
                vobservation, vreward, vterminated, vtruncated, vinfo = venv.step(
                    vaction
                )
                r = float(vreward) * (0.1 if vterminated or vtruncated else 1)
                vactions[vaction] = vactions.get(vaction, 0) + r
                venv.close()

        action = max(vactions, key=vactions.get)
        return action
