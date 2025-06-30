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
                # episodes of simulations for each action
                "act_episodes": 2,
                # steps to simulate
                "simu_steps": 1,
            }
        else:
            self.stratagy = stratagy

    def strat(self, observation, reward, terminated, truncated, info) -> int:
        reward_list = []
        for action in range(self.action_space.n):
            vobservation, vreward, vterminated, vtruncated, vinfo = self.simu(
                observation, action
            )
            reward_list.append(self.act_tree_recur(1, vobservation))
        dec_action = reward_list.index(max(reward_list))
        return dec_action

    def act_tree_recur(self, recur_level, observation):
        if recur_level >= self.stratagy["simu_steps"]:
            return self.deci_sum(observation)
        else:
            reward_list = []
            for action in range(self.action_space.n):
                vobservation, vreward, vterminated, vtruncated, vinfo = self.simu(
                    observation, action
                )
                reward_list.append(self.act_tree_recur(recur_level + 1, vobservation))
            return sum(reward_list)

    def deci_sum(self, observation):
        return sum(self.decision(observation).values())

    def decision(self, observation):
        # simulation result dict {virtual_action: virtual_reward}
        vactions: dict[int, float] = {}
        for vaction in range(self.action_space.n):
            r = self.simu_reward(
                observation, vaction, episodes=self.stratagy["act_episodes"]
            )
            vactions[vaction] = vactions.get(vaction, 0) + r
        return vactions

    def simu_reward(self, observation, action, episodes=2):
        total_reward = 0
        for _ in range(episodes):
            vobservation, vreward, vterminated, vtruncated, vinfo = self.simu(
                observation, action
            )
            r = float(vreward) * (0.1 if vterminated or vtruncated else 1)
            total_reward += r
        return total_reward

    def simu(self, observation, action):
        venv = gym.make("Game2048Env/Game2048-v0")
        venv.reset(options={"board": observation})
        vobservation, vreward, vterminated, vtruncated, vinfo = venv.step(action)
        venv.close()
        return vobservation, vreward, vterminated, vtruncated, vinfo
