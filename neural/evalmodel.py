"""
evaluate trained model
-h for help
"""

import argparse
import torch
import gymnasium as gym
import Game2048Env
from tools import evalagent
from agentnn import AgentNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate_model(model, num_episodes=4):
    """
    evaluate average reward for a model
    """
    agent = AgentNN(model)

    avg_reward = evalagent.evaluate_agent(agent, num_episodes)
    return avg_reward


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="evaluate trained model")
    parser.add_argument("path", type=str, help="path to trained model")
    parser.add_argument(
        "-e", "--episodes", type=int, default=64, help="number of episodes to eval"
    )
    args = parser.parse_args()

    policy_net = torch.load(args.path)
    policy_net.eval()

    average_reward = evaluate_model(policy_net, num_episodes=args.episodes)
    print(f"Average Reward over {args.episodes} episodes: {average_reward}")
