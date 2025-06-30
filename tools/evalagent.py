import argparse
import Game2048Env
import gymnasium as gym
from agents.greedy import *


def evaluate_agent(agent, num_episodes=4):
    total_rewards = []
    env = gym.make("Game2048Env/Game2048-v0")
    agent = agent

    for _ in range(num_episodes):
        observation, _ = env.reset()
        reward, terminated, truncated, info = 0, False, False, None
        total_reward = 0
        while True:
            action = agent.strat(observation, reward, terminated, truncated, info)
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            if terminated or truncated:
                break
        total_rewards.append(total_reward)
    avg_reward = sum(total_rewards) / num_episodes
    return avg_reward


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="evaluate agent")
    parser.add_argument("agent", type=str, help="agent class name")
    parser.add_argument(
        "-e", "--episodes", type=int, default=32, help="number of episodes to eval"
    )
    args = parser.parse_args()

    agent = eval(args.agent)()
    average_reward = evaluate_agent(agent, num_episodes=args.episodes)
    print(f"Average Reward over {args.episodes} episodes: {average_reward}")
