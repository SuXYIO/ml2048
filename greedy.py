"""
A greedy policy
-h for help
"""

import argparse
import gymnasium as gym
import Game2048Env


def evaluate_greedy(num_episodes=4, stratagy={"vaction_episodes": 2}):
    total_rewards = []
    for _ in range(num_episodes):
        env = gym.make("Game2048Env/Game2048-v0")
        observation, _ = env.reset()
        total_reward = 0

        while True:
            # virtural simulation {virtual_action: virtual_reward}
            vactions = {}
            for _ in range(stratagy["vaction_episodes"]):
                for vaction in range(env.action_space.n):
                    venv = gym.make("Game2048Env/Game2048-v0")
                    venv.reset(options={"board": observation})
                    observation, reward, terminated, truncated, _ = venv.step(vaction)
                    r = reward * (1 if terminated or truncated else 10)
                    vactions[vaction] = vactions.get(vaction, 0) + r

            action = max(vactions, key=vactions.get)
            observation, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward

            if terminated or truncated:
                total_rewards.append(total_reward)
                break
    avg_reward = sum(total_rewards) / num_episodes
    return avg_reward


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="a greedy policy")
    parser.add_argument(
        "-e", "--episodes", type=int, default=64, help="number of episodes to eval"
    )
    parser.add_argument(
        "-ve",
        "--vaction_episodes",
        type=int,
        default=2,
        help="number of episodes to virturally simulate for each action",
    )
    args = parser.parse_args()
    strat = {"vaction_episodes": args.vaction_episodes}

    average_reward = evaluate_greedy(num_episodes=args.episodes, stratagy=strat)
    print(f"Average Reward over {args.episodes} episodes: {average_reward}")
