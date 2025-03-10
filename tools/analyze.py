# WARN: this script is not complete
"""
analyze parameters for an agent
-h for help
"""

import argparse
import gymnasium as gym
import Game2048Env
import matplotlib.pyplot as plt
import pygame
from agents import *


def evaluate_greedy(agent_class, num_episodes=4, stratagy=None, demo_mode=False):
    agent = agent_class(stratagy)
    total_rewards = []
    render_mode = None
    if demo_mode:
        num_episodes = 1
        render_mode = "human"
    for _ in range(num_episodes):
        env = gym.make("Game2048Env/Game2048-v0", render_mode=render_mode)
        observation, _ = env.reset()
        reward, terminated, truncated, info = 0, False, False, None
        total_reward = 0

        while True:
            action = agent.strat(observation, reward, terminated, truncated, info)
            observation, reward, terminated, truncated, _ = env.step(action)
            total_reward += float(reward)

            if terminated or truncated:
                total_rewards.append(total_reward)
                break
            if render_mode == "human":
                wait_for_key()
    avg_reward = sum(total_rewards) / num_episodes
    return avg_reward


def wait_for_key():
    while True:
        event = pygame.event.wait()
        if event.type == pygame.KEYDOWN:
            return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="analyze parameters for an agent")
    parser.add_argument("agent", type=str, help="agent name in agents.py")
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
    parser.add_argument(
        "-s",
        "--statistic",
        type=int,
        default=None,
        nargs=3,
        metavar=("START", "END", "STEP"),
        help="enable statistic mode, which plots reward for different vaction_episodes values. Takes three numbers to range()",
    )
    parser.add_argument(
        "-d",
        "--demo-mode",
        action="store_true",
        help="enable demo mode",
    )
    args = parser.parse_args()

    agent = eval(args.agent)
    if args.demo_mode:
        strat = {"vaction_episodes": args.vaction_episodes}
        average_reward = evaluate_greedy(
            agent, num_episodes=args.episodes, stratagy=strat, demo_mode=True
        )
        print(f"Total reward: {average_reward}")
    else:
        if args.statistic == None:
            strat = {"vaction_episodes": args.vaction_episodes}
            average_reward = evaluate_greedy(
                agent, num_episodes=args.episodes, stratagy=strat
            )
            print(f"Average Reward over {args.episodes} episodes: {average_reward}")
        elif args.statistic != None:
            stat = {}
            for vaction_episodes in range(*args.statistic):
                strat = {"vaction_episodes": vaction_episodes}
                average_reward = evaluate_greedy(
                    agent, num_episodes=args.episodes, stratagy=strat
                )

                stat[vaction_episodes] = stat.get(vaction_episodes, 0) + average_reward
                print(
                    f"Average Reward over {args.episodes} episodes for vaction_episodes={vaction_episodes}: {average_reward}"
                )
            plt.plot(stat.keys(), stat.values())
            plt.xlabel("vaction_episodes")
            plt.ylabel("reward")
            plt.show()
