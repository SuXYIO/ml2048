"""
demonstrate agent
-h for help
"""

import argparse
import gymnasium as gym
from time import sleep
import Game2048Env
import pygame
from agents.greedy import *


def wait_for_key():
    while True:
        event = pygame.event.wait()
        if event.type == pygame.KEYDOWN:
            return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="demonstrate agent")
    parser.add_argument("agent", type=str, help="agent class name")
    parser.add_argument(
        "-r",
        "--render-mode",
        type=str,
        default="human",
        choices=["human", "ansi"],
        help="render mode for demo",
    )
    parser.add_argument(
        "-d",
        "--delay",
        type=float,
        default=0.5,
        help="delay between steps, ignored if -p is on",
    )
    parser.add_argument(
        "-p",
        "--pause",
        action="store_true",
        help="pause and wait for keypress after each step",
    )
    args = parser.parse_args()

    env = gym.make("Game2048Env/Game2048-v0", render_mode=args.render_mode)
    agent = eval(args.agent)()
    observation, _ = env.reset()
    reward, terminated, truncated, info = 0, False, False, None

    while True:
        action = agent.strat(observation, reward, terminated, truncated, info)
        observation, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
        if args.render_mode == "ansi":
            print(env.render())
        if args.render_mode == "human" and args.pause:
            wait_for_key()
        else:
            sleep(args.delay)
