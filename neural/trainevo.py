"""
use evolutionary algorithm to train
-h for help
"""

import argparse
import torch
import evotorch
import Game2048Env
from evalmodel import evaluate_model
import matplotlib.pyplot as plt
from network import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="train network using evo for 2048 game"
    )
    parser.add_argument("template_path", type=str, help="path to network template")
    parser.add_argument("save_path", type=str, help="path to save the trained network")
    parser.add_argument("episodes", type=int, help="episodes to train the network")
    parser.add_argument(
        "--popsize", type=int, default=64, help="population size of the algorithm"
    )
    parser.add_argument(
        "--radius_init",
        type=float,
        default=2.25,
        help="initial radius of the algorithm",
    )
    parser.add_argument(
        "--center-learning-rate",
        type=float,
        default=0.2,
        help="learning rate of the center of the algorithm",
    )
    parser.add_argument(
        "--stdev-learning-rate",
        type=float,
        default=0.1,
        help="learning rate of the stdev of the algorithm",
    )
    parser.add_argument(
        "-i",
        "--interval",
        type=int,
        default=16,
        help="interval of logging, no log if 0",
    )
    args = parser.parse_args()

    problem = evotorch.neuroevolution.NEProblem(
        "max",
        network=torch.load(args.template_path),
        network_eval_func=evaluate_model,
        num_actors="max",
        num_gpus_per_actor="max",
    )

    searcher = evotorch.algorithms.PGPE(
        problem,
        popsize=args.popsize,
        radius_init=args.radius_init,
        center_learning_rate=args.center_learning_rate,
        stdev_learning_rate=args.stdev_learning_rate,
    )

    if args.interval != 0:
        evotorch.logging.StdOutLogger(searcher, interval=args.interval)
    logger = evotorch.logging.PandasLogger(searcher)

    searcher.run(args.episodes)
    logger.to_dataframe().mean_eval.plot()
    plt.title(f"{args.save_path}, {args.episodes} episodes")
    plt.xlabel("episode")
    plt.ylabel("reward")
    plt.legend()
    plt.grid()
    plt.show()

    trained_network = problem.parameterize_net(searcher.status["center"])
    torch.save(trained_network, args.save_path)
