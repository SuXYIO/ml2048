import argparse
import torch
import evotorch
import gymnasium
import Game2048Env
from evalmodel import evaluate_model
import matplotlib.pyplot as plt
from network import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train network using evo for 2048 game')
    parser.add_argument('template_path', type=str, help='path to network template')
    parser.add_argument('save_path', type=str, help='path to save the trained network')
    parser.add_argument('episodes', type=int, help='episodes to train the network')
    args = parser.parse_args()

    problem = evotorch.neuroevolution.NEProblem(
        "max",
        network=torch.load(args.template_path),
        network_eval_func=evaluate_model
    )

    searcher = evotorch.algorithms.PGPE(
        problem,
        popsize=64,
        radius_init=2.25,
        center_learning_rate=0.2,
        stdev_learning_rate=0.1
    )

    interv = 16
    evotorch.logging.StdOutLogger(searcher, interval=interv)
    logger = evotorch.logging.PandasLogger(searcher)

    searcher.run(args.episodes)
    logger.to_dataframe().mean_eval.plot()
    plt.title(f'{args.save_path}, {args.episodes} episodes')
    plt.xlabel('episode')
    plt.ylabel('reward')
    plt.show()

    trained_network = problem.parameterize_net(searcher.status["center"])
    torch.save(trained_network, args.save_path)
