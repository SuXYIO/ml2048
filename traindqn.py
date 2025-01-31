'''
use DQN to train
-h for help
'''
# Code modified from https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
import argparse
import gymnasium as gym
import Game2048Env
import math
import random
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
from torch import nn
from torch import optim

from network import *

env = gym.make('Game2048Env/Game2048-v0')

plt.ion()

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "cpu"
)

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

net = torch.load(args.template_path)
policy_net = net().to(device)
target_net = net().to(device)
target_net.load_state_dict(policy_net.state_dict())

steps_done = 0
episode_rewards = []

class ReplayMemory:
    '''DQN replay memory'''
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *transition_args):
        '''Save a transition'''
        self.memory.append(Transition(*transition_args))

    def sample(self, batch_size):
        '''Sample from memory'''
        return random.sample(self.memory, batch_size)

    def __len__(self):
        '''Memory length'''
        return len(self.memory)

def select_action(state):
    '''select action using epsilon greedy'''
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # Ensure state has the correct shape [batch_size, n_observations]
            return policy_net(state).max(1).indices.view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)

def plot_result():
    '''plot the training result'''
    plt.figure(figsize=(6, 6))

    # Reward
    rewards_t = torch.tensor(episode_rewards, dtype=torch.float)
    plt.title(f'{args.save_path}, {num_episodes} episodes')
    plt.xlabel('episode')
    plt.ylabel('reward')
    plt.plot(rewards_t.numpy(), label='Total Reward', color='blue')

    # Take 100 episode averages and plot them too
    if len(rewards_t) >= 100:
        means = rewards_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy(), label='100 Episode Average', color='red')

    plt.legend()
    plt.grid()
    plt.show()

def optimize_model():
    '''optimize the model'''
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1).values
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train network using dqn for 2048 game')
    parser.add_argument('template_path', type=str, help='path to network template')
    parser.add_argument('save_path', type=str, help='path to save the trained network')
    parser.add_argument('episodes', type=int, help='episodes to train the network')
    parser.add_argument('--batch-size', type=int, default=64, help='batch size')
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor')
    parser.add_argument('--eps-start', type=float, default=0.9, help='epsilon start value')
    parser.add_argument('--eps-end', type=float, default=0.05, help='epsilon end value')
    parser.add_argument('--eps-decay', type=float, default=0.005, help='epsilon decay rate')
    parser.add_argument('--tau', type=float, default=1e-4, help='target network update rate')
    parser.add_argument('--lr', type=float, default=10000, help='learning rate')
    parser.add_argument('--replay-memory-size', type=int, help='capacity of replay memory')
    args = parser.parse_args()

    # BATCH_SIZE is the number of transitions sampled from the replay buffer
    # GAMMA is the discount factor as mentioned in the previous section
    # EPS_START is the starting value of epsilon
    # EPS_END is the final value of epsilon
    # EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
    # TAU is the update rate of the target network
    # LR is the learning rate of the `AdamW` optimizer
    BATCH_SIZE = args.batch_size
    GAMMA = args.gamma
    EPS_START = args.eps_start
    EPS_END = args.eps_end
    EPS_DECAY = args.eps_decay
    TAU = args.tau
    LR = args.lr
    REPLAY_MEMORY_SIZE = args.replay_memory_size

    optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
    memory = ReplayMemory(REPLAY_MEMORY_SIZE)

    LOG_INTERV = 16

    num_episodes = args.episodes
    for i_episode in range(num_episodes):
        estate, info = env.reset()
        estate = torch.tensor(
            estate.flatten(),
            dtype=torch.float32,
            device=device
        ).unsqueeze(0)
        total_reward = 0
        for t in count():
            action = select_action(estate)
            observation, reward, terminated, truncated, _ = env.step(action.item())
            total_reward += reward
            reward = torch.tensor([reward], device=device)

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(
                    observation.flatten(),
                    dtype=torch.float32,
                    device=device
                ).unsqueeze(0)

            memory.push(estate, action, next_state, reward)
            state = next_state
            optimize_model()

            # Soft update of the target network's weights
            with torch.no_grad():
                for key in target_net.state_dict():
                    target_net.state_dict()[key] = \
                        target_net.state_dict()[key] \
                        * (1 - TAU) + policy_net.state_dict()[key] \
                        * TAU

            if terminated or truncated:
                episode_rewards.append(total_reward)
                if i_episode % LOG_INTERV == 0 and i_episode != 0:
                    print(f"Episode {i_episode + 1} finished, total reward: {total_reward}")
                break

    print('Complete')
    plot_result()
    plt.ioff()
    plt.show()

    # Save network
    torch.save(target_net, args.save_path)
