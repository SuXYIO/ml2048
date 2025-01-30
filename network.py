import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

class fnn0(nn.Module):
    def __init__(self, n_observations, n_actions):
        super().__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Network export')
    parser.add_argument('network_name', type=str, help='class name of network')
    parser.add_argument('save_path', type=str, help='network save path')
    args = parser.parse_args()

    torch.save(eval(args.network_name), args.save_path)
