import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

class fnn0(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(16, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 4)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Network export')
    parser.add_argument('network_name', type=str, help='class name of network')
    parser.add_argument('save_path', type=str, help='network save path')
    args = parser.parse_args()

    torch.save(eval(args.network_name), args.save_path)
