'''
neural network definitions, and template export
-h for help
'''
import argparse
import torch
from torch import nn
import torch.nn.functional as F

class Fnn0(nn.Module):
    '''feed-forward neural network 0'''
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(16, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 4)

    def forward(self, x):
        '''forward run the network'''
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Fnn1(nn.Module):
    '''feed-forward neural network 1'''
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(16, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 48)
        self.fc4 = nn.Linear(48, 4)

    def forward(self, x):
        '''forward run the network'''
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Network export')
    parser.add_argument('network_name', type=str, help='class name of network')
    parser.add_argument('save_path', type=str, help='network save path')
    args = parser.parse_args()

    cmp_name = args.network_name.lower()
    if args.network_name == 'fnn0':
        SaveCls = Fnn0
    elif args.network_name == 'fnn1':
        SaveCls = Fnn1
    else:
        raise ValueError(f"Network '{args.network_name}' is not defined.")

    torch.save(SaveCls, args.save_path)
