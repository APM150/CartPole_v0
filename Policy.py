import torch
import torch.nn as nn

class Policy(nn.Module):

    def __init__(self, inputSize=4, outputSize=2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(inputSize, 128),
                                 nn.ReLU(),
                                 nn.Linear(128, 256),
                                 nn.ReLU(),
                                 nn.Linear(256, 512),
                                 nn.ReLU(),
                                 nn.Linear(512, 256),
                                 nn.ReLU(),
                                 nn.Linear(256, 128),
                                 nn.ReLU(),
                                 nn.Linear(128, outputSize))

    def forward(self, X):
        return self.net(X)