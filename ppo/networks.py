import torch
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np


'''
PolicyNet / PolicyNet_FC is identical theoritically.
Perfomace of two model is similar but PolicyNet was slightly better than PolicyNet_FC experimentally. 
Recommend to use PolicyNet_FC for simplicity.
Two different ways of implementation is just for my curiosity.
'''
class PolicyNet(nn.Module):
    def __init__(self, obs_dim, action_dim, config):
        super(PolicyNet, self).__init__()
        assert config["non_linearity"] in ["ReLU", "LeakyReLU"]
        assert config["hidden_layer"] >= 2

        prenet_layers = []
        prenet_layers.append(nn.Conv2d(in_channels = 1,
                              out_channels = config["hidden_units"],
                              kernel_size = 4))
        if config["non_linearity"] == "ReLU":
            prenet_layers.append(nn.ReLU(inplace=True))
        else:
            prenet_layers.append(nn.LeakyReLU(inplace=True))
        self.prenet = nn.Sequential(*prenet_layers)

        layers = []
        for i in range(config["hidden_layer"]-2):
            layers.append(nn.Linear(config["hidden_units"],config["hidden_units"]))
            if config["non_linearity"] == "ReLU":
                layers.append(nn.ReLU(inplace=True))
            elif config["non_linearity"] == "LeackyReLU":
                layers.append(nn.LeakyReLU(inplace=True))
        layers.append(nn.Linear(config["hidden_units"], action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, s):
        B, C = s.shape
        s = s.reshape(B, 1, int(np.sqrt(C)), -1)
        x = self.prenet(s)
        x = x.squeeze().reshape(B, -1)
        x = self.net(x)

        return x

class PolicyNet_FC(nn.Module):
    def __init__(self, obs_dim, action_dim, config):
        super(PolicyNet_FC, self).__init__()
        assert config["non_linearity"] in ["ReLU", "LeakyReLU"]
        assert config["hidden_layer"] >= 2

        layers = []
        layers.append(nn.Linear(obs_dim, config["hidden_units"]))
        if config["non_linearity"] == "ReLU":
            layers.append(nn.ReLU(inplace=True))
        else:
            layers.append(nn.LeakyReLU(inplace=True))
        for i in range(config["hidden_layer"]-2):
            layers.append(nn.Linear(config["hidden_units"],config["hidden_units"]))
            if config["non_linearity"] == "ReLU":
                layers.append(nn.ReLU(inplace=True))
            elif config["non_linearity"] == "LeackyReLU":
                layers.append(nn.LeakyReLU(inplace=True))
        layers.append(nn.Linear(config["hidden_units"], action_dim))
        self.fc = nn.Sequential(*layers)

    def forward(self, s):
        x = self.fc(s)

        return x

