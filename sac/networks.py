import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

'''
SAC algorithm for discrete action space. 
'''
class ActorNet(nn.Module):
    def __init__(self, obs_dim, action_dim, config):
        super(ActorNet, self).__init__()
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
        layers.append(nn.Softmax(dim=-1))
        self.net = nn.Sequential(*layers)

    def forward(self, s):
        B, C = s.shape
        s = s.reshape(B, 1, int(np.sqrt(C)), -1)
        x = self.prenet(s)
        x = x.squeeze().reshape(B, -1)
        x = self.net(x)

        return x

class ActorNet_FC(nn.Module):
    def __init__(self, obs_dim, action_dim, config):
        super(ActorNet_FC, self).__init__()
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
        layers.append(nn.Softmax(dim=-1))
        self.fc = nn.Sequential(*layers)

    def forward(self, s, action_mask=None):
        # breakpoint()
        action_probs = self.fc(s)
        if action_mask is not None:
            action_probs = action_probs * action_mask
            if not action_mask.all():
                action_probs = (action_probs == 0.0).float() * 1e-8

        action_dist = Categorical(action_probs)
        actions = action_dist.sample().view(-1, 1)

        z = (action_probs == 0.0).float() * 1e-8
        log_action_probs = torch.log(action_probs + z)

        return actions, action_probs, log_action_probs
'''
only get observation as input, which can be the difference between implementations of discrete action space and continuous action space
'''
class CriticNet_FC(nn.Module):
    def __init__(self, obs_dim, action_dim, config):
        super(CriticNet_FC, self).__init__()
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
        q_value = self.fc(s)

        return q_value