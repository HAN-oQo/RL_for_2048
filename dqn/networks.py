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

"""
Separated Policy network and Target network
"""
class DQN(nn.Module):
    def __init__(self, obs_dim, action_dim, config):
        super(DQN, self).__init__()
        self.action_dim = action_dim
        if config["architecture"] == "conv":
            self.q_action = PolicyNet(obs_dim, action_dim, config)
            self.q_eval = PolicyNet(obs_dim, action_dim, config)
        else:
            self.q_action = PolicyNet_FC(obs_dim, action_dim, config)
            self.q_eval = PolicyNet_FC(obs_dim, action_dim, config)
        self.q_eval.load_state_dict(self.q_action.state_dict())

        self.optimizer = optim.Adam(self.q_action.parameters(), lr=config["learning_rate"]["policy"])
        self.tau = config["target_smoothing_coefficient"]
        self.gamma =config["discount"]
    
    def forward(self, s, a, s_prime):
        # breakpoint()
        a = a.type(torch.cuda.LongTensor)
        q_pred = self.q_action(s).gather(dim=-1, index=a)
        q_next = self.q_eval(s_prime).max(dim=-1, keepdim=True)[0]
        # breakpoint()
        return q_pred, q_next

    def train_net(self,mini_batch):
        s, a, r, s_prime, done = mini_batch

        self.optimizer.zero_grad()
        q_pred, q_next = self.forward(s,a,s_prime)
        # breakpoint() # check dimensions

        q_target = r + done *self.gamma * q_next
        loss = F.mse_loss(q_pred, q_target).mean() 
        loss.backward()
        self.optimizer.step()
        self.soft_update()

        return loss.item()
    
    def get_action(self, s, eps):
        with torch.no_grad():
            if np.random.random() > eps:
                q_pred = self.q_action(s)
                a = torch.argmax(q_pred, dim=-1)
                a = a.detach().cpu().numpy()
            else:
                a = np.random.choice(self.action_dim)
            
        return a

    def soft_update(self):
        for param_target, param in zip(self.q_eval.parameters(), self.q_action.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)



