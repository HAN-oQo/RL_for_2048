import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from .networks import *

class SAC(nn.Module):
    def __init__(self, obs_dim, action_dim, config):
        super(SAC, self).__init__()
        self.action_dim = action_dim
        if config["architecture"] == "conv":
            raise NotImplementedError
        else:
            self.actor = ActorNet_FC(obs_dim = obs_dim,
                                    action_dim=action_dim, config=config)
            self.critic1 = CriticNet_FC(obs_dim=obs_dim,
                                    action_dim=1,
                                    config=config)
            self.critic2 = CriticNet_FC(obs_dim=obs_dim,
                                    action_dim=1,
                                    config=config)
            self.critic1_target = CriticNet_FC(obs_dim=obs_dim,
                                    action_dim=1,
                                    config=config)
            self.critic2_target = CriticNet_FC(obs_dim=obs_dim,
                                    action_dim=1,
                                    config=config)

        self.critic1_target.load_state_dict(critic1.state_dict())
        self.critic2_target.load_state_dict(critic2.state_dict())

        init_alpha = config["init_alpha"]
        self.log_alpha = torch.tensor(np.log(init_alpha))
        self.log_alpha.requires_grad = True

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config["learning_rate"]["actor"])
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=config["learning_rate"]["critic"])
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=config["learning_rate"]["critic"])
        self.log_alpha_optimizer = optim.Adam([self.log_alpha], lr=config["learning_rate"]["alpha"])

        self.tau = config["target_smoothing_coefficient"]
        self.gamma =config["discount"]

    def forward(self, s, a, s_prime):
        pass

    def train_net(self, mini_batch):
        s, a, r, s_prime, done = mini_batch

        pass
    
    def get_action(self):
        pass
    
    def calc_target(self, r, s_prime):
        with torch.no_grad():
            a_prime, log_prob= self.actor(s_prime)
            entropy = self.log_alpha.exp() * log_prob
            entropy = entropy.sum(dim=-1, keepdim = True)
            q1_val, q2_val = self.critic1_target(s_prime,a_prime), self.critic2_target(s_prime,a_prime)
            q1_q2 = torch.cat([q1_val, q2_val], dim=1)
            min_q = torch.min(q1_q2, 1, keepdim=True)[0]
            target = r + gamma * done * (min_q + entropy)
        return target

    def soft_update(self):
        for param_target, param in zip(self.critic1_target.parameters(), self.critic1.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)
        for param_target, param in zip(self.critic2_target.parameters(), self.critic2.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)
    
    def save_ckpt(self, ckpt_path):
        torch.save({"actor": self.actor.state_dict(),
                    "critic1": self.critic1.state_dict(),
                    "critic2": self.critic2.state_dict(),
                    "critic1_target": self.critic1_target.state_dict(),
                    "critic2_target": self.critic2_target.state_dict(),
                    "optim": self.optimizer.state_dict(),
                    }, ckpt_path)
    
    def load_ckpt(self, ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
        self.actor.load_state_dict(checkpoint["actor"])
        print("Model: {} Loaded!".format(ckpt_path))




    
