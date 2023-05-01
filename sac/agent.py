import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from .networks import *
import numpy as np

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
                                    action_dim=action_dim,
                                    config=config)
            self.critic2 = CriticNet_FC(obs_dim=obs_dim,
                                    action_dim=action_dim,
                                    config=config)
            self.critic1_target = CriticNet_FC(obs_dim=obs_dim,
                                    action_dim=action_dim,
                                    config=config)
            self.critic2_target = CriticNet_FC(obs_dim=obs_dim,
                                    action_dim=action_dim,
                                    config=config)

        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        init_alpha = config["init_alpha"]
        self.log_alpha = torch.tensor(np.log(init_alpha))
        self.log_alpha.requires_grad = True

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config["learning_rate"]["actor"])
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=config["learning_rate"]["critic"])
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=config["learning_rate"]["critic"])
        self.log_alpha_optimizer = optim.Adam([self.log_alpha], lr=config["learning_rate"]["alpha"])

        self.tau = config["target_smoothing_coefficient"]
        self.gamma =config["discount"]
        self.target_entropy = -self.action_dim
        self.grad_clip_max_norm = config["grad_clip_max_norm"]

    def forward(self, s, a, target_net):
        if target_net == "critic":
            q1_val = self.critic1(s).gather(1, a.long())
            q2_val = self.critic2(s).gather(1, a.long())
            return q1_val, q2_val

        elif target_net == "actor":
            a, a_prob, log_prob = self.actor.forward(s)
            entropy = -self.log_alpha.exp() * a_prob * log_prob
            entropy = entropy.sum(dim=-1, keepdim = True)

            with torch.no_grad():
                q1_val, q2_val = self.critic1(s), self.critic2(s)
                
            min_q = (a_prob * (torch.min(q1_val, q2_val))).sum(dim=-1, keepdim=True)

            return a_prob, log_prob, entropy, min_q
        else:
            NotImplementedError

    def train_net(self, mini_batch):
        s, a, r, s_prime, done = mini_batch
        
        # train critics
        q1_val, q2_val = self.forward(s,a,target_net="critic")
        q_target = self.calc_target(r, s_prime, done)
        critic1_loss = F.smooth_l1_loss(q1_val, q_target).mean()
        critic2_loss = F.smooth_l1_loss(q2_val, q_target).mean()

        self.critic1_optimizer.zero_grad()
        self.critic2_optimizer.zero_grad()
        critic1_loss.backward()
        critic2_loss.backward()
        nn.utils.clip_grad_norm_(self.critic1.parameters(), self.grad_clip_max_norm)
        nn.utils.clip_grad_norm_(self.critic2.parameters(), self.grad_clip_max_norm)
        self.critic1_optimizer.step()
        self.critic2_optimizer.step()

        # train actor
        action_prob, log_prob, entropy, min_q = self.forward(s, a, target_net="actor")
        actor_loss = (-min_q - entropy).mean() # for gradient ascent

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip_max_norm)
        self.actor_optimizer.step()

        # anneal alpha
        alpha_loss = -(self.log_alpha.exp() * ((action_prob * log_prob).sum(dim=-1, keepdim=True) + self.target_entropy).detach()).mean()

        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        # soft update
        self.soft_update()
        
        return critic1_loss.item(), critic2_loss.item(), actor_loss.item(), alpha_loss.item(), self.log_alpha.exp(), entropy.mean()
    
    def get_action(self, s, action_mask, use_mask=True):
        if use_mask:
            with torch.no_grad():
                a, _ , _ = self.actor(s, action_mask)
                a = a.detach().cpu().numpy()
        else:
            with torch.no_grad():
                a, _, _ = self.actor(s)
                a = a.detach().cpu().numpy()

        return a
    
    def calc_target(self, r, s_prime, done):
        with torch.no_grad():
            a_prime, a_prime_prob, log_prob= self.actor(s_prime)
            entropy = -self.log_alpha.exp()* a_prime_prob * log_prob
            entropy = entropy.sum(dim=-1, keepdim=True)
            q1_val, q2_val = self.critic1_target(s_prime), self.critic2_target(s_prime)
            min_q = (a_prime_prob * (torch.min(q1_val, q2_val))).sum(dim=-1, keepdim=True)
            target = r + self.gamma * done * (min_q + entropy)
            
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
                    "log_alpha": self.log_alpha,
                    "critic1_optimizer": self.critic1_optimizer.state_dict(),
                    "critic2_optimizer": self.critic2_optimizer.state_dict(),
                    "actor_optimizer": self.actor_optimizer.state_dict(),
                    "log_alpha_optimizer": self.log_alpha_optimizer.state_dict(),
                    }, ckpt_path)
    
    def load_ckpt(self, ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
        self.actor.load_state_dict(checkpoint["actor"])
        print("Model: {} Loaded!".format(ckpt_path))




    
