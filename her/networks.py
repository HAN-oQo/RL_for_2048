import torch
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np



class PolicyNet(nn.Module):
    def __init__(self, obs_dim, action_dim, goal_dim, config):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(obs_dim+goal_dim, config["hidden_units"])
        self.fc2 = nn.Linear(config["hidden_units"], action_dim)

    def forward(self, s, g):
        x = torch.cat([s,g], dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


"""
Separated Policy network and Target network
"""
class HER(nn.Module):
    def __init__(self, obs_dim, action_dim, goal_dim, config):
        super(HER, self).__init__()
        self.action_dim = action_dim
        self.q_action = PolicyNet(obs_dim, action_dim, goal_dim, config)
        self.q_eval = PolicyNet(obs_dim, action_dim, goal_dim, config)
        self.q_eval.load_state_dict(self.q_action.state_dict())

        self.optimizer = optim.Adam(self.q_action.parameters(), lr=config["learning_rate"]["policy"])
        self.tau = config["target_smoothing_coefficient"]
        self.gamma =config["discount"]
    
    def forward(self, s, a, s_prime, goal):
        # breakpoint()
        a = a.type(torch.cuda.LongTensor)
        q_pred = self.q_action(s, goal).gather(dim=-1, index=a)
        q_next = self.q_eval(s_prime, goal).max(dim=-1, keepdim=True)[0]

        return q_pred, q_next

    def train_net(self,mini_batch):
        s, a, r, s_prime, done, goal = mini_batch

        self.optimizer.zero_grad()
        q_pred, q_next = self.forward(s,a,s_prime,goal)
        # breakpoint() # check dimensions

        q_target = r + done *self.gamma * q_next
        loss = F.mse_loss(q_pred, q_target).mean() 
        loss.backward()
        self.optimizer.step()
        self.soft_update()

        return loss.item()
    
    def get_action(self, s, g, eps):
        with torch.no_grad():
            if np.random.random() > eps:
                q_pred = self.q_action(s, g)
                a = torch.argmax(q_pred, dim=-1)
                a = a.detach().cpu().numpy()
            else:
                a = np.random.choice(self.action_dim)
            
        return a

    def soft_update(self):
        for param_target, param in zip(self.q_eval.parameters(), self.q_action.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)



