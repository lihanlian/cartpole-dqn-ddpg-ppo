import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepQNet(nn.Module):
    def __init__(self, num_obs=4, num_act=2):
        super(DeepQNet, self).__init__()
        self.fc1 = nn.Linear(num_obs, 256)
        self.leaky_relu1 = nn.LeakyReLU()
        self.fc2 = nn.Linear(256, 256)
        self.leaky_relu2 = nn.LeakyReLU()
        self.fc3 = nn.Linear(256, num_act)

    def forward(self, x):
        x = self.fc1(x)
        x = self.leaky_relu1(x)
        x = self.fc2(x)
        x = self.leaky_relu2(x)
        x = self.fc3(x)
        return x
    
# define network architecture
class DDPGCritic(nn.Module):
    def __init__(self, num_obs=4, num_act=2):
        super(DDPGCritic, self).__init__()
        self.fc1 = nn.Linear(num_obs + num_act, 256)  # state + action as input
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)  # output single value for Q

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)  # concatenate state and action
        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = F.relu(x)

        x = self.fc3(x)  # final Q-value estimation
        return x
    
class DDPGActor(nn.Module):
    def __init__(self, num_obs=4, num_act=2):
        super(DDPGActor, self).__init__()
        
        self.fc1 = nn.Linear(num_obs, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, num_act)

        self.leaky_relu = nn.LeakyReLU()

    def forward(self, x):
        x = self.leaky_relu(self.fc1(x))
        x = self.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        x = torch.tanh(x)
        return x