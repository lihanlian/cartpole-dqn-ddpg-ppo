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

class PPOPolicyNetwork(nn.Module):
    """
    Continuous action policy network.
    """
    def __init__(self, num_obs: int = 4, num_act: int = 1, hidden_size: int = 256):
        super(PPOPolicyNetwork, self).__init__()

        self.backbone = nn.Sequential(
            nn.Linear(num_obs, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
        )
        # Mean head
        self.mean_head = nn.Sequential(
            nn.Linear(hidden_size, num_act),
            nn.Tanh()             # action space is normalized to [-1,1]
        )

    def forward(self, x: torch.Tensor):
        """
        :param x:    Tensor of shape [batch_size, num_obs]
        :returns:    mean Tensor [batch_size, num_act],
                     std Tensor [num_act] (broadcastable over batch)
        """
        features = self.backbone(x)
        mean = self.mean_head(features)
        return mean

class PPOValueNetwork(nn.Module):
    """
    State‐value function approximator.
    Outputs a single scalar V(x) per input state.
    """
    def __init__(self, num_obs: int = 4, hidden_size: int = 256):
        super(PPOValueNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(num_obs, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x: torch.Tensor):
        """
        :param x:  Tensor of shape [batch_size, num_obs]
        :returns:  V(x) Tensor of shape [batch_size, 1]
        """
        return self.net(x)
       