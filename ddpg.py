import torch.distributions
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from nn import DDPGCritic, DDPGActor
from tensorboardX import SummaryWriter
from utils import soft_update, set_seed, OUActionNoise
import os

class DDPG:
    def __init__(self, cfg):
        self.cfg = cfg
        set_seed(self.cfg.seed)

        self.act_space = 1
        self.discount = 0.99
        self.mini_batch_size = 128
        self.batch_size = self.cfg.num_envs * self.mini_batch_size
        self.tau = 0.995
        self.num_eval_freq = 100
        self.lr = 3e-4
        self.reward = 0
        self.tensorboard_dir = "runs/ddpg"
        self.writer = SummaryWriter(self.tensorboard_dir)
        os.makedirs(self.tensorboard_dir, exist_ok=True)

        # define actor and critic networks
        self.actor = DDPGActor(num_act=self.act_space).to(self.cfg.sim_device)
        self.critic = DDPGCritic(num_act=self.act_space).to(self.cfg.sim_device)
        
        # use same weight at initialization
        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)
        
        self.noise = OUActionNoise(mu=np.zeros(1))
        # Only optimize actor and critc networks directly, use soft update for target networks
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr)  

        self.actor_loss = 0
        self.critic_loss = 0
        

    def update(self, replay_buffer):
        # Sample a batch of experience from the replay buffer
        with torch.no_grad():
            obs, act, reward, next_obs, done_mask = replay_buffer.sample(self.mini_batch_size)
            next_action = self.actor_target.forward(next_obs)
            q_val_next = self.critic_target.forward(next_obs, next_action)
            reward = reward.unsqueeze(1)
            done_mask = done_mask.unsqueeze(1)
            target = reward + self.discount * q_val_next * done_mask

        q_val = self.critic.forward(obs, act)
        self.critic_loss = F.mse_loss(q_val, target)
        
        self.critic_optimizer.zero_grad()                  
        # critic_loss.backward(retain_graph=True)
        self.critic_loss.backward()
        self.critic_optimizer.step()
        
        # Actor update (maximize Q-value)
        mu = self.actor.forward(obs)
        self.actor_loss = -self.critic.forward(obs, mu).mean()  # negative because we want to maximize Q
        self.actor_optimizer.zero_grad()
        self.actor_loss.backward()
        self.actor_optimizer.step()

        # Update the target networks (using soft update)
        soft_update(self.critic, self.critic_target, self.tau)
        soft_update(self.actor, self.actor_target, self.tau)

        self.reward += torch.mean(reward.float()).item() / self.num_eval_freq
        

    def get_action(self, obs, epsilon=0.0):
        with torch.no_grad():
            mu = self.actor.forward(obs).to(self.cfg.sim_device)
            # add action noise
            mu_prime = mu + torch.tensor(self.noise(),
                                    dtype=torch.float).to(self.cfg.sim_device)
            return mu_prime
        

    def log_training_info(self, replay_buffer, step, epsilon):
        if replay_buffer.size() < self.mini_batch_size:
            print(f'Steps: {step:04d} | Training is NOT started. | Replay Buffer Size: {replay_buffer.size()}')
        else:
            self.update(replay_buffer)
            if (step+1) % self.num_eval_freq == 0:
                print(f'Steps: {step+1:04d} | Reward {self.reward:.04f} | TD Loss {self.critic_loss:.04f} | ', end="")
                print(f'Replay Buffer size: {replay_buffer.size():03d}')
                self.reward = 0
        # plot training curves for tensorboard
        self.writer.add_scalar('Reward/RLTrainSteps', self.reward, step+1)
  