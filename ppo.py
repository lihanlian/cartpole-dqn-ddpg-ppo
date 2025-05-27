import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
from nn import PPOPolicyNetwork, PPOValueNetwork
from tensorboardX import SummaryWriter
from utils import set_seed
from torch.distributions import MultivariateNormal
import os

class PPO:
    def __init__(self, cfg):
        self.cfg = cfg
        set_seed(self.cfg.seed)

        self.act_space = 1
        self.obs_space = 4
        self.epoch = 5
        self.lr = 3e-4
        self.gamma = 0.99
        self.lmbda = 0.95
        self.clip = 0.3
        self.rollout_size = 128
        self.chunk_size = 32
        self.mini_chunk_size = self.rollout_size // self.chunk_size
        self.mini_batch_size = self.cfg.num_envs * self.mini_chunk_size
        self.num_eval_freq = 100
        self.tensorboard_dir = "runs/ppo"
        self.writer = SummaryWriter(self.tensorboard_dir)
        os.makedirs(self.tensorboard_dir, exist_ok=True)

        self.data = []
        self.reward = 0
        self.run_step = 0
        self.optim_step = 0

        self.policy_network = PPOPolicyNetwork(self.obs_space, self.act_space).to(self.cfg.sim_device)
        self.value_network = PPOValueNetwork(self.obs_space).to(self.cfg.sim_device)
        self.action_var = torch.full((self.act_space,), 0.1).to(self.cfg.sim_device)
        self.policy_optim = torch.optim.Adam(self.policy_network.parameters(), lr=self.lr)
        self.value_optim = torch.optim.Adam(self.value_network.parameters(), lr=self.lr)
        

    def make_data(self):
        # organise data and make batch
        data = []
        for _ in range(self.chunk_size):
            obs_lst, a_lst, r_lst, next_obs_lst, log_prob_lst, done_lst = [], [], [], [], [], []
            for _ in range(self.mini_chunk_size):
                rollout = self.data.pop(0)
                obs, action, reward, next_obs, log_prob, done = rollout

                obs_lst.append(obs)
                a_lst.append(action)
                r_lst.append(reward.unsqueeze(-1))
                next_obs_lst.append(next_obs)
                log_prob_lst.append(log_prob)
                done_lst.append(done.unsqueeze(-1))

            obs, action, reward, next_obs, done = \
                torch.stack(obs_lst), torch.stack(a_lst), torch.stack(r_lst), torch.stack(next_obs_lst), torch.stack(done_lst)

            # compute reward-to-go (target)
            with torch.no_grad():
                target = reward + self.gamma * self.value_network(next_obs) * done
                delta = target - self.value_network(obs)

            # compute advantage
            advantage_lst = []
            advantage = 0.0
            for delta_t in reversed(delta):
                advantage = self.gamma * self.lmbda * advantage + delta_t
                advantage_lst.insert(0, advantage)

            advantage = torch.stack(advantage_lst)
            log_prob = torch.stack(log_prob_lst)

            mini_batch = (obs, action, log_prob, target, advantage)
            data.append(mini_batch)
        return data

    def update(self):
        # update actor and critic network
        data = self.make_data()

        for i in range(self.epoch):
            for mini_batch in data:
                obs, action, old_log_prob, target, advantage = mini_batch

                mu = self.policy_network(obs)
                cov_mat = torch.diag(self.action_var)
                dist = MultivariateNormal(mu, cov_mat)
                log_prob = dist.log_prob(action)

                ratio = torch.exp(log_prob - old_log_prob).unsqueeze(-1)

                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * advantage

                loss_policy = -torch.min(surr1, surr2)
                loss_value = F.mse_loss(self.value_network(obs), target)

                self.policy_network.zero_grad(); self.value_network.zero_grad()
                loss_policy.mean().backward(); loss_value.mean().backward()
                nn.utils.clip_grad_norm_(self.policy_network.parameters(), 1.0)
                nn.utils.clip_grad_norm_(self.value_network.parameters(), 1.0)
                self.policy_optim.step(); self.value_optim.step()

                self.optim_step += 1


    def get_action(self, obs):
        with torch.no_grad():
            mu = self.policy_network(obs)
            cov_mat = torch.diag(self.action_var)
            dist = MultivariateNormal(mu, cov_mat)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            action = action.clip(-1, 1)
        return action, log_prob

    def log_training_info(self, step):

        if (step+1) % self.num_eval_freq == 0:
            print(f'Steps: {step+1:04d} | Reward {self.reward:.04f} | Action Var {self.action_var[0].item():.4f}')
            self.reward = 0
        # plot training curves for tensorboard
        self.writer.add_scalar('Reward/RLTrainSteps', self.reward, step+1)
  