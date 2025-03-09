import torch.distributions
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter
import os
from utils import soft_update, set_seed
from nn import DeepQNet

class DQN:
    def __init__(self, cfg):
        self.cfg = cfg
        set_seed(self.cfg.seed)

        self.act_space = 3  # [-1, 0, 1]
        self.discount = 0.95
        self.mini_batch_size = 128
        self.batch_size = self.cfg.num_envs * self.mini_batch_size
        self.tau = 0.995
        self.num_eval_freq = 100
        self.lr = 3e-4
        self.score = 0
        self.tensorboard_dir = "runs/dqn"
        self.writer = SummaryWriter(self.tensorboard_dir)
        os.makedirs(self.tensorboard_dir, exist_ok=True)

        # define Q-network and target network
        self.q        = DeepQNet(num_act=self.act_space).to(self.cfg.sim_device)
        self.q_target = DeepQNet(num_act=self.act_space).to(self.cfg.sim_device)
        soft_update(self.q, self.q_target, tau=0.0)

        self.q_target.eval()
        self.optimizer = torch.optim.Adam(self.q.parameters(), lr=self.lr)
        self.loss = 0       

    def update(self, replay_buffer):
        # policy update using TD loss
        self.optimizer.zero_grad()

        obs, act, reward, next_obs, done_mask = replay_buffer.sample(self.mini_batch_size)
        q_table = self.q(obs)

        act_idx = act + 1 # maps back to the prediction space (0,1,2) for indexing
        q_val = q_table[torch.arange(self.batch_size), act_idx.long()]

        # Get optimal q value at next time step using target network
        with torch.no_grad():
            q_table_next = self.q_target(next_obs)
            idx_optimal = torch.argmax(q_table_next, dim=1)
            q_val_next = q_table_next[torch.arange(q_table_next.size(0)),idx_optimal]

        # If it is done, then no future reward 
        target = reward + self.discount * q_val_next * done_mask
        self.loss = F.smooth_l1_loss(q_val, target)

        self.loss.backward()
        self.optimizer.step()

        # soft update target networks
        soft_update(self.q, self.q_target, self.tau)
        self.score += torch.mean(reward.float()).item() / self.num_eval_freq
        

    def get_action(self, obs, epsilon=0.0):
        # get flag for explore or not (exploit)
        is_explore = torch.rand(self.cfg.num_envs, device=self.cfg.sim_device) < epsilon
        # generate random action frome action space [-1, 0, 1]
        random_action = torch.randint(0, 3, (self.cfg.num_envs,), device=self.cfg.sim_device) - 1

        with torch.no_grad():
            # predict q table and get the index for the optimal action (max Q-value)
            q_table = self.q(obs)
            q_max_idx = torch.argmax(q_table, dim=1)
            # convert to action space
            act_optimal = q_max_idx
            act_optimal[q_max_idx == 0] = -1
            act_optimal[q_max_idx == 1] = 0
            act_optimal[q_max_idx == 2] = 1

        act = is_explore.float() * random_action + (1 - is_explore.float()) * act_optimal
        return act
    
    def log_training_info(self, replay_buffer, step, epsilon):
        if replay_buffer.size() < self.mini_batch_size:
            print(f'Training is NOT started. Replay Buffer Size: {replay_buffer.size()}, Run Step: {step}')
        else:
            self.update(replay_buffer)
            if (step + 1) % self.num_eval_freq == 0:
                print(f'Steps: {step:04d} | Reward {self.score:.04f} | TD Loss {self.loss:.04f} | ', end="")
                print(f'Epsilon: {epsilon:.04f} | Replay Buffer size: {replay_buffer.size():03d}')
                self.score = 0
        # plot training curves for tensorboard
        self.writer.add_scalar('Reward/RLTrainSteps', self.score, step)
  