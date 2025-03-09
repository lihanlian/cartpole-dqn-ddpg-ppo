import torch
import random
import numpy as np
import collections

def set_seed(seed=42):
    random.seed(seed)  # Python random seed
    np.random.seed(seed)  # NumPy random seed
    torch.manual_seed(seed)  # PyTorch seed
    torch.cuda.manual_seed_all(seed)  # Seed all CUDA devices (if using GPU)
    torch.backends.cudnn.deterministic = True  # Make CuDNN deterministic
    torch.backends.cudnn.benchmark = False  # Disable auto-tuning

def soft_update(net, net_target, tau):
    for param_target, param in zip(net_target.parameters(), net.parameters()):
        param_target.data.copy_(param_target.data * tau + param.data * (1.0 - tau))


class ReplayBuffer:
    def __init__(self, buffer_limit=int(1000), num_envs=1):
        self.buffer = collections.deque(maxlen=buffer_limit)
        self.num_envs = num_envs

    def push(self, obs, action, reward, next_obs, done):
        self.buffer.append(tuple([obs, action, reward, next_obs, done]))

    def sample(self, mini_batch_size):
        a = random.sample(self.buffer, mini_batch_size)
        obs, action, reward, next_obs, done = zip(*random.sample(self.buffer, mini_batch_size))

        rand_idx = torch.randperm(mini_batch_size * self.num_envs)  # random shuffle tensors

        obs = torch.cat(obs)[rand_idx]
        action = torch.cat(action)[rand_idx]
        reward = torch.cat(reward)[rand_idx]
        next_obs = torch.cat(next_obs)[rand_idx]
        done = torch.cat(done)[rand_idx]
        return obs, action, reward, next_obs, done

    def size(self):
        return len(self.buffer)


class OUActionNoise(object):
    def __init__(self, mu, sigma=0.1, theta=.5, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(
                                                            self.mu, self.sigma)