
# Overview

This repository aims for provding minimal implementation of Deep Q-learning (DQN), Deep Deterministic Policy Graident (DDPG) and Proximal Policy Optimization (PPO) reinforcement learning algorithms **without** using any RL libraries (only PyTorch is used). All algorithms are tested in the cartpole environment of [Isaac Gym](https://github.com/isaac-sim/IsaacGymEnvs) simulator using **Ubuntu 20.04** operating system and **Python 3.8**. Implementation details and walk through of algorithms can be found at this [blog post](https://lihanlian.github.io/posts/blog6).

**Example (DQN and training curves)**
<p align="center">
  <img alt="dqn_result" src="/figs/dqn-result.gif" width="45%" />
  <img alt="tensorboard" src="/figs/tensorboard.png" width="45%" />
</p>

## Run Locally

Install [Isaac Gym](https://github.com/isaac-sim/IsaacGymEnvs) based on the instructions. Then clone the repository.

```bash
  git clone https://github.com/lihanlian/cartpole-dqn-ddpg-ppo.git
```

Go to project directory. To start training (dqn or ddpg agent):
```bash
  python train_dqn_ddpg.py
```
To view training result using tensorboard (--port=6006 is used for specifying port):
```bash
  tensorboard --logdir runs --port=6006
```

 - run _train_dqn_ddpg.py_ to start the main program for training agent using DQN or DDPG (both require relay buffer).
 - run _train_ppo.py_ to start the main program for training agent using PPO (no need for replay buffer). 
 - RL aglorithms are written in _dqn.py_, _ddpg.py_ and _ppo_.py.
 - Configuration files are stored in _cfg_ folder. Training information files are stored in _runs_ folder (TensorBoard events files and used yaml files for training).

## To  Do
 - Add _ppo.py_ and _train_ppo.py_
 
## References
 - [minimal-isaac-gym](https://github.com/lorenmt/minimal-isaac-gym) & [DDPG-Pytorch](https://github.com/XinJingHao/DDPG-Pytorch) [Github]
 - [How DDPG Algorithms works in reinforcement learning?](https://medium.com/@amaresh.dm/how-ddpg-deep-deterministic-policy-gradient-algorithms-works-in-reinforcement-learning-117e6a932e68) [Blog Post]
 - [Training a Deep Q-Network with Fixed Q-targets - Reinforcement Learning](https://www.youtube.com/watch?v=xVkPh9E9GfE&t=316s) [YouTube] (explanation of soft update)
 - [深度强化学习(4/5)：Actor-Critic Methods](https://www.youtube.com/watch?v=xjd7Jq9wPQY&list=PLvOO0btloRnsiqM72G4Uid0UWljikENlU&index=5) [YouTube] (in Chinese)

## License

[MIT](./LICENSE)