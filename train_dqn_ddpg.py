import hydra
from omegaconf import DictConfig, OmegaConf
from cartpole import Cartpole
from utils import ReplayBuffer
from dqn import DQN
from ddpg import DDPG

algorithm = 'ddpg' # dqn or ddpg

@hydra.main(version_base="1.1", config_name="config", config_path="./cfg")
def main(cfg: DictConfig):
    
    env = Cartpole(cfg)
    replay_buffer = ReplayBuffer(num_envs=cfg.num_envs)
    
    if algorithm == 'dqn':
        agent = DQN(cfg)
    else:
        agent = DDPG(cfg)

    # print out used config file for training
    with open(f"./runs/{algorithm}.yaml", "w") as f:
        f.write(OmegaConf.to_yaml(cfg))

    for step in range(cfg.total_steps):
        # Used or DQN action exploration, unused for DDPG
        epsilon = max(0.01, 0.8 - 0.01 * (step / 20))
        
        obs = env.obs_buf.clone()
        action = agent.get_action(obs, epsilon)
        env.step(action)       
        next_obs, reward, done = env.obs_buf.clone(), env.reward_buf.clone(), env.reset_buf.clone()
        env.reset()
        replay_buffer.push(obs, action, reward, next_obs, 1 - done)

        # Print out training information and plot tensorboard training curve
        agent.log_training_info(replay_buffer, step, epsilon)
        

if __name__ == '__main__':
    main()
