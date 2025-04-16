import hydra
from omegaconf import DictConfig, OmegaConf
from cartpole import Cartpole
from ppo import PPO
import torch


algorithm = 'ppo' 

@hydra.main(version_base="1.1", config_name="config", config_path="./cfg")
def main(cfg: DictConfig):
    
    env = Cartpole(cfg)
    agent = PPO(cfg)

    # print out used config file for training
    with open(f"./runs/{algorithm}.yaml", "w") as f:
        f.write(OmegaConf.to_yaml(cfg))

    for step in range(cfg.total_steps):
        
        obs = env.obs_buf.clone()
        action, log_prob = agent.get_action(obs)
        env.step(action)       
        next_obs, reward, done = env.obs_buf.clone(), env.reward_buf.clone(), env.reset_buf.clone()
        env.reset()

        agent.data.append((obs, action, reward, next_obs, log_prob, 1 - done))
        agent.reward += torch.mean(reward.float()).item() / agent.num_eval_freq

        agent.action_var = torch.max(0.01 * torch.ones_like(agent.action_var), agent.action_var - 0.00002)

        # training mode
        if len(agent.data) == agent.rollout_size:
            agent.update()

        # Print out training information and plot tensorboard training curve
        agent.log_training_info(step)
        

if __name__ == '__main__':
    main()
