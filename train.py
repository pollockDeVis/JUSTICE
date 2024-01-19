import pandas as pd
import numpy as np

from src.util.data_loader import DataLoader
from src.util.enumerations import *
from src.util.model_time import TimeHorizon
from src.model import JUSTICE
from tqdm.notebook import tqdm
from rl.marl.ray_wrapper import JusticeEnv, config
import os
import ray
import datetime
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from tqdm import tqdm
import wandb
def prepare_config():
    register_env("justice", env_creator )
    ppo_config = (  # 1. Configure the algorithm,
        PPOConfig()
        .environment("justice", env_config=config)
        .rollouts(num_rollout_workers=9)
        .framework("torch")
        .training(model={"fcnet_hiddens": [64, 64]})
        .evaluation(evaluation_num_workers=0)
    )
    return ppo_config

def env_creator(env_config):
    return JusticeEnv(env_config)


if __name__ == "__main__":


    check_point_dir_name = f"cp_{datetime.datetime.now().strftime('%m-%d-%Y_%H:%M:%S')}"
    check_point_dir_path = os.path.join(os.getcwd(), "rl", "marl", "checkpoints")

    wandb_api_key = "91f4b56e70eb59889967350b045b94cd0d7bcaa8"
    wandb.login(key=wandb_api_key)
    wandb.init(
                project="test",
                name='test_train',
                entity="justice-rl",
            )

    ray.init(local_mode=True)

    #load environment / config
    ppo_config = prepare_config()
    algo = ppo_config.build()  

    #train
    for _ in tqdm(range(30)):
        result = algo.train()
        wandb.log(
                {
                    "episode_reward_min": result["episode_reward_min"],
                    "episode_reward_mean": result["episode_reward_mean"],
                    "episode_reward_max": result["episode_reward_max"],
                },
                step=result["episodes_total"],
            )
        

    #save
    os.mkdir(check_point_dir_path + '/' + check_point_dir_name)
    checkpoint_path = os.path.join(check_point_dir_path,check_point_dir_name)
    algo.save(checkpoint_path)

    wandb.finish()
    