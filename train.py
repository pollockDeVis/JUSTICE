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

def prepare_config():
    register_env("justice", env_creator )
    ppo_config = (  # 1. Configure the algorithm,
        PPOConfig()
        .environment("justice", env_config=config)
        .rollouts(num_rollout_workers=0)
        .framework("torch")
        .training(model={"fcnet_hiddens": [64, 64]})
        .evaluation(evaluation_num_workers=0)
    )
    return ppo_config

def env_creator(env_config):
    return JusticeEnv(env_config)

check_point_dir_name = f"cp_{datetime.datetime.now().strftime('%m-%d-%Y_%H:%M:%S')}"
check_point_dir_path = os.path.join(os.getcwd(), "rl", "marl", "checkpoints")




if __name__ == "__main__":

    ray.init(local_mode=True)

    #load environment / config
    ppo_config = prepare_config()
    algo = ppo_config.build()  

    #train
    for _ in tqdm(range(1)):
        print(algo.train()) 

    #save
    os.mkdir(check_point_dir_path + '/' + check_point_dir_name)
    checkpoint_path = os.path.join(check_point_dir_path,check_point_dir_name)
    algo.save(checkpoint_path)

    #eval
    