import pandas as pd
import numpy as np

from src.util.data_loader import DataLoader
from src.util.enumerations import *
from src.util.model_time import TimeHorizon
from src.model import JUSTICE
from tqdm.notebook import tqdm
from rl.marl.ray_wrapper import JusticeEnv, config

import ray


#register env
from ray.tune.registry import register_env

def env_creator(env_config):
    return JusticeEnv(env_config)

register_env("justice", env_creator )

from ray.rllib.algorithms.ppo import PPOConfig

ppo_config = (  # 1. Configure the algorithm,
    PPOConfig()
    .environment("justice", env_config=config)
    .rollouts(num_rollout_workers=0)
    .framework("torch")
    .training(model={"fcnet_hiddens": [64, 64]})
    .evaluation(evaluation_num_workers=0)
)

ray.init(local_mode=True)

algo = ppo_config.build()  # 2. build the algorithm,

for _ in range(5):
    print(algo.train())  # 3. train it,

algo.evaluate() 