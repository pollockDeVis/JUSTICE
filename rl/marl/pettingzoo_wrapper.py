import functools
import pickle
from pathlib import Path

import numpy as np
from gymnasium.spaces import Box
from gymnasium.utils import seeding
from pettingzoo import ParallelEnv

from src.model import JUSTICE
from src.util.enumerations import Abatement, DamageFunction, Economy

CONFIG = {
    "start_year": 2015,
    "end_year": 2300,
    "timestep": 1,
    "scenario": 7,
    "economy_type": Economy.NEOCLASSICAL,
    "damage_function_type": DamageFunction.KALKUHL,
    "abatement_type": Abatement.ENERDATA,
    "num_agents": 57,
    "model_pickle_path": Path("pickles") / "JUSTICE.pkl",
    "config_pickle_path": Path("pickles") / "config.pkl",
}

OBSERVATIONS = [
    "net_economic_output",
    "emissions",
    "regional_temperature",
    "economic_damage",
    "abatement_cost",
]
# REWARD="welfare_utilitarian_temporal"
REWARD = "disentangled_utility"
GLOBAL_OBSERVATIONS = ["global_temperature"]


class JusticeEnv(ParallelEnv):
    metadata = {"name": "justice_env"}

    def __init__(self, env_config, render_mode=None):
        self.possible_agents = [f"agent_{i}" for i in range(env_config["num_agents"])]
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )
        self.render_mode = render_mode

        self.model_pickle_path: Path = env_config["model_pickle_path"]

        self.timestep = 0
        self.start_year = env_config["start_year"]
        self.end_year = env_config["end_year"]

    @staticmethod
    def pickle_model(env_config):
        model_pickle_path: Path = env_config["model_pickle_path"]
        config_pickle_path: Path = env_config["config_pickle_path"]
        if config_pickle_path.exists():
            with config_pickle_path.open("rb") as f:
                pickle_config = pickle.load(f)
        else:
            pickle_config = None
        if not model_pickle_path.exists() or pickle_config != env_config:
            model = JUSTICE(
                start_year=env_config["start_year"],
                end_year=env_config["end_year"],
                timestep=env_config["timestep"],
                scenario=env_config["scenario"],
                economy_type=env_config["economy_type"],
                damage_function_type=env_config["damage_function_type"],
                abatement_type=env_config["abatement_type"],
            )
            with model_pickle_path.open("wb") as f:
                pickle.dump(model, f)
            with config_pickle_path.open("wb") as f:
                pickle.dump(env_config, f)

    # lru_cache allows observation and action spaces to be memoized, reducing clock cycles required to get each agent's space.
    # If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(OBSERVATIONS + GLOBAL_OBSERVATIONS),),
            dtype=np.float32,
        )

    # If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Box(0.0001, 0.9999, shape=(2,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._np_random, seed = seeding.np_random(seed)

        # TODO: add JUSTICE reset
        with self.model_pickle_path.open("rb") as f:
            self.model = pickle.load(f)

        self.agents = self.possible_agents
        self.timestep = 0
        data = self.model.stepwise_evaluate(timestep=self.timestep)

        infos = {agent: {} for agent in self.agents}
        obs = self.generate_observations(data)
        return obs, infos
    
    def generate_reward(self, vector):
        rews = np.full((self.num_agents,), vector[self.timestep, 0], dtype=np.float32)
        return rews

    def generate_observations(self, data):
        global_obs = np.array([data[key][self.timestep, 0] for key in GLOBAL_OBSERVATIONS], dtype=np.float32)
        local_obs = np.array([data[key][:, self.timestep, 0] for key in OBSERVATIONS], dtype=np.float32).T

        obs = {agent: np.concatenate((local_obs[i], global_obs)) for i, agent in enumerate(self.agents)}

        return obs

    def step(self, actions: dict):
        savings_rate = np.array([actions[agent][0] for agent in self.agents], dtype=np.float32)
        emissions_rate = np.array([actions[agent][1] for agent in self.agents], dtype=np.float32)

        self.model.stepwise_run(
            savings_rate=savings_rate,
            emissions_control_rate=emissions_rate,
            timestep=self.timestep,
        )

        data = self.model.stepwise_evaluate(timestep=self.timestep)
        obs = self.generate_observations(data)

        done = self.start_year + self.timestep >= self.end_year
        truncated = {agent: done for agent in self.agents}
        terminated = {agent: done for agent in self.agents}
        rewards = {agent: data[REWARD][i, self.timestep, 0] for i, agent in enumerate(self.agents)}
        infos = {agent: {} for agent in self.agents}

        self.timestep += 1  # NOTE: update timestep after stepwise_run?

        return obs, rewards, terminated, truncated, infos
