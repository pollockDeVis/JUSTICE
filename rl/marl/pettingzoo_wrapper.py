import functools
import pickle
from pathlib import Path

import numpy as np
from gymnasium.spaces import Box, MultiDiscrete
from gymnasium.utils import seeding
from pettingzoo import ParallelEnv

from src.model import JUSTICE

OBSERVATIONS = [
    "net_economic_output",
    "emissions",
    "regional_temperature",
    "economic_damage",
    "abatement_cost",
]
REWARD="welfare_utilitarian_regional_temporal"
#REWARD = "disentangled_utility"
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
        self.env_config = env_config
        self.timestep = 0
        self.start_year = env_config["start_year"]
        self.end_year = env_config["end_year"]

        self.pct_change = env_config["pct_change"]
        self.continuous_actions = env_config["continuous_actions"]
        self.num_discrete_actions = env_config["num_discrete_actions"]
        self.actions_clip = env_config["actions_clip"]
        self.num_years_per_step = env_config["num_years_per_step"]

    @staticmethod
    def pickle_model(env_config):
        model_pickle_path: Path = env_config["model_pickle_path"]
        config_pickle_path: Path = env_config["config_pickle_path"]
        if config_pickle_path.exists():
            with config_pickle_path.open("rb") as f:
                pickle_config = pickle.load(f)
        else:
            pickle_config = None
        if pickle_config != env_config:
            model = JUSTICE(
                start_year=env_config["start_year"],
                end_year=env_config["end_year"],
                timestep=env_config["timestep"],
                scenario=env_config["scenario"],
                economy_type=env_config["economy_type"],
                damage_function_type=env_config["damage_function_type"],
                abatement_type=env_config["abatement_type"],
                pure_rate_of_social_time_preference=env_config["pure_rate_of_social_time_preference"],
                inequality_aversion=env_config["inequality_aversion"],
                climate_ensembles=env_config["climate_ensembles"],
                elasticity_of_marginal_utility_of_consumption = env_config["elasticity_of_marginal_utility_of_consumption"]
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
            shape=(len(OBSERVATIONS + GLOBAL_OBSERVATIONS + self.possible_agents),),
            dtype=np.float32,
        )

    # If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        if self.continuous_actions:
            return Box(self.actions_clip[0], self.actions_clip[1], shape=(2,), dtype=np.float32)
        return MultiDiscrete([self.num_discrete_actions] * 2)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._np_random, seed = seeding.np_random(seed)

        # TODO: add JUSTICE reset
        # with self.model_pickle_path.open("rb") as f:
        #     self.model: JUSTICE = pickle.load(f)
        ensemble = np.random.randint(1,1001)
        self.model = JUSTICE(
                start_year=self.env_config["start_year"],
                end_year=self.env_config["end_year"],
                timestep=self.env_config["timestep"],
                scenario=self.env_config["scenario"],
                economy_type=self.env_config["economy_type"],
                damage_function_type=self.env_config["damage_function_type"],
                abatement_type=self.env_config["abatement_type"],
                pure_rate_of_social_time_preference=self.env_config["pure_rate_of_social_time_preference"],
                inequality_aversion=self.env_config["inequality_aversion"],
                climate_ensembles=ensemble
            )

        self.agents = self.possible_agents
        self.timestep = 0
        data = self.model.stepwise_evaluate(timestep=self.timestep)

        infos = {agent: {} for agent in self.agents}
        obs = self.generate_observations(data, np.zeros(len(self.agents)))
        return obs, infos
    
    def generate_reward(self, vector):
        rews = np.full((self.num_agents,), vector[self.timestep, 0], dtype=np.float32)
        return rews

    def generate_observations(self, data, emissions_rates):
        global_obs = np.array([data[key][self.timestep, 0] for key in GLOBAL_OBSERVATIONS], dtype=np.float32)
        local_obs = np.array([data[key][:, self.timestep, 0] for key in OBSERVATIONS], dtype=np.float32).T

        obs = {agent: np.concatenate((local_obs[i], global_obs, emissions_rates)) for i, agent in enumerate(self.agents)}

        return obs

    def calc_reward_diff(self, rewards):
        current_rewards = rewards[:,self.timestep]
        previous_rewards = rewards[:,self.timestep-1]
        pct_change = (current_rewards - previous_rewards) / previous_rewards
        return pct_change

    def discrete_to_float(self, discrete_action):
        float_action = discrete_action / (self.num_discrete_actions - 1)
        float_action *= (self.actions_clip[1] - self.actions_clip[0])
        float_action += self.actions_clip[0]
        return float_action

    def step(self, actions: dict):
        if self.continuous_actions:
            savings_rate = np.array([actions[agent][0] for agent in self.agents], dtype=np.float32)
            emissions_rate = np.array([actions[agent][1] for agent in self.agents], dtype=np.float32)
        else:
            savings_rate = np.array([self.discrete_to_float(actions[agent][0]) for agent in self.agents], dtype=np.float32)
            emissions_rate = np.array([self.discrete_to_float(actions[agent][1]) for agent in self.agents], dtype=np.float32)

        for year_step in range(self.num_years_per_step):
            self.model.stepwise_run(
                emission_control_rate=emissions_rate,
                timestep=self.timestep,
                savings_rate=savings_rate,
            )

            data = self.model.stepwise_evaluate(timestep=self.timestep)
            obs = self.generate_observations(data, emissions_rate)

            if self.timestep == 0:
                rewards_pct = {agent: 0.000001 for i, agent in enumerate(self.agents)}
            else:
                pct_change = self.calc_reward_diff(data[REWARD])
                rewards_pct = {agent: pct_change[i, 0] for i, agent in enumerate(self.agents)}

            if self.pct_change:
                rewards = rewards_pct
            else:
                rewards = {agent: data[REWARD][i,self.timestep, 0] for i, agent in enumerate(self.agents)}

            infos = {agent: {"mean_reward":np.mean([data[REWARD][j,self.timestep, 0] for j, _ in enumerate(self.agents)]),
                            "pct_change_reward":rewards_pct[agent],
                            "absolute_reward":data[REWARD][i,self.timestep, 0]} 
                    for i, agent in enumerate(self.agents)}

            self.timestep += 1 

            done = self.start_year + self.timestep >= self.end_year
            truncated = {agent: done for agent in self.agents}
            terminated = {agent: done for agent in self.agents}

            if done and year_step != (self.num_years_per_step - 1):
                print(f"Warning: end episode encountered before {self.num_years_per_step} years per step was completed. To fix, set num_years_per_step to a factor of 285")
                break
        
        return obs, rewards, terminated, truncated, infos
