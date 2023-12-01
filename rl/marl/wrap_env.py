from src.util.data_loader import DataLoader
from src.util.enumerations import *
from src.util.model_time import TimeHorizon
from src.model import JUSTICE
import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from ray.rllib.utils.typing import MultiAgentDict

config = {
    "start_year": 2015,
    "end_year": 2300,
    "timestep": 1,
    "scenario": 7,
    "economy_type": Economy.NEOCLASSICAL,
    "damage_function_type": DamageFunction.KALKUHL,
    "abatement_type": Abatement.ENERDATA,
    "num_agents": 1,
}


class JusticeEnv(MultiAgentEnv):
    def __init__(self, env_config):
        self.model = JUSTICE(
            start_year=env_config["start_year"],
            end_year=env_config["end_year"],
            timestep=env_config["timestep"],
            scenario=env_config["scenario"],
            economy_type=env_config["economy_type"],
            damage_function_type=env_config["damage_function_type"],
            abatement_type=env_config["abatement_type"],
        )

        self.agent_ids = set([f"agent_{i}" for i in range(env_config["num_agents"])])
        self.num_agents = env_config["num_agents"]

    def reset(self, seed=None, options=None) -> tuple[MultiAgentDict, MultiAgentDict]:
        # return multiagent observation dict and info
        super().reset(seed, options)

        # this is pseudo code, but initializing the model should return observations (obs)
        # infos is also required, but not useful
        obs_vec = self.model.start()
        obs = vec_to_dict(obs_vec)
        return obs, {}

    def vec_to_dict(self, vector):
        output_dict = {}
        for agent_idx in range(self.num_agents):
            output_dict[f"agent_{agent_idx}"] = vector[agent_idx]
        return output_dict

    def step(
        self, action_dict
    ) -> tuple(
        MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict
    ):
        savings_rate = []
        emissions_rate = []
        for agent_idx in range(self.num_agents):
            savings_rate.append(action_dict[f"agent_{agent_idx}"]["savings_rate"])
            emissions_rate.append(action_dict[f"agent_{agent_idx}"]["emissions_rate"])

        # stepwise run return vector of size num_agents x observations
        # for reward should return vector size num_agents
        # terminated is a boolean T/F is run is finished.
        # obs, rewards, terminated should come from model.evaluate_timestep,
        # which can be called inside the model.stepwise_run
        obs_vec, rewards_vec, terminated  = self.model.stepwise_run(
            np.array(savings_rate), np.array(emissions_rate), self.model.timestep
        )
        obs = self.vec_to_dict(obs_vec)
        rewards = self.vec_to_dict(rewards_vec)
        
        #broadcast across all agents
        terminated = {
            "__all__":terminated
        }

        truncateds, infos = {}, {}
        return obs, rewards, terminated, truncateds, infos
