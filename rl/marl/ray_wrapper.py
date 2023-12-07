from src.util.data_loader import DataLoader
from src.util.enumerations import *
from src.util.model_time import TimeHorizon
from src.model import JUSTICE
import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import tqdm
from ray.rllib.utils.typing import MultiAgentDict

config = {
    "start_year": 2015,
    "end_year": 2300,
    "timestep": 1,
    "scenario": 7,
    "economy_type": Economy.NEOCLASSICAL,
    "damage_function_type": DamageFunction.KALKUHL,
    "abatement_type": Abatement.ENERDATA,
    "num_agents": 57,
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
        self.reward_key = "consumption"
        self.timestep = 0
        self.start_year = env_config["start_year"]
        self.end_year = env_config["end_year"]

    def reset(self, seed=None, options=None) -> tuple[MultiAgentDict, MultiAgentDict]:
        # return multiagent observation dict and info
        super().reset(seed=seed, options=options)
        self.timestep = 0
        # this is pseudo code, but initializing the model should return observations (obs)
        # infos is also required, but not useful
        obs = self.model.evaluate()
        return obs, {}

    def vec_to_dict(self, vector):
        output_dict = {}
        for agent_idx in range(self.num_agents):
            output_dict[f"agent_{agent_idx}"] = vector[agent_idx]
        return output_dict

    def step(
        self, action_dict
    ) -> tuple[
        MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict
    ]:
        self.timestep+=1

        savings_rate = []
        emissions_rate = []
        for agent_idx in range(self.num_agents):
            savings_rate.append(action_dict[f"agent_{agent_idx}"]["savings_rate"])
            emissions_rate.append(action_dict[f"agent_{agent_idx}"]["emissions_rate"])

        self.model.stepwise_run(savings_rate = np.array(savings_rate),
                                 emissions_control_rate = np.array(emissions_rate),
                                   timestep=self.timestep)
                                
        observations = self.model.stepwise_evaluate(timestep=self.timestep)

        rewards = self.vec_to_dict(observations[self.reward_key])
        #broadcast across all agents
        terminated = {
            "__all__":self.start_year + self.timestep > self.end_year
        }

        truncateds, infos = {}, {}
        return observations, rewards, terminated, truncateds, infos

