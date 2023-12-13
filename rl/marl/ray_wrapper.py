from src.util.data_loader import DataLoader
from src.util.enumerations import *
from src.util.model_time import TimeHorizon
from src.model import JUSTICE
import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import tqdm
from ray.rllib.utils.typing import MultiAgentDict
from gymnasium.spaces import Dict, Box
from numpy import finfo

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

OBSERVATIONS = ['net_economic_output', 'emissions', 'regional_temperature', 'economic_damage', 'abatement_cost']
REWARD="welfare_utilitarian_temporal"
GLOBAL_OBSERVATIONS = ['global_temperature']

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
        self.timestep = 0
        self.start_year = env_config["start_year"]
        self.end_year = env_config["end_year"]

        obs_space = {
            key: Box(low=finfo("float32").min,
                      high=finfo("float32").max,
                      shape = (1,),
                      dtype=np.float32) 
                        for key in OBSERVATIONS + GLOBAL_OBSERVATIONS
        }
        self.observation_space = Dict(
            obs_space
        )
        self.action_space = Dict(
            {
                "savings_rate":Box(0,1),
                "emissions_rate":Box(0,1)
            }
        )


    def reset(self, seed=None, options=None) -> tuple[MultiAgentDict, MultiAgentDict]:
        # return multiagent observation dict and info

        #TODO: add JUSTICE reset
        super().reset(seed=seed, options=options)
        self.timestep = 0
        # this is pseudo code, but initializing the model should return observations (obs)
        # infos is also required, but not useful
        data = self.model.stepwise_evaluate(timestep=self.timestep)
        obs = self.generate_observations(data)
        return obs, {}

    def generate_reward(self, vector):
        output_dict = {}
        for agent_idx in range(self.num_agents):
            output_dict[f"agent_{agent_idx}"] = vector[self.timestep, 0]
        return output_dict
    
    def generate_observations(self, data):
        obs = {k:data[k] for k in OBSERVATIONS}
        obs_per_agent = {}
        for agent_idx in range(self.num_agents):
            #TODO: parameterize number of ensembles for MARL during initialization
            local_observations_per_agent = {k:np.array([v[agent_idx,self.timestep,0]]).astype("float32") for k,v in obs.items()}
            global_observations_per_agent = {key:np.array([data[key][self.timestep,0]]).astype("float32") for key in GLOBAL_OBSERVATIONS}

            obs_per_agent[f"agent_{agent_idx}"] = {**local_observations_per_agent, **global_observations_per_agent}
        return obs_per_agent


    def step(
        self, action_dict
    ) -> tuple[
        MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict
    ]:            

        savings_rate = []
        emissions_rate = []
        for agent_idx in range(self.num_agents):
            savings_rate.append(action_dict[f"agent_{agent_idx}"]["savings_rate"])
            emissions_rate.append(action_dict[f"agent_{agent_idx}"]["emissions_rate"])

        self.model.stepwise_run(savings_rate = np.squeeze(np.array(savings_rate)),
                                 emissions_control_rate = np.squeeze(np.array(emissions_rate)),
                                   timestep=self.timestep)
                                
        data = self.model.stepwise_evaluate(timestep=self.timestep) #57 x 286 x 1001
        obs = self.generate_observations(data)
        rewards = self.generate_reward(data[REWARD])
        #broadcast across all agents
        terminated = {
            "__all__":self.start_year + self.timestep >= self.end_year
        }
        truncateds = {
             "__all__":False
        }
        infos = {}
        return obs, rewards, terminated, truncateds, infos
    

