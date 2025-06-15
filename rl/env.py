
"""Defines the JUSTICE IAM as a MOMARL environment using the MOMALand API."""

import functools
import pickle
from pathlib import Path
from dataclasses import asdict

import numpy as np
from gymnasium.spaces import Box, MultiDiscrete
from gymnasium.utils import seeding
from momaland.utils.env import MOParallelEnv

from justice.util.enumerations import Rewards
from justice.model import JUSTICE

OBSERVATIONS = [  # local observations, not shared with other agents
    "net_economic_output",
    "emissions",
    "regional_temperature",
    "economic_damage",
    "abatement_cost",
]

GLOBAL_OBSERVATIONS = ["global_temperature"]  # global observations, same for all agents

BILATERAL_OBSERVATIONS = [  # negotiation observations, shared with other agents, modelling the negotiation process
    "promised_mitigation_rate",
    "requested_mitigation_rate",
    "proposal_decisions",
]

class JusticeMOMARLEnv(MOParallelEnv):
    
    """
    The environment class for the multi-agent justice environment.
    """

    metadata = {"name": "justice_env"}

    def __init__(self, env_config, render_mode=None):

        self.negotiation_on = env_config[
            "negotiation_on"
        ]  # if the negotiation mechanism is on

        if self.negotiation_on:
            self.reduced_space = env_config[
                "reduced_space"
            ]  # if we use the reduced observation space for negotiation

        self.fixed_savings_rate = env_config[
            "fixed_savings_rate"
        ]  # if we use fixed savings rates

        self.clustering = env_config[
            "clustering"
        ]  # if we are using a cluster of regions instead of all of them

        self.possible_agents = [f"agent_{i}" for i in range(env_config["num_agents"])]
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )
        self.agents = self.possible_agents  # renamed for convenience

        self.action_window = env_config[
            "action_window"
        ]  # if we want to restrict the action space to a certain range around the previous action
        self.masking = (
            self.action_window or self.negotiation_on
        )  # if we are using action masking (just window or negotiation mechanism)

        if self.masking:
            self.window_actions = env_config[
                "window_actions"
            ]  # actions for which we want to restrict the action space

        self.render_mode = render_mode

        self.env_config = env_config

        self.timestep = (
            0  # current timestep, including negotiation stages and simulation stages
        )
        self.activity_timestep = 0  # current simulation timestep. Becomes standard timestep when negotiation is off

        self.start_year = env_config["start_year"]  # start year of the simulation
        self.end_year = env_config["end_year"]  # end year of the simulation

        self.num_discrete_action_levels = env_config[
            "num_discrete_actions"
        ]  # number of mitigation and saving rate actions

        self.rates_possible_actions = (
            [self.num_discrete_action_levels] * 2
            if not self.fixed_savings_rate
            else [self.num_discrete_action_levels]
        )
        self.proposal_possible_actions = (
            [self.num_discrete_action_levels] * 2 * len(self.possible_agents)
        )  # promised and requested mitigation rates
        self.evaluation_possible_actions = [2] * len(
            self.possible_agents
        )  # proposal decisions

        self.total_possible_actions = (
            (
                self.rates_possible_actions
                + self.proposal_possible_actions  # promised and requested mitigation rates
                + self.evaluation_possible_actions  # proposal decisions
            )
            if self.negotiation_on
            else self.rates_possible_actions
        )

        self.actions_clip = env_config["actions_clip"]  # action range
        self.num_years_per_step = env_config[
            "num_years_per_step"
        ]  # number of simulation years per timestep

        self.num_activity_steps = (
            self.end_year - self.start_year
        ) // self.num_years_per_step  # number of simulation steps. Becomes the total number of steps when negotiation is off
        self.num_steps = (
            (self.num_activity_steps * 3) - 2
            if self.negotiation_on
            else self.num_activity_steps
        )  # number of total steps

        self.global_observations = []
        self.local_observations = []
        self.saving_rates = []
        self.emission_control_rates = []

        if self.negotiation_on:
            self.requested_emission_control_rates = (
                []
            )  # (i,j) -> requested emission control rate from agent i to agent j
            self.promised_emission_control_rates = (
                []
            )  # (i,j) -> promised emission control rate from agent i to agent j
            self.negotiation_decisions = (
                []
            )  # (i,j) -> decision of agent i on the proposal of agent j
            self.min_emission_control_rates = (
                []
            )  # minimum emission control rate that an agent can accept based on the requested and promised rates

        self.rewards = env_config["rewards"]
        self.num_rewards = len(self.rewards)

        # === Initialize the objectives ===
        self.objectives = env_config["rewards"]
        
        if "regional_welfare" in self.objectives:
            self.objectives.remove("regional_welfare")
            for i in range(env_config["num_agents"]):
                self.objectives.append("regional_welfare"+str(i))
        self.num_objectives = len(self.objectives)

        objective_signs = []

        if "years_above_threshold" in self.objectives:
            objective_signs.append(-1.0) # lower is better
        if "years_below_threshold" in self.objectives:
            objective_signs.append(1.0) # higher is better
        if "total_damage" in self.objectives:
            objective_signs.append(-1.0) # lower is better
        if "yearly_damage" in self.objectives:
            objective_signs.append(-1.0) # lower is better
        if "total_abatement" in self.objectives:
            objective_signs.append(-1.0) # lower is better
        if "yearly_abatement" in self.objectives:
            objective_signs.append(-1.0) # lower is better
        if "gini_c1" in self.objectives:
            objective_signs.append(-1.0) # lower is better
        if "gini_c2" in self.objectives:
            objective_signs.append(-1.0) # lower is better
        if "welfare" in self.objectives:
            objective_signs.append(1.0) # higher is better
        if "delta" in self.objectives:
            objective_signs.append(-1.0) # lower is better
        if "regional_welfare0" in self.objectives:
            for i in range(self.num_groups):
                objective_signs.append(1.0) # higher is better

        self.objective_signs = np.array(objective_signs)

        self.threshold_temperature = 2.0
        self.log_histories = False

    @staticmethod
    def pickle_model(args, exp_name):
        env_config = asdict(args.env_config)
        env_config["config_pickle_path"] = Path(args.base_save_path) / "pickles"
        config_pickle_path: Path = env_config["config_pickle_path"] / f"{exp_name}.pkl"

        with config_pickle_path.open("wb") as f:
            pickle.dump(args, f)

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return (
            Box(
                low=-np.inf,
                high=np.inf,
                shape=(
                    len(OBSERVATIONS)  # local observations
                    + len(GLOBAL_OBSERVATIONS)  # global observations
                    + 1  # negotiation stage
                    + self.num_agents  # emission control rates of all the agents
                    + 2
                    * self.num_agents  # promised (promised and got promised) emission control rates for the next activity timestep
                    + 2
                    * self.num_agents  # requested (requested and got requested) emission control rates for the next activity timestep
                    + 2
                    * self.num_agents  # proposal decisions (proposed ones that were accepted and requested ones that the agent is accepting)
                    + 1,  # min emission rate among the agreed ones
                ),
                dtype=np.float32,
            )
            if self.negotiation_on and not self.reduced_space

            else Box(
                low=-np.inf,
                high=np.inf,
                shape=(
                    len(OBSERVATIONS)  # local observations
                    + len(GLOBAL_OBSERVATIONS)  # global observations
                    + self.num_agents,  # emission control rates of all the agents
                ),
                dtype=np.float32,
            )
        )

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return MultiDiscrete(self.total_possible_actions)

    @functools.lru_cache(maxsize=None)
    def reward_space(self, agent):

        # Define a Box space with two scalar rewards, each unbounded (can be adjusted as needed)
        return Box(
            low=np.array(
                [-np.inf for _ in range(self.num_rewards)]
            ),  # Lower bound for each reward
            high=np.array(
                [np.inf for _ in range(self.num_rewards)]
            ),  # Upper bound for each reward
            shape=(self.num_rewards,),  # N-dimensional space
            dtype=np.float32,
        )

    def state(self):
        """
        Returns a global observation for centralized training or other global state-based logic.
        This is a single numpy array, not a dictionary.
        """
        # Combine global observations and other global features like emission control rates
        global_state = np.concatenate(
            [
                np.array(self.global_observations).flatten(),  # Global observations
                np.array(
                    self.emission_control_rates
                ).flatten(),  # Emission control rates
            ]
        )

        return global_state

    def reset(self, seed=None, options=None):

        # print("Resetting environment")

        if seed is not None:
            self._np_random, seed = seeding.np_random(seed)

        ensemble = self.env_config["climate_ensembles"]

        self.model = JUSTICE(
            start_year=self.env_config["start_year"],
            end_year=self.env_config["end_year"],
            timestep=self.env_config["timestep"],
            scenario=self.env_config["scenario"],
            economy_type=self.env_config["economy_type"],
            damage_function_type=self.env_config["damage_function_type"],
            abatement_type=self.env_config["abatement_type"],
            pure_rate_of_social_time_preference=self.env_config[
                "pure_rate_of_social_time_preference"
            ],
            inequality_aversion=self.env_config["inequality_aversion"],
            climate_ensembles=ensemble,
            clustering=self.env_config["clustering"],
            cluster_level=self.num_agents,
            data_path=self.env_config["data_path"],
        )

        self.timestep = 0
        self.activity_timestep = 0

        if self.negotiation_on:
            self.set_stage()  # sets to model stage (negotiation_stage = 0)

        if self.masking:
            self.default_action_mask = self.set_default_agent_action_mask()
            self.action_mask = (
                self.default_action_mask.copy()
            )  # all actions are possible at the beginning

        self.saving_rates = np.zeros((self.num_steps, self.num_agents))
        self.emission_control_rates = np.zeros((self.num_steps, self.num_agents))

        if self.negotiation_on:  # initialize the negotiation observations
            self.global_observations = np.zeros(
                (self.num_steps, len(GLOBAL_OBSERVATIONS))
            )
            self.local_observations = np.zeros(
                (self.num_steps, self.num_agents, len(OBSERVATIONS))
            )
            self.requested_emission_control_rates = np.zeros(
                (self.num_steps, self.num_agents, self.num_agents)
            )
            self.promised_emission_control_rates = np.zeros(
                (self.num_steps, self.num_agents, self.num_agents)
            )
            self.negotiation_decisions = np.zeros(
                (self.num_steps, self.num_agents, self.num_agents)
            )
            self.min_emission_control_rates = np.zeros(
                (self.num_steps, self.num_agents)
            )

        # initialize the local and global observations
            
        self.data = self.model.stepwise_evaluate(timestep=self.activity_timestep)

        self.data["inverse_global_temperature"] = np.zeros((self.num_activity_steps))
        self.data["inverse_local_temperature"] = np.zeros((57, self.num_activity_steps))
        self.data["net_economic_output_rev"] = np.zeros((57, self.num_activity_steps))
        self.data["global_economic_output"] = np.zeros((self.num_activity_steps))

        for r in self.rewards:
            if r in ["inverse_global_temperature","delta", "years_above_threshold", "years_below_threshold", "ginic1", "ginic2", "welfare", "global_economic_output"]:
                self.data[r] = np.zeros((self.num_activity_steps))
            else:
                self.data[r] = np.zeros((57, self.num_activity_steps))

        obs = self.get_observations()

        if self.masking:
            self.action_mask = self.calc_action_mask()

        # the action mask is passed to the infos dictionary to be used in the training loop
        infos = {
            agent: {"action_mask": self.action_mask[i]}
            for i, agent in enumerate(self.agents)
            if self.masking
        }

        return obs, infos

    def step(self, actions: dict):
        """
        The step function of the environment.
        If the negotiation mechanism is on, the environment goes through three stages: model stage, proposal stage, evaluation stage.
        @param actions: dict -> the actions taken by the agents
        """

        if self.negotiation_on:
            self.set_stage()  # sets the step stage (0: model stage, 1: proposal stage, 2: evaluation stage)

        # set variables for this timestep to the previous timestep values
        if self.timestep > 0 and self.negotiation_on:
            self.requested_emission_control_rates[self.timestep, :, :] = (
                self.requested_emission_control_rates[self.timestep - 1, :, :]
            )
            self.promised_emission_control_rates[self.timestep, :, :] = (
                self.promised_emission_control_rates[self.timestep - 1, :, :]
            )
            self.negotiation_decisions[self.timestep, :, :] = (
                self.negotiation_decisions[self.timestep - 1, :, :]
            )
            self.min_emission_control_rates[self.timestep, :] = (
                self.min_emission_control_rates[self.timestep - 1, :]
            )

        if self.activity_timestep > 0:
            self.saving_rates[self.activity_timestep, :] = self.saving_rates[
                self.activity_timestep - 1, :
            ]
            self.emission_control_rates[self.activity_timestep, :] = (
                self.emission_control_rates[self.activity_timestep - 1, :]
            )
            if self.negotiation_on:
                self.global_observations[self.activity_timestep, :] = (
                    self.global_observations[self.activity_timestep - 1, :]
                )
                self.local_observations[self.activity_timestep, :, :] = (
                    self.local_observations[self.activity_timestep - 1, :, :]
                )

        if not self.negotiation_on or self.is_model_stage():  # model stage


            savings_rate = (
                self.get_actions("savings", actions)
                if not self.fixed_savings_rate
                else None
            )
            emission_control_rates = self.get_actions("emissions_rate", actions)

            self.emission_control_rates[self.activity_timestep, :] = (
                emission_control_rates
            )
            self.saving_rates[self.activity_timestep, :] = savings_rate

            self.model.stepwise_run(
                emission_control_rate=emission_control_rates,
                timestep=self.activity_timestep,
                savings_rate=savings_rate,
                endogenous_savings_rate=self.fixed_savings_rate,
            )

            data = self.model.stepwise_evaluate(timestep=self.activity_timestep)

            obs = self.get_observations()

            # creating the two individual rewards for each agent
            data["inverse_local_temperature"][:, self.activity_timestep] = (
                1 / data["regional_temperature"][:, self.activity_timestep, :].mean(
                    axis=1
                )
            )
            
            data["net_economic_output_rev"][:, self.activity_timestep] = (
                data["net_economic_output"][:, self.activity_timestep, :].mean(
                    axis=1
                )
            )
            data["inverse_global_temperature"][self.activity_timestep] = (
                1/data["global_temperature"][self.activity_timestep].mean(
                axis=0)
            )
            data["global_economic_output"][self.activity_timestep] = (
                data["net_economic_output"][:, self.activity_timestep, :].mean(axis=0).sum(axis=0)
            )
            
            # creating the reward vector for each agent
            agents_reward = self.generate_momarl_reward()

            rewards = {
                agent: agents_reward[i, :] for i, agent in enumerate(self.agents)
            }

            infos = (
                {
                    agent: {
                        "mean_reward": np.mean(agents_reward),
                        "absolute_reward": rewards[agent],
                        "action_mask": self.action_mask[i],
                    }
                    for i, agent in enumerate(self.agents)
                }
                if self.masking
                else {
                    agent: {
                        "mean_reward": np.mean(agents_reward),
                        "absolute_reward": rewards[agent],
                    }
                    for i, agent in enumerate(self.agents)
                }
            )

            done = (
                self.timestep >= self.num_steps - 1
            )  # ends when the last year is reached

            self.timestep += 1
            self.activity_timestep += 1

            # ends when the last year is reached

            truncated = {agent: done for agent in self.agents}
            terminated = {agent: done for agent in self.agents}

        elif self.is_proposal_stage():  # proposal stage
            obs, rewards, terminated, truncated, infos = self.step_propose(actions)

            self.timestep += 1

        elif self.is_evaluation_stage():  # evaluation stage
            obs, rewards, terminated, truncated, infos = self.step_evaluate_proposal(
                actions
            )

            self.timestep += 1

        else:
            raise ValueError("Invalid stage: something went wrong")

        return obs, rewards, terminated, truncated, infos

    def step_propose(self, actions):
        """
        The proposal stage of the negotiation mechanism.
        Agents send proposals to each other in the form of a requested and promised emission control rate.
        This means that each agent i sends a tuple (requested, promised) to each agent j.

        @param actions: dict -> the actions taken by the agents
        """

        promised_emission_control_rates = self.get_actions(
            "promised_mitigation_rate", actions
        )

        for i, agent in enumerate(self.agents):
            self.promised_emission_control_rates[self.timestep, i, :] = (
                promised_emission_control_rates[i]
            )

        requested_emission_control_rates = self.get_actions(
            "requested_mitigation_rate", actions
        )

        for i, agent in enumerate(self.agents):
            self.requested_emission_control_rates[self.timestep, i, :] = (
                requested_emission_control_rates[i]
            )

        obs = self.get_observations()

        # create dummy agents rewards
        agents_reward = np.zeros((self.num_agents, 2))

        rewards = {agent: agents_reward[i, :] for agent in self.agents}
        terminated = {agent: False for agent in self.agents}
        truncated = {agent: False for agent in self.agents}
        infos = {
            agent: {"action_mask": self.action_mask[i]}
            for i, agent in enumerate(self.agents)
        }

        return obs, rewards, terminated, truncated, infos

    def step_evaluate_proposal(self, actions):
        """
        The evaluation stage of the negotiation mechanism.
        Agents evaluate the proposals they received and decide whether to accept them or not.

        @param actions: dict -> the actions taken by the agents
        """

        proposal_decisions = self.get_actions("proposal_decisions", actions)
        self.negotiation_decisions[self.timestep, :, :] = proposal_decisions

        for i, agent in enumerate(self.agents):
            min_mitigation = self.calc_mitigation_rate_lower_bound(i)
            self.min_emission_control_rates[self.timestep, i] = min_mitigation

        obs = self.get_observations()

        agents_reward = np.zeros((self.num_agents, 2))

        rewards = {agent: agents_reward[i, :] for agent in self.agents}
        terminated = {agent: False for agent in self.agents}
        truncated = {agent: False for agent in self.agents}

        infos = {
            agent: {"action_mask": self.action_mask[i]}
            for i, agent in enumerate(self.agents)
        }

        return obs, rewards, terminated, truncated, infos

    def generate_momarl_reward(self):

        """
        Generate the reward for each agent in the environment.

        """
        rewards = []

        for reward in self.rewards:
            reward_vector = self.data[reward]
            if reward in [
                "inverse_global_temperature",
                "global_economic_output",
                "years_above_threshold",
                "years_below_threshold",
                "delta",
                "ginic1",
                "ginic2",
                "welfare",
                ]:

                agent_rewards = [np.full(
                    (self.num_agents,),
                    reward_vector[self.activity_timestep],
                    dtype=np.float32
                )]
            elif reward in [
                "net_economic_output_rev",
                "inverse_local_temperature",
                "stepwise_marl_reward",
                "total_abatement",
                "yearly_abatement",
                "total_damage",
                "yearly_damage",
                ]: 
                if self.clustering:
                    cluster_sizes = np.array(
                        [
                            len(self.model.cluster_to_country[cluster_idx])
                            for cluster_idx in range(len(self.model.clusters))
                        ]
                    )

                    clustered_agents_reward = np.zeros((self.num_agents))
                    agent_rewards = []
                    for region_idx, cluster_idx in self.model.country_to_cluster.items():
                        clustered_agents_reward[cluster_idx] += reward_vector[
                            region_idx, self.activity_timestep
                        ]

                    agent_rewards = [clustered_agents_reward / cluster_sizes]
                    
                else:

                    agent_rewards = np.array(
                        [
                            reward_vector[i, self.activity_timestep]
                            for i, _ in enumerate(self.agents)
                        ],
                        dtype=np.float32,
                    )
                
            rewards.append(agent_rewards)
            
        rewards = np.stack(rewards, axis=1)
        
        if len(rewards.shape) == 3:
            rewards = rewards.squeeze(0).T
        
        return rewards

    def get_observations(self):
        """
        This method handles the correct generation of observations for each agent.
        It is called in the reset() and step() methods of the environment.
        """

        if not self.negotiation_on or self.is_model_stage():
            # If we are in a (justice) model stage, fetch the observations from the model
            
            global_obs = np.array(
                [
                    self.data[key][self.activity_timestep, :].mean(axis=0)
                    for key in GLOBAL_OBSERVATIONS
                ],
                dtype=np.float32,
            )
            local_obs = np.array(
                [
                    self.data[key][:, self.activity_timestep, :].mean(axis=1)
                    for key in OBSERVATIONS
                ],
                dtype=np.float32,
            ).T

            
        else:
            # Fetch pre-stored observations for negotiation stages from the last model stage
            global_obs = self.global_observations[self.activity_timestep, :]
            local_obs = self.local_observations[self.activity_timestep, :, :]

        # If we are in a model stage, then the observations have to be clustered into the correct number of clusters
        if self.clustering and (not self.negotiation_on or self.is_model_stage()):

            # Initialize the clustered observations array
            clustered_local_obs = np.zeros((self.num_agents, len(OBSERVATIONS)))

            # Sum local observations by cluster
            for region_idx, cluster_idx in self.model.country_to_cluster.items():
                clustered_local_obs[cluster_idx, :] += local_obs[region_idx, :]

            # Precompute cluster sizes to avoid recalculating
            cluster_sizes = np.array(
                [
                    len(self.model.cluster_to_country[cluster_idx])
                    for cluster_idx in range(len(self.model.clusters))
                ]
            )
            # Compute the average by dividing by the number of regions per cluster

            clustered_local_obs[:, :] /= cluster_sizes[:, np.newaxis]

            # Update local_obs with clustered observations
            local_obs = clustered_local_obs

        # Update stored observations in case of negotiation and model stage
        if self.negotiation_on and self.is_model_stage():
            self.global_observations[self.activity_timestep, :] = global_obs
            self.local_observations[self.activity_timestep, :, :] = local_obs

        if self.negotiation_on and not self.reduced_space:
            obs = {
                agent: np.concatenate(
                    (
                        local_obs[i],
                        global_obs,
                        np.array([self.negotiation_stage]),
                        self.emission_control_rates[self.activity_timestep, :],
                        self.promised_emission_control_rates[self.timestep, i, :],
                        self.promised_emission_control_rates[self.timestep, :, i],
                        self.requested_emission_control_rates[self.timestep, i, :],
                        self.requested_emission_control_rates[self.timestep, :, i],
                        self.negotiation_decisions[self.timestep, i, :],
                        self.negotiation_decisions[self.timestep, :, i],
                        [self.min_emission_control_rates[self.timestep, i]],
                    )
                )
                for i, agent in enumerate(self.agents)
            }

        else:  # if we use the reduced space, we only pass the standard observations despite using the negotiation mechanism

            obs = {
                agent: np.concatenate(
                    (
                        local_obs[i],
                        global_obs,
                        self.emission_control_rates[self.activity_timestep, :],
                    )
                )
                for i, agent in enumerate(self.agents)
            }

        if self.masking:
            self.action_mask = self.calc_action_mask()
        # print(negotiation_stage, self.timestep, self.activity_timestep)
        return obs

    def get_actions_index(self, action_type):
        """
        Get the index of the first action of a certain type in the actions array.

        @param action_type: str -> the type of action to be retrieved
        """

        if action_type == "savings":
            return 0
        if action_type == "emissions_rate":
            return (
                int(len(self.rates_possible_actions) / 2)
                if not self.fixed_savings_rate
                else 0
            )
        if action_type == "proposal":
            return len(self.rates_possible_actions)
        if action_type == "proposal_decisions":
            return len(self.rates_possible_actions + self.proposal_possible_actions)

    def get_actions(self, action_type, actions):

        """
        Get the actions of a certain type from the actions dict.
        @param action_type: str -> the type of action to be retrieved
        @param actions: dict -> the actions dict coming from the agent's policy (i.e. the nn output)
        """

        if action_type == "savings":
            savings_actions_index = self.get_actions_index("savings")
            # print(actions)
            return np.array(
                [
                    self.discrete_to_float(actions[agent][savings_actions_index])
                    for i, agent in enumerate(self.agents)
                ],
                dtype=np.float32,
            )

        if action_type == "emissions_rate":
            mitigation_rate_action_index = self.get_actions_index("emissions_rate")

            return np.array(
                [
                    self.discrete_to_float(actions[agent][mitigation_rate_action_index])
                    for i, agent in enumerate(self.agents)
                ],
                dtype=np.float32,
            )

        if action_type == "promised_mitigation_rate":
            proposal_actions_index_start = self.get_actions_index("proposal")
            num_proposal_actions = len(self.proposal_possible_actions)

            return np.array(
                [
                    self.discrete_to_float(
                        actions[agent][
                            proposal_actions_index_start : proposal_actions_index_start
                            + num_proposal_actions : 2
                        ]
                    )
                    for i, agent in enumerate(self.agents)
                ]
            )

        if action_type == "requested_mitigation_rate":
            proposal_actions_index_start = self.get_actions_index("proposal")
            num_proposal_actions = len(self.proposal_possible_actions)

            return np.array(
                [
                    self.discrete_to_float(
                        actions[agent][
                            proposal_actions_index_start
                            + 1 : proposal_actions_index_start
                            + num_proposal_actions : 2
                        ]
                    )
                    for i, agent in enumerate(self.agents)
                ]
            )

        if action_type == "proposal_decisions":
            proposal_decisions_index_start = self.get_actions_index(
                "proposal_decisions"
            )
            num_evaluation_actions = len(self.evaluation_possible_actions)

            proposal_decisions = np.array(
                [
                    actions[agent][
                        proposal_decisions_index_start : proposal_decisions_index_start
                        + num_evaluation_actions
                    ]
                    for i, agent in enumerate(self.agents)
                ]
            )

            for i, agent in enumerate(self.agents):
                proposal_decisions[i, i] = 0  # we set the agent's own decision to 0

            return proposal_decisions

    def is_model_stage(self):
        return self.negotiation_stage == 0

    def is_proposal_stage(self):
        return self.negotiation_stage == 1

    def is_evaluation_stage(self):
        return self.negotiation_stage == 2

    def set_stage(self):

        """
        Set the negotiation stage based on the current timestep.
        """

        self.negotiation_stage = self.timestep % 3

    def calc_mitigation_rate_lower_bound(self, agent):
        """
        Computes the minimum emission control rate that an agent can accept.
        Takes into account the (accepted) requested and promised mitigation rates.
        Then it returns the maximum value among the elements of the two arrays.

        @param agent: int -> the agent for which the minimum mitigation rate is calculated
        """

        outgoing_accepted_mitigation_rates = (
            self.get_outgoing_accepted_mitigation_rates(agent)
        )

        incoming_accepted_mitigation_rates = (
            self.get_incoming_accepted_mitigation_rates(agent)
        )

        min_mitigation = max(
            outgoing_accepted_mitigation_rates + incoming_accepted_mitigation_rates
        )

        return min_mitigation

    def get_incoming_accepted_mitigation_rates(self, agent_id):
        """
        Get the emission control rates that the agent has received and accepted from other agents.

        @param agent: int -> the agent who proposed the emission control rates
        """

        return [
            self.promised_emission_control_rates[self.timestep, j, agent_id]
            * self.negotiation_decisions[self.timestep, agent_id, j]
            for j, agent in enumerate(self.agents)
        ]

    def get_outgoing_accepted_mitigation_rates(self, agent_id):
        """
        Get the accepted emission control rates that an agent has proposed to other agents.

        @param agent: int -> the agent who received the emission control rates proposals
        """

        return [
            self.requested_emission_control_rates[self.timestep, agent_id, j]
            * self.negotiation_decisions[self.timestep, j, agent_id]
            for j, agent in enumerate(self.agents)
        ]

    def calc_action_mask(self):
        """
        Generate action masks. This includes the action window if we want to restrict
        then actions to a certain range around the previous action, and the action mask
        coming from the negotiation mechanism.
        """

        mask_dict = {i: None for i, agent in enumerate(self.agents)}

        for i, agent in enumerate(self.agents):

            if self.action_window:
                mask = self.calc_action_window(i)

            else:
                mask = self.default_action_mask.copy()

            if self.negotiation_on:

                mask_start = self.get_mask_index("emissions_rate")[0]
                mask_end = mask_start + self.num_discrete_action_levels
                if not self.continuous_actions:
                    minimum_mitigation_rate = int(
                        round(
                            self.min_emission_control_rates[self.timestep, i]
                            * self.num_discrete_action_levels
                        )
                    )
                    # print(minimum_mitigation_rate)
                    mitigation_mask = np.array(
                        [0 for _ in range(minimum_mitigation_rate)]
                        + [
                            1
                            for _ in range(
                                self.num_discrete_action_levels
                                - minimum_mitigation_rate
                            )
                        ]
                    )

                    old_mask = mask[mask_start:mask_end].copy()

                    # the mask values are 1 only if they are 1 in the new mask and in the window mask

                    mask[mask_start:mask_end] = np.logical_and(
                        mask[mask_start:mask_end], mitigation_mask
                    ) if np.random.rand() < (1 - self.dprob) else mask[mask_start:mask_end] # Agents ignore negotiation agreements with probability self.dprob

                    # if no action is possible, we keep the mask given by the window
                    if not np.any(mask[mask_start:mask_end]):
                        mask[mask_start:mask_end] = old_mask.copy()

                elif self.continuous_actions:
                    ValueError("Continuous action space not implemented")

            mask_dict[i] = mask

        return mask_dict

    def set_default_agent_action_mask(self):
        """
        Set the default action mask for each agent (all actions are possible).
        """

        self.possible_actions_length = sum(self.total_possible_actions)
        if not self.continuous_actions:
            return np.ones(self.possible_actions_length, dtype=bool)
        else:
            ValueError("Continuous action space not implemented")

    def calc_action_window(self, agent_id):
        """
        Calculate the action window for a certain agent.
        The action window is a range of possible actions around the previous action.
        """

        base_mask = self.default_action_mask.copy()

        for action in self.window_actions:

            previous_action = (
                self.emission_control_rates[self.activity_timestep, agent_id]
                if action == "emissions_rate"
                else self.saving_rates[self.activity_timestep, agent_id]
            )

            scaled_action = self.float_to_discrete(previous_action)

            mask_start, mask_end = self.get_mask_index(action)

            current_mask = base_mask[mask_start:mask_end]
            current_mask[:] = 0

            current_mask[
                max(0, scaled_action - 2) : min(
                    self.num_discrete_action_levels, scaled_action + 3
                )
            ] = 1

            base_mask[mask_start:mask_end] = current_mask

        return base_mask.astype(int)

    def get_mask_index(self, action_type):
        """
        Get the index of the action mask for a certain action type.
        """

        move_index = (
            0 if not self.fixed_savings_rate else self.num_discrete_action_levels
        )

        if action_type == "savings":
            return 0, self.num_discrete_action_levels
        if action_type == "emissions_rate":
            return (
                self.num_discrete_action_levels - move_index,
                2 * self.num_discrete_action_levels - move_index,
            )
        if action_type == "promise":
            return (
                2 * self.num_discrete_action_levels - move_index,
                2 * self.num_discrete_action_levels
                + len(self.possible_agents) * self.num_discrete_action_levels
                - move_index,
            )
        if action_type == "request":
            return (
                2 * self.num_discrete_action_levels
                + len(self.possible_agents) * self.num_discrete_action_levels
                - move_index,
                2 * self.num_discrete_action_levels
                + 2 * len(self.possible_agents) * self.num_discrete_action_levels
                - move_index,
            )
        if action_type == "proposal_decisions":
            return (
                2 * self.num_discrete_action_levels
                + 2 * len(self.possible_agents) * self.num_discrete_action_levels
                - move_index,
                2 * self.num_discrete_action_levels
                + 2 * len(self.possible_agents) * self.num_discrete_action_levels
                + len(self.possible_agents) * 2
                - move_index,
            )

    def discrete_to_float(self, discrete_action):
        """
        Convert a discrete action to a float action.
        This has to be done because the action space is discrete,
        but the model requires continuous actions.
        """

        float_action = discrete_action / (self.num_discrete_action_levels - 1)
        float_action *= self.actions_clip[1] - self.actions_clip[0]
        float_action += self.actions_clip[0]
        return float_action

    def float_to_discrete(self, float_action):
        """
        Convert back a float action to a discrete action.
        """

        float_action -= self.actions_clip[0]
        float_action /= self.actions_clip[1] - self.actions_clip[0]
        discrete_action = int(
            round(float_action * (self.num_discrete_action_levels - 1))
        )
        return discrete_action