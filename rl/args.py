
"""Defines the dataclasses for training args and environment configuration."""
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, List

from justice.util.enumerations import Abatement, DamageFunction, Economy, Rewards

@dataclass
class EnvConfig:

    """
    JUSTICE MOMARL Environment configuration parameters.
    """

    """ JUSTICE parameters """
    economy_type: Enum = Economy.NEOCLASSICAL # Economy model
    damage_function_type: Enum = DamageFunction.KALKUHL # Damage function
    abatement_type: Enum = Abatement.ENERDATA # Abatement function
    pure_rate_of_social_time_preference: float = 0.02 # Rate of social time preference
    elasticity_of_marginal_utility_of_consumption: float = 1.45 # Elasticity of marginal utility of consumption
    inequality_aversion: float = 0.5 # Normative parameter
    fixed_savings_rate: bool = False # Whether to fix the savings rates to predefined values (endogenous savings rates)
    time_preference: float = 0.015 # Discount rate of the JUSTICE model
    scenario: int = 2 # SSP RCP combinations
    start_year: int = 2015 # Start year of the simulation
    end_year: int = 2300 # End year of the simulation
    timestep: int = 1 # Timestep of the simulation

    """ Paths """
    data_path: Path = Path("justice/data")
    
    climate_ensembles: Optional[list] = [500]

    """ RL parameters """
    num_discrete_actions: int = 11 # Number of discrete actions
    actions_clip: tuple[float, float] = (0, 1) # Min and max values for the actions
    num_years_per_step: int = 1 # Number of simulation years per rl step
    num_agents: int = 12 # Number of agents
    clustering: bool = True # Set to true if you are using a clustering of the original regions
    negotiation_on: bool = False # Whether to enable the negotiation mechanism
    reduced_space: bool = False # Whether to enable a reduced observation space in the negotiation case
    action_window: bool = False # Whether to enable the action window mechanism, which constrains the agents actions.
    window_actions: list = field(default_factory=lambda: ["savings", "emissions_rate"])
    rewards: list = field( # Objectives to use for the training
        default_factory=lambda: [Rewards.INVERSE_GLOBAL_TEMPERATURE.value[1], Rewards.GLOBAL_ECONOMIC_OUTPUT.value[1]]
    )
    normalise_eval_env: bool = False # Whether to normalise the evaluation environment


@dataclass
class Args:

    """
    MOMARL training parameters.
    """

    """ Environment """
    env_config: EnvConfig = EnvConfig() # Environment configuration

    """ Seeding and device """
    seed: int = 1 # Experiments seed
    torch_deterministic: bool = True # If toggled, PyTorch will be deterministic
    cuda: bool = True # If toggled, cuda will be used

    """ Wandb Tracking"""
    track: bool = False # If toggled, the experiment will be tracked with wandb
    wandb_project_name: str = "paper-experiments" # wandb project name
    wandb_entity: str = "ai2p-team" # wandb entity

    """ MOMARL specific arguments """
    num_weights: None | int = 10 # Number of different weights to train on for this run.
    start_uniform_weight: None | int = 0 # Start weight for uniform weight generation. 
    end_uniform_weight: None | int = 10 # End weight for uniform weight generation. Should be start_uniform_weight + num_weights
    total_uniform_weights: None | int = 100 # Total number of weights to generate
    timesteps_per_weight: None | int = 1000000 # Training steps per weight vector
    weights_generation: None | str = "uniform" # The method to generate the weights - 'OLS' or 'uniform'
    n_sample_weights: None | int = 10 # Number of weights to sample for EUM and MUL computation
    ref_point: None | List[float] = field(default_factory=lambda: [0.0, 0.0]) # Reference point for hypervolume calculation
    save_policies: None | bool = True # If toggled, the policies will be saved
    base_save_path: str = "results" # The base path where checkpoints, pickles and artefacts will be saved
    evaluation_iterations: int = 6 # The number of training iterations after which the policy will be evaluated
    debug: bool = False

    """ PPO specific arguments """
    learning_rate: float = 2.5e-4 # Learning rate
    num_envs: int = 1 # Number of parallel environments
    anneal_lr: bool = True # Toggle learning rate annealing for policy and value networks
    gamma: float = 0.99 # Discount factor
    gae_lambda: float = 0.95 # GAE lambda
    num_minibatches: int = 285 # The number of mini-batches. This has been made equal to the number of time step so that we take one timestep per batch"""
    update_epochs: int = 20 # The K epochs to update the policy  
    norm_adv: bool = False # Toggle advantage normalisation
    clip_coef: float = 0.2 # The PPO clip parameter
    clip_vloss: bool = True # Toggle value loss clipping
    ent_coef: float = 0.001 # Entropy coefficient
    vf_coef: float = 1 # Value function coefficient
    max_grad_norm: float = 0.5 # Maximum gradient norm
    target_kl: float | None = None # Target KL divergence threshold
    network_model_config: None | str = ("rl/nn/params/mlp.json") # The path to the config of the network model

    """ Filled at runtime """
    num_steps: int = 0 # Number of steps to run in each environment per policy rollout
    batch_size: int = 0 # The batch size (computed in runtime)
    minibatch_size: int = 0 # The mini-batch size (computed in runtime)
    num_iterations: int = 0 # The number of iterations to run
