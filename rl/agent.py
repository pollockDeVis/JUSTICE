"""Defines the class for the agent."""
import numpy as np
from rl.distributions import MultiCategorical
from rl.nn.enumerations import NetworkModelNames
from rl.nn.models import MagicNetwork, MLP
from rl.nn.configs import NetworkModelConfig
import torch.nn as nn
from torch import Tensor

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer

class AgentBase(nn.Module):
    def __init__(
        self, envs, model_config: NetworkModelConfig, num_timesteps: int, num_agents
    ):
        super().__init__()

        self.envs = envs
        self.num_timesteps = num_timesteps
        self.num_agents = num_agents # Only used for MAGIC

        self._load_network_model(model_config=model_config)

        self.critic = nn.Sequential( # Critic network, takes the actor output as input
            layer_init(nn.Linear(self.network.output_dim, 1), std=1.0),
        )

    def get_value(self, x, timestep: int) -> Tensor:
        return self.critic(self.network(x, timestep=timestep))

    def _load_network_model(self, model_config: NetworkModelConfig):
        model_type = model_config.model.network_model

        if model_type == NetworkModelNames.MAGIC:
            self.network = MagicNetwork(
                magic_network_config=model_config.model,
                input_dim=self.envs.single_observation_space.shape[0],
                num_agents=self.num_agents,
                num_timesteps=self.num_timesteps,
            )
        elif model_type == NetworkModelNames.MLP:
            self.network = MLP(
                mlp_config=model_config.model.mlp,
                input_dim=self.envs.single_observation_space.shape[0],
            )
        else:
            raise NotImplementedError(
                f"{model_type} is not supported for the network model."
            )

class AgentDiscrete(AgentBase):
    def __init__(self, envs, *args, **kwargs):
        super().__init__(envs, *args, **kwargs)
        self.action_nvec = tuple(envs.single_action_space.nvec)
        self.actor = nn.Sequential(
            layer_init(
                nn.Linear(self.network.output_dim, sum(self.action_nvec)), std=0.01
            ),
        )

    def get_action_and_value(
        self,
        x,
        timestep: int,
        update_lstm_hidden_state: bool = False,
        action=None,
        action_mask=None,
    ):
        network_output = self.network(
            x, timestep=timestep, update_lstm_hidden_state=update_lstm_hidden_state
        )

        action_logits = self.actor(network_output)

        # Apply the action mask
        if action_mask is not None:
            # Set logits of invalid actions to a very negative number (effectively -inf)
            action_logits = action_logits.masked_fill(action_mask == 0, -1e9)

        # Get probabilities
        probs = MultiCategorical(action_logits, self.action_nvec)

        if action is None:
            action = probs.sample()

        return (
            action,
            probs.log_prob(action),
            probs.entropy(),
            self.critic(self.network(x, timestep=timestep)),
        )