import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm as LinearLayerNorm
from rl.nn.configs import (
    MLPBlock,
    EncoderConfig,
    SubSchedulerConfig,
    SubProcessorConfig,
    MagicNetworkConfig,
)
from rl.nn.utils import get_activation_function_from_enum
from rl.nn.gnn_model import GraphAttention

class MLP(nn.Module):
    def __init__(self, mlp_config: MLPBlock, input_dim):
        super().__init__()
        hidden_dims = mlp_config.mlp_hidden_dims
        self.output_dim = mlp_config.output_dim

        self.activation = get_activation_function_from_enum(
            activation=mlp_config.activation
        )

        self.MLP = nn.ModuleList()
        in_features_for_last_layer = input_dim
        if hidden_dims:
            self.MLP.extend(
                [
                    nn.Linear(
                        in_features=input_dim,
                        out_features=hidden_dims[0],
                    ),
                    self.activation,
                ]
            )

            for h_index in range(1, len(hidden_dims)):
                self.MLP.extend(
                    [
                        nn.Linear(
                            in_features=hidden_dims[h_index - 1],
                            out_features=hidden_dims[h_index],
                        ),
                        self.activation,
                    ]
                )
            in_features_for_last_layer = hidden_dims[-1]

        self.MLP.append(
            nn.Linear(
                in_features=in_features_for_last_layer, out_features=self.output_dim
            )
        )

        if mlp_config.use_layer_norm:
            self.MLP.append(LinearLayerNorm([self.output_dim]))

    def forward(self, X: torch.Tensor, *args, **kwargs):
        for layer in self.MLP:
            X = layer(X)

        return X


class Encoder(nn.Module):
    def __init__(
        self,
        encoder_config: EncoderConfig,
        input_dim: int,
        num_timesteps: int,
        num_agents: int,
    ):
        super().__init__()
        self.obs_encoder = MLP(
            mlp_config=encoder_config.obs_encoder, input_dim=input_dim
        )
        self.lstm_cell = None
        self.message_encoder = None
        self.num_agents = num_agents

        self.output_dim = self.obs_encoder.output_dim

        if encoder_config.lstm_cell:
            self.lstm_hidden_dim = encoder_config.lstm_cell.hidden_dim
            self.lstm_cell = nn.LSTMCell(
                self.lstm_hidden_dim,
                self.output_dim,
            )
            self.lstm_hidden_states = torch.zeros(
                num_timesteps, num_agents, self.lstm_hidden_dim
            )
            self.lstm_cell_states = torch.zeros(
                num_timesteps, num_agents, self.lstm_hidden_dim
            )

            self.output_dim = encoder_config.lstm_cell.hidden_dim

        if encoder_config.message_encoder:
            self.message_encoder = MLP(
                mlp_config=encoder_config.message_encoder, input_dim=self.output_dim
            )
            self.output_dim = self.message_encoder.output_dim

    def forward(
        self, X: torch.Tensor, timestep: int, update_lstm_hidden_state: bool = False
    ):
        obs = self.obs_encoder(X)

        if self.lstm_cell:
            if timestep == 0:
                previous_hidden_state = torch.zeros(
                    self.num_agents, self.lstm_hidden_dim
                )
                previous_cell_state = torch.zeros(self.num_agents, self.lstm_hidden_dim)
            else:
                previous_hidden_state = self.lstm_hidden_states[timestep - 1]
                previous_cell_state = self.lstm_cell_states[timestep - 1]

            new_hidden_state, new_cell_state = self.lstm_cell(
                obs, (previous_hidden_state, previous_cell_state)
            )

            if update_lstm_hidden_state:
                self.lstm_hidden_states[timestep] = new_hidden_state.squeeze(0)
                self.lstm_cell_states[timestep] = new_cell_state.squeeze(0)

        if self.message_encoder:
            obs = self.message_encoder(obs)

        return obs


class SubScheduler(nn.Module):
    def __init__(
        self, sub_scheduler_config: SubSchedulerConfig, input_dim: int, num_agents: int
    ):
        super().__init__()

        self.num_agents = num_agents
        self.learn_graph = sub_scheduler_config.learn_graph
        self.directed = sub_scheduler_config.directed
        self.gnn_encoder = None

        if sub_scheduler_config.gnn_encoder:
            self.gnn_encoder = GraphAttention(
                graph_config=sub_scheduler_config.gnn_encoder, input_dim=input_dim
            )
            self.output_dim = self.gnn_encoder.out_features * 2

        self.mlp = MLP(mlp_config=sub_scheduler_config.mlp, input_dim=self.output_dim)

    def get_complete_graph(self):
        """
        Function to generate a complete graph.
        """
        n = self.num_agents
        adj = torch.ones(n, n)
        return adj

    def forward(self, X: torch.Tensor):

        # Return fully connected graph if the graph is not supposed to be dynamically learnt
        if not self.learn_graph:
            return self.get_complete_graph()

        encoding = X

        if self.gnn_encoder:
            adj_complete = self.get_complete_graph()
            encoding = self.gnn_encoder(X=encoding, adj=adj_complete)

        n = self.num_agents
        hid_size = encoding.size(-1)

        # hard_attn_input: [n * n * (2*hid_size)]
        hard_attn_input = torch.cat(
            [encoding.repeat(1, n).view(n * n, -1), encoding.repeat(n, 1)], dim=1
        ).view(n, -1, 2 * hid_size)

        # hard_attn_output: [n * n * 2]
        if self.directed:
            hard_attn_output = F.gumbel_softmax(self.mlp(hard_attn_input), hard=True)
        else:
            hard_attn_output = F.gumbel_softmax(
                0.5 * self.mlp(hard_attn_input)
                + 0.5 * self.mlp(hard_attn_input.permute(1, 0, 2)),
                hard=True,
            )

        # hard_attn_output: [n * n * 1]
        hard_attn_output = torch.narrow(hard_attn_output, 2, 1, 1)

        # adj: [n * n]
        adj = hard_attn_output.squeeze()

        return adj


class SubProcessor(nn.Module):
    def __init__(self, sub_processor_config: SubProcessorConfig, input_dim: int):
        super().__init__()

        self.gnn_encoder = GraphAttention(
            graph_config=sub_processor_config.gnn_encoder, input_dim=input_dim
        )
        self.output_dim = self.gnn_encoder.out_features

    def forward(self, X: torch.Tensor, adj: torch.Tensor):

        return self.gnn_encoder(X=X, adj=adj)


class MagicNetwork(nn.Module):
    def __init__(
        self,
        magic_network_config: MagicNetworkConfig,
        input_dim: int,
        num_agents: int,
        num_timesteps: int,
    ):
        super().__init__()

        self.num_comminucation_rounds = len(magic_network_config.processors)

        self.encoder = Encoder(
            encoder_config=magic_network_config.encoder,
            input_dim=input_dim,
            num_timesteps=num_timesteps,
            num_agents=num_agents,
        )
        input_dims = [
            (
                self.encoder.output_dim
                if round == 0
                else magic_network_config.processors[round - 1].gnn_encoder.out_features
            )
            for round in range(self.num_comminucation_rounds)
        ]

        self.schedulers = [
            SubScheduler(
                sub_scheduler_config=magic_network_config.schedulers[i],
                input_dim=input_dims[i],
                num_agents=num_agents,
            )
            for i in range(self.num_comminucation_rounds)
        ]
        self.processors = [
            SubProcessor(
                sub_processor_config=magic_network_config.processors[i],
                input_dim=input_dims[i],
            )
            for i in range(self.num_comminucation_rounds)
        ]

        self.sub_schedulers_density = [[] for _ in range(self.num_comminucation_rounds)]

        self.output_dim = self.processors[-1].output_dim
        self.decoder = None

        if magic_network_config.decoder:
            self.decoder = MLP(
                mlp_config=magic_network_config.decoder,
                input_dim=self.processors[-1].output_dim,
            )
            self.output_dim = self.decoder.output_dim

        # Since final output is concatenation of LSTM hidden state and the processor/decoder output
        if self.encoder.lstm_cell:
            self.output_dim = self.output_dim + self.encoder.lstm_hidden_dim

    def _calculate_density(self, adj_matrix):
        adj_matrix = adj_matrix.cpu().detach().numpy()

        total_elements = adj_matrix.size
        one_elements = np.count_nonzero(adj_matrix == 1)
        sparsity = one_elements / total_elements
        return sparsity

    def forward(
        self, X: torch.Tensor, timestep: int, update_lstm_hidden_state: bool = False
    ):
        X = X.squeeze(0)
        comm = self.encoder(
            X, timestep=timestep, update_lstm_hidden_state=update_lstm_hidden_state
        )

        for round in range(self.num_comminucation_rounds):
            adj = self.schedulers[round](X=comm)
            density = self._calculate_density(adj_matrix=adj)
            self.sub_schedulers_density[round].append(density)
            comm = self.processors[round](X=comm, adj=adj)

        if self.decoder:
            comm = self.decoder(comm)

        if self.encoder.lstm_cell:
            comm = torch.cat([self.encoder.lstm_hidden_states[timestep], comm], dim=-1)

        return comm
