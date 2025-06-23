from pydantic import BaseModel, Field
from typing import Optional, List, Literal, Union
from rl.nn.enumerations import Activations, NetworkModelNames


class MLPBlock(BaseModel):
    """This defines the configuration for an MLPBlock.

    mlp_hidden_dims: Optional[List[int]]
        The hidden dimensions in the MLP. Can be empty for single layer. PReLU is applied after every hidden layer.
    output_dim: int
        The output dim for the MLP block
    use_layer_norm: bool
        Whether to use layer norm or not. It is applied after every layer in the MLP.
    activation: Activations
        The activation function to use after every linear layer.
    """

    output_dim: int
    use_layer_norm: bool = True
    mlp_hidden_dims: Optional[List[int]] = None
    activation: Optional[Activations] = Activations.RELU


class LSTMCell(BaseModel):
    """This defines the configuration for an LSTM Cell. Input dim is automatically taken from the previos layer.
    hidden_dim: int
        The size of the hidden dimension of the LSTM.
    """

    hidden_dim: int


class GraphAttentionBlock(BaseModel):
    """This defines the configuration for the graph attention block.

    out_features (int): number of features in each output node
    dropout (int/float): dropout probability for the coefficients
    negative_slope (int/float): control the angle of the negative slope in leakyrelu
    number_heads (int): number of heads of attention
    bias (bool): if adding bias to the output
    self_loop_type (int): 0 -- force no self-loop; 1 -- force self-loop; other values (2)-- keep the input adjacency matrix unchanged
    average (bool): if averaging all attention heads
    normalize (bool): if normalizing the coefficients after zeroing out weights using the communication graph
    """

    out_features: int
    dropout: float
    negative_slope: float
    number_heads: int
    bias: bool = True
    self_loop_type: int = 1
    average: bool = True
    normalize: bool = False


class EncoderConfig(BaseModel):
    """
    Following the implementation of MAGIC:

    obs_encoder: MLPBlock
        This is the observation encoder. In the paper, they use a simple Linear Layer. MLP blocks allows for multiple layers as well.
    lstm_cell: Optional[LSTMCell]
        This is an optional LSTM cell that allows for observations to be encoded through time.
    message_encoder: Optional[MLPBlock]
        This is an optional message encoder as in MAGIC. Similar to obs_encoder, MLP block allows for multiple layers.
    """

    obs_encoder: MLPBlock
    lstm_cell: Optional[LSTMCell] = None
    message_encoder: Optional[MLPBlock] = None


class SubSchedulerConfig(BaseModel):
    """
    The scheduler learns the adjcency matrix for agent communication. It learns "who to communicate with".

    learn_graph: bool
        If set to True, the graph is learnt. Otherwise, a fully connected graph is returned.
    learn_graph: bool
        Whether or not to learn a directed graph.
    gnn_encoder: Optional[GraphAttentionBlock]
        The config for the graph block used in the sub-scheduler.
    mlp: MLPBlock
        The config for the MLP in the sub-scheduler.

    """

    learn_graph: bool
    directed: bool
    gnn_encoder: Optional[GraphAttentionBlock]
    mlp: MLPBlock


class SubProcessorConfig(BaseModel):
    """
    The prcoessor learns the message passing in MARL communication. It learns "what to communicate".

    gnn_encoder: GraphAttentionBlock
        This is the graph block that learns the message passing.

    """

    gnn_encoder: GraphAttentionBlock


class MagicNetworkConfig(BaseModel):
    """
    In MAGIC's implementation, the observation sequentially passes through sub-schedulers and sub-processors.
    This defines the entire network model using sub-schedulers and sub-processors. The list allows multi-round
    communication as implemented in MAGIC.

    encoder: EncoderConfig
        Module for encoding observations

    schedulers: List[SubSchedulerConfig]
        A list of schedulers

    processors: List[SubProcessorConfig]
        A list of processors

    decoder: Optional[MLPBlock]
        Optionally, set a decoder which is just an MLP block.

    The output at each
    """

    network_model: Literal[NetworkModelNames.MAGIC]
    encoder: EncoderConfig
    schedulers: List[SubSchedulerConfig]
    processors: List[SubProcessorConfig]
    decoder: Optional[MLPBlock]


class MLPModelConfig(BaseModel):
    """This is the configuration for a simple MLP model."""

    network_model: Literal[NetworkModelNames.MLP]
    mlp: MLPBlock


class NetworkModelConfig(BaseModel):
    model: Union[MagicNetworkConfig, MLPModelConfig] = Field(
        ..., discriminator="network_model"
    )
    
    class Config:
        use_enum_values = True
        smart_union = True
