from enum import Enum


class Activations(str, Enum):
    """
    Enumeration for activation functions.
    """

    RELU = "relu"
    PRELU = "prelu"


class NetworkModelNames(str, Enum):
    """
    Enumerations for the model types we support for the netork model
    """

    MAGIC = "magic"
    MLP = "mlp"
