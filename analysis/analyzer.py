"""
This module contains the uncertainty analysis for the JUSTICE model using EMA Workbench.
"""

from ema_workbench import (
    Model,
    RealParameter,
    ScalarOutcome,
    Constant,
    perform_experiments,
    ema_logging,
    ArrayOutcome,
)
