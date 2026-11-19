"""Tests for the annual growth constraint on the emission control rate."""

import numpy as np

from justice.util.emission_control_constraint import EmissionControlConstraint


def constrain_two_years(first_year, second_year):
    """Constrain two consecutive years for two regions, starting control in year 0."""
    constraint = EmissionControlConstraint(
        max_annual_growth_rate=0.04,
        emission_control_start_timestep=0,
        min_emission_control_rate=0.01,
    )
    # In the start year the rates are rescaled to [0.01, 0.04]
    constraint.constrain_emission_control_rate(np.array(first_year, dtype=float), 0)
    second = np.array(second_year, dtype=float)
    constrained = constraint.constrain_emission_control_rate(second, 1)
    return constrained.ravel(), second.ravel()


def test_increases_below_the_cap_are_applied():
    constrained, _ = constrain_two_years([[0.0], [1.0]], [[0.03], [0.06]])
    np.testing.assert_allclose(constrained, [0.03, 0.06])


def test_increases_above_the_cap_are_capped():
    constrained, _ = constrain_two_years([[0.0], [1.0]], [[0.03], [0.20]])
    np.testing.assert_allclose(constrained, [0.03, 0.08])


def test_decreases_are_held():
    constrained, _ = constrain_two_years([[0.0], [1.0]], [[0.00], [0.05]])
    np.testing.assert_allclose(constrained, [0.01, 0.05])


def test_input_is_not_modified():
    _, second = constrain_two_years([[0.0], [1.0]], [[0.03], [0.20]])
    np.testing.assert_allclose(second, [0.03, 0.20])
