import pytest
from rl.marl.wrap_env import JusticeEnv
from src.util.enumerations import Economy, DamageFunction, Abatement, WelfareFunction


@pytest.fixture
def env() -> JusticeEnv:
    env_config = {
        "start_year": 2015,
        "end_year": 2300,
        "timestep": 1,
        "scenario": 0,
        "economy_type": Economy.NEOCLASSICAL,
        "damage_function_type": DamageFunction.KALKUHL,
        "abatement_type": Abatement.ENERDATA,
        "social_welfare_function": WelfareFunction.UTILITARIAN,
        "num_agents": 57,
    }

    return JusticeEnv(env_config)


def test_smoke(env: JusticeEnv):
    env


def test_reset(env: JusticeEnv):
    obs, rew = env.reset()
