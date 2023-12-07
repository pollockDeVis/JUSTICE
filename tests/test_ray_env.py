import pytest
from rl.marl.ray_wrapper import JusticeEnv
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

def test_step(env: JusticeEnv):
    action = {
        "savings_rate":0.2,
        "emissions_rate":0.5
    }
    actions = {f"agent_{i}":action for i in range(env.num_agents)}
    for i in range(100):
        observations, rewards, terminated, truncateds, infos = env.step(actions)
