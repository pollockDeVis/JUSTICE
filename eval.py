from ray.rllib.algorithms.algorithm import Algorithm
from rl.marl.ray_wrapper import config, JusticeEnv
from ray.tune.registry import register_env
import ray
from train import env_creator
ray.init(local_mode=True, ignore_reinit_error= True)

#this is a placeholder
checkpoint_path = "/Users/pwozny/Repos/justice/JUSTICE/rl/marl/checkpoints/cp_12-22-2023_10:36:49"

def load_saved_model(checkpoint_path):
    register_env("justice", env_creator)
    algo = Algorithm.from_checkpoint(checkpoint_path)
    env=JusticeEnv(config)
    return algo, env

def get_actions(algo, obs):
    actions = {}
    for agent_id in range(env.num_agents):
        actions[f"agent_{agent_id}"] = algo.compute_single_action(obs[f"agent_{agent_id}"])
    return actions
    
if __name__ == "__main__":

    episode_reward = 0
    terminated = truncated = False
    algo, env = load_saved_model(checkpoint_path)
    obs, info = env.reset()

    while not terminated:
        actions = get_actions(algo,obs)
        obs, reward, terminated, truncated, info = env.step(actions)
        episode_reward += reward["agent_0"]
        terminated = terminated["__all__"]
        print(episode_reward, terminated)
    