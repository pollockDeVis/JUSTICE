from ray.rllib.algorithms.algorithm import Algorithm
from rl.marl.ray_wrapper import config, JusticeEnv
from ray.tune.registry import register_env
import ray
from train import env_creator
import wandb
import numpy as np
ray.init(local_mode=True, ignore_reinit_error= True)

#this is a placeholder can be replaced with script argument
checkpoint_path = "/Users/pwozny/Repos/justice/JUSTICE/rl/marl/checkpoints/cp_01-17-2024_18:13:55"

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

    wandb_api_key = "91f4b56e70eb59889967350b045b94cd0d7bcaa8"
    wandb.login(key=wandb_api_key)
    wandb.init(
                project="test",
                name='test_eval',
                entity="justice-rl",
            )

    outputs = {
            "actions":[],
            "obs":[],
            "reward":[]
        }
    
    episode_reward = 0
    terminated = truncated = False
    algo, env = load_saved_model(checkpoint_path)

    obs, info = env.reset()
    #generate eval rollout
    while not terminated:
        actions = get_actions(algo,obs)
        outputs["actions"].append(actions)
        obs, reward, terminated, truncated, info = env.step(actions)
        outputs["reward"].append(reward)
        outputs["obs"].append(obs)
        terminated = terminated["__all__"]

    OBSERVATIONS = ['net_economic_output', 'emissions', 'regional_temperature', 'economic_damage', 'abatement_cost']
    REWARD="welfare_utilitarian_temporal"
    GLOBAL_OBSERVATIONS = ['global_temperature']

    def invert_dict(obs_dict, key):
        output = {}
        output[key] = {agent:obs_dict[agent][key][0] for agent in obs_dict.keys()}
        return output

    #agent observations
    for observation in OBSERVATIONS:
        xs = []
        ys  = []
        for idx, step in enumerate(outputs["obs"]):
            obstep = invert_dict(step, observation)
            step_values = [x for x in obstep[observation].values()]
            xs.append(idx)
            ys.append(step_values)
            mean_obs = np.mean(step_values)
            wandb.log({f"{observation}_average": mean_obs})
        wandb.log({observation:wandb.plot.line_series(
            xs=xs,
            ys=np.array(ys).T,
            keys=list(step.keys()),
            title=observation)
            })

    #agent reward
    xs = []
    ys  = []
    for idx, step in enumerate(outputs["reward"]):
        reward_values = [x for x in step.values()]
        xs.append(idx)
        ys.append(reward_values)
        mean_reward = np.mean(reward_values)
        wandb.log({"reward_average": mean_reward})
    wandb.log({observation:wandb.plot.line_series(
            xs=xs,
            ys=np.array(ys).T,
            keys=list(step.keys()),
            title="Reward")
            })


    #global
    for observation in GLOBAL_OBSERVATIONS:
        for step in outputs["obs"]:
            gt = step["agent_0"][observation][0]
            wandb.log({"global_temp":gt})

    #actions
    savings_ys = []
    emissions_ys = []
    xs = []
    for idx, step in enumerate(outputs["actions"]):
        savings_rates = invert_dict(step, "savings_rate")["savings_rate"]
        savings_rate_values = [x for x in savings_rates.values()]
        mean_savings_rate = np.mean(savings_rate_values)
        savings_ys.append(savings_rate_values)
        xs.append(idx)
        wandb.log({"savings_rate_average":mean_savings_rate})
        emissions_rates = invert_dict(step, "emissions_rate")["emissions_rate"]
        emission_rate_values = [x for x in emissions_rates.values()]
        mean_emissions_rate = np.mean(emission_rate_values)
        emissions_ys.append(emission_rate_values)
        wandb.log({"emissions_rate_average":mean_emissions_rate})

    wandb.log({"Savings Rates":wandb.plot.line_series(
            xs=xs,
            ys=np.array(savings_ys).T,
            keys=list(step.keys()),
            title="Savings Rates")
            })


    wandb.log({"Emissions Rates":wandb.plot.line_series(
            xs=xs,
            ys=np.array(emissions_ys).T,
            keys=list(step.keys()),
            title="Emissions Rates")
            })


    wandb.finish()
    