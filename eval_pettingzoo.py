import time
from rl.marl.pettingzoo_wrapper import JusticeEnv, CONFIG
import supersuit as ss
import gymnasium as gym
from train_ppo_pettingzoo import Agent
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import tyro
import torch
import os
import numpy as np
import wandb
#glboals
NUM_ENVS = 1

wandb_api_key = "91f4b56e70eb59889967350b045b94cd0d7bcaa8"
wandb.login(key=wandb_api_key)
wandb.init(
            project="test_cleanrl",
            name='test_eval',
            entity="justice-rl",
        )

CHECKPOINT = Path("rl") / "marl" / "checkpoints" / "train_ppo_pettingzoo_2024-01-24_16:34:51.pt"

OBSERVATIONS = [
    "net_economic_output",
    "emissions",
    "regional_temperature",
    "economic_damage",
    "abatement_cost",
]
GLOBAL_OBSERVATIONS = ["global_temperature"]
ALL_OBSERVATIONS = OBSERVATIONS + GLOBAL_OBSERVATIONS

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]\
        +"_"+datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "JusticeEnv"
    """the wandb's project name"""
    wandb_entity: str = "brenting"
    """the entity (team) of wandb's project"""

    # Algorithm specific arguments
    total_timesteps: int = 50000000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 2
    """the number of parallel game environments"""
    num_steps: int = 286
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 20
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""

# env setup
JusticeEnv.pickle_model(CONFIG) # needed to pickle the JUSTICE model
env = JusticeEnv(CONFIG)
env = ss.pettingzoo_env_to_vec_env_v1(env)
env = gym.wrappers.ClipAction(env)
envs = ss.concat_vec_envs_v1(env, NUM_ENVS, num_cpus=NUM_ENVS, base_class="gymnasium")
envs.single_observation_space = envs.observation_space
envs.single_action_space = envs.action_space
envs.is_vector_env = True



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

agent = Agent(envs).to(device)
agent.load_state_dict(torch.load(CHECKPOINT))
agent.eval()
# TRY NOT TO MODIFY: start the game
args = tyro.cli(Args)
global_step = 0
start_time = time.time()
next_obs, _ = envs.reset(seed=args.seed)
next_obs = torch.Tensor(next_obs).to(device)
next_done = torch.zeros(envs.num_envs).to(device)

# ALGO Logic: Storage setup
obs: torch.Tensor = torch.zeros((args.num_steps, envs.num_envs) + envs.single_observation_space.shape).to(device)
actions: torch.Tensor = torch.zeros((args.num_steps, envs.num_envs) + envs.single_action_space.shape).to(device)
logprobs = torch.zeros((args.num_steps, envs.num_envs)).to(device)
rewards = torch.zeros((args.num_steps, envs.num_envs)).to(device)
dones = torch.zeros((args.num_steps, envs.num_envs)).to(device)
values = torch.zeros((args.num_steps, envs.num_envs)).to(device)

def remap_observation(observation):
    output = {}
    for idx, observation_name in enumerate(ALL_OBSERVATIONS):
        output[observation_name] = observation[idx].cpu().numpy()
    return output

def invert_dict(obs_dict, key):
    output = {}
    output[key] = {agent:obs_dict[agent][key] for agent in obs_dict.keys()}
    return output

outputs = {
        "actions":[],
        "obs":[],
        "reward":[]
    }

#eval loop
for step in range(0, args.num_steps):
    global_step += envs.num_envs
    obs[step] = next_obs
    dones[step] = next_done

    # ALGO LOGIC: action logic
    with torch.no_grad():
        action, logprob, _, value = agent.get_action_and_value(next_obs)
        values[step] = value.flatten()
    actions[step] = action
    logprobs[step] = logprob
    
    
    next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
    next_done_bool = np.logical_or(terminations, truncations)
    rewards[step] = torch.tensor(reward).to(device).view(-1)
    next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done_bool).to(device)
    remapped_observations = {f"agent_{idx}": remap_observation(next_obs[idx,:]) for idx in range(next_obs.shape[0])}
    outputs["obs"].append(remapped_observations)
    outputs["reward"].append({f"agent_{idx}":reward[idx] for idx in range(next_obs.shape[0])})

    
    outputs["actions"].append({
        f"agent_{idx}":{
            "savings_rate":np.clip(action.cpu().numpy()[idx,0], a_min=0.00001, a_max=.99999),
            "emissions_rate":np.clip(action.cpu().numpy()[idx,1], a_min=0.00001, a_max=.99999)
        } for idx in range(next_obs.shape[0])
    })




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
    
#global
for observation in GLOBAL_OBSERVATIONS:
    for step in outputs["obs"]:
        gt = step["agent_0"][observation]
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

    
    

