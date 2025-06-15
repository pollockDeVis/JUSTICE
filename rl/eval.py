"""

#######################
## EVALUATION SCRIPT ##
#######################

This script can be used to evaluate a trained policy on a given evaluation seed and for a particular weight combination.

"""


import random, numpy as np, torch, supersuit as ss, json, re

from dataclasses import asdict
from paper.eval_utils import *
from rl.env import JusticeMOMARLEnv
from rl.agent import AgentDiscrete
from rl.args import Args
from copy import deepcopy
from momaland.utils.parallel_wrappers import (
    LinearizeReward,
    NormalizeReward,
    RecordEpisodeStatistics,
)
from rl.nn.configs import NetworkModelConfig
from argparse import ArgumentParser
from rl.args import Args

from pathlib import Path

OBSERVATIONS = [
    "net_economic_output",
    "emissions",
    "regional_temperature",
    "economic_damage",
    "abatement_cost",
]
GLOBAL_OBSERVATIONS = ["global_temperature"]

ALL_OBSERVATIONS = OBSERVATIONS + GLOBAL_OBSERVATIONS

def remap_observation(observation):
    output = {}
    for idx, observation_name in enumerate(ALL_OBSERVATIONS):
        output[observation_name] = observation[idx].cpu().numpy()
    return output

def invert_dict(obs_dict, key):
    output = {}
    output[key] = {agent: obs_dict[agent][key] for agent in obs_dict.keys()}
    return output

def set_random_seeds(seed: int):

    """

    Set the random seeds for reproducibility.

    @param seed: The seed to set

    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False

def evaluate_seed(base_emissions, loaded_args: Args, checkpoint_path: str, eval_seed: int):

    """
    Evaluate a trained policy on a given seed.

    @param base_emissions: The base emissions data for abated emissions calculations
    @param loaded_args: The configuration arguments for the evaluation
    @param checkpoint_path: The path to the checkpoint file
    @param eval_seed: The seed to evaluate the policy on

    """

    set_random_seeds(eval_seed)
    
    env_config = asdict(loaded_args.env_config) # Prepare the evaluation environment

    loaded_args.num_steps = 285 # Full JUSTICE simulation
    
    # Obtain the weights combination from the name of the checkpoint path, as a regex matching w=[]
    match = re.search(r"w=\[([^\]]+)\]", checkpoint_path)
    weights_str = match.group(1)
    weights = [float(w) for w in weights_str.split()]

    # Initialize the environment and normalize the rewards
    jenv = JusticeMOMARLEnv(env_config)
    env = deepcopy(jenv)
    for agent in env.possible_agents:
        for idx in range(env.unwrapped.reward_space(agent).shape[0]):
            env = NormalizeReward(env, agent, idx)
    _weights = {agent: weights for agent in env.possible_agents}
    env = LinearizeReward(env, _weights)
    env = RecordEpisodeStatistics(env)
        
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    
    envs = ss.concat_vec_envs_v1(
        env, loaded_args.num_envs, num_cpus=loaded_args.num_envs, base_class="gymnasium"
    )
    envs.single_observation_space = envs.observation_space
    envs.single_action_space = envs.action_space
    envs.is_vector_env = True
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    Agent = AgentDiscrete

    with open(loaded_args.network_model_config) as f:
        network_model_config = json.load(f)

    network_model_config = NetworkModelConfig(**network_model_config)

    agent = Agent(
        envs,
        network_model_config,
        num_timesteps=loaded_args.num_steps,
        num_agents=env_config["num_agents"],
    ).to(device)

    agent.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    agent.eval()

    # TRY NOT TO MODIFY: start the game
    next_obs, infos = envs.reset(seed=loaded_args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(envs.num_envs).to(device)

    # ALGO Logic: Storage setup
    obs: torch.Tensor = torch.zeros(
        (loaded_args.num_steps, envs.num_envs) + envs.single_observation_space.shape
    ).to(device)
    actions: torch.Tensor = torch.zeros(
        (loaded_args.num_steps, envs.num_envs) + envs.single_action_space.shape
    ).to(device)
    logprobs = torch.zeros((loaded_args.num_steps, envs.num_envs)).to(device)
    rewards = torch.zeros((loaded_args.num_steps, envs.num_envs)).to(device)
    dones = torch.zeros((loaded_args.num_steps, envs.num_envs)).to(device)
    values = torch.zeros((loaded_args.num_steps, envs.num_envs)).to(device)

    outputs = {"actions": [], "obs": [], "reward": []}
    
    # Evaluation Loop
    for step in range(0, loaded_args.num_steps):

        obs[step] = next_obs
        dones[step] = next_done

        action_masks = (
            torch.tensor(
                np.array([info.get("action_mask") for info in infos]), dtype=torch.int32
            ).to(device)
            if "action_mask" in infos[0]
            else None
        )
        
        with torch.no_grad():
            action, logprob, _, value = agent.get_action_and_value(
                next_obs,
                action_mask=action_masks,
                timestep=step,
                update_lstm_hidden_state=True,
            )
            values[step] = value.flatten()
        
        actions[step] = action
        logprobs[step] = logprob

        next_obs, reward, terminations, truncations, infos = envs.step(
            action.cpu().numpy()
        )

        next_done_bool = np.logical_or(terminations, truncations)
        rewards[step] = torch.tensor(reward).to(device).view(-1)
        next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(
            next_done_bool
        ).to(device)

        remapped_observations = {
            f"agent_{idx}": remap_observation(next_obs[idx, :])
            for idx in range(next_obs.shape[0])
        }

        # Extract the env.data array from the environment

        outputs["obs"].append(remapped_observations)
        outputs["reward"].append(
            {f"agent_{idx}": reward[idx] for idx in range(next_obs.shape[0])}
        )

        outputs["actions"].append(
            {
                f"agent_{idx}": {
                    "savings_rate": jenv.discrete_to_float(action[idx, 0]),
                    "emissions_rate": jenv.discrete_to_float(action[idx, 1]),
                }
                for idx in range(next_obs.shape[0])
            }
        )
    
    num_agents = env_config["num_agents"]

    emissions_data = np.zeros([num_agents, loaded_args.num_steps])
    abated_emissions_data = np.zeros([num_agents, loaded_args.num_steps])
    economic_data = np.zeros([num_agents, loaded_args.num_steps])
    global_temperature_data = np.zeros([loaded_args.num_steps])
    regional_temperature_data = np.zeros([num_agents, loaded_args.num_steps])
    savings_rates_data = np.zeros([num_agents, loaded_args.num_steps])
    emission_control_rates_data = np.zeros([num_agents, loaded_args.num_steps])
    
    for timestep in range(0, loaded_args.num_steps):
        
        step_action = outputs["actions"][timestep]

        emission_control_rates = invert_dict(step_action, "emissions_rate")["emissions_rate"]
        emission_control_rate_values = [x for x in emission_control_rates.values()]
        
        savings_rates = invert_dict(step_action, "savings_rate")["savings_rate"]
        savings_rate_values = [x for x in savings_rates.values()]

        global_temperature_data[timestep] = float(outputs["obs"][timestep]["agent_0"]["global_temperature"]) # Same for all agents
        
        for agent_index in range(num_agents):
            emissions_data[agent_index][timestep] = float(
                outputs["obs"][timestep][f"agent_{agent_index}"]["emissions"]
            )
            economic_data[agent_index][timestep] = float(
                outputs["obs"][timestep][f"agent_{agent_index}"]["net_economic_output"]
            )
            regional_temperature_data[agent_index][timestep] = float(
                outputs["obs"][timestep][f"agent_{agent_index}"]["regional_temperature"]
            )
            abated_emissions_data[agent_index][timestep] = float(emission_control_rate_values[agent_index].numpy()) * float(base_emissions[agent_index][timestep])
            
            savings_rates_data[agent_index][timestep] = savings_rate_values[agent_index]
            emission_control_rates_data[agent_index][timestep] = emission_control_rate_values[agent_index]
        
    
    eval_seed_data = {
        "emissions": emissions_data,
        "abated_emissions": abated_emissions_data,
        "net_economic_output": economic_data,
        "global_temperature": global_temperature_data,
        "regional_temperature": regional_temperature_data,
        "savings_rates": savings_rates_data,
        "emission_control_rates": emission_control_rates_data
    }
    
    return eval_seed_data

if __name__ == "__main__":

    CURRENT_WORKING_DIR = os.path.dirname(os.path.realpath(__file__))
    
    # Take a checkpoint path as input from command line
    parser = ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str,required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--seed", default=0, type=int, required=False)
    
    args = parser.parse_args()

    # Load the configuration arguments
    loaded_args = Args()

    # Load base emissions for abated emissions calculations
    baselines_emissions_data_path = os.path.join(CURRENT_WORKING_DIR, "paper", "baseline_emissions.csv")
    base_emissions = pd.read_csv(baselines_emissions_data_path, delimiter=";")
    base_emissions = base_emissions.iloc[:, 1:]
    base_emissions = base_emissions.to_numpy()

    eval_seed_data = evaluate_seed(base_emissions, loaded_args, args.checkpoint_path, args.seed)

    # Save the evaluation data
    eval_seed_data_path = Path(args.output_path) / f"eval_seed_{args.seed}.pkl"

    with open(eval_seed_data_path, "wb") as f:
        pickle.dump(eval_seed_data, f)

    



