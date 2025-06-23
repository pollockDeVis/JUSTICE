""" 

#####################
## TRAINING SCRIPT ##
#####################

This script trains the MOMARL algorithm on JUSTICE

"""

import random, numpy as np, tyro, numpy as np, wandb, torch

from dataclasses import asdict
from datetime import datetime
from pathlib import Path


from rl.moma_ppo import train_momappo
from rl.args import Args

from torch.utils.tensorboard import SummaryWriter

def set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def run_training(args):

    # If base_save_path does not exist, create it
    Path(args.base_save_path).mkdir(parents=True, exist_ok=True)

    # If pickles folder in base_save_path does not exist, create it
    Path(args.base_save_path, "pickles").mkdir(parents=True, exist_ok=True)

    # If policies folder in base_save_path does not exist, create it
    Path(args.base_save_path, "checkpoints").mkdir(parents=True, exist_ok=True)

    # If results folder in base_save_path does not exist, create it
    Path(args.base_save_path, "artefacts").mkdir(parents=True, exist_ok=True)
    
    assert (
        args.env_config.num_agents == 57
        or (args.env_config.num_agents == 12 and args.env_config.clustering)
        or (args.env_config.num_agents == 5 and args.env_config.clustering)
    ), "You either use all the 57 agents or use the clustering mechanism with 12 agents or 5 agents"

    # ensure a full episode in the batch
    num_model_steps = (
        args.env_config.end_year - args.env_config.start_year
    ) // args.env_config.num_years_per_step

    args.num_steps = (
        (
            3
            * (args.env_config.end_year - args.env_config.start_year)
            // args.env_config.num_years_per_step
        )
        - 2
        if args.env_config.negotiation_on
        else num_model_steps
    )

    args.batch_size = args.num_envs * num_model_steps

    args.minibatch_size = args.batch_size // args.num_minibatches

    writer = SummaryWriter(f"runs/{args.exp_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    set_seeds(seed=args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    torch.backends.cudnn.benchmark = False

    args.env_config.config_pickle_path = Path(args.base_save_path) / "pickles"
   
    train_momappo(args=args, writer=writer)

if __name__ == "__main__":

    args = tyro.cli(Args)

    network_model_name = args.network_model_config.split("/")[-1][:-5]

    args.exp_name = (
        f"{network_model_name}_(negotiation={args.env_config.negotiation_on})_"
        f"(num_agents={args.env_config.num_agents})_"
        f"(window={args.env_config.action_window})_"
        f"(reduced_space={args.env_config.reduced_space})_"
        f"(fixed_savings_rate={args.env_config.fixed_savings_rate})_"
        f"(rewards={args.env_config.rewards})"
        f"(wgen={args.weights_generation})"
        f"(num_ensembles={len(args.env_config.climate_ensembles)})"
        f"(seed={args.seed})_"
        f"{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}"
    )

    args.cuda = str(torch.cuda.is_available())

    if args.track:
        print(f"Tracking experiment with wandb")

        wandb.login()

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=asdict(args),
            name=f"{args.exp_name}",
            save_code=True,
        )
    else:
        print(f"Experiment NOT tracked with wandb")

    run_training(args=args)
