
"""

######################
## MOMAPPO Training ##
######################

Pytorch implementation of the MOMAPPO algorithm.

"""

import time, numpy as np, torch.nn as nn, morl_baselines.common.weights, numpy as np, wandb, json, torch, supersuit as ss, torch.nn as nn

from morl_baselines.common.evaluation import log_all_multi_policy_metrics
from morl_baselines.multi_policy.linear_support.linear_support import LinearSupport
from momaland.utils.env import MOParallelEnv

from momaland.utils.parallel_wrappers import (
    LinearizeReward,
    NormalizeReward,
    RecordEpisodeStatistics,
)
from rl.env import JusticeMOMARLEnv
from rl.agent import AgentDiscrete
from rl.nn.configs import NetworkModelConfig

from torch import optim
from copy import deepcopy

from pathlib import Path
from dataclasses import asdict

def train_mappo_step(args, jenv, eval_env, num_objectives, weights, network_model_config, device, writer, eval_results, eval_results_path, eval_rep=10):

    """
    Single training step of MAPPO for a chosen objective weights combination.

    @param args: The arguments for the training
    @param jenv: The environment to train on
    @param eval_env: The environment to evaluate the agent on
    @param num_objectives: The number of objectives
    @param weights: The MORL weights
    @param network_model_config: The network model configuration
    @param device: The device to train on
    @param writer: The tensorboard writer
    @param eval_results: The evaluation results
    @param eval_results_path: Where to save the evaluation results
    @param eval_rep: The number of seeds to evaluate the agent with
    
    """

    env = deepcopy(jenv)
    for agent in env.possible_agents:
        for idx in range(env.unwrapped.reward_space(agent).shape[0]):
            env = NormalizeReward(env, agent, idx)
    _weights = {agent: weights for agent in env.possible_agents}
    env = LinearizeReward(env, _weights)  # linearizing the rewards given the weights
    env = RecordEpisodeStatistics(env)

    env = ss.pettingzoo_env_to_vec_env_v1(env)
    envs = ss.concat_vec_envs_v1(
        env, args.num_envs, num_cpus=args.num_envs, base_class="gymnasium"
    )
    envs.single_observation_space = envs.observation_space
    envs.single_action_space = envs.action_space
    envs.is_vector_env = True

    agent_class = AgentDiscrete

    agent = agent_class(
        envs,
        network_model_config,
        num_timesteps=args.num_steps,
        num_agents=args.env_config.num_agents,
    ).to(device)

    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # INIT ENV
    next_obs, infos = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(env.num_envs).to(device)

    # ALGO Logic: Storage setup
    obs: torch.Tensor = torch.zeros(
        (args.num_steps, envs.num_envs) + envs.single_observation_space.shape
    ).to(device)
    actions: torch.Tensor = torch.zeros(
        (args.num_steps, envs.num_envs) + envs.single_action_space.shape
    ).to(device)
    logprobs = torch.zeros((args.num_steps, envs.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, envs.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, envs.num_envs)).to(device)
    values = torch.zeros((args.num_steps, envs.num_envs)).to(device)

    # stats setup
    episode_returns = torch.zeros((envs.num_envs,)).to(device)
    episode_lengths = torch.zeros((envs.num_envs,), dtype=torch.int32).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    next_obs, infos = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(envs.num_envs).to(device)

    for iteration in range(1, args.num_iterations + 1):  # iterations that we choose
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):

            global_step += env.num_envs

            obs[step] = (
                next_obs  # shape (num_agents, num_obs) where num_obs = num_agents + global_obs + local_obs
            )

            dones[step] = next_done

            action_masks = (
                torch.tensor(
                    np.array([info.get("action_mask") for info in infos]),
                    dtype=torch.int32,
                ).to(device)
                if "action_mask" in infos[0]
                else None
            )

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(
                    next_obs,
                    timestep=step,
                    update_lstm_hidden_state=True,
                    action_mask=action_masks,
                )
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(
                action.cpu().numpy()
            )

            next_done_bool = np.logical_or(terminations, truncations)
            rewards[step] = (
                torch.tensor(reward).to(device).view(-1)
            )  # Need to get team reward here
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(
                next_done_bool
            ).to(device)

            episode_returns += rewards[step]
            episode_lengths += 1

            if next_done_bool.any():

                mean_episodic_return = episode_returns[next_done_bool].mean()
                print(
                    f"global_step= {global_step}\nmean_episodic_return={mean_episodic_return}"
                )
                writer.add_scalar(
                    f"charts/mean_episodic_return/{str(weights)}",
                    mean_episodic_return,
                    global_step,
                )
                episode_returns[next_done_bool] = 0
                episode_lengths[next_done_bool] = 0

                savings_rates = (
                    jenv.discrete_to_float(actions[step][:, 0])
                    if not args.env_config.fixed_savings_rate
                    else None
                )

                if savings_rates is not None:
                    mean_savings_rate = torch.mean(savings_rates)
                    writer.add_scalar(
                        f"charts/savings_rate_average/{str(weights)}",
                        mean_savings_rate,
                        global_step,
                    )
                    print(f"mean_savings_rate={mean_savings_rate}")

                emissions_rates = (
                    jenv.discrete_to_float(actions[step][:, 1])
                    if not args.env_config.fixed_savings_rate
                    else jenv.discrete_to_float(actions[step][:, 0])
                )

                mean_emissions_rate = torch.mean(emissions_rates)
                writer.add_scalar(
                    f"charts/emissions_rate_average/{str(weights)}",
                    mean_emissions_rate,
                    global_step,
                )
                print(f"mean_emissions_rate={mean_emissions_rate}")

                global_temperature = obs[step][:, 5]
                mean_global_temperature = global_temperature[0]  # same for all agents
                writer.add_scalar(
                    f"charts/global_temperature/{str(weights)}",
                    mean_global_temperature,
                    global_step,
                )
                print(f"global_temperature={mean_global_temperature}")

                net_economic_output = obs[step][:, 0]
                mean_net_economic_output = torch.mean(net_economic_output)
                writer.add_scalar(
                    f"charts/net_economic_output_average/{str(weights)}",
                    mean_net_economic_output,
                    global_step,
                )
                print(f"mean_net_economic_output={mean_net_economic_output}")

                emissions = obs[step][:, 1]
                mean_emissions = torch.mean(emissions)
                writer.add_scalar(
                    f"charts/emissions_average/{str(weights)}",
                    mean_emissions,
                    global_step,
                )
                print(f"mean_emissions={mean_emissions}")

                regional_temperature = obs[step][:, 2]
                mean_regional_temperature = torch.mean(regional_temperature)
                writer.add_scalar(
                    f"charts/regional_temperature_average/{str(weights)}",
                    mean_regional_temperature,
                    global_step,
                )
                print(f"mean_regional_temperature={mean_regional_temperature}")

                economic_damage = obs[step][:, 3]
                mean_economic_damage = torch.mean(economic_damage)
                writer.add_scalar(
                    f"charts/economic_damage_average/{str(weights)}",
                    mean_economic_damage,
                    global_step,
                )
                print(f"mean_economic_damage={mean_economic_damage}")

                abatement_cost = obs[step][:, 4]
                mean_abatement_cost = torch.mean(abatement_cost)
                writer.add_scalar(
                    f"charts/abatement_cost_average/{str(weights)}",
                    mean_abatement_cost,
                    global_step,
                )
                print(f"mean_abatement_cost={mean_abatement_cost}")

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(
                next_obs, timestep=(args.num_steps - 1)
            ).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = (
                    rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                )
                advantages[t] = lastgaelam = (
                    delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                )
            returns = advantages + values

        b_obs = obs
        b_logprobs = logprobs
        b_actions = actions
        b_advantages = advantages
        b_returns = returns
        b_values = values

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for _ in range(args.update_epochs):
            np.random.shuffle(b_inds)
            count = 0
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]
                count += 1
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    x=b_obs[mb_inds],
                    action=b_actions[mb_inds],
                    timestep=mb_inds[0],
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [
                        ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()
                    ]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - args.clip_coef, 1 + args.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar(
            f"charts/learning_rate/{str(weights)}",
            optimizer.param_groups[0]["lr"],
            global_step,
        )
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        
        # Run evaluation for weight at every n-th iteration
        if iteration % args.evaluation_iterations == 1:
            vec_return, disc_vec_return = policy_evaluation_momarl(
                agent=agent, env=eval_env, rep=eval_rep, num_obj=num_objectives, start_seed=0
            )
            
            if str(weights) not in eval_results.keys():
                eval_results[str(weights)] = {str(global_step): vec_return.tolist()}
            else:
                eval_results[str(weights)].update({str(global_step): vec_return.tolist()})

            # Write the updated dictionary back to the file
            with eval_results_path.open("w") as file:
                json.dump(eval_results, file, indent=4)  

    envs.close()

    return agent, eval_results


def policy_evaluation_momarl(agent, env, num_obj, rep=10, gamma_decay=0.99, start_seed=0):

    """

    Evaluate the current policy for multiple seeds and return the undiscounted and discounted mean returns.

    @param agent: The agent to evaluate
    @param env: The environment to evaluate the agent in
    @param num_obj: The number of objectives
    @param rep: The number of seeds to evaluate the agent with
    @param gamma_decay: The discount factor
    @param start_seed: The starting seed for the evaluation

    """

    env.reset()

    evals = []
    seed = start_seed
    for _ in range(rep):
        evals.append(
            eval_moma_torch(agent, env, num_obj, gamma_decay=gamma_decay, seed=seed)
        )
        seed += 1

    avg_vec_return = np.mean([eval[0] for eval in evals], axis=(0, 1))
    disc_vec_return = np.mean([eval[1] for eval in evals], axis=(0, 1))


    return avg_vec_return, disc_vec_return


def eval_moma_torch(agent, env, num_obj, gamma_decay=0.99, seed=42):

    """

    Evaluate the current policy for a single seed and return the undiscounted and discounted mean returns.

    @param agent: The agent to evaluate
    @param env: The environment to evaluate the agent in
    @param num_obj: The number of objectives
    @param gamma_decay: The discount factor
    @param seed: The seed for the evaluation
    
    """

    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )

    agent.eval()

    obs, infos = env.reset(seed=seed)
    obs = torch.Tensor(obs).to(device)
    done = torch.zeros(env.num_envs).to(device)

    action_masks = (
        torch.tensor(
            np.array([info.get("action_mask") for info in infos]), dtype=torch.int32
        ).to(device)
        if "action_mask" in infos[0]
        else None
    )

    vec_returns = np.zeros((env.num_envs, num_obj))
    disc_returns = np.zeros((env.num_envs, num_obj))

    step = 0
    gamma = 1.0

    while not done.all():
        with torch.no_grad():
            action, logprob, _, value = agent.get_action_and_value(
                obs,
                timestep=step,
                action_mask=action_masks,
                update_lstm_hidden_state=True,
            )
        obs, rew, done, trunc, infos = env.step(action.cpu().numpy())

        obs = torch.Tensor(obs).to(device)
        done = torch.Tensor(done).to(device)
        trunc = torch.Tensor(trunc).to(device)

        action_masks = (
            torch.tensor(
                np.array([info.get("action_mask") for info in infos]), dtype=torch.int32
            ).to(device)
            if "action_mask" in infos[0]
            else None
        )

        done = torch.logical_or(done, trunc)

        reward = np.mean(rew, axis=0)

        vec_returns += reward
        disc_returns += reward * gamma

        step += 1
        gamma *= gamma_decay

    return vec_returns, disc_returns


def train_momappo(args, writer):

    """

    Runs MOMAPPO training.

    @param args: The arguments for the training
    @param writer: The tensorboard writer

    """

    print(f"Training MOMARL on {args.env_config.rewards}")

    # Ensure all MOMARL-specific arguments are set
    assert (
        args.num_weights is not None
    ), "The number of weights (num_weights) must be specified for MOMARL."
    assert (
        args.timesteps_per_weight is not None
    ), "Timesteps per weight vector (timesteps_per_weight) must be specified for MOMARL."
    assert (
        args.weights_generation is not None
    ), "The method to generate the weights (weights_generation) must be specified for MOMARL."
    assert (
        args.n_sample_weights is not None
    ), "The number of weights to sample (n_sample_weights) must be specified for MOMARL."
    assert (
        args.ref_point is not None
    ), "The reference point (ref_point) must be specified for hypervolume calculation for MOMARL."

    args.num_iterations = args.timesteps_per_weight // (
        args.num_envs * args.env_config.num_agents * args.num_steps
    )

    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}")
    
    exp_name_without_date = "".join(args.exp_name.split("_")[:-2])
    
    eval_results_path = Path(args.base_save_path) / "artefacts" / f"{exp_name_without_date}_start={args.start_uniform_weight}.json"
    
    eval_results = {}

    # env_constructor = all_environments[args.env_id].parallel_env
    env_constructor = JusticeMOMARLEnv
    start_time = time.time()

    # env setup
    env_config = asdict(args.env_config)
    JusticeMOMARLEnv.pickle_model(args, args.exp_name)

    # Env init
    env: MOParallelEnv = env_constructor(env_config)

    reward_dim = env.unwrapped.reward_space(env.possible_agents[0]).shape[0]

    with open(args.network_model_config) as f:
        network_model_config = json.load(f)

    network_model_config = NetworkModelConfig(**network_model_config)
    
    ols = LinearSupport(num_objectives=reward_dim, epsilon=0.0, verbose=args.debug)
    weight_number = 0

    value = []
    if args.weights_generation == "OLS":
        print(f"Using OLS weight generation.")
        w = ols.next_weight()
    elif args.weights_generation == "uniform":
        print(
            f"Using Uniform weight generation. A total of {args.total_uniform_weights} weights will be generated and this experimnent will train policies from weight {args.start_uniform_weight} to {args.end_uniform_weight}"
        )
        all_weights = morl_baselines.common.weights.equally_spaced_weights(
            reward_dim, args.total_uniform_weights
        )
        all_weights = all_weights[args.start_uniform_weight: args.end_uniform_weight]
        w = all_weights[weight_number]
        args.num_weights = args.end_uniform_weight - args.start_uniform_weight
    else:
        raise ValueError("Weights generation method not recognized")

    while (args.weights_generation != "OLS" or not ols.ended()) and weight_number < args.num_weights:

        print(
            f"Weight Iteration {weight_number + 1}/{args.num_weights} for MOMARL. Training objectives with weights {w}"
        )

        eval_env: MOParallelEnv = env_constructor(env_config)

        if args.env_config.normalise_eval_env:
            print(F"Normalising eval environment.")
            for agent in env.possible_agents:
                for idx in range(env.unwrapped.reward_space(agent).shape[0]):
                    eval_env = NormalizeReward(eval_env, agent, idx)
            eval_env = RecordEpisodeStatistics(eval_env)

        eval_env = ss.pettingzoo_env_to_vec_env_v1(eval_env)
        eval_env = ss.concat_vec_envs_v1(
            eval_env, args.num_envs, num_cpus=args.num_envs, base_class="gymnasium"
        )
        eval_env.single_observation_space = eval_env.observation_space
        eval_env.single_action_space = eval_env.action_space
        eval_env.is_vector_env = True

        agent_state, eval_results = train_mappo_step(
            args, env, eval_env, ols.num_objectives, w, network_model_config, device, writer, eval_results, eval_results_path
        )

        # We evaluate the policy for 10 seeds
        vec_return, _ = policy_evaluation_momarl(
            agent=agent_state, env=eval_env, rep=10, num_obj=ols.num_objectives, start_seed=0
        )

        value.append(vec_return)
        print(f"Weight {weight_number + 1}/{args.num_weights} done!")
        print(f"Value: {vec_return}, weight: {w}")  
        
        
        if str(w) in eval_results.keys():
            eval_results[str(w)].update({"post": vec_return.tolist()}) 

        ols.add_solution(value[-1], w)

        # Write the final eval results in the file dictionary back to the file
        with eval_results_path.open("w") as file:
            json.dump(eval_results, file, indent=4)  
        
        if args.track:
            log_all_multi_policy_metrics(
                current_front=ols.ccs,
                hv_ref_point=np.array(args.ref_point),
                reward_dim=reward_dim,
                global_step=(weight_number + 1) * args.timesteps_per_weight,
                n_sample_weights=args.n_sample_weights,
            )
            
        if args.save_policies:
            # save checkpoint
            torch.save(
                agent_state.state_dict(),
                Path(args.base_save_path) / "checkpoints" / f"{args.exp_name}_w={w}.pt",
            )

        weight_number += 1
        if args.weights_generation == "OLS":
            w = ols.next_weight()
        elif args.weights_generation == "uniform":
            if (
                weight_number >= args.num_weights
            ):  
                break
            w = all_weights[weight_number]

    env.close()
    wandb.finish()
    print(f"total time: {time.time() - start_time}")
