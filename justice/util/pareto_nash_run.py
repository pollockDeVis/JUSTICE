import os
import json
import csv
import time
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product
import multiprocessing as mp
from typing import Any, Dict, List, Tuple, Optional

from solvers.emodps.rbf import RBF

from justice.util.enumerations import (
    Economy,
    DamageFunction,
    Abatement,
    WelfareFunction,
)
from justice.model import JUSTICE
from justice.util.emission_control_constraint import EmissionControlConstraint
from justice.util.regional_configuration import (
    aggregate_by_macro_region,
    build_macro_region_mapping,
)
from justice.objectives.objective_functions import fraction_of_ensemble_above_threshold
from justice.util.data_loader import DataLoader
from justice.util.model_time import TimeHorizon


# ------------------------ policy bank ------------------------


def build_policy_bank_from_5row_csv(
    policy_csv_path: str,
    config_path: str,
    n_agents: int = 5,
) -> Tuple[np.ndarray, Dict]:
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    df = pd.read_csv(policy_csv_path)
    if len(df) != 5:
        raise ValueError(
            f"Expected CSV to have exactly 5 rows (5 actions), got {len(df)}"
        )

    n_inputs = int(config["n_inputs"])
    rbf_template = RBF(n_rbfs=(n_inputs + 2), n_inputs=n_inputs, n_outputs=1)
    centers_shape, radii_shape, weights_shape = rbf_template.get_shape()
    centers_len, radii_len, weights_len = (
        centers_shape[0],
        radii_shape[0],
        weights_shape[0],
    )
    decision_var_len = centers_len + radii_len + weights_len

    policy_bank = np.empty((n_agents, 5, decision_var_len), dtype=np.float64)

    for a in range(5):  # action index = row index
        row = df.iloc[a]
        for i in range(n_agents):
            centers = np.array(
                [row[f"center_{i}_{j}"] for j in range(centers_len)], dtype=np.float64
            )
            radii = np.array(
                [row[f"radii_{i}_{j}"] for j in range(radii_len)], dtype=np.float64
            )
            weights = np.array(
                [row[f"weights_{i}_{j}"] for j in range(weights_len)], dtype=np.float64
            )
            policy_bank[i, a, :] = np.concatenate([centers, radii, weights])

    return policy_bank, config


# ------------------------ multiprocessing worker ------------------------

_G: Dict[str, Any] = {}


def _init_worker(
    policy_bank: np.ndarray, config: Dict, scenario: int, mapping_base_path: str
):
    # prevent BLAS oversubscription inside each process
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

    _G["policy_bank"] = policy_bank
    _G["config"] = config
    _G["scenario"] = scenario
    _G["mapping_base_path"] = mapping_base_path

    # --- config values used repeatedly ---
    start_year = config["start_year"]
    end_year = config["end_year"]
    data_timestep = config["data_timestep"]
    timestep = config["timestep"]
    emission_control_start_year = config["emission_control_start_year"]

    n_inputs = int(config["n_inputs"])
    temperature_year_of_interest = config["temperature_year_of_interest"]
    stochastic_run = config["stochastic_run"]
    climate_members = config.get("climate_ensemble_members")

    min_temperature = config["min_temperature"]
    max_temperature = config["max_temperature"]
    min_temperature_change = config["min_temperature_change"]
    max_temperature_change = config["max_temperature_change"]
    consumption_min = config["consumption_min"]
    consumption_max = config["consumption_max"]

    _G["n_inputs"] = n_inputs
    _G["temperature_year_of_interest"] = temperature_year_of_interest
    _G["min_temperature"] = min_temperature
    _G["max_temperature"] = max_temperature
    _G["min_temperature_change"] = min_temperature_change
    _G["max_temperature_change"] = max_temperature_change
    _G["consumption_min"] = consumption_min
    _G["consumption_max"] = consumption_max

    # --- mapping / regions (build once) ---
    data_loader = DataLoader()
    region_list = data_loader.REGION_LIST
    n_regions = len(region_list)

    time_horizon = TimeHorizon(
        start_year=start_year,
        end_year=end_year,
        data_timestep=data_timestep,
        timestep=timestep,
    )
    emission_start_ts = time_horizon.year_to_timestep(
        year=emission_control_start_year, timestep=timestep
    )
    temperature_year_index = time_horizon.year_to_timestep(
        year=temperature_year_of_interest, timestep=timestep
    )
    n_timesteps = len(time_horizon.model_time_horizon)

    r5_json = Path(mapping_base_path) / "R5_regions.json"
    rice50_json = Path(mapping_base_path) / "rice50_regions_dict.json"
    region_to_macro, macro_region_names = build_macro_region_mapping(
        region_list=region_list,
        r5_json_path=r5_json,
        rice50_json_path=rice50_json,
    )
    n_macro = len(macro_region_names)
    if n_macro != 5:
        raise RuntimeError(f"Expected 5 macro regions, got {n_macro}")

    _G["region_to_macro"] = region_to_macro
    _G["n_regions"] = n_regions
    _G["n_macro"] = n_macro
    _G["n_timesteps"] = n_timesteps
    _G["temperature_year_index"] = temperature_year_index
    _G["emission_start_ts"] = emission_start_ts

    # --- instantiate model ONCE per worker ---
    model = JUSTICE(
        scenario=scenario,
        economy_type=Economy.NEOCLASSICAL,
        damage_function_type=DamageFunction.KALKUHL,
        abatement_type=Abatement.ENERDATA,
        social_welfare_function=WelfareFunction.UTILITARIAN,
        stochastic_run=stochastic_run,
        climate_ensembles=climate_members,
    )
    _G["model"] = model
    _G["no_of_ensembles"] = int(model.__getattribute__("no_of_ensembles"))

    # --- constraint (once) ---
    _G["emission_constraint"] = EmissionControlConstraint(
        max_annual_growth_rate=0.04,
        emission_control_start_timestep=emission_start_ts,
        min_emission_control_rate=0.01,
    )

    # --- precompute constants & preallocate buffers (once) ---
    inv_temperature_range = 1.0 / (max_temperature - min_temperature)
    inv_temperature_change_range = 1.0 / (
        max_temperature_change - min_temperature_change
    )
    inv_consumption_range = 1.0 / (consumption_max - consumption_min)

    _G["inv_temperature_range"] = inv_temperature_range
    _G["inv_temperature_change_range"] = inv_temperature_change_range
    _G["inv_consumption_range"] = inv_consumption_range

    noE = _G["no_of_ensembles"]
    _G["rbf_in"] = np.empty((n_inputs, noE), dtype=np.float64)
    _G["prev_temp"] = np.zeros(noE, dtype=np.float64)
    _G["prev_dtemp"] = np.zeros(noE, dtype=np.float64)

    _G["regional_ecr"] = np.zeros((n_regions, n_timesteps, noE), dtype=np.float64)
    _G["constrained_ecr"] = np.zeros((n_regions, n_timesteps, noE), dtype=np.float64)
    _G["macro_ecr"] = np.zeros((n_macro, n_timesteps, noE), dtype=np.float64)
    _G["macro_cpc_hist"] = np.zeros((n_macro, n_timesteps, noE), dtype=np.float64)

    # region population is fixed conditional on scenario/config; compute once
    _G["region_population"] = model.economy.get_population()


def _simulate_profile(
    actions: Tuple[int, int, int, int, int],
) -> Tuple[Tuple[int, int, int, int, int], np.ndarray, float]:
    """
    actions: (a0..a4) each in {0..4}
    Returns: (actions, welfare_vec(5,), fraction_above_threshold)
    """
    policy_bank = _G["policy_bank"]
    model: JUSTICE = _G["model"]

    n_inputs = _G["n_inputs"]
    region_to_macro = _G["region_to_macro"]
    emission_constraint: EmissionControlConstraint = _G["emission_constraint"]

    n_regions = _G["n_regions"]
    n_macro = _G["n_macro"]
    n_timesteps = _G["n_timesteps"]
    noE = _G["no_of_ensembles"]

    min_temperature = _G["min_temperature"]
    min_temperature_change = _G["min_temperature_change"]
    consumption_min = _G["consumption_min"]

    inv_temperature_range = _G["inv_temperature_range"]
    inv_temperature_change_range = _G["inv_temperature_change_range"]
    inv_consumption_range = _G["inv_consumption_range"]

    temperature_year_index = _G["temperature_year_index"]

    # --- reset model state (key change) ---
    model.reset()

    # --- build RBFs for this profile (small overhead; could be further cached if needed) ---
    macro_rbfs: List[RBF] = []
    for i in range(n_macro):
        rbf = RBF(n_rbfs=(n_inputs + 2), n_inputs=n_inputs, n_outputs=1)
        rbf.set_decision_vars(policy_bank[i, actions[i], :])
        macro_rbfs.append(rbf)

    # --- reuse buffers ---
    regional_ecr = _G["regional_ecr"]
    regional_ecr.fill(0.0)
    constrained_ecr = _G["constrained_ecr"]
    constrained_ecr.fill(0.0)
    macro_ecr = _G["macro_ecr"]
    macro_ecr.fill(0.0)
    macro_cpc_hist = _G["macro_cpc_hist"]
    macro_cpc_hist.fill(0.0)

    rbf_in = _G["rbf_in"]
    prev_temp = _G["prev_temp"]
    prev_temp.fill(0.0)
    prev_dtemp = _G["prev_dtemp"]
    prev_dtemp.fill(0.0)

    region_population = _G["region_population"]

    # --- run ---
    for t in range(n_timesteps):
        constrained_ecr[:, t, :] = emission_constraint.constrain_emission_control_rate(
            regional_ecr[:, t, :], t, allow_fallback=False
        )

        model.stepwise_run(
            emission_control_rate=constrained_ecr[:, t, :],
            timestep=t,
            endogenous_savings_rate=True,
        )
        ds_t = model.stepwise_evaluate(timestep=t)

        temp = ds_t["global_temperature"][t, :]
        cons = ds_t["consumption"][:, t, :]

        if t == 0:
            dtemp = np.zeros_like(temp)
            prev_temp[:] = temp
            prev_dtemp[:] = dtemp
        elif t % 5 == 0:
            dtemp = temp - prev_temp
            prev_temp[:] = temp
            prev_dtemp[:] = dtemp
        else:
            dtemp = prev_dtemp

        rbf_in[0, :] = np.clip(
            (temp - min_temperature) * inv_temperature_range, 0.0, 1.0
        )
        rbf_in[1, :] = np.clip(
            (dtemp - min_temperature_change) * inv_temperature_change_range, 0.0, 1.0
        )

        pop_t = region_population[:, t, :]
        macro_pop = aggregate_by_macro_region(pop_t, region_to_macro)
        macro_total_cons = aggregate_by_macro_region(cons, region_to_macro)
        macro_cpc = (macro_total_cons / macro_pop) * 1e3
        macro_cpc_hist[:, t, :] = macro_cpc

        norm_macro_cpc = np.clip(
            (macro_cpc - consumption_min) * inv_consumption_range, 0.0, 1.0
        )

        if t < n_timesteps - 1:
            for i, rbf in enumerate(macro_rbfs):
                rbf_in[2, :] = norm_macro_cpc[i, :]
                macro_ecr[i, t + 1, :] = rbf.apply_rbfs(rbf_in)
            regional_ecr[:, t + 1, :] = macro_ecr[region_to_macro, t + 1, :]

    ds = model.evaluate()

    welfare_vec = model.welfare_function.calculate_spatially_disaggregated_welfare(
        macro_cpc_hist
    )
    welfare_vec = np.asarray(welfare_vec, dtype=np.float64)

    frac = fraction_of_ensemble_above_threshold(
        temperature=ds["global_temperature"],
        temperature_year_index=temperature_year_index,
        threshold=2.0,
    )

    return actions, welfare_vec, float(frac)


# ------------------------ benchmark + full run ------------------------


def benchmark_profiles(
    policy_csv_path: str,
    config_path: str,
    scenario: int = 2,
    mapping_base_path: str = "data/input",
    n_samples: int = 20,
    seed: int = 0,
) -> None:
    """
    Runs n_samples profiles sequentially in THIS process to estimate time/profile.
    Uses the same init/reset logic as workers.
    """
    rng = np.random.default_rng(seed)
    policy_bank, config = build_policy_bank_from_5row_csv(
        policy_csv_path, config_path, n_agents=5
    )

    _init_worker(policy_bank, config, scenario, mapping_base_path)

    samples = [tuple(rng.integers(0, 5, size=5).tolist()) for _ in range(n_samples)]

    t0 = time.perf_counter()
    for a in samples:
        _simulate_profile(a)
    t1 = time.perf_counter()

    per = (t1 - t0) / n_samples
    est_total = per * (5**5)
    print(f"Benchmark: {n_samples} samples in {t1-t0:.3f}s")
    print(f"Time/profile: {per:.4f}s")
    print(f"Estimated time for 5^5=3125 profiles: {est_total/60:.2f} minutes")


def run_5pow5_payoff_table(
    policy_csv_path: str,
    config_path: str,
    out_csv_path: str,
    scenario: int = 2,
    mapping_base_path: str = "data/input",
    n_workers: Optional[int] = None,
    chunksize: int = 8,
) -> None:
    policy_bank, config = build_policy_bank_from_5row_csv(
        policy_csv_path, config_path, n_agents=5
    )

    profiles = list(product(range(5), repeat=5))  # 3125

    out_path = Path(out_csv_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    header = [
        "a0",
        "a1",
        "a2",
        "a3",
        "a4",
        "welfare_0",
        "welfare_1",
        "welfare_2",
        "welfare_3",
        "welfare_4",
        "fraction_above_threshold",
    ]

    # If you are on Linux and want faster startup, you can try "fork".
    # "spawn" is safest cross-platform.
    ctx = mp.get_context("spawn")

    if n_workers is None:
        n_workers = max(1, mp.cpu_count() - 1)

    t0 = time.perf_counter()
    completed = 0
    report_every = 100

    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        with ctx.Pool(
            processes=n_workers,
            initializer=_init_worker,
            initargs=(policy_bank, config, scenario, mapping_base_path),
        ) as pool:
            for actions, welfare_vec, frac in pool.imap_unordered(
                _simulate_profile, profiles, chunksize=chunksize
            ):
                writer.writerow([*actions, *welfare_vec.tolist(), frac])
                completed += 1

                if completed % report_every == 0:
                    t = time.perf_counter() - t0
                    rate = completed / max(t, 1e-9)
                    eta = (len(profiles) - completed) / max(rate, 1e-9)
                    print(
                        f"{completed}/{len(profiles)} done | {rate:.2f} проф/s | ETA {eta/60:.1f} min"
                    )

    t1 = time.perf_counter()
    print(f"Finished {len(profiles)} profiles in {(t1-t0)/60:.2f} minutes")


def nondominated_mask(points: np.ndarray) -> np.ndarray:
    """
    points: (m, k) array, all objectives assumed to be MAXIMIZED.
    Returns mask of rows that are NOT Pareto-dominated by any other row.
    """
    m = points.shape[0]
    dominated = np.zeros(m, dtype=bool)

    # O(m^2 * k) but here groups are tiny (e.g., 5 actions), so it's fast.
    for i in range(m):
        if dominated[i]:
            continue
        ge = (points >= points[i]).all(axis=1)
        gt = (points > points[i]).any(axis=1)
        if np.any(ge & gt):
            dominated[i] = True
    return ~dominated


def infer_agent_count(
    df: pd.DataFrame,
    welfare_prefix: str = "welfare_",
    action_prefix: str = "a",
) -> int:
    """
    Infers n_agents from columns welfare_0..welfare_{n-1} or a0..a{n-1}.
    """
    welfare_idx = []
    for c in df.columns:
        if c.startswith(welfare_prefix):
            suf = c[len(welfare_prefix) :]
            if suf.isdigit():
                welfare_idx.append(int(suf))

    action_idx = []
    for c in df.columns:
        if c.startswith(action_prefix):
            suf = c[len(action_prefix) :]
            if suf.isdigit():
                action_idx.append(int(suf))

    candidates = []
    if welfare_idx:
        candidates.append(max(welfare_idx) + 1)
    if action_idx:
        candidates.append(max(action_idx) + 1)

    if not candidates:
        raise ValueError(
            "Could not infer number of agents from columns (no welfare_* or a* columns found)."
        )

    n_agents = max(candidates)
    return n_agents


def pareto_nash_set(
    payoff_csv_path: str,
    fraction_col: str = "fraction_above_threshold",
    minimize_fraction: bool = True,
    welfare_prefix: str = "welfare_",
    action_prefix: str = "a",
    n_agents: Optional[int] = None,
) -> pd.DataFrame:
    """
    Computes the Pareto–Nash set for an n-player game where each player's outcome is 2D:
      [welfare_i (maximize), -fraction (maximize)]  if minimize_fraction=True

    Returns a dataframe subset with boolean columns br_i and is_pareto_nash.

    Notes
    -----
    Requires action columns a0..a{n-1} and welfare columns welfare_0..welfare_{n-1}.
    """
    df = pd.read_csv(payoff_csv_path)

    if n_agents is None:
        n_agents = infer_agent_count(
            df, welfare_prefix=welfare_prefix, action_prefix=action_prefix
        )

    action_cols = [f"{action_prefix}{i}" for i in range(n_agents)]
    welfare_cols = [f"{welfare_prefix}{i}" for i in range(n_agents)]

    missing = [
        c for c in action_cols + welfare_cols + [fraction_col] if c not in df.columns
    ]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    actions_arr = df[action_cols].to_numpy(dtype=np.int64)
    welfare_arr = df[welfare_cols].to_numpy(dtype=np.float64)

    frac_raw = df[fraction_col].to_numpy(dtype=np.float64)
    frac_obj = (
        -frac_raw if minimize_fraction else frac_raw
    )  # convert to maximize objective

    br_masks: List[np.ndarray] = []

    for i in range(n_agents):
        others = [j for j in range(n_agents) if j != i]
        keys = actions_arr[:, others]  # (N, n_agents-1)

        # Fast grouping by "others": turn each key row into bytes
        keys_struct = np.ascontiguousarray(keys).view(
            np.dtype((np.void, keys.dtype.itemsize * keys.shape[1]))
        )
        _, group_id = np.unique(keys_struct, return_inverse=True)

        br = np.zeros(len(df), dtype=bool)

        # iterate groups; each group contains all unilateral actions for player i
        # (typically size = number of actions per player)
        for gid in np.unique(group_id):
            idx = np.where(group_id == gid)[0]
            pts = np.column_stack([welfare_arr[idx, i], frac_obj[idx]])
            br[idx] = nondominated_mask(pts)

        df[f"br_{i}"] = br
        br_masks.append(br)

    df["is_pareto_nash"] = np.logical_and.reduce(br_masks)
    return df[df["is_pareto_nash"]].copy()


def main():
    policy_csv_path = (
        "data/temporary/MOMA_DATA/200k/COMBINED_MOMA_epsilon_nondominated_set.csv"
    )
    config_path = "analysis/momadps_config.json"
    mapping_base_path = "data/input"

    # 1) quick sequential benchmark (no multiprocessing)
    benchmark_profiles(
        policy_csv_path=policy_csv_path,
        config_path=config_path,
        scenario=2,
        mapping_base_path=mapping_base_path,
        n_samples=10,
        seed=0,
    )

    # 2) full parallel run (5^5 profiles)
    out_csv_path = "data/temporary/MOMA_DATA/200k/out/payoff_table_5x5x5x5x5.csv"
    Path(out_csv_path).parent.mkdir(parents=True, exist_ok=True)

    run_5pow5_payoff_table(
        policy_csv_path=policy_csv_path,
        config_path=config_path,
        out_csv_path=out_csv_path,
        scenario=2,
        mapping_base_path=mapping_base_path,
        n_workers=4,  # tune
        chunksize=8,  # tune
    )


if __name__ == "__main__":
    # # Required on macOS/Windows when using "spawn"
    # mp.freeze_support()
    # # Optional: be explicit (recommended for reproducibility)
    # mp.set_start_method("spawn", force=True)

    # main()
    payoff_csv = "data/temporary/MOMA_DATA/200k/out/payoff_table_5x5x5x5x5.csv"

    pn = pareto_nash_set(
        payoff_csv_path=payoff_csv,
        minimize_fraction=True,  # usually you want to minimize fraction above threshold
        fraction_col="fraction_above_threshold",
        welfare_prefix="welfare_",
        action_prefix="a",
        n_agents=None,  # infer from columns
    )

    print("Pareto–Nash profiles:", len(pn))
    cols_to_show = [
        c for c in pn.columns if c.startswith("a") or c.startswith("welfare_")
    ] + ["fraction_above_threshold"]
    print(pn[cols_to_show].head())

    # Save Pareto–Nash profiles to CSV
    pn_out_path = "data/temporary/MOMA_DATA/200k/out/pareto_nash_profiles.csv"
    pn.to_csv(pn_out_path, index=False)
    print(f"Saved Pareto–Nash profiles to {pn_out_path}")
