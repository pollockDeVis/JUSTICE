"""
justice/util/pareto_nash_simulate.py

Simulate selected Pareto-Nash profiles through the JUSTICE model and save
output arrays to disk.

Main entry point:
    simulate_nash_profiles_by_row_index(...)
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

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


def load_nash_profile(out_dir: str, profile_index: int, DATASET_KEYS: list) -> dict:
    """Load all saved arrays for a given Nash profile index."""
    return {
        key: np.load(Path(out_dir) / f"{key}_nash_profile_{profile_index}.npy")
        for key in DATASET_KEYS
    }


def get_array_path(out_dir: str, profile_index: int, key: str) -> str:
    """Return the file path for a specific array of a Nash profile."""
    return str(Path(out_dir) / f"{key}_nash_profile_{profile_index}.npy")


def infer_n_agents_from_actions(df: pd.DataFrame, action_prefix: str = "a") -> int:
    idx = [
        int(c[len(action_prefix) :])
        for c in df.columns
        if c.startswith(action_prefix) and c[len(action_prefix) :].isdigit()
    ]
    if not idx:
        raise ValueError("No action columns like a0, a1, ... found.")
    return max(idx) + 1


def build_policy_bank_from_5row_csv(
    policy_csv_path: str,
    config_path: str,
    n_agents: int,
) -> Tuple[np.ndarray, Dict]:
    """
    CSV must have exactly 5 rows (one per discrete action).
    Returns policy_bank[agent, action, decision_vars] and config dict.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    df = pd.read_csv(policy_csv_path)
    if len(df) != 5:
        raise ValueError(f"Expected 5 rows (actions), got {len(df)}")

    n_inputs = int(config["n_inputs"])
    rbf = RBF(n_rbfs=(n_inputs + 2), n_inputs=n_inputs, n_outputs=1)
    c_len, r_len, w_len = (s[0] for s in rbf.get_shape())
    dv_len = c_len + r_len + w_len

    policy_bank = np.empty((n_agents, 5, dv_len), dtype=np.float64)
    for a in range(5):
        row = df.iloc[a]
        for i in range(n_agents):
            centers = np.array([row[f"center {i} {j}"] for j in range(c_len)])
            radii = np.array([row[f"radii {i} {j}"] for j in range(r_len)])
            weights = np.array([row[f"weights {i} {j}"] for j in range(w_len)])
            policy_bank[i, a, :] = np.concatenate([centers, radii, weights])

    return policy_bank, config


DEFAULT_DATASETS_TO_SAVE = (
    "constrained_emission_control_rate",
    "emissions",
    "economic_damage",
    "abatement_cost",
    "global_temperature",
    "net_economic_output",
    "gross_economic_output",
)


def save_selected_datasets(
    datasets: Dict[str, np.ndarray],
    out_dir: str,
    profile_index: int,
    datasets_to_save: Sequence[str] = DEFAULT_DATASETS_TO_SAVE,
) -> None:
    """Save the requested model output arrays for one Nash profile as .npy files.

    datasets_to_save defaults to all outputs. Pass a subset (e.g. ["emissions"])
    to save disk space when simulating many profiles.
    """
    unknown = [name for name in datasets_to_save if name not in datasets]
    if unknown:
        raise KeyError(
            f"Unknown dataset(s) {unknown}. Available: {sorted(datasets.keys())}"
        )
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    for name in datasets_to_save:
        np.save(out_path / f"{name}_nash_profile_{profile_index}.npy", datasets[name])


@dataclass
class NashProfileSimulator:
    policy_bank: np.ndarray
    config: Dict
    scenario: int = 2
    mapping_base_path: str = "data/input"

    def __post_init__(self) -> None:
        cfg = self.config
        self.n_agents = int(self.policy_bank.shape[0])
        self.n_actions = int(self.policy_bank.shape[1])

        self.start_year = cfg["start_year"]
        self.end_year = cfg["end_year"]
        self.data_timestep = cfg["data_timestep"]
        self.timestep = cfg["timestep"]
        self.emission_control_start_year = cfg["emission_control_start_year"]
        self.n_inputs = int(cfg["n_inputs"])
        self.temperature_year_of_interest = cfg["temperature_year_of_interest"]
        self.stochastic_run = cfg["stochastic_run"]
        self.climate_members = cfg.get("climate_ensemble_members")
        self.min_temperature = cfg["min_temperature"]
        self.max_temperature = cfg["max_temperature"]
        self.min_temperature_change = cfg["min_temperature_change"]
        self.max_temperature_change = cfg["max_temperature_change"]
        self.consumption_min = cfg["consumption_min"]
        self.consumption_max = cfg["consumption_max"]

        self.inv_temperature_range = 1.0 / (self.max_temperature - self.min_temperature)
        self.inv_temperature_change_range = 1.0 / (
            self.max_temperature_change - self.min_temperature_change
        )
        self.inv_consumption_range = 1.0 / (self.consumption_max - self.consumption_min)

        data_loader = DataLoader()
        self.region_list = data_loader.REGION_LIST
        self.n_regions = len(self.region_list)

        self.time_horizon = TimeHorizon(
            start_year=self.start_year,
            end_year=self.end_year,
            data_timestep=self.data_timestep,
            timestep=self.timestep,
        )
        self.n_timesteps = len(self.time_horizon.model_time_horizon)
        self.emission_start_ts = self.time_horizon.year_to_timestep(
            year=self.emission_control_start_year, timestep=self.timestep
        )
        self.temperature_year_index = self.time_horizon.year_to_timestep(
            year=self.temperature_year_of_interest, timestep=self.timestep
        )

        r5_json = Path(self.mapping_base_path) / "R5_regions.json"
        rice50_json = Path(self.mapping_base_path) / "rice50_regions_dict.json"
        self.region_to_macro, self.macro_region_names = build_macro_region_mapping(
            region_list=self.region_list,
            r5_json_path=r5_json,
            rice50_json_path=rice50_json,
        )
        if len(self.macro_region_names) != self.n_agents:
            raise RuntimeError(
                f"Mismatch: policy_bank has {self.n_agents} agents, "
                f"mapping has {len(self.macro_region_names)} macro regions."
            )

        self.model = JUSTICE(
            scenario=self.scenario,
            economy_type=Economy.NEOCLASSICAL,
            damage_function_type=DamageFunction.KALKUHL,
            abatement_type=Abatement.ENERDATA,
            social_welfare_function=WelfareFunction.UTILITARIAN,
            stochastic_run=self.stochastic_run,
            climate_ensembles=self.climate_members,
        )
        self.no_of_ensembles = int(self.model.__getattribute__("no_of_ensembles"))

        self.emission_constraint = EmissionControlConstraint(
            max_annual_growth_rate=0.04,
            emission_control_start_timestep=self.emission_start_ts,
            min_emission_control_rate=0.01,
        )
        self.region_population = self.model.economy.get_population()

        E, T, R, M = (
            self.no_of_ensembles,
            self.n_timesteps,
            self.n_regions,
            self.n_agents,
        )
        self.regional_ecr = np.zeros((R, T, E), dtype=np.float64)
        self.constrained_ecr = np.zeros((R, T, E), dtype=np.float64)
        self.macro_ecr = np.zeros((M, T, E), dtype=np.float64)
        self.macro_cpc_hist = np.zeros((M, T, E), dtype=np.float64)
        self.rbf_in = np.empty((self.n_inputs, E), dtype=np.float64)
        self.prev_temp = np.zeros(E, dtype=np.float64)
        self.prev_dtemp = np.zeros(E, dtype=np.float64)

        self.macro_rbfs: List[RBF] = [
            RBF(n_rbfs=(self.n_inputs + 2), n_inputs=self.n_inputs, n_outputs=1)
            for _ in range(self.n_agents)
        ]

    def simulate(
        self, actions: Sequence[int]
    ) -> Tuple[Tuple[float, ...], Dict[str, np.ndarray]]:
        if len(actions) != self.n_agents:
            raise ValueError(
                f"actions must have length {self.n_agents}, got {len(actions)}"
            )
        actions = tuple(int(a) for a in actions)
        if any(a < 0 or a >= self.n_actions for a in actions):
            raise ValueError(f"Action out of range [0,{self.n_actions-1}]: {actions}")

        self.model.reset()
        self.regional_ecr.fill(0.0)
        self.constrained_ecr.fill(0.0)
        self.macro_ecr.fill(0.0)
        self.macro_cpc_hist.fill(0.0)
        self.prev_temp.fill(0.0)
        self.prev_dtemp.fill(0.0)

        for i in range(self.n_agents):
            self.macro_rbfs[i].set_decision_vars(self.policy_bank[i, actions[i], :])

        for t in range(self.n_timesteps):
            self.constrained_ecr[:, t, :] = (
                self.emission_constraint.constrain_emission_control_rate(
                    self.regional_ecr[:, t, :],
                    t,
                    allow_fallback=False,
                )
            )
            self.model.stepwise_run(
                emission_control_rate=self.constrained_ecr[:, t, :],
                timestep=t,
                endogenous_savings_rate=True,
            )
            ds_t = self.model.stepwise_evaluate(timestep=t)
            global_temp = ds_t["global_temperature"][t, :]
            consumption = ds_t["consumption"][:, t, :]

            if t == 0:
                temperature_change = np.zeros_like(global_temp)
                self.prev_temp[:] = global_temp
                self.prev_dtemp[:] = temperature_change
            elif t % 5 == 0:
                temperature_change = global_temp - self.prev_temp
                self.prev_temp[:] = global_temp
                self.prev_dtemp[:] = temperature_change
            else:
                temperature_change = self.prev_dtemp

            self.rbf_in[0, :] = np.clip(
                (global_temp - self.min_temperature) * self.inv_temperature_range,
                0.0,
                1.0,
            )
            self.rbf_in[1, :] = np.clip(
                (temperature_change - self.min_temperature_change)
                * self.inv_temperature_change_range,
                0.0,
                1.0,
            )

            pop_t = self.region_population[:, t, :]
            macro_pop = aggregate_by_macro_region(pop_t, self.region_to_macro)
            macro_cons = aggregate_by_macro_region(consumption, self.region_to_macro)
            macro_cpc = (macro_cons / macro_pop) * 1e3
            self.macro_cpc_hist[:, t, :] = macro_cpc

            norm_macro_cpc = np.clip(
                (macro_cpc - self.consumption_min) * self.inv_consumption_range,
                0.0,
                1.0,
            )

            if t < self.n_timesteps - 1:
                for i in range(self.n_agents):
                    self.rbf_in[2, :] = norm_macro_cpc[i, :]
                    self.macro_ecr[i, t + 1, :] = self.macro_rbfs[i].apply_rbfs(
                        self.rbf_in
                    )
                self.regional_ecr[:, t + 1, :] = self.macro_ecr[
                    self.region_to_macro, t + 1, :
                ]

        datasets = self.model.evaluate()
        datasets["constrained_emission_control_rate"] = self.constrained_ecr.copy()

        spatial_welfare = (
            self.model.welfare_function.calculate_spatially_disaggregated_welfare(
                self.macro_cpc_hist
            )
        )
        frac = fraction_of_ensemble_above_threshold(
            temperature=datasets["global_temperature"],
            temperature_year_index=self.temperature_year_index,
            threshold=2.0,
        )
        objectives = tuple(float(x) for x in spatial_welfare) + (float(frac),)
        return objectives, datasets


def simulate_nash_profiles_by_row_index(
    nash_profiles_csv: str,
    policy_5row_csv: str,
    config_path: str,
    out_dir: str,
    selected_profile_indices: Optional[Sequence[int]] = None,
    scenario: int = 2,
    mapping_base_path: str = "data/input",
    datasets_to_save: Sequence[str] = DEFAULT_DATASETS_TO_SAVE,
) -> None:
    """
    Main entry point. Simulates selected (or all) Pareto-Nash profiles and
    saves .npy arrays + JSON sidecar per profile to out_dir.

    Parameters
    ----------
    nash_profiles_csv        : Path to pareto_nash_profiles.csv
    policy_5row_csv          : Path to 5-row policy CSV (COMBINED_MOMA_epsilon_nondominated_set.csv)
    config_path              : Path to momadps_config.json
    out_dir                  : Output directory for .npy and .json files
    selected_profile_indices : Row indices to simulate. None = simulate all.
    scenario                 : SSP scenario (default 2)
    mapping_base_path        : Folder with R5_regions.json and rice50_regions_dict.json
    datasets_to_save         : Output arrays to save per profile. Defaults to all
                               (see DEFAULT_DATASETS_TO_SAVE); pass e.g. ["emissions"]
                               to save disk space when simulating many profiles.
    """
    nash_df = pd.read_csv(nash_profiles_csv)
    n_agents = infer_n_agents_from_actions(nash_df)

    if selected_profile_indices is None:
        selected_profile_indices = list(range(len(nash_df)))

    print(f"Nash profiles    : {nash_profiles_csv}  ({len(nash_df)} total)")
    print(f"Policy CSV       : {policy_5row_csv}")
    print(f"Profiles to run  : {selected_profile_indices}")
    print(f"Output dir       : {out_dir}")
    print(f"Datasets saved   : {list(datasets_to_save)}")
    print(f"⚠ This may take several minutes per profile.\n")

    policy_bank, config = build_policy_bank_from_5row_csv(
        policy_5row_csv, config_path, n_agents
    )
    sim = NashProfileSimulator(
        policy_bank=policy_bank,
        config=config,
        scenario=scenario,
        mapping_base_path=mapping_base_path,
    )

    for idx in selected_profile_indices:
        row = nash_df.iloc[int(idx)]
        actions = tuple(int(row[f"a{i}"]) for i in range(n_agents))

        print(f"  Profile {idx:>3} | actions: {actions}", end=" ... ", flush=True)
        objectives, datasets = sim.simulate(actions)
        save_selected_datasets(
            datasets,
            out_dir=out_dir,
            profile_index=int(idx),
            datasets_to_save=datasets_to_save,
        )

        meta = {
            "nash_profile_row_index": int(idx),
            "actions": actions,
            "objectives": objectives,
        }
        (Path(out_dir) / f"meta_nash_profile_{int(idx)}.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )
        print(f"done  |  frac_above_2C: {objectives[-1]:.3f}")

    print(f"\nAll done. Files saved to: {out_dir}")
