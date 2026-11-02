"""
Orchestrates a multi-objective optimization of the JUSTICE model using Borg.
- Builds the EMA Model (constants, levers, outcomes).
- Registers the model wrapper with the Borg adapter (direct evaluation).
- Defines MPI (MS/MM) subclasses that only tweak Borg’s library/settings.
- Drives EMA’s optimize(...) with convergence metrics (rank 0 only).
"""

import datetime
import json
import os
import random
import pandas as pd
import warnings
from pathlib import Path
from typing import Optional
from justice.util.regional_configuration import build_macro_region_mapping
from solvers.emodps.rbf import RBF
import numpy as np
from ema_workbench import (
    Model,
    RealParameter,
    ScalarOutcome,
    CategoricalParameter,
    ema_logging,
    MultiprocessingEvaluator,
    SequentialEvaluator,
    MPIEvaluator,
    Constant,
    Sample,
)
from platypus import EpsNSGAII

from justice.util.EMA_model_wrapper import (
    model_wrapper_emodps,
    model_wrapper_momadps,
    model_wrapper_momadps_single_agent,
)
from justice.util.data_loader import DataLoader
from justice.util.enumerations import (
    Abatement,
    DamageFunction,
    Economy,
    Evaluator,
    Optimizer,
    WelfareFunction,
)
from justice.util.model_time import TimeHorizon

from solvers.moea.borg_platypus_adapter import (
    BorgMOEA,
    set_ema_context,
    _ArchiveView,
    _AlgorithmStub,
    _create_intermediate_archives,
)
from platypus import Solution

SMALL_NUMBER = 1e-9
warnings.filterwarnings("ignore")

_dir = os.path.dirname(os.path.abspath(__file__))


def _mpi_rank() -> int:
    """Determine MPI/Slurm rank if present, else 0."""
    for key in ("OMPI_COMM_WORLD_RANK", "PMI_RANK", "SLURM_PROCID", "MPI_RANK"):
        val = os.environ.get(key)
        if val is not None:
            try:
                return int(val)
            except ValueError:
                pass
    return 0


class MSBorgMOEA(BorgMOEA):
    """Master–Slave Borg (islands = 1)."""

    def __init__(self, problem, epsilons, population_size=None, **kwargs):
        super().__init__(
            problem,
            epsilons,
            population_size=population_size,
            # borg_library_path="./libborgms.so",  # NOTE: For mac, use "./libborg.dylib", for linux use "./libborgms.so"
            borg_library_path=os.path.join(_dir, "solvers", "moea", "libborgms.so"),
            solve_settings={},
            seed=None,  # keep Borg's internal RNG
            direct_evaluation=True,  # use the evaluation function registered in context
            **kwargs,
        )

    def run(self, max_evaluations: int):
        """Run Borg in master–slave MPI mode (islands = 1)."""
        from solvers.moea.borg import Borg, Configuration

        if self.borg_library_path:
            Configuration.setBorgLibrary(self.borg_library_path)

        nvars = self.problem.nvars
        nobjs = self.problem.nobjs
        nconstr = getattr(self.problem, "nconstr", 0)
        callback = self._make_callback()
        borg = Borg(nvars, nobjs, nconstr, callback)

        self._set_bounds(borg)
        borg.setEpsilons(*self.epsilons)

        if _mpi_rank() == 0:
            print(f"[MSBorgMOEA] epsilons = {self.epsilons}", flush=True)

        try:
            Configuration.startMPI()
            borg_result = borg.solveMPI(
                islands=1,
                maxEvaluations=int(max_evaluations),
                runtime=b"ms.runtime",
            )
        finally:
            Configuration.stopMPI()

        self.result = []
        if borg_result is not None:
            for s_borg in borg_result:
                sol = Solution(self.problem)
                sol.variables = list(s_borg.getVariables())
                sol.objectives = list(s_borg.getObjectives())
                if nconstr:
                    sol.constraints = list(s_borg.getConstraints())
                self.result.append(sol)
            self.nfe = int(max_evaluations)
        else:
            self.nfe = 0

        # Update stub
        self.archive = _ArchiveView(self.result, improvements=len(self.result))
        self.algorithm = _AlgorithmStub(self.archive)

    def step(self):
        return


class MMBorgMOEA(BorgMOEA):
    """Multi-Master Borg (requires islands*(workers+1)+1 MPI ranks)."""

    def __init__(self, problem, epsilons, population_size=None, **kwargs):
        islands = int(os.environ.get("BORG_ISLANDS", "2"))
        super().__init__(
            problem,
            epsilons,
            population_size=population_size,
            # borg_library_path="./libborgmm.so",
            borg_library_path=os.path.join(_dir, "solvers", "moea", "libborgmm.so"),
            solve_settings={"islands": islands},
            seed=None,
            direct_evaluation=True,
            **kwargs,
        )
        self._islands = islands

    def run(self, max_evaluations: int):
        """Run Borg in multi-master MPI mode."""
        from solvers.moea.borg import Borg, Configuration

        if self.borg_library_path:
            Configuration.setBorgLibrary(self.borg_library_path)

        nvars = self.problem.nvars
        nobjs = self.problem.nobjs
        nconstr = getattr(self.problem, "nconstr", 0)
        callback = self._make_callback()
        borg = Borg(nvars, nobjs, nconstr, callback)

        self._set_bounds(borg)
        borg.setEpsilons(*self.epsilons)

        if _mpi_rank() == 0:
            print(
                f"[MMBorgMOEA] epsilons = {self.epsilons}, islands = {self._islands}",
                flush=True,
            )

        runtime_dir = os.environ.get("BORG_RUNTIME_DIR")
        if runtime_dir is None:
            raise RuntimeError(
                "BORG_RUNTIME_DIR is not set. "
                "Set this environment variable to the desired output folder before running the optimizer."
            )

        runtime_template = os.path.join(runtime_dir, "mm_%d.runtime").encode("utf-8")

        try:
            Configuration.startMPI()
            borg_result = borg.solveMPI(
                islands=self._islands,
                maxEvaluations=int(max_evaluations),
                runtime=runtime_template,
            )
        finally:
            Configuration.stopMPI()

        self.result = []
        if borg_result is not None:
            for s_borg in borg_result:
                sol = Solution(self.problem)
                sol.variables = list(s_borg.getVariables())
                sol.objectives = list(s_borg.getObjectives())
                if nconstr:
                    sol.constraints = list(s_borg.getConstraints())
                self.result.append(sol)
            self.nfe = int(max_evaluations)
        else:
            self.nfe = 0

        self.archive = _ArchiveView(self.result, improvements=len(self.result))
        self.algorithm = _AlgorithmStub(self.archive)

    def step(self):
        return


def run_optimization_adaptive(
    config_path,
    nfe=None,
    population_size=100,
    swf=0,
    seed=None,
    datapath="./data",
    filename=None,
    folder=None,
    economy_type=Economy.NEOCLASSICAL,
    damage_function_type=DamageFunction.KALKUHL,
    abatement_type=Abatement.ENERDATA,
    optimizer=Optimizer.EpsNSGAII,
    evaluator=Evaluator.SequentialEvaluator,
    reference_ssp_rcp_scenario_index=2,
):
    """Set up the EMA Model, register evaluation context, and run optimize()."""
    with open(config_path, "r") as file:
        config = json.load(file)

    start_year = config["start_year"]
    end_year = config["end_year"]
    data_timestep = config["data_timestep"]
    timestep = config["timestep"]
    emission_control_start_year = config["emission_control_start_year"]
    n_rbfs = config["n_rbfs"]
    n_inputs = config["n_inputs"]
    epsilons = config["epsilons"]
    temperature_year_of_interest = config["temperature_year_of_interest"]
    reference_index = reference_ssp_rcp_scenario_index
    stochastic_run = config["stochastic_run"]
    climate_members = config.get("climate_ensemble_members")

    social_welfare_function = WelfareFunction.from_index(swf)
    swf_type = social_welfare_function.value[0]

    model = Model("JUSTICE", function=model_wrapper_emodps)

    data_loader = DataLoader()
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

    model.constants = [
        Constant("n_regions", len(data_loader.REGION_LIST)),
        Constant("n_timesteps", len(time_horizon.model_time_horizon)),
        Constant("emission_control_start_timestep", emission_start_ts),
        Constant("n_rbfs", n_rbfs),
        Constant("n_inputs_rbf", n_inputs),
        Constant("n_outputs_rbf", len(data_loader.REGION_LIST)),
        Constant("social_welfare_function_type", swf_type),
        Constant("economy_type", economy_type.value),
        Constant("damage_function_type", damage_function_type.value),
        Constant("abatement_type", abatement_type.value),
        Constant("temperature_year_of_interest_index", temperature_year_index),
        Constant("stochastic_run", stochastic_run),
        Constant("climate_ensemble_members", climate_members),
    ]

    model.uncertainties = [CategoricalParameter("ssp_rcp_scenario", tuple(range(8)))]

    centers_shape = n_rbfs * n_inputs
    weights_shape = len(data_loader.REGION_LIST) * n_rbfs

    centers = [RealParameter(f"center_{i}", -1.0, 1.0) for i in range(centers_shape)]
    radii = [
        RealParameter(f"radii_{i}", SMALL_NUMBER, 1.0) for i in range(centers_shape)
    ]
    weights = [
        RealParameter(f"weights_{i}", SMALL_NUMBER, 1.0) for i in range(weights_shape)
    ]
    model.levers = centers + radii + weights

    model.outcomes = [
        ScalarOutcome("welfare", variable_name="welfare", kind=ScalarOutcome.MINIMIZE),
        ScalarOutcome(
            "fraction_above_threshold",
            variable_name="fraction_above_threshold",
            kind=ScalarOutcome.MINIMIZE,
        ),
    ]

    reference_scenario = Sample("reference", ssp_rcp_scenario=reference_index)

    filename = f"{social_welfare_function.value[1]}_{nfe}_{seed}.tar.gz"
    timestamp = datetime.datetime.now().strftime("%Y_%m")  # _%d_%H_%M_%S
    random_number = random.randint(0, 10000)
    directory_name = os.path.abspath(
        os.path.join(
            datapath,
            f"{social_welfare_function.value[1]}_{timestamp}_{random_number}_ref{reference_ssp_rcp_scenario_index}_{seed}",
        )
    )

    os.environ["BORG_RUNTIME_DIR"] = directory_name
    os.makedirs(directory_name, exist_ok=True)

    rank = _mpi_rank()

    if optimizer == Optimizer.EpsNSGAII:
        algorithm_class = EpsNSGAII
    elif optimizer == Optimizer.MMBorgMOEA:
        algorithm_class = MMBorgMOEA
    elif optimizer == Optimizer.MSBorgMOEA:
        algorithm_class = MSBorgMOEA
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer}")

    set_ema_context(
        model=model,
        reference=reference_scenario,
        evaluation=model_wrapper_emodps,
        reference_index=reference_index,
    )

    if evaluator == Evaluator.MPIEvaluator:
        with MPIEvaluator(model) as _evaluator:
            results = _evaluator.optimize(
                searchover="levers",
                nfe=nfe,
                epsilons=epsilons,
                reference=reference_scenario,
                filename=filename,
                directory=directory_name,
                population_size=population_size,
                algorithm=algorithm_class,
            )
    elif evaluator == Evaluator.MultiprocessingEvaluator:
        with MultiprocessingEvaluator(model) as _evaluator:
            results = _evaluator.optimize(
                searchover="levers",
                nfe=nfe,
                epsilons=epsilons,
                reference=reference_scenario,
                filename=filename,
                directory=directory_name,
                population_size=population_size,
                algorithm=algorithm_class,
            )
    else:
        with SequentialEvaluator(model) as _evaluator:
            results = _evaluator.optimize(
                searchover="levers",
                nfe=nfe,
                epsilons=epsilons,
                reference=reference_scenario,
                filename=filename,
                directory=directory_name,
                population_size=population_size,
                algorithm=algorithm_class,
            )

    if (
        rank == 0
        and optimizer == Optimizer.MMBorgMOEA
        and os.path.isdir(directory_name)
    ):
        header = [lever.name for lever in model.levers] + [
            outcome.name for outcome in model.outcomes
        ]
        islands = int(os.environ.get("BORG_ISLANDS", "2"))
        _create_intermediate_archives(directory_name, filename, islands, header)

    return results


################################################################################################################################


def run_optimization_momadps(
    config_path,
    nfe=None,
    population_size=100,
    seed=None,
    datapath="./data",
    economy_type=Economy.NEOCLASSICAL,
    damage_function_type=DamageFunction.KALKUHL,
    abatement_type=Abatement.ENERDATA,
    optimizer=Optimizer.EpsNSGAII,
    evaluator=Evaluator.SequentialEvaluator,
    reference_ssp_rcp_scenario_index=2,
    mapping_base_path="data/input",
):
    """Configure and run the MOMADPS optimization experiment."""
    with open(config_path, "r", encoding="utf-8") as file:
        config = json.load(file)

    start_year = config["start_year"]
    end_year = config["end_year"]
    data_timestep = config["data_timestep"]
    timestep = config["timestep"]
    emission_control_start_year = config["emission_control_start_year"]
    # n_rbfs = config["n_rbfs"]
    n_inputs = config["n_inputs"]  # expect 3 (T, ΔT, consumption)
    epsilons = config["epsilons"]
    temperature_year_of_interest = config["temperature_year_of_interest"]
    stochastic_run = config["stochastic_run"]
    climate_members = config.get("climate_ensemble_members")

    min_temperature = config["min_temperature"]
    max_temperature = config["max_temperature"]
    min_temperature_change = config["min_temperature_change"]
    max_temperature_change = config["max_temperature_change"]
    consumption_min = config["consumption_min"]
    consumption_max = config["consumption_max"]

    model = Model("JUSTICE", function=model_wrapper_momadps)

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

    # Build macro-region mapping
    r5_json = Path(mapping_base_path) / "R5_regions.json"
    rice50_json = Path(mapping_base_path) / "rice50_regions_dict.json"
    region_to_macro, macro_region_names = build_macro_region_mapping(
        region_list=region_list,
        r5_json_path=r5_json,
        rice50_json_path=rice50_json,
    )
    n_macro_regions = len(macro_region_names)

    # Constants fed into the wrapper
    model.constants = [
        Constant("n_regions", n_regions),
        Constant("n_timesteps", len(time_horizon.model_time_horizon)),
        Constant("emission_control_start_timestep", emission_start_ts),
        # Constant("n_rbfs", n_rbfs),
        Constant("n_inputs_rbf", n_inputs),
        Constant("n_outputs_rbf", 1),
        Constant(
            "social_welfare_function_type", WelfareFunction.from_index(0).value[0]
        ),
        Constant("economy_type", economy_type.value),
        Constant("damage_function_type", damage_function_type.value),
        Constant("abatement_type", abatement_type.value),
        Constant("temperature_year_of_interest_index", temperature_year_index),
        Constant("stochastic_run", stochastic_run),
        Constant("climate_ensemble_members", climate_members),
        Constant("region_to_macro", region_to_macro.tolist()),
        Constant("macro_region_names", list(macro_region_names)),
        Constant("n_macro_regions", n_macro_regions),
        Constant("min_temperature", min_temperature),
        Constant("max_temperature", max_temperature),
        Constant("min_temperature_change", min_temperature_change),
        Constant("max_temperature_change", max_temperature_change),
        Constant("consumption_min", consumption_min),
        Constant("consumption_max", consumption_max),
    ]

    model.uncertainties = [CategoricalParameter("ssp_rcp_scenario", tuple(range(8)))]

    # Determine lever shapes from an RBF template
    rbf_probe = RBF(
        n_rbfs=(n_inputs + 2),
        n_inputs=n_inputs,
        n_outputs=1,
    )
    centers_shape, radii_shape, weights_shape = rbf_probe.get_shape()
    centers_len, radii_len, weights_len = (
        centers_shape[0],
        radii_shape[0],
        weights_shape[0],
    )

    levers = []
    for macro_idx in range(n_macro_regions):
        levers.extend(
            RealParameter(f"center_{macro_idx}_{i}", -1.0, 1.0)
            for i in range(centers_len)
        )
        levers.extend(
            RealParameter(f"radii_{macro_idx}_{i}", SMALL_NUMBER, 1.0)
            for i in range(radii_len)
        )
        levers.extend(
            RealParameter(f"weights_{macro_idx}_{i}", SMALL_NUMBER, 1.0)
            for i in range(weights_len)
        )
    model.levers = levers

    model.outcomes = [
        ScalarOutcome(
            "macro_welfare_R5ASIA",
            variable_name="macro_welfare_R5ASIA",
            kind=ScalarOutcome.MAXIMIZE,
        ),
        ScalarOutcome(
            "macro_welfare_R5LAM",
            variable_name="macro_welfare_R5LAM",
            kind=ScalarOutcome.MAXIMIZE,
        ),
        ScalarOutcome(
            "macro_welfare_R5MAF",
            variable_name="macro_welfare_R5MAF",
            kind=ScalarOutcome.MAXIMIZE,
        ),
        ScalarOutcome(
            "macro_welfare_R5OECD",
            variable_name="macro_welfare_R5OECD",
            kind=ScalarOutcome.MAXIMIZE,
        ),
        ScalarOutcome(
            "macro_welfare_R5REF",
            variable_name="macro_welfare_R5REF",
            kind=ScalarOutcome.MAXIMIZE,
        ),
        ScalarOutcome(
            "fraction_above_threshold",
            variable_name="fraction_above_threshold",
            kind=ScalarOutcome.MINIMIZE,
        ),
    ]

    reference_scenario = Sample(
        "reference", ssp_rcp_scenario=reference_ssp_rcp_scenario_index
    )

    filename = f"MOMADPS_{nfe}_{seed}.tar.gz"
    timestamp = datetime.datetime.now().strftime("%Y_%m")
    random_number = random.randint(0, 10000)
    directory_name = os.path.abspath(
        os.path.join(
            datapath,
            f"MOMADPS_{timestamp}_{random_number}_ref{reference_ssp_rcp_scenario_index}_{seed}",
        )
    )
    os.environ["BORG_RUNTIME_DIR"] = directory_name
    os.makedirs(directory_name, exist_ok=True)

    rank = _mpi_rank()
    lever_names = [lever.name for lever in model.levers]
    outcome_names = [outcome.name for outcome in model.outcomes]

    optimizer_map = {
        Optimizer.EpsNSGAII: EpsNSGAII,
        Optimizer.MMBorgMOEA: MMBorgMOEA,
        Optimizer.MSBorgMOEA: MSBorgMOEA,
    }
    algorithm_class = optimizer_map.get(optimizer)
    if algorithm_class is None:
        raise ValueError(f"Unsupported optimizer: {optimizer}")

    set_ema_context(
        model=model,
        reference=reference_scenario,
        evaluation=model_wrapper_momadps,
        reference_index=reference_ssp_rcp_scenario_index,
    )

    evaluator_map = {
        Evaluator.MPIEvaluator: MPIEvaluator,
        Evaluator.MultiprocessingEvaluator: MultiprocessingEvaluator,
        Evaluator.SequentialEvaluator: SequentialEvaluator,
    }
    evaluator_cls = evaluator_map[evaluator]

    with evaluator_cls(model) as eval_ctx:
        results = eval_ctx.optimize(
            searchover="levers",
            nfe=nfe,
            epsilons=epsilons,
            reference=reference_scenario,
            filename=filename,
            directory=directory_name,
            population_size=population_size,
            algorithm=algorithm_class,
        )

    if (
        rank == 0
        and optimizer == Optimizer.MMBorgMOEA
        and os.path.isdir(directory_name)
    ):
        header = lever_names + outcome_names
        islands = int(os.environ.get("BORG_ISLANDS", "2"))
        _create_intermediate_archives(directory_name, filename, islands, header)

    return results


###############################################################################################################################
def run_single_agent_momadps(
    config_path: str,
    nash_profiles_path: str,
    policy_bank_path: str,
    policy_index: int,
    variable_macro_index: int,
    nfe: Optional[int] = None,
    population_size: int = 100,
    seed: Optional[int] = None,
    datapath: str = "./data",
    economy_type: Economy = Economy.NEOCLASSICAL,
    damage_function_type: DamageFunction = DamageFunction.KALKUHL,
    abatement_type: Abatement = Abatement.ENERDATA,
    optimizer: Optimizer = Optimizer.EpsNSGAII,
    evaluator: Evaluator = Evaluator.SequentialEvaluator,
    reference_ssp_rcp_scenario_index: int = 2,
    mapping_base_path: str = "data/input",
    epsilons: Optional[list[float]] = None,
):
    """
    Re-run an adaptive MOMADPS optimization where only one macro agent is allowed to
    adjust its RBF parameters, while the other four remain fixed at a known Pareto-Nash
    solution.

    The fixed parameters for each agent are looked up independently:
      - `nash_profiles_path` (pareto_nash_profiles.csv): row `policy_index` gives
        per-agent action indices a0..a4.
      - `policy_bank_path` (COMBINED_MOMA_epsilon_nondominated_set.csv): each agent i
        reads its RBF parameters from row a_i, which may differ across agents.

    The objectives:
        * Maximize the selected agent's welfare (macro-specific).
        * Minimize the fraction of ensemble runs above the temperature threshold.
    """

    with open(config_path, "r", encoding="utf-8") as file:
        config = json.load(file)

    # --- Load Nash profile row to get per-agent action indices ---
    nash_df = pd.read_csv(nash_profiles_path)
    if policy_index < -len(nash_df) or policy_index >= len(nash_df):
        raise IndexError(
            f"policy_index={policy_index} is out of bounds for CSV with {len(nash_df)} rows."
        )
    nash_row = nash_df.iloc[int(policy_index)]

    # --- Load the policy bank (5-row CSV) ---
    policy_bank_df = pd.read_csv(policy_bank_path)

    start_year = config["start_year"]
    end_year = config["end_year"]
    data_timestep = config["data_timestep"]
    timestep = config["timestep"]
    emission_control_start_year = config["emission_control_start_year"]
    n_inputs = config["n_inputs"]
    temperature_year_of_interest = config["temperature_year_of_interest"]
    stochastic_run = config["stochastic_run"]
    climate_members = config.get("climate_ensemble_members")

    min_temperature = config["min_temperature"]
    max_temperature = config["max_temperature"]
    min_temperature_change = config["min_temperature_change"]
    max_temperature_change = config["max_temperature_change"]
    consumption_min = config["consumption_min"]
    consumption_max = config["consumption_max"]

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

    r5_json = Path(mapping_base_path) / "R5_regions.json"
    rice50_json = Path(mapping_base_path) / "rice50_regions_dict.json"
    region_to_macro, macro_region_names = build_macro_region_mapping(
        region_list=region_list,
        r5_json_path=r5_json,
        rice50_json_path=rice50_json,
    )
    n_macro_regions = len(macro_region_names)

    if not (0 <= variable_macro_index < n_macro_regions):
        raise ValueError(
            f"variable_macro_index={variable_macro_index} invalid; must be in [0, {n_macro_regions-1}]"
        )

    # --- Extract per-agent action indices from the Nash profile row ---
    actions = [int(nash_row[f"a{i}"]) for i in range(n_macro_regions)]

    model = Model("JUSTICE", function=model_wrapper_momadps_single_agent)

    rbf_probe = RBF(
        n_rbfs=(n_inputs + 2),
        n_inputs=n_inputs,
        n_outputs=1,
    )
    centers_shape, radii_shape, weights_shape = rbf_probe.get_shape()
    centers_len, radii_len, weights_len = (
        centers_shape[0],
        radii_shape[0],
        weights_shape[0],
    )

    fixed_centers = np.zeros((n_macro_regions, centers_len), dtype=float)
    fixed_radii = np.zeros((n_macro_regions, radii_len), dtype=float)
    fixed_weights = np.zeros((n_macro_regions, weights_len), dtype=float)

    # --- Each agent reads from its own row in the policy bank ---
    for macro_idx in range(n_macro_regions):
        action_row = policy_bank_df.iloc[actions[macro_idx]]
        for i in range(centers_len):
            fixed_centers[macro_idx, i] = action_row[f"center_{macro_idx}_{i}"]
            fixed_radii[macro_idx, i] = action_row[f"radii_{macro_idx}_{i}"]
        for i in range(weights_len):
            fixed_weights[macro_idx, i] = action_row[f"weights_{macro_idx}_{i}"]

    model.constants = [
        Constant("n_regions", n_regions),
        Constant("n_timesteps", len(time_horizon.model_time_horizon)),
        Constant("emission_control_start_timestep", emission_start_ts),
        Constant("n_inputs_rbf", n_inputs),
        Constant("n_outputs_rbf", 1),
        Constant("social_welfare_function_type", WelfareFunction.UTILITARIAN.value[0]),
        Constant("economy_type", economy_type.value),
        Constant("damage_function_type", damage_function_type.value),
        Constant("abatement_type", abatement_type.value),
        Constant("temperature_year_of_interest_index", temperature_year_index),
        Constant("stochastic_run", stochastic_run),
        Constant("climate_ensemble_members", climate_members),
        Constant("region_to_macro", region_to_macro.tolist()),
        Constant("macro_region_names", list(macro_region_names)),
        Constant("n_macro_regions", n_macro_regions),
        Constant("min_temperature", min_temperature),
        Constant("max_temperature", max_temperature),
        Constant("min_temperature_change", min_temperature_change),
        Constant("max_temperature_change", max_temperature_change),
        Constant("consumption_min", consumption_min),
        Constant("consumption_max", consumption_max),
        Constant("variable_macro_index", variable_macro_index),
        Constant("fixed_centers", fixed_centers.tolist()),
        Constant("fixed_radii", fixed_radii.tolist()),
        Constant("fixed_weights", fixed_weights.tolist()),
    ]

    algorithm_map = {
        Optimizer.EpsNSGAII: EpsNSGAII,
        Optimizer.MMBorgMOEA: MMBorgMOEA,
        Optimizer.MSBorgMOEA: MSBorgMOEA,
    }
    algorithm_class = algorithm_map.get(optimizer)
    if algorithm_class is None:
        raise ValueError(f"Unsupported optimizer: {optimizer}")

    model.uncertainties = [CategoricalParameter("ssp_rcp_scenario", tuple(range(8)))]

    levers = []
    for i in range(centers_len):
        levers.append(RealParameter(f"center_{variable_macro_index}_{i}", -1.0, 1.0))
    for i in range(radii_len):
        levers.append(
            RealParameter(f"radii_{variable_macro_index}_{i}", SMALL_NUMBER, 1.0)
        )
    for i in range(weights_len):
        levers.append(
            RealParameter(f"weights_{variable_macro_index}_{i}", SMALL_NUMBER, 1.0)
        )
    model.levers = levers

    if epsilons is None:
        epsilons = [1e-3, 1e-3]

    agent_name = macro_region_names[variable_macro_index]
    model.outcomes = [
        ScalarOutcome(f"macro_welfare_{agent_name}", kind=ScalarOutcome.MAXIMIZE),
        ScalarOutcome("fraction_above_threshold", kind=ScalarOutcome.MINIMIZE),
    ]

    reference_scenario = Sample(
        "reference", ssp_rcp_scenario=reference_ssp_rcp_scenario_index
    )

    filename = f"SingleAgent_{variable_macro_index}_{nfe}_{seed}.tar.gz"
    timestamp = datetime.datetime.now().strftime("%Y_%m")
    random_number = random.randint(0, 10000)
    directory_name = os.path.abspath(
        os.path.join(
            datapath,
            f"single_agent_{variable_macro_index}_{timestamp}_{random_number}_ref"
            f"{reference_ssp_rcp_scenario_index}_{seed}",
        )
    )
    os.environ["BORG_RUNTIME_DIR"] = directory_name
    os.makedirs(directory_name, exist_ok=True)

    rank = _mpi_rank()

    lever_names = [lever.name for lever in model.levers]
    outcome_names = [outcome.name for outcome in model.outcomes]

    set_ema_context(
        model=model,
        reference=reference_scenario,
        evaluation=model_wrapper_momadps_single_agent,
        reference_index=reference_ssp_rcp_scenario_index,
    )

    evaluator_map = {
        Evaluator.MPIEvaluator: MPIEvaluator,
        Evaluator.MultiprocessingEvaluator: MultiprocessingEvaluator,
        Evaluator.SequentialEvaluator: SequentialEvaluator,
    }
    evaluator_cls = evaluator_map[evaluator]

    with evaluator_cls(model) as eval_ctx:
        results = eval_ctx.optimize(
            searchover="levers",
            nfe=nfe,
            epsilons=epsilons,
            reference=reference_scenario,
            filename=filename,
            directory=directory_name,
            population_size=population_size,
            algorithm=algorithm_class,
        )

    if (
        rank == 0
        and optimizer == Optimizer.MMBorgMOEA
        and os.path.isdir(directory_name)
    ):
        header = lever_names + outcome_names
        islands = int(os.environ.get("BORG_ISLANDS", "2"))
        _create_intermediate_archives(directory_name, filename, islands, header)

    return results


###############################################################################################################################


def build_random_policy(model, seed=1234):
    """
    This is for testing purposes only: builds a random policy within the lever bounds.

    Args:
        model (ema_workbench.Model): The EMA Workbench model with defined levers.
        seed (int): Random seed for reproducibility.
    """

    rng = np.random.default_rng(seed)
    policy_data = {}
    for lever in model.levers:
        low, high = lever.lower_bound, lever.upper_bound
        policy_data[lever.name] = rng.uniform(low, high)
    return Sample("random_policy", **policy_data)


def setup_model_from_config(config_path, mapping_base_path="data/input"):
    with open(config_path, "r", encoding="utf-8") as file:
        config = json.load(file)

    start_year = config["start_year"]
    end_year = config["end_year"]
    data_timestep = config["data_timestep"]
    timestep = config["timestep"]
    emission_control_start_year = config["emission_control_start_year"]
    n_rbfs = config["n_rbfs"]
    n_inputs = config["n_inputs"]
    temperature_year_of_interest = config["temperature_year_of_interest"]
    stochastic_run = config["stochastic_run"]
    climate_members = config.get("climate_ensemble_members")

    min_temperature = config["min_temperature"]
    max_temperature = config["max_temperature"]
    min_temperature_change = config["min_temperature_change"]
    max_temperature_change = config["max_temperature_change"]
    consumption_min = config["consumption_min"]
    consumption_max = config["consumption_max"]

    model = Model("JUSTICE", function=model_wrapper_momadps)

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

    r5_json = Path(mapping_base_path) / "R5_regions.json"
    rice50_json = Path(mapping_base_path) / "rice50_regions_dict.json"
    region_to_macro, macro_region_names = build_macro_region_mapping(
        region_list=region_list,
        r5_json_path=r5_json,
        rice50_json_path=rice50_json,
    )
    n_macro_regions = len(macro_region_names)

    model.constants = [
        Constant("n_regions", n_regions),
        Constant("n_timesteps", len(time_horizon.model_time_horizon)),
        Constant("emission_control_start_timestep", emission_start_ts),
        Constant("n_rbfs", n_rbfs),
        Constant("n_inputs_rbf", n_inputs),
        Constant("n_outputs_rbf", 1),
        Constant(
            "social_welfare_function_type", WelfareFunction.from_index(0).value[0]
        ),
        Constant("economy_type", Economy.NEOCLASSICAL.value),
        Constant("damage_function_type", DamageFunction.KALKUHL.value),
        Constant("abatement_type", Abatement.ENERDATA.value),
        Constant("temperature_year_of_interest_index", temperature_year_index),
        Constant("stochastic_run", stochastic_run),
        Constant("climate_ensemble_members", climate_members),
        Constant("region_to_macro", region_to_macro.tolist()),
        Constant("macro_region_names", list(macro_region_names)),
        Constant("n_macro_regions", n_macro_regions),
        Constant("min_temperature", min_temperature),
        Constant("max_temperature", max_temperature),
        Constant("min_temperature_change", min_temperature_change),
        Constant("max_temperature_change", max_temperature_change),
        Constant("consumption_min", consumption_min),
        Constant("consumption_max", consumption_max),
    ]

    model.uncertainties = [CategoricalParameter("ssp_rcp_scenario", tuple(range(8)))]

    rbf_probe = model_wrapper_momadps.__globals__["RBF"](
        n_rbfs=(n_inputs + 2),
        n_inputs=n_inputs,
        n_outputs=1,
    )
    centers_shape, radii_shape, weights_shape = rbf_probe.get_shape()
    centers_len, radii_len, weights_len = (
        centers_shape[0],
        radii_shape[0],
        weights_shape[0],
    )

    levers = []
    for macro_idx in range(n_macro_regions):
        levers.extend(
            RealParameter(f"center_{macro_idx}_{i}", -1.0, 1.0)
            for i in range(centers_len)
        )
        levers.extend(
            RealParameter(f"radii_{macro_idx}_{i}", SMALL_NUMBER, 1.0)
            for i in range(radii_len)
        )
        levers.extend(
            RealParameter(f"weights_{macro_idx}_{i}", SMALL_NUMBER, 1.0)
            for i in range(weights_len)
        )
    model.levers = levers

    model.outcomes = [
        ScalarOutcome(
            "macro_welfare_R5ASIA",
            variable_name="macro_welfare_R5ASIA",
            kind=ScalarOutcome.MAXIMIZE,
        ),
        ScalarOutcome(
            "macro_welfare_R5LAM",
            variable_name="macro_welfare_R5LAM",
            kind=ScalarOutcome.MAXIMIZE,
        ),
        ScalarOutcome(
            "macro_welfare_R5MAF",
            variable_name="macro_welfare_R5MAF",
            kind=ScalarOutcome.MAXIMIZE,
        ),
        ScalarOutcome(
            "macro_welfare_R5OECD",
            variable_name="macro_welfare_R5OECD",
            kind=ScalarOutcome.MAXIMIZE,
        ),
        ScalarOutcome(
            "macro_welfare_R5REF",
            variable_name="macro_welfare_R5REF",
            kind=ScalarOutcome.MAXIMIZE,
        ),
        ScalarOutcome(
            "fraction_above_threshold",
            variable_name="fraction_above_threshold",
            kind=ScalarOutcome.MINIMIZE,
        ),
    ]

    return model, macro_region_names


###############################################################################################################################

# EMODPS Runs
if __name__ == "__main__":
    config_path = "analysis/normative_uncertainty_optimization.json"

    ema_logging.log_to_stderr(ema_logging.INFO)

    run_optimization_adaptive(
        config_path=config_path,
        nfe=10,
        swf=0,
        seed=10,  # None for Borg. Any integer for reproducibility with other optimizers
        datapath="./data",
        optimizer=Optimizer.MSBorgMOEA,  # Optimizer.MMBorgMOEA, Optimizer.EpsNSGAII
        population_size=2,  # default is 100. Test locally with 2
        reference_ssp_rcp_scenario_index=2,  # NOTE #TODO Get this from config json
        evaluator=Evaluator.SequentialEvaluator,
    )


# Single Agent Runs
# if __name__ == "__main__":
# config_path = "analysis/momadps_config.json"

# ema_logging.log_to_stderr(ema_logging.INFO)

# results = run_single_agent_momadps(
#     config_path="analysis/momadps_config.json",
#     nash_profiles_path="pareto_nash_profiles.csv",
#     policy_bank_path="COMBINED_MOMA_epsilon_nondominated_set.csv",
#     policy_index=9,  # row in pareto_nash_profiles.csv
#     variable_macro_index=0,  # 0=R5ASIA, 1=R5LAM, etc.
#     nfe=50,
#     population_size=2,
#     epsilons=[1e-3, 1e-3],
#     seed=10,
#     datapath="./data",
#     optimizer=Optimizer.MSBorgMOEA,
#     reference_ssp_rcp_scenario_index=2,
# )

# Multi-Agent Runs (full MOMADPS optimization)
# if __name__ == "__main__":
#     config_path = "analysis/momadps_config.json"

#     ema_logging.log_to_stderr(ema_logging.INFO)

#     run_optimization_momadps(
#         config_path=config_path,
#         nfe=50,
#         # swf=0,
#         seed=10,  # None for Borg. Any integer for reproducibility with other optimizers
#         datapath="./data",
#         optimizer=Optimizer.MSBorgMOEA,  # Optimizer.MMBorgMOEA, Optimizer.EpsNSGAII
#         population_size=2,  # default is 100. Test locally with 2
#         reference_ssp_rcp_scenario_index=2,  # NOTE #TODO Get this from config json
#         evaluator=Evaluator.SequentialEvaluator,
#     )


###############################################################################################################################

# if __name__ == "__main__":
#     config_path = "analysis/momadps_config.json"

#     model, macro_region_names = setup_model_from_config(config_path)

#     reference_scenario = Scenario("debug", ssp_rcp_scenario=2)
#     policy = build_random_policy(model, seed=42)

#     with SequentialEvaluator(model) as evaluator:
#         experiments, outcomes = perform_experiments(
#             model,
#             scenarios=[reference_scenario],
#             policies=[policy],
#             evaluator=evaluator,
#         )

#     print("Experiment outcomes:")
#     for name, values in outcomes.items():
#         print(f"{name}: {values}")

#     print("\nPolicy used:")
#     print(dict(policy))
