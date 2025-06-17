# Check this method out too: #NOTE ROUGH

from justice.util.output_data_processor import process_scenario

import os
import filecmp
import pandas as pd
from pathlib import Path
import multiprocessing as mp
from functools import partial
from justice.util.enumerations import WelfareFunction, SSP
from justice.util.output_data_processor import process_scenario


if __name__ == "__main__":
    ######################################
    scenario_list = ["SSP126", "SSP245", "SSP370", "SSP460", "SSP534"]

    social_welfare_function = WelfareFunction.PRIORITARIAN
    nfe = 50_000
    ssp = SSP.SSP2

    sw_name = social_welfare_function.value[1]
    path = f"data/temporary/NU_DATA/combined/{str(ssp).split('.')[1]}/"
    filename = f"{sw_name}_reference_set.csv"

    # Print the selected policy indices values of last 4 columns
    # Find the min and max welfare values for utilitarian and prioritarian
    loaded_df = pd.read_csv(path + filename)
    policy_indices = list(range(len(loaded_df)))

    sw_name = social_welfare_function.value[1]

    filename = f"{sw_name}_reference_set.csv"

    loaded_df = pd.read_csv(path + filename)
    print(f"Loading data for {sw_name} from {path+filename}")
    print("Selected policy‑indices last 2 columns:")
    print(loaded_df.iloc[policy_indices, -2:])

    # spawn‐based pool so that each worker is a fresh interpreter
    mp.set_start_method("spawn")
    # Bind the fixed arguments swf and policy_indices to process_scenario
    bound_process_scenario = partial(
        process_scenario, social_welfare_function, path, policy_indices
    )
    with mp.Pool(processes=len(scenario_list)) as pool:
        pool.map(bound_process_scenario, scenario_list)
