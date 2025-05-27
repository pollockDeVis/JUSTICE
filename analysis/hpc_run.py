from analysis.analyzer import run_optimization_adaptive
import sys
import random
import numpy as np
from justice.util.enumerations import Optimizer, Evaluator

config_path = (
    "analysis/limitarian_opt_config.json"  # This loads the config used in the Paper
)

if __name__ == "__main__":
    nfe = int(sys.argv[1]) if len(sys.argv) > 1 else 5  # default value 5
    swf = int(sys.argv[2]) if len(sys.argv) > 2 else 0  # default value 0
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 5000  # default value 5000

    # Setting the seed
    random.seed(seed)
    np.random.seed(seed)
    run_optimization_adaptive(
        config_path=config_path,
        nfe=nfe,
        swf=swf,
        seed=seed,
        datapath="./data",
        optimizer=Optimizer.EpsNSGAII,  # Optimizer.BorgMOEA,
        evaluator=Evaluator.SequentialEvaluator,
    )
