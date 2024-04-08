from analysis.analyzer import run_optimization_adaptive
import sys


if __name__ == "__main__":
    nfe = int(sys.argv[1]) if len(sys.argv) > 1 else 5  # default value 5
    swf = int(sys.argv[2]) if len(sys.argv) > 2 else 0  # default value 0
    run_optimization_adaptive(n_rbfs=4, n_inputs=2, nfe=nfe, swf=swf)
