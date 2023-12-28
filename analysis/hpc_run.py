from analysis.analyzer import perform_exploratory_analysis, run_optimization_adaptive
import sys


if __name__ == "__main__":
    # stat = sys.argv[1] if len(sys.argv) > 1 else "mean" #HPC shell script input
    run_optimization_adaptive(n_rbfs=4, n_inputs=2, nfe=5)
