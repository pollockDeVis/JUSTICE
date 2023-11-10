from analysis.analyzer import perform_exploratory_analysis
import sys

if __name__ == "__main__":
    # stat = sys.argv[1] if len(sys.argv) > 1 else "mean" #HPC shell script input
    perform_exploratory_analysis(number_of_experiments=2000)
