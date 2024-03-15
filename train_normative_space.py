import numpy as np
import subprocess
# Example usage
time_preferences = np.linspace(.96, 0.99, 4)
inequality_aversions = np.linspace(0,2,4)
#scenarios = range(0,7)
for tp in time_preferences:
    for ia in inequality_aversions:
        subprocess.run([
            "python",
            "train_ppo_pettingzoo.py",
            "--gamma",
            str(tp),
            "--inequality_aversion",
            str(ia)
        ])