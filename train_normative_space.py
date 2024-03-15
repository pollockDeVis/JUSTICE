import numpy as np
import subprocess
# Example usage
time_preferences = np.linspace(0, 0.03, 4)
inequality_aversions = np.linspace(0,2,4)
#scenarios = range(0,7)
for tp in time_preferences:
    for ia in inequality_aversions:
        subprocess.run([
            "python",
            "train_ppo_pettingzoo.py",
            "--time_preference",
            str(tp),
            "--inequality_aversion",
            str(ia)
        ])