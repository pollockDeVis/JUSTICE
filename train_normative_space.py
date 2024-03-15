import numpy as np
import subprocess
# Example usage
time_preferences = np.linspace(0, 0.03, 10)
inequality_aversions = np.linspace(0,2,10)

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