import time

from src.drafts_and_tests.utils_save import visualize_policy
from src.drafts_and_tests.utils_save_household_thresholds import (
    visualize_household_thresholds,
)

path = "../../data/output/SAVE_2024_09_19_1013_TOKEEP/"
region_list = [32]
print("--> Visualizing results for regions: ", region_list)
print("   -> Save directory is: ", path)
for region in region_list:
    print("      -> Region ", region)
    time.sleep(1)
    visualize_household_thresholds(path, region)
    time.sleep(1)
    visualize_policy(path, region)
    print("         L> OK")
