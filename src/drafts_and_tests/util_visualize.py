import time

from src.drafts_and_tests.utils_save_household_thresholds import (
    visualize_household_thresholds,
)
from src.drafts_and_tests.utils_visualize_HK_opinion_dynamics import visualize_HK_OpDyn
from src.drafts_and_tests.utils_visualize_policy import visualize_policy

path = "../../data/output/SAVE_2024_11_07_0918/"
region_list = [54]
print("--> Visualizing results for regions: ", region_list)
print("   -> Save directory is: ", path)
for region in region_list:
    print("      -> Region ", region)
    time.sleep(1)
    visualize_HK_OpDyn(path, region)
    time.sleep(1)
    visualize_policy(path, region)
    print("         L> OK")
