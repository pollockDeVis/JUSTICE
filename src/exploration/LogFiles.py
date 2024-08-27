from datetime import datetime
import os
import csv
import json
import pandas as pd
import numpy as np
from src.exploration.DataLoaderTwoLevelGame import XML_init_values
from src.exploration.household import Household


class LogFiles:
    def __init__(self):
        # -> Create folder for current simulation
        path = "data/output/" + datetime.now().strftime("SAVE_%Y_%m_%d_%H%M") + "/"
        os.makedirs(path, exist_ok=True)

        # average net consumption, loss and damage per quintile, abatement costs per quintile, post-damage concumption per quintile, average of post damage concumption per quintile
        f1 = open(path + "regions.csv", "w", newline="")
        self.f_region = (
            f1,
            csv.writer(f1, delimiter=",", quotechar="|", quoting=csv.QUOTE_MINIMAL),
        )
        self.f_region[1].writerow(
            [
                "Timestep",
                "Region ID",
                "Region code",
                "First 20%",
                "Second 20%",
                "Third 20%",
                "Fourth 20%",
                "Fifth 20%",
                "JUSTICE net avg. cons.",
                "JUSTICE dmg.",
                "JUSTICE abt.",
                "First dmg.",
                "Second dmg.",
                "Third dmg.",
                "Fourth dmg.",
                "Fifth dmg.",
                "First abt.",
                "Second abt.",
                "Third abt.",
                "Fourth abt.",
                "Fifth abt.",
                "First cons. pre",
                "Second cons. pre",
                "Third cons. pre",
                "Fourth cons. pre",
                "Fifth cons. pre",
                "First cons. post",
                "Second cons. post",
                "Third cons. post",
                "Fourth cons. post",
                "Fifth cons. post",
                "net avg. cons.",
            ]
        )

        # self.f_policy[1].writerow(['Region ID', 'Timestep', 'Policy Size', 'Range of Shift', 'Delta Shift', 'Policy goals', 'Policy Years', 'Support'])
        f2 = open(path + "policy.csv", "w", newline="")
        self.f_policy = (
            f2,
            csv.writer(f2, delimiter=",", quotechar="|", quoting=csv.QUOTE_MINIMAL),
        )

        f3 = open(path + "negotiator.csv", "w", newline="")
        self.f_negotiator = (
            f3,
            csv.writer(f3, delimiter=",", quotechar="|", quoting=csv.QUOTE_MINIMAL),
        )

        f4 = open(path + "information.csv", "w", newline="")
        self.f_information = (
            f4,
            csv.writer(f4, delimiter=",", quotechar="|", quoting=csv.QUOTE_MINIMAL),
        )

        f5 = open(path + "household.csv", "w", newline="")
        self.f_household = (
            f5,
            csv.writer(f5, delimiter=",", quotechar="|", quoting=csv.QUOTE_MINIMAL),
        )
        self.f_household[1].writerow(
            ["Timestep", "Region"]
            + [
                "Household Threshold"
                for i in range(XML_init_values.Region_n_households)
            ]
            + ["B0 Climate Change" for i in range(XML_init_values.Region_n_households)]
            + [
                "Emotion Climate Change"
                for i in range(XML_init_values.Region_n_households)
            ]
            + [
                "Opinion Climate Change"
                for i in range(XML_init_values.Region_n_households)
            ]
            + ["B0 Economy" for i in range(XML_init_values.Region_n_households)]
            + ["Emotion Economy" for i in range(XML_init_values.Region_n_households)]
            + ["Opinion Economy" for i in range(XML_init_values.Region_n_households)]
        )

        f5_beliefs = open(path + "household_beliefs.csv", "w", newline="")
        self.f_household_beliefs = (
            f5_beliefs,
            csv.writer(
                f5_beliefs, delimiter=",", quotechar="|", quoting=csv.QUOTE_MINIMAL
            ),
        )
        self.f_household_beliefs[1].writerow(
            ["Timestep", "Region", "Household ID"]
            + [
                "belief T(y+99)=" + str(i)
                for i in np.arange(
                    Household.DISTRIB_MIN_VALUE,
                    Household.DISTRIB_MAX_VALUE,
                    Household.DISTRIB_RESOLUTION,
                )
            ]
        )
        f_household_assessment = open(
            path + "household_policy_assessment.csv", "w", newline=""
        )
        self.f_household_assessment = (
            f_household_assessment,
            csv.writer(
                f_household_assessment,
                delimiter=",",
                quotechar="|",
                quoting=csv.QUOTE_MINIMAL,
            ),
        )
        self.f_household_assessment[1].writerow(
            [
                "Timestep",
                "Region",
                "Household ID",
                "sensitivity to costs",
                "loss_and_damages",
                "mitigation_costs",
                "experienced_economic_context",
                "expected_loss_and_damages",
                "expected_mitigation_costs",
                "expected_economic_context",
                "expected temperature",
                "current policy level",
                "Resulting Utility",
            ]
        )

        # id region, emission control rate profile (historical + pledges)
        f6 = open(path + "emissions.csv", "w", newline="")
        self.f_emissions = (
            f6,
            csv.writer(f6, delimiter=",", quotechar="|", quoting=csv.QUOTE_MINIMAL),
        )

        self.f_parameters = open(path + "parameters.txt", "w", newline="")

    def close_files(self):
        self.f_policy[0].close()
        self.f_region[0].close()
        self.f_household[0].close()
        self.f_negotiator[0].close()
        self.f_information[0].close()
        self.f_household_assessment[0].close()
        self.f_household_beliefs[0].close()
        self.f_parameters.close()
        return
