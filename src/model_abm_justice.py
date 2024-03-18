from src.model import JUSTICE
from src.exploration.information import Information
from src.exploration.twolevelsgame import TwoLevelsGame
from src.util.enumerations import Economy, DamageFunction, Abatement, WelfareFunction
import numpy as np


class AbmJustice(JUSTICE):
    def __init__(
        self,
        start_year=2015,  # Model is only tested for start year 2015
        end_year=2300,  # Model is only tested for end year 2300
        timestep=1,  # Model is only tested for timestep 1
        scenario=0,
        climate_ensembles=None,
        economy_type=Economy.NEOCLASSICAL,
        damage_function_type=DamageFunction.KALKUHL,
        abatement_type=Abatement.ENERDATA,
        social_welfare_function=WelfareFunction.UTILITARIAN,
        seed=0,
        **kwargs,
    ):
        # INSTANTIATE JUSTICE MODULE
        print("--> Initialisation of ABM-JUSTICE model")
        self.rng = np.random.default_rng(seed=seed);
        print("   -> Instantiation of JUSTICE module")
        JUSTICE.__init__(self)
        # Populating data with new informations
        self.data["emission_cutting_rate"] = np.zeros(
            (
                len(self.data_loader.REGION_LIST),
                len(self.time_horizon.model_time_horizon),
                self.no_of_ensembles,
            )
        )
        print("      OK")
        # INSTANTIATE POLICY MODULE
        print("   -> Instantiation of policy module")
        self.two_levels_game = TwoLevelsGame(self, timestep=timestep)
        print("      OK")

        # INSTANTIATE INFORMATION MODULE
        print("   -> Instantiation of information module")
        self.information_model = Information(
            self,
            start_year=2015,  # Model is only tested for start year 2015
            end_year=2300,  # Model is only tested for end year 2300
            timestep=1,  # Model is only tested for timestep 1
            scenario=0,
            climate_ensembles=None,
            economy_type=Economy.NEOCLASSICAL,
            damage_function_type=DamageFunction.KALKUHL,
            abatement_type=Abatement.ENERDATA,
            social_welfare_function=WelfareFunction.UTILITARIAN,
            **kwargs,
        )

        self.information_model.generate_information(self, self.emission_control_rate)
        print("      OK")
        print("--> ABM-JUSTICE MODEL INSTANTIATED: OK")

    def abm_stepwise_run(
        self,
        timestep,
        savings_rate=None,
        endogenous_savings_rate=False,
    ):
        self.information_model.step(timestep)
        self.two_levels_game.step(timestep)
        emission_control_rate = self.emission_control_rate[:, timestep]
        self.stepwise_run(
            emission_control_rate,
            timestep,
            savings_rate=None,
            endogenous_savings_rate=False,
        )

    def close_files(self):
        self.two_levels_game.close_files()
