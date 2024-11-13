from src.exploration.DataLoaderTwoLevelGame import XML_init_values
from src.exploration.LogFiles import print_log
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
        seed=None,
        **kwargs,
    ):

        #If **kwargs exist, then modify the paramters accordingly
        XML_init_values.modify(kwargs)
        XML_init_values.dict['seed'] = seed
        # Save simulation parameters
        print_log.f_parameters.write("scenario: "+str(scenario)+"\n")
        for key, value in XML_init_values.dict.items():
            print_log.f_parameters.write(f"{key}: {value}\n")

        # INSTANTIATE JUSTICE MODULE
        print("--> Initialisation of ABM-JUSTICE model")
        self.rng = np.random.default_rng(seed=seed)
        print("   -> Instantiation of JUSTICE module")
        JUSTICE.__init__(self)

        consumption_per_capita = []

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
        self.two_levels_game = TwoLevelsGame(self.rng, self, timestep=timestep)
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

        self.information_model.generate_information()
        print("      OK")
        print("--> ABM-JUSTICE MODEL INSTANTIATED: OK")

    def abm_stepwise_run(
        self,
        timestep,
        savings_rate=None,
        endogenous_savings_rate=False,
    ):
        emission_control_rate = self.emission_control_rate[:, timestep]
        self.stepwise_run(
            emission_control_rate=emission_control_rate,
            timestep=timestep,
            savings_rate=savings_rate,
            endogenous_savings_rate=endogenous_savings_rate,
        )
        print_log.f_output_fair[1].writerow(
            [timestep] + self.data["global_temperature"][(timestep), :].tolist()
        )
        self.information_model.step(timestep)
        self.two_levels_game.step(timestep)

    def full_run(self, max_time_steps=85):
        print("Step-by-step run:")
        for timestep in range(max_time_steps):

            self.abm_stepwise_run(
                timestep=timestep, endogenous_savings_rate=True
            )  # savings_rate = fixed_savings_rate[:, timestep],
            # datasets = self.stepwise_evaluate(timestep=timestep)
            if timestep % 5 == 0:
                print("      >>> ", timestep, "of ", max_time_steps)

        self.close_files()
        print("DONE! :D")

    def close_files(self):
        print_log.close_files()

