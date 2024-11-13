import xml.etree.ElementTree as ET


class DataLoaderTwoLevelGame:
    def __init__(self, src):
        self.tree = ET.parse(src)
        self.root = self.tree.getroot()

        self.dict = {}

        self.dict["seed"] = self.get_value(
            'Class/[@name="General"]/Attribute/[@name="seed"]'
        )
        self.dict["model_HK"] = self.get_value(
            'Class/[@name="General"]/Attribute/[@name="model_HK"]'
        )
        self.dict["loss_and_damages_neutral"] = self.get_value(
            'Class/[@name="General"]/Attribute/[@name="loss_and_damages_neutral"]'
        )

        self.dict["TwoLevelsGame_Y_nego"] = self.get_value(
            'Class/[@name="TwoLevelsGame"]/Attribute/[@name="Y_nego"]'
        )
        self.dict["TwoLevelsGame_Y_policy"] = self.get_value(
            'Class/[@name="TwoLevelsGame"]/Attribute/[@name="Y_policy"]'
        )

        self.dict["Region_n_households"] = self.get_value(
            'Class/[@name="Region"]/Attribute/[@name="n_households"]'
        )
        self.dict["Region_opdyn_max_iter"] = self.get_value(
            'Class/[@name="Region"]/Attribute/[@name="opdyn_max_iter"]'
        )
        self.dict["Region_opdyn_influence_close"] = self.get_value(
            'Class/[@name="Region"]/Attribute/[@name="opdyn_influence_close"]'
        )
        self.dict["Region_opdyn_influence_far"] = self.get_value(
            'Class/[@name="Region"]/Attribute/[@name="opdyn_influence_far"]'
        )
        self.dict["Region_opdyn_agreement"] = self.get_value(
            'Class/[@name="Region"]/Attribute/[@name="opdyn_agreement"]'
        )
        self.dict["Region_opdyn_threshold_close"] = self.get_value(
            'Class/[@name="Region"]/Attribute/[@name="opdyn_threshold_close"]'
        )
        self.dict["Region_opdyn_threshold_far"] = self.get_value(
            'Class/[@name="Region"]/Attribute/[@name="opdyn_threshold_far"]'
        )

        self.dict["Negotiator_policy_start_year"] = self.get_value(
            'Class/[@name="Negotiator"]/Attribute/[@name="policy_start_year"]'
        )
        self.dict["Negotiator_policy_end_year"] = self.get_value(
            'Class/[@name="Negotiator"]/Attribute/[@name="policy_end_year"]'
        )
        self.dict["Negotiator_ecr_start_year"] = self.get_value(
            'Class/[@name="Negotiator"]/Attribute/[@name="ecr_start_year"]'
        )
        self.dict["Negotiator_ecr_first_term"] = self.get_value(
            'Class/[@name="Negotiator"]/Attribute/[@name="ecr_first_term"]'
        )
        self.dict["Negotiator_ecr_end_year"] = self.get_value(
            'Class/[@name="Negotiator"]/Attribute/[@name="ecr_end_year"]'
        )
        self.dict["Negotiator_max_cutting_rate_gradient"] = self.get_value(
            'Class/[@name="Negotiator"]/Attribute/[@name="max_cutting_rate_gradient"]'
        )

        self.dict["Household_DISTRIB_RESOLUTION"] = self.get_value(
            'Class/[@name="Household"]/Attribute/[@name="DISTRIB_RESOLUTION"]'
        )
        self.dict["Household_DISTRIB_MIN_VALUE"] = self.get_value(
            'Class/[@name="Household"]/Attribute/[@name="DISTRIB_MIN_VALUE"]'
        )
        self.dict["Household_DISTRIB_MAX_VALUE"] = self.get_value(
            'Class/[@name="Household"]/Attribute/[@name="DISTRIB_MAX_VALUE"]'
        )
        self.dict["Household_BELIEF_YEAR_OFFSET"] = self.get_value(
            'Class/[@name="Household"]/Attribute/[@name="BELIEF_YEAR_OFFSET"]'
        )
        self.dict["Household_DEFAULT_INFORMATION_STD"] = self.get_value(
            'Class/[@name="Household"]/Attribute/[@name="DEFAULT_INFORMATION_STD"]'
        )
        self.dict["Household_P_INFORMATION"] = self.get_value(
            'Class/[@name="Household"]/Attribute/[@name="P_INFORMATION"]'
        )
        self.dict["Household_GAMMA"] = self.get_value(
            'Class/[@name="Household"]/Attribute/[@name="GAMMA"]'
        )
        self.dict["Household_climate_init_mean_beliefs_floor"] = self.get_value(
            'Class/[@name="Household"]/Attribute/[@name="climate_init_mean_beliefs_floor"]'
        )
        self.dict["Household_climate_init_mean_beliefs_var"] = self.get_value(
            'Class/[@name="Household"]/Attribute/[@name="climate_init_mean_beliefs_var"]'
        )
        self.dict["Household_climate_init_var_beliefs_current"] = self.get_value(
            'Class/[@name="Household"]/Attribute/[@name="climate_init_var_beliefs_current"]'
        )
        self.dict["Household_climate_init_var_beliefs_offset1"] = self.get_value(
            'Class/[@name="Household"]/Attribute/[@name="climate_init_var_beliefs_offset1"]'
        )
        self.dict["Household_ecr_end_year"] = self.get_value(
            'Class/[@name="Household"]/Attribute/[@name="ecr_end_year"]'
        )
        self.dict["sentiment_temperature_increase"] = self.get_value(
            'Class/[@name="Household"]/Attribute/[@name="sentiment_temperature_increase"]'
        )
        self.dict["sentiment_willingness_to_pay"] = self.get_value(
            'Class/[@name="Household"]/Attribute/[@name="sentiment_willingness_to_pay"]'
        )

        self.dict["Region_alpha1"] = self.get_value(
            'Class/[@name="Region"]/Attribute/[@name="alpha1"]'
        )
        self.dict["Region_alpha2"] = self.get_value(
            'Class/[@name="Region"]/Attribute/[@name="alpha2"]'
        )
        self.dict["Region_beta1"] = self.get_value(
            'Class/[@name="Region"]/Attribute/[@name="beta1"]'
        )
        self.dict["Region_beta2"] = self.get_value(
            'Class/[@name="Region"]/Attribute/[@name="beta2"]'
        )
        self.dict["Region_gamma"] = self.get_value(
            'Class/[@name="Region"]/Attribute/[@name="gamma"]'
        )
        self.dict["IPCC_report_period"] = self.get_value(
            'Class/[@name="TwoLevelsGame"]/Attribute/[@name="IPCC_report_period"]'
        )
        self.dict["weight_info_dmg_local"] = self.get_value(
            'Class/[@name="Household"]/Attribute/[@name="weight_info_dmg_local"]'
        )
        self.dict["factor_conflict_coefficient"] = self.get_value(
            'Class/[@name="Household"]/Attribute/[@name="factor_conflict_coefficient"]'
        )
        self.dict["HK_epsilon_dmg"] = self.get_value(
            'Class/[@name="Region"]/Attribute/[@name="HK_epsilon_dmg"]'
        )
        self.dict["HK_epsilon_support"] = self.get_value(
            'Class/[@name="Region"]/Attribute/[@name="HK_epsilon_support"]'
        )
        self.dict["HK_influence_close"] = self.get_value(
            'Class/[@name="Region"]/Attribute/[@name="HK_influence_close"]'
        )

    def get_value(self, expression):
        findings = self.root.find(expression)
        type_ = findings.attrib["type"]
        # print(findings.text)
        match type_:
            case "float":
                value = float(findings.text)
            case "int":
                value = int(findings.text)
            case "List[float]":
                value = [float(e) for e in findings.text.split(",")]
            case _:
                raise Exception(
                    type_
                    + " ::: Is not known as a correct type when reading the values in 'init_values.xml'."
                )
        return value

    def modify(self, dict_changes):
        for key, value in dict_changes.items():
            self.dict[key] = value


XML_init_values = DataLoaderTwoLevelGame(
    "data/input/inputs_ABM/init_values_default.xml"
)
