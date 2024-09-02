import xml.etree.ElementTree as ET


class DataLoaderTwoLevelGame:
    def __init__(self):
        self.tree = ET.parse("data/input/inputs_ABM/init_values.xml")
        self.root = self.tree.getroot()

        self.TwoLevelsGame_Y_nego = self.get_value(
            'Class/[@name="TwoLevelsGame"]/Attribute/[@name="Y_nego"]'
        )
        self.TwoLevelsGame_Y_policy = self.get_value(
            'Class/[@name="TwoLevelsGame"]/Attribute/[@name="Y_policy"]'
        )

        self.Region_n_households = self.get_value(
            'Class/[@name="Region"]/Attribute/[@name="n_households"]'
        )
        self.Region_opdyn_max_iter = self.get_value(
            'Class/[@name="Region"]/Attribute/[@name="opdyn_max_iter"]'
        )
        self.Region_opdyn_influence_close = self.get_value(
            'Class/[@name="Region"]/Attribute/[@name="opdyn_influence_close"]'
        )
        self.Region_opdyn_influence_far = self.get_value(
            'Class/[@name="Region"]/Attribute/[@name="opdyn_influence_far"]'
        )
        self.Region_opdyn_agreement = self.get_value(
            'Class/[@name="Region"]/Attribute/[@name="opdyn_agreement"]'
        )
        self.Region_opdyn_threshold_close = self.get_value(
            'Class/[@name="Region"]/Attribute/[@name="opdyn_threshold_close"]'
        )
        self.Region_opdyn_threshold_far = self.get_value(
            'Class/[@name="Region"]/Attribute/[@name="opdyn_threshold_far"]'
        )

        self.Negotiator_policy_start_year = self.get_value(
            'Class/[@name="Negotiator"]/Attribute/[@name="policy_start_year"]'
        )
        self.Negotiator_policy_end_year = self.get_value(
            'Class/[@name="Negotiator"]/Attribute/[@name="policy_end_year"]'
        )
        self.Negotiator_policy_period = self.get_value(
            'Class/[@name="Negotiator"]/Attribute/[@name="policy_period"]'
        )
        self.Negotiator_ecr_start_year = self.get_value(
            'Class/[@name="Negotiator"]/Attribute/[@name="ecr_start_year"]'
        )
        self.Negotiator_ecr_first_term = self.get_value(
            'Class/[@name="Negotiator"]/Attribute/[@name="ecr_first_term"]'
        )
        self.Negotiator_ecr_end_year = self.get_value(
            'Class/[@name="Negotiator"]/Attribute/[@name="ecr_end_year"]'
        )
        self.Negotiator_max_cutting_rate_gradient = self.get_value(
            'Class/[@name="Negotiator"]/Attribute/[@name="max_cutting_rate_gradient"]'
        )

        self.Household_DISTRIB_RESOLUTION = self.get_value(
            'Class/[@name="Household"]/Attribute/[@name="DISTRIB_RESOLUTION"]'
        )
        self.Household_DISTRIB_MIN_VALUE = self.get_value(
            'Class/[@name="Household"]/Attribute/[@name="DISTRIB_MIN_VALUE"]'
        )
        self.Household_DISTRIB_MAX_VALUE = self.get_value(
            'Class/[@name="Household"]/Attribute/[@name="DISTRIB_MAX_VALUE"]'
        )
        self.Household_BELIEF_YEAR_OFFSET = self.get_value(
            'Class/[@name="Household"]/Attribute/[@name="BELIEF_YEAR_OFFSET"]'
        )
        self.Household_DEFAULT_INFORMATION_STD = self.get_value(
            'Class/[@name="Household"]/Attribute/[@name="DEFAULT_INFORMATION_STD"]'
        )
        self.Household_P_INFORMATION = self.get_value(
            'Class/[@name="Household"]/Attribute/[@name="P_INFORMATION"]'
        )
        self.Household_GAMMA = self.get_value(
            'Class/[@name="Household"]/Attribute/[@name="GAMMA"]'
        )
        self.Household_climate_init_mean_beliefs_floor = self.get_value(
            'Class/[@name="Household"]/Attribute/[@name="climate_init_mean_beliefs_floor"]'
        )
        self.Household_climate_init_mean_beliefs_var = self.get_value(
            'Class/[@name="Household"]/Attribute/[@name="climate_init_mean_beliefs_var"]'
        )
        self.Household_climate_init_var_beliefs_current = self.get_value(
            'Class/[@name="Household"]/Attribute/[@name="climate_init_var_beliefs_current"]'
        )
        self.Household_climate_init_var_beliefs_offset1 = self.get_value(
            'Class/[@name="Household"]/Attribute/[@name="climate_init_var_beliefs_offset1"]'
        )
        self.Household_ecr_end_year = self.get_value(
            'Class/[@name="Household"]/Attribute/[@name="ecr_end_year"]'
        )
        self.Household_ecr_end_year = self.get_value(
            'Class/[@name="Household"]/Attribute/[@name="ecr_end_year"]'
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
            case _:
                raise Exception(
                    type_
                    + " ::: Is not known as a correct type when reading the values in 'init_values.xml'."
                )
        return value


XML_init_values = DataLoaderTwoLevelGame()
