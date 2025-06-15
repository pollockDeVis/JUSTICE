class PolicyNames:
    CLIMATE = "Climate Policy"
    ECONOMIC = "Economic Policy"
    COMPROMISE = "Compromise Policy"


class VariableCodes:
    NET_ECONOMIC_OUTPUT = "net_economic_output"
    EMISSIONS = "emissions"
    ABATED_EMISSIONS = "abated_emissions"


class VariableLabels:
    NET_ECONOMIC_OUTPUT = "Net Economic Output"
    EMISSIONS = "Emissions"
    ABATED_EMISSIONS = "Abated Emissions"
    ECR = "Emission Control Rate"


class PolicyCodes:
    CLIMATE = "climate"
    ECONOMIC = "economic"
    COMPROMISE = "compromise"


policy_name_dict = {
    PolicyCodes.CLIMATE: PolicyNames.CLIMATE,
    PolicyCodes.ECONOMIC: PolicyNames.ECONOMIC,
    PolicyCodes.COMPROMISE: PolicyNames.COMPROMISE,
}

policy_color_dict = {
    PolicyCodes.CLIMATE: "purple",
    PolicyCodes.ECONOMIC: "lightgreen",
    PolicyCodes.COMPROMISE: "orange",
}

variable_to_label_dict = {
    VariableCodes.NET_ECONOMIC_OUTPUT: VariableLabels.NET_ECONOMIC_OUTPUT,
    VariableCodes.EMISSIONS: VariableLabels.EMISSIONS,
    VariableCodes.ABATED_EMISSIONS: VariableLabels.ABATED_EMISSIONS,
}
