# JUSTICE Integrated Assessment Framework

| Module         | Status                                                                                                                                                                  |
| -------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| JUSTICE Module | [![GitHub Actions build status](https://github.com/pollockDeVis/JUSTICE/workflows/Build%20JUSTICE%20Module/badge.svg)](https://github.com/pollockDeVis/JUSTICE/actions) |

JUSTICE (JUST Integrated Climate Economy) is an open-source Integrated Assessment Modelling Framework for Normative Uncertainty Analysis

JUSTICE is designed to explore the influence on distributive justice outcomes due to underlying modelling assumptions across model components and functions: the economy and climate components, emissions, abatement, damage and social welfare functions. JUSTICE is a simple IAM inspired by the long-established RICE, and RICE50+, and is designed to be a surrogate for more complex IAMs for eliciting normative insights.

<img title="JUSTICE Framework" alt="Flowchart of JUSTICE" src="/docs/diagrams/JUSTICE Flowchart.jpeg">

The following is the repository structure. JUSTICE is modular and each module is contained in a separate folder. The modules are: economy, emissions, climate, damage and welfare. The data folder contains input and output data. The docs folder contains the documentation. The tests folder contains unit tests. The .github folder contains the GitHub Actions for CI/CD workflows.

```plaintext
📂 JUSTICE
┣ 📂 .github
┃ ┗ 📂 workflows
┃    ┗ 📜 main.yml         # GitHub Actions for CI/CD workflows
┣ 📂 src
┃ ┣ 📂 economy
┃ ┃  ┗ 📜 neoclassical.py
┃ ┣ 📂 emissions
┃ ┃  ┗ 📜 emission.py
┃ ┣ 📂 climate
┃ ┃  ┗ 📜 coupled_fair.py
┃ ┃  ┗ 📜 temperature_downscaler.py
┃ ┃  ┗ 📜 download_fair_configurations.py
┃ ┣ 📂 damage
┃ ┃  ┗ 📜 kalkuhl.py
┃ └ 📂 welfare
┃   ┣ 📜 utilitarian.py
┣ 📂 data
┃ ┣ 📂 input
┃ ┗ 📂 output
┣ 📂 docs                  # Documentation using sphinx/read-the-docs
┃ ┗ 📂 source
┣ 📂 tests                     # Unit tests
┣ 📜 .gitignore
┣ 📜 README.md
┗ 📜 LICENSE.md
```

# DICE/RICE Documentation

Compiled by: Palok Biswas \
Date: 31st May 2023

Resources:

1. [DICE 2013R User Manual](https://yale.app.box.com/s/whlqcr7gtzdm4nxnrfhvap2hlzebuvvm/file/1044222401276)
2. [DICE 2023 User Manual](https://yale.app.box.com/s/whlqcr7gtzdm4nxnrfhvap2hlzebuvvm/file/1254456088939)

# Data of DICE/RICE

- DICE and RICE fundamentals are similar except the Output, Population, Emissions, Damages and abatement are disaggregated into Regions in RICE
- Output, population, and emissions variables are built from national data. It is aggregated into major regions (US, China, EU, India etc). RICE projects these variables separately whereas it is aggregated in DICE
- Regional outputs adn capital stocks are aggregated using Purchasing Power Parity (PPP)

### Exogenous variable

1. Technological Change
2. Population Growth / Labour
3. Carbon intensity at time t, σ(t)

### Endogenous variable

1.  Capital Accumulation [neoclassical]

# Submodels

### Economy

Labour Population growth (Logistic Function)

$$
L(t) = L(t - 1)[1 + g_{L}(t)]
$$

Growth rate of population (calibrated to a value of 13.4% per 5 years so that population equals UN projection for 2050)

$$
g_{L}(t) = g_{L}{(t - 1)}/(1 + \delta_{L})
$$

- Energy takes two forms

  - Carbon-based
  - non Carbon-based

- Technological change takes two forms:
  _ Economy-wide technological change
  _ Carbon saving technological change
  Total Factor Productivity (TFP) represented by a logistic function and is represented by A(t). A(2010) is calibrated to gross world product in 2010 and g\_{A} is set to 0.6% per 5 years. This leads to 1.9% consumption per capita growth from 2010-2100 and 0.9% per year from 2100 to 2200.

$$
A(t) = A(t - 1)[1 + g_{A}(t)]
$$

Where

$$
g_{A}(t) = g_{A}{(t - 1)}/(1 + \delta_{A})
$$

Economic Output (represented by Q(t)) is produced using Cobb-Douglas production function which takes in capital, Labour and energy. Outputs are measured in PPP exchange rates using IMF estimates. Total output for each region is projected using a partial convergence model and output are aggregated to world total in DICE. Damage is in the denominator so that damages to not exceed 100% of the output

$$
Q(t) = [1- \Lambda(t)] A(t) K(t)^{\gamma} L(t)^{1 - \gamma} / [1+\Omega(t)]
$$

Here:

- `Q(t)` represents the global output at time t,
- `A(t)` is the total factor productivity at time t,
- `K(t)` is the capital stock at time t,
- `γ`
- `L(t)` is the labor input at time t.

- Ω(t) represents the damage function at time t,
- λ(t) is the abatement cost at time t.

Abatement cost is a function of emissions reduction rate $\mu(t). The backstop technology is introduced into the model by setting the time path of the parameters in the abatement-cost equation so that the marginal cost of abatement at a control rate of 100 percent is equal to the backstop price for a given year. The backstop price is assumed to be initially high and to decline over time with carbon-saving technological change. Backstop price in DICE 2013R is $344 per ton CO2 at 100% removal. The cost of the backstop technology is assumed to decline at 0.5% per year.

Marginal cost of emissions is calculated from the abatement cost equation by substituting output equations (?)

$$
\Lambda(t) = \theta_{1}(t) \cdot \mu(t)^{\theta_{2}}
$$

- $\Lambda(t)$ is the abatement cost at time t,
- $\theta_{1}(t)$ is
- $\mu(t)$ is the emissions reduction rate at time t,
- $\theta_{2}$ is

#### Standard Economic Accounting Equations [Neoclassical]

- Output, Q(t), is the sum of consumption, C(t), and gross investment, I(t).

$$
Q(t) = C(t) + I(t)
$$

- Per capita consumption, c(t), is the ratio of total consumption, C(t), to population, L(t).

$$
c(t) = \frac{C(t)}{L(t)}
$$

- Capital stock dynamics, K(t), follows a perpetual inventory method, is determined by the difference between investment, I(t), and depreciated capital from the previous period, $(\delta_{K} K(t - 1))$. Here, $\delta_{K}$ is the depreciation rate.

$$
K(t) = I(t) - \delta_{K} K(t - 1)
$$

Emissions: CO2 emissions are projected as a function of total Output, a time-varying emissions-output ratio and the emission control rate. The emissions-output ratio is estimated for individual regions and is then aggregated to the global ratio (for DICE). The emissions-control rate is determined by the climate-change policy under examination [Policy Lever].

Cost of emission reduction is parameterized by a log-linear function and calibrated using [EMF](https://emf.stanford.edu/) 22 report by [Clarke et al. 2009](https://www.researchgate.net/profile/John-Weyant/publication/228944221_Overview_of_EMF_22_US_transition_scenarios/links/5a8887ab458515b8af920ebf/Overview-of-EMF-22-US-transition-scenarios.pdf)

- Older versions of DICE/RICE used emissions control rate as the control variable in optimization. Newer versions incorporated carbon tax as a control variable.

The carbon price is determined by assuming that the price is equal to the marginal cost of emissions. Marginal cost is calculated from the abatement cost equation.

This equation represents the Baseline industrial CO2 emissions and is a function of level of carbon intensiy and economic output. Carbon intensity is exogenous and aggregated (in DICE) from emissions estimates of 12 regions. Emissions are reduced by [1 - μ(t)], which is the emissions reduction rate.

$$
E_{\text{Ind}}(t) = \sigma(t)[1 - \mu(t)]A(t) K(t)^{\gamma}  L(t)^{(1-\gamma)}
$$

- $E_{\text{Ind}}(t)$ denotes industrial CO2 emissions at time t,
- $\sigma(t)$ is the carbon intensity at time t,
- $\mu(t)$ is the emissions reduction rate at time t,
- $A(t)$ is the total factor productivity at time t,
- $K(t)$ is the capital stock at time t,
- $\gamma$ is ,
- $L(t)$ is the labor input at time t.

Baseline carbon intensity estimates are assumed to be logistic type equation. σ(2010) is set to equal the carbon intensity in 2010, 0.549 tons of CO2 per $1000 of GDP. gσ(2015) is calibrated to -1.0% per year; and δσ = -0.1% per five years. This specification leads to rate of change of carbon intensity (with no climate change policies) of -0.95% per year from 2010 to 2100 and -0.87% per year from 2100 to 2200.

$$
\sigma(t) = \sigma(t - 1)[1 + g_{\sigma}(t)]
$$

Where

$$
g_{\sigma}(t) = g_{\sigma}{(t - 1)}/(1 + \delta_{\sigma})
$$

Limitations of Total resources of Carbon fuels, CCum is given in the equation below. In earlier versions, the carbon constraint was binding, but it is not in the current version. The model assumes that incremental extraction costs are zero and that carbon fuels are efficiently allocated over time by the market. Limit set to 6000 tons of carbon content. Cumulative carbon emissions from 2010 to 2100 in the baseline DICE-2013R model are projected to be 1870 GtC, and for the entire period 4800 GtC. Estimates for 2100 are slightly higher than the models surveyed in the IPCC Fifth Assessment Report.

$$
\text{CCum}\; E(t) \geq \sum_{t=1}^{T_{\text{max}}} E_{\text{Ind}}(t)
$$

This inequality represents the limitation of the total resources of carbon fuels, where:

- $\text{CCum}\; E(t)$ denotes the cumulative total resources of carbon fuels up to time t,
- $T_{\text{max}}$ is the maximum time period,
- $E_{\text{Ind}}(t)$ represents industrial CO2 emissions at time t,
- The summation symbol $\sum_{t=1}^{T_{\text{max}}}$ indicates that industrial CO2 emissions are summed from the first time period (t=1) to the maximum time period (t=$T_{\text{max}}$).

# Functions (Social Welfare & Damage)

Welfare, W is maximised by the chosen policies. W is the discounted sum of the population-weighted utility of per capita consumption

$$
W = \sum_{t=1}^{T_{\text{max}}} U[c(t), L(t)] \cdot R(t)
$$

- c(t) - per capita consumption
- L(t) - population/labour inputs
- R(t) - Discount factor [ Π(t) in DICE 2023 - Make up your mind economists]

Utility is represented by a constant elasticity of utility function or constant elasticity of the marginal utility of consumption α.

$$
U[c(t), L(t)] = L(t) \cdot \frac{c(t)^{(1-\alpha)}}{(1-\alpha)}
$$

-     α - marginal utility of consumption (in DICE 2023 it is 'φ' [-_-] )

N.B. The elasticity is a parameter that represents the extent of substitutability of the consumption of different years or generations. If α/𝜑 is close to zero, then the consumptions of different generations are close substitutes; if α/𝜑 is high, then the consumptions are not close substitutes. α/𝜑 is calibrated in conjunction with the pure rate of time preference and the riskiness of climate investments

R(t) is the discount factor where 'ρ' is the pure rate of social time preference

$$
\Pi(t) or R(t) = {(1+\rho)^{-t}}
$$

Damage Function

$$
\Omega(t) = \psi_{1} \cdot T_{AT}(t) + \psi_{1} \cdot [T_{AT}(t)]^2
$$

## Inputs of DICE/RICE submodels

## Outputs of DICE/RICE submodels

## List of Assumptions in DICE/RICE

1. Economic and Climate policies are designed to optimize the flow of consumption over time
2. Neoclassical Growth Theory by Solow assumes reducing consumption today to invest in capital, education and technologies
3. The world or individual regions have well-defined preferences represented by the SWF
4. Utility is represented by a constant elasticity of utility function
5. Value of consumption in a period is proportional to the population
6. A single commodity is used for consumption (food, shelter and non-market environmental amenities and services), investment or abatement
7. Population growth assumed to follow a logistic curve
8. Total Factor Productivity is also assumed to follow a logistic curve
9. Production function are assumed to be constant-returns-to-scale Cobb-Douglas production function in capital, labour, and Hicks-neutral technological change. Hicks-neutral technological change is a form of productivity improvement that equally affects all factors of production, increasing overall output without altering the balance between labor and capital inputs.
10. Damage function are assumed to be a quadratic function of temperature change and does not include sharp thresholds or tipping points.
11. 25% of adjusted damages to reflect non-monetized impacts in: economic value losses from biodiversity, ocean acidification, political reactions, catastrophic events, long-term warming and uncertainty.
12. The backstop price is assumed to be initially high and to decline over time with carbon-saving technological change. Backstop price in DICE 2013R is $344 per ton CO2 at 100% removal.
13. The cost of the backstop technology is assumed to decline at 0.5% per year.
14. The abatement cost function assumes that abatement costs are proportional to output and to a power function of the reduction rate.
15. The carbon price is determined by assuming that the price is equal to the marginal cost of emissions.
16. Baseline carbon intensity estimates are assumed to be logistic type equation
17. Incremental extraction costs of carbon fuel are assumed to be zero and that carbon fuels are efficiently allocated over time by the market

## List of Normative Variables

1. 'ρ' : Pure rate of social time preference
2. 'α' : Elasticity of Marginal utility of consumption. α represents aversion to generational inequality, which is the diminishing social valuations of consumption of different generations

## List of Acronyms

1. IAM - Integrated Assessment Model
2. SWF - Social Welfare Function
3. PPP - Purchasing Power Parity
4. TFP - Total Factor Productivity
5. IMF - International Monetary Fund
6. EMF - Energy Modeling Forum
7. IPCC - Intergovernmental Panel on Climate Change
