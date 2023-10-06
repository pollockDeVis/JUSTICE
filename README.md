# JUSTICE Integrated Assessment Framework

JUSTICE (JUST Integrated Climate Economy) is an open-source Integrated Assessment Modeling Framework for Normative Uncertainty Analysis

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
┃ ┣ 📂 emissions
┃ ┣ 📂 climate
┃ ┣ 📂 damage
┃ └ 📂 welfare
┣ 📂 data
┃ ┣ 📂 input
┃ └ 📂 output
┣ 📂 docs                  # Documentation using sphinx/read-the-docs
┃ ┗ 📂 source
┃    ┣ 📜 conf.py          # Sphinx config
┃    ┣ 📜 index.rst        # Documentation home page
┃    ┣ 📜 economy.rst      # Documentation for economy module
┃    ┣ 📜 emissions.rst    # Documentation for emissions module
┃    ┣ 📜 climate.rst      # Documentation for climate module
┃    ┣ 📜 damage.rst    # Documentation for damage module
┃    └ 📜 welfare.rst   # Documentation for welfare module
┣ 📂 tests                     # Unit tests 
┃   ┣ 📜 test_economy.py
┃   ┣ 📜 test_emissions.py 
┃   ┣ 📜 test_climate.py
┃   ┣ 📜 test_damage.py
┃   └ 📜 test_welfare.py
┣ 📜 .gitignore                
┣ 📜 README.md                 
┗ 📜 LICENSE.md                
```