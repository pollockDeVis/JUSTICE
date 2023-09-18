# JUSTICE
JUSTICE is an open-source Integrated Assessment Modeling Framework for Normative Uncertainty Analysis

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
┃    ┣ 📜 emissions.rst    # Documentation for emissions module, and so on..
┃    ┣ 📜 climate.r
┃    ┣ 📜 damage.rst    # Documentation for damage module
┃    └ 📜 welfare.rst   # Documentation for welfare module
┣ 📂 tests                     # Unit tests for your project
┃   ┣ 📜 test_economy.py
┃   ┣ 📜 test_emissions.py 
┃   ┣ 📜 test_climate.py
┃   ┣ 📜 test_damage.py
┃   └ 📜 test_welfare.py
┣ 📜 .gitignore                # File listing what to ignore in Git
┣ 📜 README.md                 # Readme for your repo, can also include project wide documentation
┗ 📜 LICENSE.md                # File that contains license for the project
```