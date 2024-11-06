<p align="center">
    <h1 align="center">EXPECTED-THROWING-VALUE</h1>
</p>
<br>

#####  Table of Contents

- [ Overview](#-overview)
- [ Features](#-features)
- [ Repository Structure](#-repository-structure)
- [ Modules](#-modules)
- [ Getting Started](#-getting-started)
    - [ Prerequisites](#-prerequisites)
    - [ Installation](#-installation)
    - [ Usage](#-usage)
- [ Project Roadmap](#-project-roadmap)
- [ Acknowledgments](#-acknowledgments)

---

##  Overview

<code>❯ REPLACE-ME</code>

---

##  Features

<code>❯ REPLACE-ME</code>

---

##  Repository Structure

```sh
└── Expected-Throwing-Value/
    ├── data
    │   └── throws.csv
    ├── dataset_description
    │   ├── descriptive_charts.ipynb
    │   └── descriptive_tables.ipynb
    ├── derived_metrics
    │   └── expected_contribution.ipynb
    ├── figures
    │   └── etv_plots.ipynb
    ├── helper_files
    │   ├── etv_model.py
    │   ├── modelling_configs.py
    │   ├── modelling_functions.py
    │   └── plotting_functions.py
    ├── meta_metrics
    │   └── etv_stability.ipynb
    ├── models
    │   └── etv_training.ipynb
    └── processing
        ├── processing.ipynb
        └── processing_functions
```

---

##  Modules

<details closed><summary>derived_metrics</summary>

| File | Summary |
| --- | --- |
| [expected_contribution.ipynb](https://github.com/BradenEberhard/Expected-Throwing-Value/blob/main/derived_metrics/expected_contribution.ipynb) | <code>❯ REPLACE-ME</code> |

</details>

<details closed><summary>dataset_description</summary>

| File | Summary |
| --- | --- |
| [descriptive_charts.ipynb](https://github.com/BradenEberhard/Expected-Throwing-Value/blob/main/dataset_description/descriptive_charts.ipynb) | <code>❯ REPLACE-ME</code> |
| [descriptive_tables.ipynb](https://github.com/BradenEberhard/Expected-Throwing-Value/blob/main/dataset_description/descriptive_tables.ipynb) | <code>❯ REPLACE-ME</code> |

</details>

<details closed><summary>processing</summary>

| File | Summary |
| --- | --- |
| [processing.ipynb](https://github.com/BradenEberhard/Expected-Throwing-Value/blob/main/processing/processing.ipynb) | <code>❯ This Jupyter Notebook contains the main data processing pipeline for analyzing throwing data. It includes data loading, cleaning, categorization, and feature extraction processes to prepare the dataset for modeling.</code> |

</details>

<details closed><summary>processing.processing_functions</summary>

| File | Summary |
| --- | --- |
| [tf_idf_functions.py](https://github.com/BradenEberhard/Expected-Throwing-Value/blob/main/processing/processing_functions/tf_idf_functions.py) | <code>❯ This Python module defines functions for calculating TF-IDF scores based on throw distances and directions. It categorizes throws, computes term frequencies, and merges results with training data to enhance the analysis of throwing performance.</code> |

</details>

<details closed><summary>helper_files</summary>

| File | Summary |
| --- | --- |
| [plotting_functions.py](https://github.com/BradenEberhard/Expected-Throwing-Value/blob/main/helper_files/plotting_functions.py) | <code>❯ REPLACE-ME</code> |
| [modelling_configs.py](https://github.com/BradenEberhard/Expected-Throwing-Value/blob/main/helper_files/modelling_configs.py) | <code>❯ REPLACE-ME</code> |
| [etv_model.py](https://github.com/BradenEberhard/Expected-Throwing-Value/blob/main/helper_files/etv_model.py) | <code>❯ REPLACE-ME</code> |
| [modelling_functions.py](https://github.com/BradenEberhard/Expected-Throwing-Value/blob/main/helper_files/modelling_functions.py) | <code>❯ REPLACE-ME</code> |

</details>

<details closed><summary>models</summary>

| File | Summary |
| --- | --- |
| [etv_training.ipynb](https://github.com/BradenEberhard/Expected-Throwing-Value/blob/main/models/etv_training.ipynb) | <code>❯ REPLACE-ME</code> |

</details>

<details closed><summary>figures</summary>

| File | Summary |
| --- | --- |
| [etv_plots.ipynb](https://github.com/BradenEberhard/Expected-Throwing-Value/blob/main/figures/etv_plots.ipynb) | <code>❯ REPLACE-ME</code> |

</details>

<details closed><summary>meta_metrics</summary>

| File | Summary |
| --- | --- |
| [etv_stability.ipynb](https://github.com/BradenEberhard/Expected-Throwing-Value/blob/main/meta_metrics/etv_stability.ipynb) | <code>❯ REPLACE-ME</code> |

</details>

---

##  Getting Started

###  Prerequisites

**JupyterNotebook**: `version x.y.z`

###  Installation

Build the project from source:

1. Clone the Expected-Throwing-Value repository:
```sh
❯ git clone https://github.com/BradenEberhard/Expected-Throwing-Value
```

2. Navigate to the project directory:
```sh
❯ cd Expected-Throwing-Value
```

3. Install the required dependencies:
```sh
❯ pip install -r requirements.txt
```

###  Usage

To run the project, execute the following command:

```sh
❯ jupyter nbconvert --execute notebook.ipynb
```

###  Tests

Execute the test suite using the following command:

```sh
❯ pytest notebook_test.py
```

---

##  Project Roadmap

- [X] **`Task 1`**: <strike>Implement feature one.</strike>
- [ ] **`Task 2`**: Implement feature two.
- [ ] **`Task 3`**: Implement feature three.

---

##  Contributing

Contributions are welcome! Here are several ways you can contribute:

- **[Report Issues](https://github.com/BradenEberhard/Expected-Throwing-Value/issues)**: Submit bugs found or log feature requests for the `Expected-Throwing-Value` project.
- **[Submit Pull Requests](https://github.com/BradenEberhard/Expected-Throwing-Value/blob/main/CONTRIBUTING.md)**: Review open PRs, and submit your own PRs.
- **[Join the Discussions](https://github.com/BradenEberhard/Expected-Throwing-Value/discussions)**: Share your insights, provide feedback, or ask questions.

<details closed>
<summary>Contributing Guidelines</summary>

1. **Fork the Repository**: Start by forking the project repository to your github account.
2. **Clone Locally**: Clone the forked repository to your local machine using a git client.
   ```sh
   git clone https://github.com/BradenEberhard/Expected-Throwing-Value
   ```
3. **Create a New Branch**: Always work on a new branch, giving it a descriptive name.
   ```sh
   git checkout -b new-feature-x
   ```
4. **Make Your Changes**: Develop and test your changes locally.
5. **Commit Your Changes**: Commit with a clear message describing your updates.
   ```sh
   git commit -m 'Implemented new feature x.'
   ```
6. **Push to github**: Push the changes to your forked repository.
   ```sh
   git push origin new-feature-x
   ```
7. **Submit a Pull Request**: Create a PR against the original project repository. Clearly describe the changes and their motivations.
8. **Review**: Once your PR is reviewed and approved, it will be merged into the main branch. Congratulations on your contribution!
</details>

<details closed>
<summary>Contributor Graph</summary>
<br>
<p align="left">
   <a href="https://github.com{/BradenEberhard/Expected-Throwing-Value/}graphs/contributors">
      <img src="https://contrib.rocks/image?repo=BradenEberhard/Expected-Throwing-Value">
   </a>
</p>
</details>

---

##  License

This project is protected under the [SELECT-A-LICENSE](https://choosealicense.com/licenses) License. For more details, refer to the [LICENSE](https://choosealicense.com/licenses/) file.

---

##  Acknowledgments

- List any resources, contributors, inspiration, etc. here.

---
