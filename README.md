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

# A Machine Learning Approach to Player Value and Decision Making in Professional Ultimate Frisbee  

## Overview  

<code>❯ REPLACE-ME</code>  
This project introduces a machine learning framework to evaluate player contributions and decision-making in professional Ultimate Frisbee, leveraging a dataset spanning four seasons (2021–2024) provided by the Ultimate Frisbee Association. The dataset includes over 327,000 throws from 604 games, offering unprecedented spatial and contextual detail.  

## Objectives  

- <code>❯ REPLACE-ME</code> **Completion Probability (CP) Model**: Estimates the likelihood of a throw being successfully completed based on spatial and contextual features.  
- <code>❯ REPLACE-ME</code> **Field Value (FV) Model**: Quantifies the positional value of the field in terms of scoring probability.  
- <code>❯ REPLACE-ME</code> **Metrics Development**: Combines insights from CP and FV models to assess player value, throwing performance, and team strategy.  

## Key Contributions  

- <code>❯ REPLACE-ME</code> **Dataset Insights**: Largest throw-level dataset in professional ultimate, enabling comprehensive spatial and contextual analysis.  
- <code>❯ REPLACE-ME</code> **Advanced Analytics**: Moves beyond traditional counting statistics like goals and assists, offering more nuanced metrics for evaluating gameplay.  
- <code>❯ REPLACE-ME</code> **Practical Applications**: Demonstrates use cases for identifying MVPs, optimizing team strategy, and uncovering undervalued players.  

## Features Modeled  

- <code>❯ REPLACE-ME</code> **Thrower and Receiver Context**: Coordinates, angles, and distances.  
- <code>❯ REPLACE-ME</code> **Game Context**: Quarter, score differential, and time remaining.  

## Methodology  

- <code>❯ REPLACE-ME</code> **CP Model**: Predicts throw completion using features such as throw distance and angle.  
- <code>❯ REPLACE-ME</code> **FV Model**: Evaluates field positioning and its contribution to scoring probability.  
- <code>❯ REPLACE-ME</code> **Modeling Approach**: XGBoost and ensemble decision trees for robust predictions.  

## Outcomes  

<code>❯ REPLACE-ME</code>  
The framework provides actionable metrics that redefine player evaluation and decision-making in professional ultimate, offering insights that parallel advanced analytics in other sports like baseball and football.  

For more details, refer to the full paper.

---

##  Features

<code>❯ REPLACE-ME</code>

---

##  Repository Structure

```sh
└── Expected-Throwing-Value/
    ├── data/processed
    │   └── throws.csv
    ├── figures
    │   ├── descriptive_charts.ipynb
    │   ├── descriptive_tables.ipynb
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

<details closed><summary>data</summary>

| File | Summary |
| --- | --- |
| [throws.csv](https://github.com/BradenEberhard/Expected-Throwing-Value/blob/main/data/throws.csv) | <code>❯ Example data from the dataset. Full data can be found. </code> |
| [descriptive_tables.ipynb](https://github.com/BradenEberhard/Expected-Throwing-Value/blob/main/figures/descriptive_tables.ipynb) | <code>❯ Generates key descriptions for UFA data including number of games, points, players etc. </code> |
| [etv_plots.ipynb](https://github.com/BradenEberhard/Expected-Throwing-Value/blob/main/figures/etv_plots.ipynb) | <code>❯ Generates key plots showcasing use cases for Expected Throw Value using a heatmap on the playing field for FV, CP and ETV. </code> |

</details>

<details closed><summary>figures</summary>

| File | Summary |
| --- | --- |
| [descriptive_charts.ipynb](https://github.com/BradenEberhard/Expected-Throwing-Value/blob/main/figures/descriptive_charts.ipynb) | <code>❯ Generates key visualizations for UFA data including example point, radial histogram and 3d radial chart of location and direction frequency. </code> |
| [descriptive_tables.ipynb](https://github.com/BradenEberhard/Expected-Throwing-Value/blob/main/figures/descriptive_tables.ipynb) | <code>❯ Generates key descriptions for UFA data including number of games, points, players etc. </code> |
| [etv_plots.ipynb](https://github.com/BradenEberhard/Expected-Throwing-Value/blob/main/figures/etv_plots.ipynb) | <code>❯ Generates key plots showcasing use cases for Expected Throw Value using a heatmap on the playing field for FV, CP and ETV. </code> |

</details>

<details closed><summary>helper_files</summary>

| File | Summary |
| --- | --- |
| [etv_model.py](https://github.com/BradenEberhard/Expected-Throwing-Value/blob/main/helper_files/etv_model.py) | <code>❯ Class for ETV. Natively handles CP, FV and data interactions for predictions. </code> |
| [helper_files_metrics.py](https://github.com/BradenEberhard/Expected-Throwing-Value/blob/main/helper_files/helper_files_metrics.py) | <code>❯ contains functions for generating both novel and traditional player level metrics. </code> |
| [modelling_configs.py](https://github.com/BradenEberhard/Expected-Throwing-Value/blob/main/helper_files/modelling_configs.py) | <code>❯ Contains the config information for training CP, FV and ETV models. </code> |
| [modelling_functions.py](https://github.com/BradenEberhard/Expected-Throwing-Value/blob/main/helper_files/modelling_functions.py) | <code>❯ Functions for training model such as data processing pipeline, hyperparameter tuning, etc. </code> |
| [plotting_functions.py](https://github.com/BradenEberhard/Expected-Throwing-Value/blob/main/helper_files/plotting_functions.py) | <code>❯ Functions for plotting heatmaps. Calculates data for full field grid. </code> |

</details>

<details closed><summary>models</summary>

| File | Summary |
| --- | --- |
| [etv_training.ipynb](https://github.com/BradenEberhard/Expected-Throwing-Value/blob/main/models/etv_training.ipynb) | <code>❯ Model training file. Saves models and generates performance metrics. </code> |
| [feature_importance.ipynb](https://github.com/BradenEberhard/Expected-Throwing-Value/blob/main/models/feature_importance.ipynb) | <code>❯ SHAP implementation and exploration over different features. </code> |

</details>


<details closed><summary>processing</summary>

| File | Summary |
| --- | --- |
| [processing.ipynb](https://github.com/BradenEberhard/Expected-Throwing-Value/blob/main/processing/processing.ipynb) | <code>❯ Contains the main data processing pipeline for analyzing throwing data. It includes data loading, cleaning, categorization, and feature extraction processes to prepare the dataset for modeling. </code> |

</details>

---

##  Getting Started

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

3. Install the required dependencies (currently no requirements file):
```sh
❯ pip install -r requirements.txt
```
