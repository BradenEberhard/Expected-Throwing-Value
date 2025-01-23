<p align="center">
    <h1 align="center">A Machine Learning Approach to Player Value and Decision Making in Professional Ultimate Frisbee</h1>
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

# Overview
This project introduces novel metrics for ultimate frisbee, designed to capture aspects of player decision-making, execution quality, and positional value. These metrics offer fresh insights into player evaluation and strategic decision-making by addressing the limitations of traditional statistics. Our framework combines data-driven modeling with practical applications, aiming to enhance understanding and performance within the sport.

# Models
We employ advanced statistical and machine learning models to develop and evaluate the proposed metrics. Key methodologies include Expected Throw Value (ETV) modeling, Completion Percentage Over Expected (CPOE) calculations, regression-based adjustments for Enhanced Completion (EC) and Adjusted Enhanced Completion (aEC), and shot selection analysis using Expected Completion Percentage (xCP). These models leverage large datasets from competitive play to ensure reliability and validity.

# Metrics
The newly developed metrics include:
- **Expected Throw Value (ETV):** A probabilistic measure of the value added by each throw, incorporating thrower, receiver, throw and game context.
- **Completion Percentage Over Expected (CPOE):** A metric isolating the execution quality of throws relative to baseline expectations.
- **Expected Contribution (EC) and Adjusted Expected Contribution (aEC):** Metrics that quantify positional value differences and normalize contributions across game contexts.
- **Expected Completion Percentage (xCP):** A metric measuring the difficulty of attempted throws, focusing on shot selection and decision-making.

These metrics aim to complement traditional measures like goals and assists, providing a more comprehensive view of performance.

# Metric Analysis

## Discrimination and Stability
Discrimination measures a metric's ability to differentiate between players based on performance, while stability reflects its consistency over time. Comparative analysis with traditional metrics indicates that the novel metrics, such as ETV and CPOE, exhibit similar levels of discrimination and stability, making them reliable tools for player evaluation.

## Independence
Independence evaluates whether a metric provides unique insights. Using a Gaussian copula model, we assess the dependency of each metric on traditional measures. Most novel metrics demonstrate strong independence, with exceptions like ETV, which primarily captures overall player involvement rather than unique decision-making insights. Adjusted metrics like aEC show greater independence, isolating specific performance aspects effectively.

## Relation to Established Metrics
We use hierarchical clustering and dendrogram visualization to analyze the relationships between new and traditional metrics. This analysis reveals expected alignments, such as R-EC with receiver-based metrics, and highlights distinct contributions of metrics like CPOE and xCP, which emphasize execution quality and shot selection.

# Applications

## Decision-Making
The metrics enable teams to make data-driven decisions:
- **Field Plots:** ETV-based visualizations identify optimal throw targets and strategies, enhancing situational awareness and offensive efficiency.
- **Feature Trends and Interactions:** SHAP analysis uncovers key factors influencing throw value, such as throw distance, positioning, and game state.

## Player Assessment
The EC and aEC metrics provide a more nuanced evaluation of player contributions compared to traditional yardage statistics. By incorporating positional and situational context, these metrics align closely with recognized elite performance, offering valuable insights for player development and team strategy.

For more details, refer to the full paper.

---

##  Repository Structure

```sh
└── Expected-Throwing-Value/
    ├── data
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
| [all_games_1024.csv](https://github.com/BradenEberhard/Expected-Throwing-Value/blob/main/data/all_games_1024.csv) | <code>❯ Processed UFA DATA from 2021 to 2024. </code> |


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
