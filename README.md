# Used Car Price Prediction

**Ridge → CatBoost → FT-Transformer** · Predicting used-car listing prices from structured vehicle attributes

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![CatBoost](https://img.shields.io/badge/CatBoost-FFCC00?logo=catboost&logoColor=black)](https://catboost.ai)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org)
[![Quarto](https://img.shields.io/badge/Quarto-Notebook-75AADB)](https://quarto.org)

[View the full analysis →](link-to-github-page)

---

## Overview

This project predicts used-car listing prices from 20 vehicle attributes using three progressively complex modeling approaches — Linear baseline establishing a performance 
floor from an interpretable linear baseline through gradient-boosted trees to a deep learning transformer. The goal is not simply to minimize error, but to evaluate the practical tradeoffs between interpretability, feature engineering effort, and predictive performance on real-world tabular data.

**The final model (CatBoost) predicts within ~$1,300 of the true listing price at the median**, across 29 manufacturers and 5,613 model variants — accurate enough to automate first-pass pricing, reduce manual appraisals, and surface arbitrage opportunities for dealerships, online marketplaces, and auto lenders.

## Results

| Metric                          | Ridge   | CatBoost (final)  | FT-Transformer |
| ------------------------------- | ------- | ----------------- | -------------- |
| **Test RMSE (full)**            | \$7,824 | \$**3,935**       | \$4,255        |
| **Test MAE (full)**             | \$3,011 | \$**1,916**       | \$1,934        |
| **Test Median AE (full)**       | \$1,759 | \$**1,280**       | \$1,305        |
| Test RMSE (clipped 1–99%)       | \$4,211 | \$**2,665**       | \$2,677        |
| Test MAE (clipped 1–99%)        | \$2,653 | \$**1,782**       | \$1,800        |
| Test Median AE (clipped 1–99%)  | \$1,736 | \$**1,267**       | \$1,291        |

CatBoost outperforms Ridge by ~50% on every metric and edges out the FT-Transformer by a meaningful margin, particularly on full-distribution RMSE where sensitivity to rare high-value vehicles amplifies differences.

## Business Context

Used-car pricing is a high-stakes, high-volume problem. Dealerships, online marketplaces, and lenders need fast, accurate price estimates to set competitive listing prices, flag underpriced inventory for acquisition, detect overpriced listings, and underwrite auto loans. A model predicting within ~$1,300 at the median is accurate enough to drive real operational decisions across portfolios of thousands of vehicles.

## Key Findings

**Feature engineering was essential across all model classes.** Collapsing 1,808 raw engine strings into five structured features, normalizing 221 transmission variants into type and gear count, and engineering domain-informed interactions (e.g., luxury-brand depreciation curves) gave all three models a clean, informative feature set. Even CatBoost relied heavily on these engineered features among its top predictors.

**CatBoost's native categorical handling is a structural advantage.** With 5,613 unique models and 1,808 unique engine strings, CatBoost avoids the dimensionality explosion of one-hot encoding (972 dummy columns for Ridge) while capturing nonlinear interactions — like the relationship between vehicle age, luxury status, and price — without explicit interaction terms.

**Deep learning is competitive but not dominant on tabular data.** The FT-Transformer came within ~$20 of CatBoost on clipped MAE, confirming that learned embeddings capture meaningful structure. But CatBoost's lower tuning burden, built-in regularization, and robustness across the full price range make it the practical production choice.

**Luxury vehicles are the hardest to price accurately.** All three models show increasing error above ~$100k, likely because luxury pricing depends on factors not in the dataset — trim levels, option packages, and collector-market dynamics.

## Technical Approach

### Data

The dataset is the Kaggle [Used Cars Dataset](https://www.kaggle.com/datasets/andreinovikov/used-cars-dataset) (~762k listings, 20 attributes). After cleaning and restricting to fully observed records, **243,500 listings** remained, split 64/16/20 into train/validation/test sets with persisted indices so all models train and evaluate on identical rows.

### Feature Engineering

Raw listing fields were transformed into a clean, model-ready feature set:

- **Engine parsing**: 1,808 unique engine description strings → displacement, cylinder count, fuel type, turbo/supercharger flags
- **Transmission normalization**: 221 raw transmission strings → type (automatic/manual/CVT) and gear count
- **Color collapsing**: Rare exterior/interior colors grouped into canonical categories
- **Domain features**: Vehicle age, luxury brand flag, depreciation interaction terms, accident/damage indicators

### Models

- **Ridge Regression** — Linear baseline establishing a performance floor. Quantifies how far a simple additive model can go on the engineered feature set.

- **CatBoost** — Gradient-boosted trees with native categorical feature handling. Selected as the **final model** based on superior generalization across all metrics.
- **FT-Transformer (Keras)** — Feature Tokenizer + Transformer architecture with learned categorical embeddings. Demonstrates advanced experimentation with deep learning on tabular data.

### Evaluation Methodology

Because used-car prices are highly skewed (median ~$27k, tail past $100k reaching $1.9M), metrics are reported in two views: **full distribution** (reflecting real-world performance including rare luxury vehicles) and **clipped 1–99%** (reflecting typical consumer vehicles). The gap between the two (e.g., $3,935 vs. $2,665 RMSE for CatBoost) quantifies the impact of tail observations.

## Reproducibility

The project uses a lightweight experiment management system designed for full reproducibility:

- **Data pipeline versioning**: Cleaning and feature engineering results are cached as Parquet files, controlled by a `FEATURE_PIPELINE_VERSION` flag that triggers automatic rebuilds when logic changes.
- **Timestamped run artifacts**: Each training run writes model weights, hyperparameters, metrics, and schema metadata to a versioned directory (e.g., `models/catboost/run_YYYYMMDD_HHMMSS/`).
- **Schema validation**: Loading a saved model validates that the current feature pipeline matches the schema used during training, preventing accidental evaluation against incompatible features.
- **Training control flags**: `TRAIN_MODE` (smoke/full/skip) and per-model `RETRAIN_*` flags allow fast iteration without overwriting production artifacts.

## Project Structure

```
├── data/
│   ├── raw/                    # Raw dataset (auto-downloaded via Kaggle CLI)
│   └── processed/              # Versioned Parquet files and split indices
├── models/
│   ├── ridge/                  # Ridge pipeline artifacts and run history
│   ├── catboost/               # CatBoost model artifacts and run history
│   ├── keras/                  # FT-Transformer weights, vocabs, and run history
│   └── metrics/                # Cross-model comparison metrics
│   ├── api_artifacts/          # Serving-layer reference artifacts (vocabs, medians, metadata)
├── notebooks/
│   └── 01_used_car_price_regression.qmd   # Full analysis notebook (Quarto)
├── src/
│   ├── __init__.py
│   ├── pipeline.py             # Cleaning and feature engineering (active)
│   ├── data_prep.py            # Data ingestion stage (scaffolded for Part 2)
│   ├── features.py             # Feature engineering stage (scaffolded for Part 2)
│   ├── train_ridge.py          # Ridge training entrypoint (scaffolded for Part 2)
│   ├── train_catboost.py       # CatBoost training entrypoint (scaffolded for Part 2)
│   └── train_keras.py          # FT-Transformer training entrypoint (scaffolded for Part 2)
├── .gitignore
├── pyproject.toml              # Project metadata and tool configuration
├── README.md
├── requirements.in             # Direct dependencies
├── requirements.txt            # Pinned dependencies
└── requirements.freeze.txt     # Full environment snapshot
```

## Getting Started

### Prerequisites

- Python 3.10+
- [Kaggle API credentials](https://www.kaggle.com/docs/api) (for automatic dataset download)

### Installation

```bash
git clone https://github.com/<username>/used-car-price-prediction.git
cd used-car-price-prediction
pip install -r requirements.txt
```

### Running the Analysis

```bash
# Render the full Quarto notebook
quarto render notebooks/01_used_car_price_regression.qmd

# Or open in Jupyter and run interactively
jupyter lab notebooks/01_used_car_price_regression.qmd
```

The notebook will automatically download the dataset via the Kaggle CLI on first run. To download manually, place `cars.csv` in `data/raw/`.

## Tech Stack

| Category                  | Tools                                                 |
| ------------------------- | ----------------------------------------------------- |
| **Languages**       | Python                                                |
| **ML / Modeling**   | scikit-learn, CatBoost, TensorFlow/Keras, Keras Tuner |
| **Data**            | pandas, NumPy, Parquet                                |
| **Visualization**   | Matplotlib, Seaborn                                   |
| **Reproducibility** | Quarto, joblib, JSON artifact versioning              |

## Future Work

**Part 2 — Containerized Pipeline (planned):** Refactor the notebook-based workflow into a modular, deployable ML pipeline using Docker and Kubernetes. The `src/` module stubs (`data_prep.py`, `features.py`, `train_*.py`) are scaffolded for this phase — each becomes an independent pipeline stage that can be containerized, orchestrated, and scaled separately. The goal is to demonstrate production ML infrastructure alongside the modeling work in Part 1.

Additional modeling improvements under consideration:

- **Geographic features**: Incorporating location data (state, ZIP, metro) to capture regional pricing variation
- **Temporal modeling**: Time-aware approaches to account for market dynamics like pandemic-era supply constraints
- **Ensemble methods**: Weighted CatBoost + FT-Transformer blend to improve tail performance where the two models disagree
- **Vehicle condition**: Incorporating granular condition descriptors that buyers weigh heavily in practice
- **Luxury vehicle features**: Trim level, option packages, certification status, and collector-market relevance to reduce error for vehicles above $100k

## Author

**Erin Weiss** · [Portfolio](link) · [LinkedIn](link) · [GitHub](link)

- [Live Notebook](link-to-github-page) — Full rendered analysis with interactive code
- [Source Code](link-to-repo) — GitHub repository

---

*Built as a portfolio project demonstrating end-to-end ML development: data engineering, feature design, model comparison across paradigms, and production-oriented experiment management.*
