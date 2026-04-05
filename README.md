# Credit Default Prediction via NLP-Based Sentiment Analysis

**Course:** BU MF810 Advanced Programming  
**Authors:** Zhankai Zhang, Weiyu Wang

## Overview

Hybrid credit scoring framework integrating traditional financial indicators with NLP-derived sentiment features to predict loan defaults. Uses the Lending Club dataset (~2.26M observations, 151 variables).

**Expected Loss = PD × EAD × LGD**

## Quick Start

1. Place `accepted_2007_to_2018Q4.csv` and `rejected_2007_to_2018Q4.csv` in `data/raw/`
2. Run:
```bash
docker-compose up
```
3. Open the Jupyter URL printed in the terminal
4. Run notebooks in order: `01_data_loading` → `02_eda_cleaning` → `03_feature_engineering` → `04_nlp_features` → `05_modeling` → `06_evaluation`

## Project Structure

```
project/
├── Dockerfile & docker-compose.yml   # Docker environment
├── requirements.txt                  # Python dependencies
├── data/raw/                         # Raw CSVs (not in git)
├── data/parquet/                     # Intermediate Parquet files (generated)
├── src/                              # Reusable Python modules
├── notebooks/                        # 6 analysis notebooks (run in order)
├── outputs/figures/                  # Saved plots
├── outputs/tables/                   # Saved metric tables
└── docs/                             # Proposal & reference docs
```

## Tech Stack

PySpark, scikit-learn, statsmodels, NLTK/VADER/TextBlob, matplotlib/seaborn
