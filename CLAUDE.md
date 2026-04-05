# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Credit Default Prediction via NLP-Based Sentiment Analysis** - Final project for BU MF810 (Advanced Programming: Data Structures and Algorithms). By Zhankai Zhang and Weiyu Wang.

Hybrid credit scoring framework that integrates traditional financial indicators with NLP-derived sentiment and textual features to predict loan defaults using the Lending Club dataset (~2.26M observations, 151 variables).

## Architecture

Two-model comparison approach:
- **Baseline Model** (structured features only): PD via Logistic Regression, LGD via Beta Regression, EAD via Linear Regression (CCF)
- **Hybrid Model** (structured + NLP): Adds sentiment analysis scores and TF-IDF vectors from borrower text fields (`desc`, `emp_title`, `title`) into the PD model

Expected Loss framework: `EL = PD x EAD x LGD`

## Implementation Steps

### Step 1: Data Loading & Storage
- Load Lending Club CSV (~2.26M rows, 151 cols, 2.5+ GB memory)
- Convert to Parquet format for columnar storage
- Use PySpark (`SparkSession`, `local[*]` mode) for all data processing
- Target variable: `loan_status` → binary (Default vs Non-Default)

### Step 2: EDA & Data Cleaning
- Explore missing patterns (especially `desc` ~94% missing, `mths_since_last_major_derog` non-random missing)
- Analyze default rates by segments (e.g., individual 13.19% vs joint app 7.06%)
- Handle ~120K joint applications (entity resolution at household level)
- Impute missing values for structured features

### Step 3: Feature Engineering (Baseline — Structured Only)
- **Structured features**: `loan_amnt`, `int_rate`, `annual_inc`, `dti`, `revol_util`, etc.
- **WoE encoding**: Convert categorical variables using Weight of Evidence
- **IV feature selection**: Use Information Value to rank and select predictive features
  - IV > 0.3 = strong predictor, 0.1–0.3 = medium, < 0.02 = drop

### Step 4: NLP Preprocessing (Hybrid Model Enhancement)
- **Text cleaning**: HTML tag removal (`<br>` etc.), stopword removal, lowercasing on `desc` and `title`
- **Occupational clustering**: Reduce `emp_title` from 512K+ unique values into standardized occupation groups to solve high-cardinality problem
- Handle 94% missing rate in `desc` — strategy needed (flag missing as feature, impute, or use only non-missing subset)

### Step 5: NLP Feature Extraction
- **Sentiment Analysis**: Apply VADER/TextBlob to `desc` and `title`, generate sentiment scores [-1, 1] as psychological proxy for borrower risk
- **TF-IDF Vectorization**: Extract topical risk signals from text (e.g., "debt", "medical", "consolidation")
- Integrate NLP features into the structured feature set for the hybrid PD model

### Step 6: Baseline Model Training (Structured Features Only)
- **PD Model**: Logistic Regression → predict probability of default
- **LGD Model**: Beta Regression (bounded 0-1) → predict loss severity given default
- **EAD Model**: Linear Regression using Credit Conversion Factors (CCF) → predict exposure at default
- Compute: `EL = PD x EAD x LGD`

### Step 7: Hybrid Model Training (Structured + NLP Features)
- Add sentiment scores + TF-IDF vectors + occupation clusters to PD Logistic Regression
- Compare hybrid PD model vs baseline PD model
- Re-compute EL with hybrid PD

### Step 8: Model Evaluation & Comparison
- **AUC-ROC**: Discriminatory power (target: hybrid > baseline)
- **Gini Coefficient**: `Gini = 2 * AUC - 1`
- **KS Statistic**: Max separation between cumulative default/non-default distributions
- **Lift Analysis**: Performance in top deciles
- Visualizations: ROC curves, KS plots, lift charts, feature importance

### Step 9: Visualization & Report
- Default rate plots by segment (application type, grade, etc.)
- LGD distribution histogram (zero-inflated/bimodal pattern)
- Comparison tables: baseline vs hybrid metrics
- Final write-up: 6-8 page scientific paper format with appendix for extra charts

## Technical Stack

- **Data storage**: Parquet (columnar format)
- **Processing**: PySpark with MapReduce paradigms, `local[*]` mode
- **Environment**: Docker (`jupyter/all-spark-notebook`) + Jupyter
- **NLP**: NLTK, TextBlob, VADER for sentiment; scikit-learn TF-IDF
- **ML**: scikit-learn (Logistic Regression), statsmodels (Beta Regression)
- **Viz**: matplotlib, seaborn

## Data Challenges

- `desc` field has ~94% missing rate — limits NLP effectiveness
- `emp_title` has 512K+ unique values (high cardinality → occupational clustering needed)
- ~120K joint applications require household-level entity resolution
- LGD distribution is zero-inflated/bimodal (not suited for linear regression → use Beta Regression)
- Non-random missing patterns (e.g., `mths_since_last_major_derog`)
- Dataset is 2.5+ GB in memory — requires Spark for efficient processing

## Grading Criteria (for context on priorities)

- Structure (2 pts): Scientific paper format, 6-8 page main body
- Data complexity (5 pts): Demonstrate difficulty and show preliminary results on sample data
- Data processing technologies (3 pts): Tidy data, Spark/MapReduce usage
- Data manipulation (2 pts): Grammar-based manipulation or SQL
- Evaluation & visualization (3 pts): Algorithm justification, plots (time-series, scatter, histograms)
- Code must run and recreate results in Docker, well-documented environment
