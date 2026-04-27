# Credit Default Prediction via NLP-Based Sentiment Analysis

**Course:** BU MF810 Advanced Programming — Final Project
**Authors:** Zhankai Zhang, Weiyu Wang

A hybrid credit-scoring framework that augments traditional WoE-binned features with NLP-derived signals (TF-IDF, Word2Vec, CNN-distilled scores, occupation/title embeddings) to estimate Probability of Default (PD), Loss Given Default (LGD), and Exposure at Default (EAD), then aggregates them into per-loan Expected Loss:

```
EL = PD × LGD × EAD
```

Built on the Lending Club dataset (~2.26M loans, 151 columns, 2007–2018Q4), with PySpark + MLlib for distributed feature engineering and modeling.

---

## Quick Start (Docker)

The entire pipeline runs inside a Docker container that ships PySpark + Jupyter Lab + all Python deps.

### 1. Place raw data

Put the Lending Club CSVs (gzipped) under `data/raw/`:

```
data/raw/accepted_2007_to_2018Q4.csv.gz
data/raw/rejected_2007_to_2018Q4.csv.gz
```

### 2. Build & launch

```bash
docker-compose up
```

> **First-time build takes 10–20 minutes.** The image installs TensorFlow (~600MB), PySpark, NLTK corpora, and a few NLP libraries. Subsequent launches are instant (image is cached).

Once it boots, copy the `http://127.0.0.1:8888/lab?token=...` URL printed in the terminal and open it in your browser.

### 3. Run notebooks in order

```
00_file_convert            # decompress .csv.gz → .csv
01_data_loading            # CSV → Parquet, train/test split
02_general_prep            # cleaning + EDA
03_feature_engineering     # WoE binning + IV-based selection
04_nlp_pipeline_dev        # TF-IDF, Word2Vec, CNN distillation, occupation clusters
05_hybrid_pd_model         # Hybrid PD (Phase A/B) + LGD + EAD + EL
```

For a non-interactive end-to-end run:

```bash
docker exec mf810-project jupyter nbconvert --to notebook \
    --execute notebooks/05_hybrid_pd_model.ipynb --inplace
```

---

## Methodology

### Why a two-phase design?

The free-text `desc` column is missing in **~94%** of loans (Lending Club removed it after 2014). Training NLP features on the full dataset would drown the signal in zero-fill noise. So the PD model is trained as **two independent experiments**:

| Phase | Subset | Size | Variants |
|---|---|---|---|
| **A** — desc-present | rows where `desc_missing == 0` | ~135K (re-split 80/20) | 6 (M0 → M5) |
| **B** — full data | original train/test split | 1.81M / 452K | 2 (baseline + categorical) |

Phase A isolates the genuine NLP lift; Phase B shows what survives when applied at production scale.

### NLP pipeline (notebook 04)

A three-stage progression on the `desc` / `title` / `emp_title` columns:

| Stage | Features | Model | Framework |
|---|---|---|---|
| 1 | TF-IDF (HashingTF 5000-dim + IDF) | LogisticRegression | MLlib |
| 2 | TF-IDF + Word2Vec (50-dim mean) via VectorAssembler | LogisticRegression | MLlib |
| 3 | Word2Vec → CNN, distilled to a scalar `cnn_score` | CNN | TensorFlow/Keras |

CNN is distilled to a single column so it can be fed back through MLlib `VectorAssembler` alongside structured features.

### PD variants (notebook 05)

Built incrementally on top of the WoE-binned structured baseline (top 60 features by IV proxy):

```
M0_baseline       structured only
M1_+tfidf         + TF-IDF vector
M2_+w2v           + Word2Vec vector
M3_+cnn           + CNN distilled score
M4_+categorical   + title_ohe + emp_ohe (occupation/title one-hot)
M5_full           everything combined
```

### LGD & EAD (notebook 05, Section 8)

| Component | Method | Justification |
|---|---|---|
| LGD | Tweedie GLM (`var_power=1.5`) via `statsmodels` | No Beta/Tweedie GLM in MLlib; fit on defaulted loans only |
| EAD | Spark MLlib LinearRegression on CCF target | Stays distributed; CCF = `total_rec_prncp / funded_amnt` |

LGD/EAD are fit once on the full training set and predicted on the full test set, then inner-joined back per phase to compute EL.

---

## Key Results

### Phase A — desc-present subset (5,109 test loans)

| Variant | AUC | Gini | KS | Δ AUC |
|---|---|---|---|---|
| M0_baseline | 0.6870 | 0.3741 | 0.2760 | — |
| M1_+tfidf | 0.6906 | 0.3812 | 0.2992 | +0.0036 |
| **M2_+w2v** | **0.7009** | **0.4017** | **0.3153** | **+0.0138** |
| M3_+cnn | 0.7005 | 0.4009 | 0.3034 | +0.0134 |
| M4_+categorical | 0.6958 | 0.3916 | 0.2983 | +0.0088 |
| M5_full | 0.6923 | 0.3846 | 0.3095 | +0.0052 |

**Finding**: Word2Vec gives the biggest lift (+1.38pp AUC). M5_full is *worse* than M2/M3 — naïvely concatenating all NLP signals introduces redundancy that hurts logistic regression.

### Phase B — full data (452,931 test loans)

| Variant | AUC | Gini | KS |
|---|---|---|---|
| M0_baseline_full | 0.7367 | 0.4733 | 0.3475 |
| M_categorical_full | 0.7377 | 0.4754 | 0.3493 |

Categorical NLP signals (title_ohe + emp_ohe) deliver a small but consistent +0.001 AUC at production scale. tfidf/w2v/cnn intentionally not used here — they're zero-filled for 94% of rows.

### Expected Loss

| | Loans | Total EL — Baseline | Total EL — Hybrid | Δ Mean EL / loan |
|---|---|---|---|---|
| Phase A (M0 vs M5_full) | 5,109 | $5,843,359 | $5,821,872 | −$4.21 |
| Phase B (M0 vs M_cat) | 452,931 | $424,805,793 | $424,755,497 | −$0.11 |

The hybrid models reduce projected losses in both phases. Phase A shows the larger per-loan effect because the NLP signal is unblurred.

---

## Project Structure

```
project/
├── docker-compose.yml & Dockerfile        # Reproducible environment
├── requirements.txt                       # Python deps
├── data/
│   ├── raw/                               # Lending Club CSVs (gzipped, gitignored)
│   └── parquet/                           # Intermediate Parquet (gitignored)
├── notebooks/                             # Pipeline (run in order)
│   ├── 00_file_convert.ipynb
│   ├── 01_data_loading.ipynb
│   ├── 02_general_prep.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_nlp_pipeline_dev.ipynb
│   └── 05_hybrid_pd_model.ipynb
├── src/                                   # Reusable Python modules
│   ├── spark_utils.py                     #   SparkSession factory
│   ├── general_prep.py                    #   train/test split + cleanup
│   ├── pd_feature_engineering.py          #   WoE binning + IV selection
│   └── evaluation.py                      #   AUC, Gini, KS, lift
├── outputs/
│   ├── figures/                           # All PNG plots
│   ├── tables/                            # Comparison + EL CSVs, WoE reports
│   └── models/                            # Saved MLlib pipelines + cnn_model.h5
└── docs/                                  # Proposal, grading rubric, reference paper
```

---

## Tech Stack

- **Spark / MLlib** — distributed feature engineering, LogisticRegression, LinearRegression, VectorAssembler, Pipeline
- **TensorFlow / Keras** — CNN distillation on Word2Vec embeddings
- **statsmodels** — Tweedie GLM for LGD (no Spark equivalent)
- **scikit-learn** — used only where MLlib lacks an equivalent
- **NLTK / VADER / TextBlob** — text cleaning + sentiment baselines
- **matplotlib / seaborn** — visualization

## Reproducibility Checklist

- [x] Docker-pinned Python and Spark versions
- [x] All raw → final pipeline runnable from `docker-compose up`
- [x] Train/test split deterministic (`seed=42`)
- [x] Schema-alignment assertion guards notebook 05 against upstream drift
- [x] Two showcase models persisted: `outputs/models/hybrid_pd_M5_subset/` (Phase A) and `outputs/models/hybrid_pd_categorical_full/` (Phase B)
