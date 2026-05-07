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
| **A** — desc-present | rows where `desc_missing == 0` | ~25K (re-split 80/20, ~5K test) | 6 (M0 → M5) |
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

### Phase A — desc-present subset (5,080 test loans)

| Variant | AUC | Gini | KS | Δ AUC |
|---|---|---|---|---|
| M0_baseline | 0.6871 | 0.3741 | 0.2711 | — |
| **M1_+tfidf** | **0.6877** | **0.3755** | **0.2891** | **+0.0007** |
| M2_+w2v | 0.6876 | 0.3751 | 0.2772 | +0.0005 |
| M3_+cnn | 0.6834 | 0.3667 | 0.2669 | −0.0037 |
| M4_+categorical | 0.6821 | 0.3643 | 0.2714 | −0.0049 |
| M5_full | 0.6859 | 0.3719 | 0.2790 | −0.0011 |

**Findings.** Three observations matter more than the absolute numbers:

1. **TF-IDF and Word2Vec contribute a real but small AUC lift** (+0.0007 / +0.0005), with a more visible KS improvement (+1.8pp at M1). On a ~5K-loan test fold, this is roughly the noise floor — the NLP signal is *directionally* consistent but not large enough to dominate a strong WoE-binned baseline.
2. **M3 (CNN) and M4 (categorical OHE) underperform the baseline.** This is informative, not a failure: the CNN is distilled to a single scalar that correlates with the W2V mean it was trained on, so adding it to a model that already contains W2V offers no new dimension; and `title_ohe` / `emp_ohe` are very high-cardinality and sparse on a 25K-row subset, where logistic regression overfits the rarely-occurring levels. Both effects vanish at full scale (Phase B).
3. **M5_full does not stack monotonically.** Concatenating every NLP feature into one LR introduces collinearity that the linear model cannot disentangle — this is exactly the regime where a non-linear PD model (GBM, NN) or per-feature regularization would extract more value, and is the most concrete direction for follow-up work.

The honest read: the WoE baseline is already near the information ceiling that linear models can extract from this dataset. NLP features add a small, measurable lift on the subset where `desc` exists and a smaller-but-consistent lift from occupational categoricals at full scale.

### Phase B — full data (452,931 test loans)

| Variant | AUC | Gini | KS |
|---|---|---|---|
| M0_baseline_full | 0.7415 | 0.4831 | 0.3548 |
| M_categorical_full | 0.7428 | 0.4856 | 0.3576 |

Categorical NLP signals (title_ohe + emp_ohe) deliver a small but consistent +0.0013 AUC and +0.003 KS at production scale. tfidf/w2v/cnn intentionally not used here — they're zero-filled for 94% of rows.

### Expected Loss

| | Loans | Total EL — Baseline | Total EL — Hybrid | Δ Mean EL / loan |
|---|---|---|---|---|
| Phase A (M0 vs M5_full) | 5,080 | $6,033,160 | $6,032,042 | −$0.22 |
| Phase B (M0 vs M_cat) | 452,931 | $428,704,435 | $428,671,886 | −$0.07 |

Hybrid models reduce projected losses in both phases, but the per-loan effect is small. The M5_full variant — chosen as the Phase A "showcase" because it bundles every NLP signal — gives roughly the same EL as the baseline despite its richer feature stack, consistent with M1/M2 being the actual driver of any lift on the subset.

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
