# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Credit Default Prediction via NLP-Based Sentiment Analysis** — BU MF810 final project (Zhankai Zhang, Weiyu Wang). Hybrid credit-scoring framework on Lending Club data (~2.26M loans, 151 cols) that adds NLP features from `desc` / `title` / `emp_title` to a traditional WoE-binned baseline.

Loss decomposition: **EL = PD × LGD × EAD**. Hybrid effort focuses on the PD model; LGD (Tweedie GLM) and EAD (LinearRegression) are fit once on full data.

## Common Commands

```bash
# Bring up Jupyter + Spark (binds repo to /home/jovyan/work)
docker-compose up                     # then open the printed URL

# Run a notebook end-to-end inside the container
docker exec mf810-project jupyter nbconvert --to notebook \
    --execute notebooks/05_hybrid_pd_model.ipynb --inplace

# Drop into the container shell
docker exec -it mf810-project bash
```

`SPARK_DRIVER_MEMORY` is set to `7g` in `docker-compose.yml`. Notebook 05 overrides this to `4g` heap + `1g` off-heap because it caches large feature matrices — don't bump it back without checking what the notebook does.

There is no test suite; no linter is configured. Validation = run-all the notebook chain top-to-bottom and check artifacts in `outputs/`.

## Notebook Pipeline (run in order)

| Notebook | Purpose | Key outputs |
|---|---|---|
| `00_file_convert` | CSV → Parquet (one-shot, also done in 01) | `data/parquet/accepted_*.parquet` |
| `01_data_loading` | Initial load + train/test split | `data/parquet/accepted_general_prep_{train,test}.parquet` |
| `02_general_prep` | EDA, missing-pattern analysis, cleanup | figures in `outputs/figures/` |
| `03_feature_engineering` | WoE binning + IV-based selection on structured features | `data/parquet/features_pd_{train,test}.parquet`, `outputs/tables/woe_report_*.csv` |
| `04_nlp_pipeline_dev` | TF-IDF + Word2Vec + CNN distillation, occupation clustering | `outputs/nlp_features_{train,test}.parquet`, `nlp_tfidf_*`, `nlp_w2v_*`, `outputs/models/{tfidf_model,w2v_model,cnn_model.h5}` |
| `05_hybrid_pd_model` | Hybrid PD with two-phase variant comparison + LGD/EAD + EL | comparison CSVs, ROC/lift/feat-imp PNGs, saved MLlib pipelines |

Notebook 03 produces the structured features used downstream. Train/test schemas are aligned (132 cols each); a defensive `assert` in notebook 05 Section 1 guards against future drift.

## Notebook 05 — Two-Phase Hybrid PD Architecture

This is the most complex notebook and the heart of the modeling story. It is structured as two independent experiments because `desc` is missing in ~94% of rows, which would drown the NLP signal if everything ran on full data.

- **Phase A** — *desc-present subset.* Union the train+test ids where `desc_missing == 0` (~135K rows), re-split 80/20 with `seed=42`, train all 6 variants:
  - `M0_baseline` (structured only) → `M1_+tfidf` → `M2_+w2v` → `M3_+cnn` → `M4_+categorical` (title_ohe + emp_ohe) → `M5_full` (everything).
  - `desc_missing` is excluded — it's a constant 0 in this subset.
- **Phase B** — *full data, original split (1.8M / 451K).* Only `M0_baseline_full` and `M_categorical_full`. tfidf/w2v/cnn are not used here because they're zero-filled for 94% of rows and become noise.
- **LGD / EAD** are fit once on full train and predicted on full test; `compute_el_table` inner-merges so each phase's EL uses only its own test ids.

Implementation conventions — please preserve:
- All helper functions live in **one cell** (`81c821f3`) early in the notebook: `prepare_nlp`, `build_fit_data`, `train_pd_variant`, `infer_vector_size`, `compute_metrics_table`, `plot_roc_overlay`, `plot_lift_comparison`, `extract_feature_names_for_pipeline`, `plot_feature_importance`, `compute_el_table`. Both phases reuse them.
- A **skinny NLP checkpoint** (`data/parquet/_hybrid_nlp_{train,test}.parquet`, 9 cols only) is written once and each variant inner-joins only the NLP cols it actually uses — avoids shuffling unused vector cols.
- `structured_cols` is selected via a one-pass IV proxy (`|P(c=1|default) − P(c=1|non_default)|`), capped at top 60 — this keeps the cached LR feature matrix inside the driver heap budget.
- Label semantics: `good_bad = 1` is non-default, `good_bad = 0` is default. **Probability of default is `probability[0]`** — see the `extract_p_default` UDF.
- Two showcase models are persisted: `outputs/models/hybrid_pd_M5_subset/` and `outputs/models/hybrid_pd_categorical_full/`.

## `src/` Modules

Reusable Python imported by notebooks (`sys.path.insert(0, '/home/jovyan/work/src')`):

- `spark_utils.py` — `get_spark_session()` (reads `SPARK_DRIVER_MEMORY`), Parquet helpers.
- `general_prep.py` — early-stage cleanup + train/test split (used by notebook 02).
- `pd_feature_engineering.py` — WoE binning, IV computation, feature selection (used by notebook 03).
- `evaluation.py` — `compute_auc_roc`, `compute_gini`, `compute_ks_statistic`, `compute_lift_table`, `plot_lift_chart`. Notebook 05 imports `compute_ks_statistic` and `compute_lift_table` directly.

## Constraints & Gotchas

- **PySpark + MLlib for modeling.** sklearn is allowed only where MLlib has no equivalent: LGD uses `statsmodels.GLM(family=Tweedie(var_power=1.5))` (no Beta/Tweedie GLM in MLlib) and EAD uses `sklearn.LinearRegression` on collected pandas (target needs raw `funded_amnt` × CCF, simpler in pandas after the join). Document any further deviations in the report.
- **Don't widen `structured_cols` past ~60** without re-checking memory. Wide dense int cols × 1.8M rows can OOM the driver heap when cached for LR.
- **`desc` is 94% missing.** Any NLP work that depends on `desc` (TF-IDF, Word2Vec, CNN) should report metrics on the desc-present subset to avoid dilution. Categorical NLP signals (title_ohe, emp_ohe) are dense enough to use on full data.
- **Raw CSVs are not in git** (`data/raw/accepted_2007_to_2018Q4.csv.gz`, `rejected_*.csv.gz`). Place them under `data/raw/` before running notebook 01.

## Output Layout

```
outputs/
├── nlp_features_{train,test}.parquet       # full NLP feature frames (notebook 04)
├── nlp_{tfidf,w2v}_{train,test}.parquet    # split-out vector cols
├── {train,test}_extracted_sentences*.parquet  # KB sentence extraction
├── knowledge_base.json
├── models/
│   ├── tfidf_model/, w2v_model/, cnn_model.h5    # NLP stage models
│   ├── hybrid_pd_M5_subset/                       # Phase A showcase
│   └── hybrid_pd_categorical_full/                # Phase B production
├── tables/
│   ├── woe_report_*.csv                           # per-feature WoE/IV (notebook 03)
│   ├── nlp_stage_metrics.json
│   ├── hybrid_model_comparison_{subset,full}.csv  # notebook 05 metrics
│   └── expected_loss_{subset,full}.csv
└── figures/                                       # all PNGs
```

## NLP Three-Stage Methodology (for reference)

The NLP track in notebook 04 progresses through three stages for the report:

| Stage | Features | Model | Framework |
|---|---|---|---|
| 1 | TF-IDF (HashingTF 5000-dim + IDF) | LogisticRegression | MLlib |
| 2 | TF-IDF + Word2Vec (50-dim mean) via VectorAssembler | LogisticRegression | MLlib |
| 3 | Word2Vec embedding → CNN (distilled to scalar `cnn_score`) | CNN | TensorFlow/Keras |

Stage 3 is distilled to a single `cnn_score` column so it can be fed back through MLlib `VectorAssembler` alongside the structured + tfidf + w2v features in notebook 05.

## Grading Hooks (priorities)

Per `docs/grading_rubric.pdf`: scientific paper structure (6–8 page main body), demonstrated data complexity, Spark/MapReduce usage, grammar-based manipulation or SQL, justified algorithm choices with plots, fully Docker-reproducible.
