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

## NLP Feature Engineering Reference

Techniques learned from Questrom NLP Deep Learning workshop, adapted for credit default prediction.

### Constraint: PySpark + MLlib Only

All processing and model training must use PySpark and MLlib (big data modules). No scikit-learn for model training. Use `pyspark.ml` pipeline API throughout.

### Approach 1: TF-IDF + Logistic Regression (Baseline NLP)

This is the simplest NLP approach and already planned in Step 5.

```
Text fields (desc, title, emp_title)
  → Preprocessing: lowercase, remove HTML/stopwords, lemmatization (UDF or pyspark.sql.functions)
  → pyspark.ml.feature.Tokenizer or RegexTokenizer
  → pyspark.ml.feature.StopWordsRemover
  → pyspark.ml.feature.HashingTF + pyspark.ml.feature.IDF (TF-IDF)
  → Feed into pyspark.ml.classification.LogisticRegression
```

PySpark TF-IDF pipeline:
```python
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline

tokenizer = RegexTokenizer(inputCol="desc_clean", outputCol="words", pattern="\\W")
remover = StopWordsRemover(inputCol="words", outputCol="filtered")
hashingTF = HashingTF(inputCol="filtered", outputCol="rawFeatures", numFeatures=5000)
idf = IDF(inputCol="rawFeatures", outputCol="tfidf_features")
lr = LogisticRegression(featuresCol="tfidf_features", labelCol="default")

pipeline = Pipeline(stages=[tokenizer, remover, hashingTF, idf, lr])
model = pipeline.fit(train_df)
```

Key parameters:
- `numFeatures=5000`: HashingTF hashes words into 5000 buckets (equivalent to max_features in sklearn)
- `StopWordsRemover`: built-in English stopwords list
- No need for custom lemmatizer — RegexTokenizer + StopWordsRemover is sufficient for PySpark

### Approach 2: Word2Vec Mean Vector + Logistic Regression (Semantic NLP)

TF-IDF treats each word independently — "struggling" and "difficulty" are completely different dimensions. Word2Vec captures semantic similarity.

MLlib has built-in Word2Vec:
```python
from pyspark.ml.feature import Word2Vec

# Train Word2Vec on Lending Club text corpus
word2vec = Word2Vec(vectorSize=50, minCount=10, inputCol="filtered", outputCol="w2v_features")
w2v_model = word2vec.fit(df)
# Automatically outputs mean vector per document (no manual averaging needed)
df_w2v = w2v_model.transform(df)
```

MLlib Word2Vec automatically averages word vectors per document — no need to manually compute mean.

Combine with TF-IDF using VectorAssembler:
```python
from pyspark.ml.feature import VectorAssembler

assembler = VectorAssembler(
    inputCols=["tfidf_features", "w2v_features", "structured_features"],
    outputCol="all_features"
)
lr = LogisticRegression(featuresCol="all_features", labelCol="default")
```

Key parameters:
- `vectorSize=50`: each word → 50-dim vector
- `minCount=10`: ignore words appearing fewer than 10 times
- MLlib Word2Vec uses Skip-Gram internally

### Approach 3: Knowledge Base Sentence Extraction + Deep Learning (Advanced)

For long text fields like `desc`, most content is noise. Extract only default-relevant sentences first.

**Step 1: Build default risk knowledge base**
```
Seed words (manual): ['default', 'missed', 'late', 'payment', 'delinquent',
                       'overdue', 'bankrupt', 'deferred', 'collection']
→ Use Word2Vec to auto-expand with nearest neighbors
→ Knowledge base: {default, missed, late, struggling, arrears, unpaid, ...}
```

**Step 2: Score and extract sentences**
```
Distance method: cosine distance between sentence vector centroid and knowledge base centroid
Match method: count intersection of sentence words with knowledge base
→ Keep top N most relevant sentences per borrower
```

**Step 3: Feed into MLlib classifier**
```
Extracted sentences
  → RegexTokenizer → StopWordsRemover → HashingTF + IDF
  → VectorAssembler (combine with structured features)
  → LogisticRegression or GBTClassifier or RandomForestClassifier (MLlib)
```

Note: MLlib does not have CNN/LSTM. For deep learning within PySpark, options are limited.
The practical approach is: use knowledge base extraction to boost signal, then classify with MLlib models.

```python
from pyspark.ml.classification import GBTClassifier, RandomForestClassifier

# GBT often outperforms Logistic Regression on mixed features
gbt = GBTClassifier(featuresCol="all_features", labelCol="default", maxIter=50)
```

### NLP Pipeline: Three-Stage Comparison

The NLP pipeline progresses from traditional statistics to deep learning. Each stage builds on the previous. Detailed implementation steps follow the Questrom NLP Deep Learning workshop notebook (`/Users/yishanranxin./Desktop/NLP/NLP_app/NLP_Deep_Learning_Questrom_WS_03_2020.ipynb`).

**Stage 1: TF-IDF → Logistic Regression (MLlib)**
```
text → PySpark HashingTF + IDF → 5000-dim sparse vector
     → LogisticRegression (MLlib)
     → baseline AUC
```
Pure word frequency signal. No semantic understanding.

**Stage 2: TF-IDF + Word2Vec → Logistic Regression (MLlib)**
```
text → TF-IDF (5000-dim) + Word2Vec mean (50-dim)
     → VectorAssembler → combined 5050-dim vector
     → LogisticRegression (MLlib)
     → compare AUC vs Stage 1
```
Adds semantic features. "struggling" and "difficulty" now contribute similarly.

**Stage 3: Word2Vec Embedding → CNN/BiLSTM (TensorFlow)**
```
text → PySpark preprocessing (tokenize, clean, pad sequences)
     → collect() to numpy
     → Embedding layer (Word2Vec or GloVe)
     → CNN (local word group patterns) or BiLSTM (sequence understanding)
     → Dense(1, sigmoid) → default probability
     → compare AUC vs Stage 1 & 2
```
Understands word order and context. "not defaulting" vs "defaulting" are distinguished.

**Final comparison table in report:**
| Stage | Features | Model | Framework | AUC |
|-------|----------|-------|-----------|-----|
| 1 | TF-IDF | LR | MLlib | ? |
| 2 | TF-IDF + Word2Vec | LR | MLlib | ? |
| 3 | Word2Vec Embedding | CNN/BiLSTM | TensorFlow | ? |

Each stage demonstrates a clear methodological advancement.

### Practical Notes

- `desc` has 94% missing — for Word2Vec/sentence extraction, only use non-missing subset or combine with `title` field
- PySpark Pipeline handles train/test split correctly — `pipeline.fit(train)` then `model.transform(test)` avoids data leakage
- For Word2Vec: cosine distance measures semantic similarity (direction matters, not magnitude)
- MLlib NaiveBayes is a fast alternative classifier — assumes word independence, just counts word frequencies per class
- All MLlib classifiers work with Pipeline API — chain tokenizer → TF-IDF → assembler → classifier in one pipeline
- Use `pyspark.ml.evaluation.BinaryClassificationEvaluator` for AUC, `MulticlassClassificationEvaluator` for accuracy/F1
- Use `pyspark.ml.tuning.CrossValidator` or `TrainValidationSplit` for hyperparameter tuning (equivalent to sklearn GridSearchCV)
