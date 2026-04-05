"""Data cleaning, target encoding, and derived column creation."""

from pyspark.sql import DataFrame
from pyspark.sql import functions as F


# Loan statuses considered as default
DEFAULT_STATUSES = [
    "Charged Off",
    "Default",
    "Late (31-120 days)",
    "Does not meet the credit policy. Status:Charged Off",
]


def encode_target(df, col="loan_status"):
    """Convert loan_status to binary: 1 = Default, 0 = Non-Default."""
    df = df.withColumn(
        "default_flag",
        F.when(F.col(col).isin(DEFAULT_STATUSES), 1).otherwise(0),
    )
    return df


def compute_missing_rate(df):
    """Return a dict of {column: missing_rate} for all columns."""
    total = df.count()
    missing = {}
    for col_name in df.columns:
        null_count = df.filter(F.col(col_name).isNull() | (F.col(col_name) == "")).count()
        missing[col_name] = round(null_count / total, 4)
    return dict(sorted(missing.items(), key=lambda x: x[1], reverse=True))


def create_lgd_column(df):
    """Compute LGD = 1 - (total_rec_prncp / funded_amnt) for defaulted loans."""
    df = df.withColumn(
        "lgd",
        F.when(
            F.col("default_flag") == 1,
            1 - F.col("total_rec_prncp") / F.col("funded_amnt"),
        ).otherwise(None),
    )
    # Clamp to [0, 1]
    df = df.withColumn("lgd", F.greatest(F.lit(0.0), F.least(F.lit(1.0), F.col("lgd"))))
    return df


def create_ead_column(df):
    """Compute EAD using Credit Conversion Factor: EAD = funded_amnt * CCF."""
    df = df.withColumn(
        "ccf",
        F.col("total_rec_prncp") / F.col("funded_amnt"),
    )
    df = df.withColumn("ccf", F.greatest(F.lit(0.0), F.least(F.lit(1.0), F.col("ccf"))))
    df = df.withColumn("ead", F.col("funded_amnt") * F.col("ccf"))
    return df


def handle_missing_numeric(df, columns, strategy="median"):
    """Impute missing numeric columns with median or mean."""
    for col_name in columns:
        if strategy == "median":
            val = df.approxQuantile(col_name, [0.5], 0.01)[0]
        else:
            val = df.select(F.mean(col_name)).first()[0]
        if val is not None:
            df = df.withColumn(col_name, F.when(F.col(col_name).isNull(), val).otherwise(F.col(col_name)))
    return df


def handle_missing_categorical(df, columns, fill_value="Unknown"):
    """Fill missing categorical columns with a default value."""
    for col_name in columns:
        df = df.withColumn(
            col_name,
            F.when(F.col(col_name).isNull() | (F.col(col_name) == ""), fill_value).otherwise(F.col(col_name)),
        )
    return df
