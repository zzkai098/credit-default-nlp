"""General preprocessing helpers ported from the original preparation notebook."""

import re

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window


DEFAULT_STATUSES = [
    "Charged Off",
    "Default",
    "Late (31-120 days)",
    "Does not meet the credit policy. Status:Charged Off",
]


def _safe_name(value: str) -> str:
    """Convert a category value into a safe Spark column suffix."""
    text = "missing" if value is None else str(value).strip()
    text = re.sub(r"[^0-9A-Za-z]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text.lower() if text else "missing"

def add_int_by_extracting_number(
    df: DataFrame,
    source_col: str,
    target_col: str,
    special_map: dict[str, int] | None = None,
) -> DataFrame:
    
    extracted = F.regexp_extract(F.col(source_col), r"(\d+)", 1)
    if special_map:
        expr = None
        for old_value, new_value in special_map.items():
            condition = F.col(source_col) == old_value
            value_expr = F.lit(new_value)
            expr = F.when(condition, value_expr) if expr is None else expr.when(condition, value_expr)
        expr = expr.otherwise(extracted.cast("int"))
        return df.withColumn(target_col, expr)

    return df.withColumn(target_col, extracted.cast("int"))

def add_months_since_column(
    df: DataFrame,
    source_col: str,
    output_col: str,
    reference_date: str = "2017-12-01",
    fmt: str = "MMM-yy",
) -> DataFrame:
    """
    Add months-since column from date-like text and clamp negatives to max observed value.

    This mirrors the original notebook behavior that replaced negative
    `mths_since_earliest_cr_line` values with the max value.
    """
    ref = F.to_date(F.lit(reference_date))
    parsed = F.to_date(F.col(source_col), fmt)
    df = df.withColumn(output_col, F.floor(F.months_between(ref, parsed)).cast("int"))

    max_non_negative = F.max(F.when(F.col(output_col) >= 0, F.col(output_col))).over(
        Window.rowsBetween(Window.unboundedPreceding, Window.unboundedFollowing)
    )
    df = df.withColumn(
        output_col,
        F.when(F.col(output_col) < 0, max_non_negative).otherwise(F.col(output_col)),
    )
    return df

def safe_name(value) -> str:
    s = str(value).strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^0-9A-Za-z_]", "_", s)
    s = re.sub(r"_+", "_", s)
    s = s.strip("_")
    return s

def fit_dummy_spec(df: DataFrame, columns: list[str], max_categories: int = 60) -> dict[str, list]:
    dummy_spec = {}

    for col_name in columns:
        if col_name not in df.columns:
            continue

        categories = (
            df.select(col_name)
            .where(F.col(col_name).isNotNull())
            .groupBy(col_name)
            .count()
            .orderBy(F.desc("count"), F.asc(col_name))
            .limit(max_categories)
            .collect()
        )

        dummy_spec[col_name] = [row[col_name] for row in categories]

    return dummy_spec

def apply_dummy_columns(df: DataFrame, dummy_spec: dict[str, list]) -> DataFrame:
    out = df

    for col_name, categories in dummy_spec.items():
        if col_name not in out.columns:
            continue

        for raw_val in categories:
            safe_val = safe_name(raw_val)
            new_col = f"{col_name}:{safe_val}"

            out = out.withColumn(
                new_col,
                F.when(F.col(col_name) == F.lit(raw_val), F.lit(1)).otherwise(F.lit(0))
            )

    return out

def get_missing_summary(df):
    total_rows = df.count()
    spark = df.sparkSession

    missing_exprs = [
        F.sum(F.when(F.col(c).isNull(), 1).otherwise(0)).alias(c)
        for c in df.columns
    ]

    missing_row = df.select(missing_exprs).collect()[0].asDict()

    result = []
    for col_name, missing_count in missing_row.items():
        if missing_count > 0:
            result.append((col_name, missing_count, missing_count / total_rows))

    return spark.createDataFrame(result, ["column", "missing_count", "missing_ratio"]) \
                .orderBy(F.desc("missing_count"))

def fill_null_with_column(df: DataFrame, target_col: str, source_col: str) -> DataFrame:
    return df.withColumn(
        target_col,
        F.when(F.col(target_col).isNull(), F.col(source_col)).otherwise(F.col(target_col))
    )

def fit_mean_imputer(train_df: DataFrame, target_col: str) -> float:
    mean_value = (
        train_df
        .select(F.avg(F.col(target_col)).alias("mean_value"))
        .collect()[0]["mean_value"]
    )
    return mean_value

def apply_mean_imputer(df: DataFrame, target_col: str, mean_value: float) -> DataFrame:
    return df.fillna({target_col: mean_value})

def fill_null_with_zero(df: DataFrame, columns: list[str]) -> DataFrame:
    fill_map = {c: 0 for c in columns if c in df.columns}
    return df.fillna(fill_map)

def check_null_count(df: DataFrame, columns: list[str], name: str = "df") -> None:
    exprs = [
        F.sum(F.when(F.col(c).isNull(), 1).otherwise(0)).alias(c)
        for c in columns if c in df.columns
    ]
    print(f"=== null check: {name} ===")
    df.select(exprs).show(truncate=False)

def add_good_bad_target(df: DataFrame, status_col: str = "loan_status") -> DataFrame:
    """
    Add binary target column:
    good_bad = 0 for default / bad statuses
    good_bad = 1 otherwise
    """
    return df.withColumn(
        "good_bad",
        F.when(F.col(status_col).isin(DEFAULT_STATUSES), F.lit(0)).otherwise(F.lit(1))
    )