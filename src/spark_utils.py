"""Spark session management and Parquet I/O utilities."""

from pyspark.sql import SparkSession, DataFrame


def get_spark_session(app_name="MF810_CreditDefault", memory="4g"):
    """Create and return a SparkSession configured for local mode."""
    spark = (
        SparkSession.builder
        .appName(app_name)
        .master("local[*]")
        .config("spark.driver.memory", memory)
        .config("spark.sql.parquet.compression.codec", "snappy")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark


def csv_to_parquet(spark, csv_path, parquet_path, header=True, infer_schema=True):
    """Load a CSV file and save as Parquet."""
    df = spark.read.csv(csv_path, header=header, inferSchema=infer_schema)
    df.write.mode("overwrite").parquet(parquet_path)
    print(f"Saved {df.count()} rows to {parquet_path}")
    return df


def load_parquet(spark, path):
    """Load a Parquet file and return a Spark DataFrame."""
    df = spark.read.parquet(path)
    print(f"Loaded {df.count()} rows from {path}")
    return df


def save_parquet(df, path):
    """Save a Spark DataFrame as Parquet."""
    df.write.mode("overwrite").parquet(path)
    print(f"Saved to {path}")
