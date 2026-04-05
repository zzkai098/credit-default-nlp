"""Weight of Evidence (WoE) encoding and Information Value (IV) feature selection."""

import numpy as np
import pandas as pd


def calculate_woe(df_pd, feature, target="default_flag", n_bins=10):
    """
    Calculate Weight of Evidence for a feature.

    Args:
        df_pd: pandas DataFrame
        feature: column name
        target: binary target column
        n_bins: number of bins for continuous features

    Returns:
        DataFrame with bin, count, events, non-events, WoE, IV per bin
    """
    df = df_pd[[feature, target]].copy()

    # Bin continuous features
    if df[feature].dtype in ["float64", "float32", "int64", "int32"]:
        df["bin"] = pd.qcut(df[feature], q=n_bins, duplicates="drop")
    else:
        df["bin"] = df[feature]

    grouped = df.groupby("bin")[target].agg(["count", "sum"])
    grouped.columns = ["total", "events"]
    grouped["non_events"] = grouped["total"] - grouped["events"]

    total_events = grouped["events"].sum()
    total_non_events = grouped["non_events"].sum()

    # Avoid division by zero
    grouped["dist_events"] = (grouped["events"] / total_events).replace(0, 0.0001)
    grouped["dist_non_events"] = (grouped["non_events"] / total_non_events).replace(0, 0.0001)

    grouped["woe"] = np.log(grouped["dist_non_events"] / grouped["dist_events"])
    grouped["iv"] = (grouped["dist_non_events"] - grouped["dist_events"]) * grouped["woe"]

    return grouped.reset_index()


def calculate_iv(woe_table):
    """Calculate total Information Value from a WoE table."""
    return woe_table["iv"].sum()


def select_features_by_iv(df_pd, features, target="default_flag", threshold=0.02):
    """
    Compute IV for each feature and return those above threshold.

    IV interpretation:
        < 0.02: not useful
        0.02 - 0.1: weak predictor
        0.1 - 0.3: medium predictor
        > 0.3: strong predictor

    Returns:
        DataFrame with feature name, IV, and strength category, sorted by IV descending
    """
    results = []
    for feat in features:
        try:
            woe_table = calculate_woe(df_pd, feat, target)
            iv = calculate_iv(woe_table)
            results.append({"feature": feat, "iv": iv})
        except Exception:
            continue

    iv_df = pd.DataFrame(results).sort_values("iv", ascending=False)

    # Add strength category
    iv_df["strength"] = pd.cut(
        iv_df["iv"],
        bins=[-np.inf, 0.02, 0.1, 0.3, np.inf],
        labels=["not useful", "weak", "medium", "strong"],
    )

    selected = iv_df[iv_df["iv"] >= threshold]
    return selected


def apply_woe_encoding(df_pd, feature, target="default_flag", n_bins=10):
    """Replace a feature's values with their WoE scores."""
    woe_table = calculate_woe(df_pd, feature, target, n_bins)
    woe_map = dict(zip(woe_table["bin"], woe_table["woe"]))

    if df_pd[feature].dtype in ["float64", "float32", "int64", "int32"]:
        bins = pd.qcut(df_pd[feature], q=n_bins, duplicates="drop")
        df_pd[f"{feature}_woe"] = bins.map(woe_map)
    else:
        df_pd[f"{feature}_woe"] = df_pd[feature].map(woe_map)

    return df_pd
