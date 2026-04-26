
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd
from pyspark.ml.feature import Bucketizer
from pyspark.sql import DataFrame
from pyspark.sql import functions as F


EPS = 1e-9


# ============================================================
# Generic helpers
# ============================================================

def _validate_spark_df(df: DataFrame, name: str = "df") -> None:
    if not isinstance(df, DataFrame):
        raise TypeError(f"{name} must be a pyspark.sql.DataFrame, got {type(df)}")


def _safe_str_col(col_name: str) -> str:
    return (
        str(col_name)
        .replace(" ", "")
        .replace("/", "_")
        .replace(".", "_")
        .replace(":", "_")
        .replace("%", "pct")
    )


def _to_good_indicator(target_col: str, good_value: Any = 1) -> F.Column:
    return F.when(F.col(target_col) == F.lit(good_value), F.lit(1.0)).otherwise(F.lit(0.0))


def _create_map_expr(mapping: Mapping[str, Any]) -> F.Column:
    items: List[F.Column] = []
    for k, v in mapping.items():
        items.extend([F.lit(str(k)), F.lit(v)])
    return F.create_map(*items)


def _numeric_from_any(col_name: str) -> F.Column:
    as_str = F.trim(F.col(col_name).cast("string"))
    cleaned = F.regexp_replace(F.regexp_replace(as_str, ",", ""), "%", "")
    extracted = F.regexp_extract(cleaned, r"[-+]?(?:\d+\.?\d*|\.\d+)", 0)
    return F.when(
        as_str.isNull() | as_str.isin("", "nan", "NaN", "none", "None", "null", "NULL", "N/A", "NA"),
        F.lit(None).cast("double"),
    ).otherwise(F.when(extracted == "", F.lit(None).cast("double")).otherwise(extracted.cast("double")))


def _compute_numeric_splits(
    df: DataFrame,
    feature: str,
    n_bins: int = 50,
    relative_error: float = 0.001,
) -> List[float]:
    quantiles = [i / n_bins for i in range(1, n_bins)]
    values = df.where(F.col(feature).isNotNull()).select(feature).approxQuantile(feature, quantiles, relative_error)
    clean = sorted({float(v) for v in values if v is not None and not math.isnan(float(v))})
    if not clean:
        return [-float("inf"), 0.0, float("inf")]
    splits = [-float("inf")] + clean + [float("inf")]
    # Guard against accidental invalid/degenerate split arrays.
    if len(splits) < 3:
        center = clean[0] if clean else 0.0
        return [-float("inf"), center, float("inf")]
    return splits


def _interval_label(index: int, splits: Sequence[float]) -> str:
    left = splits[index]
    right = splits[index + 1]

    def _fmt(x: float) -> str:
        if math.isinf(x):
            return "-inf" if x < 0 else "inf"
        if float(x).is_integer():
            return str(int(x))
        return f"{x:.6g}"

    return f"({_fmt(left)}, {_fmt(right)}]"


def _bin_numeric_feature(
    df: DataFrame,
    feature: str,
    splits: Sequence[float],
    output_col: str,
    missing_label: str = "MISSING",
) -> DataFrame:
    bucket_idx_col = f"__bucket_idx_{_safe_str_col(feature)}"
    bucketizer = Bucketizer(
        splits=list(splits),
        inputCol=feature,
        outputCol=bucket_idx_col,
        handleInvalid="keep",
    )
    out = bucketizer.transform(df)

    mapping_expr = F.create_map(
        *[
            x
            for i in range(len(splits) - 1)
            for x in (F.lit(float(i)), F.lit(_interval_label(i, splits)))
        ]
    )

    return (
        out.withColumn(
            output_col,
            F.when(F.col(feature).isNull(), F.lit(missing_label)).otherwise(mapping_expr[F.col(bucket_idx_col)]),
        ).drop(bucket_idx_col)
    )


def build_woe_iv_table(
    df: DataFrame,
    group_col: str,
    target_col: str,
    good_value: Any = 1,
    ordered_labels: Optional[Sequence[str]] = None,
    sort_by: str = "woe",
) -> pd.DataFrame:
    """
    Build a standard WoE / IV table from a pre-grouped column.
    """
    _validate_spark_df(df)

    tmp = df.withColumn("__is_good__", _to_good_indicator(target_col, good_value))
    stats = (
        tmp.groupBy(group_col)
        .agg(
            F.count(F.lit(1)).alias("n_obs"),
            F.avg("__is_good__").alias("prop_good"),
        )
        .withColumn("n_good", F.col("n_obs") * F.col("prop_good"))
        .withColumn("n_bad", F.col("n_obs") - F.col("n_good"))
    )

    totals = stats.agg(
        F.sum("n_obs").alias("total_obs"),
        F.sum("n_good").alias("total_good"),
        F.sum("n_bad").alias("total_bad"),
    ).first()

    total_obs = float(totals["total_obs"] or 0.0)
    total_good = float(totals["total_good"] or 0.0)
    total_bad = float(totals["total_bad"] or 0.0)

    stats = (
        stats.withColumn("prop_n_obs", F.col("n_obs") / F.lit(total_obs if total_obs else 1.0))
        .withColumn("prop_n_good", F.col("n_good") / F.lit(total_good if total_good else 1.0))
        .withColumn("prop_n_bad", F.col("n_bad") / F.lit(total_bad if total_bad else 1.0))
        .withColumn("WoE", F.log((F.col("prop_n_good") + F.lit(EPS)) / (F.col("prop_n_bad") + F.lit(EPS))))
    )

    pdf = stats.toPandas().rename(columns={group_col: "bin"})
    if pdf.empty:
        return pdf

    pdf["bin"] = pdf["bin"].astype(str)

    if ordered_labels is not None:
        order_map = {str(label): i for i, label in enumerate(ordered_labels)}
        pdf["__order__"] = pdf["bin"].map(order_map)
        pdf = pdf.sort_values("__order__", kind="stable").reset_index(drop=True)
    elif sort_by == "feature":
        pdf = pdf.sort_values("bin", kind="stable").reset_index(drop=True)
    else:
        pdf = pdf.sort_values("WoE", kind="stable").reset_index(drop=True)

    pdf["diff_prop_good"] = pdf["prop_good"].diff().abs()
    pdf["diff_WoE"] = pdf["WoE"].diff().abs()
    pdf["IV_component"] = (pdf["prop_n_good"] - pdf["prop_n_bad"]) * pdf["WoE"]
    pdf["IV"] = float(pdf["IV_component"].sum())

    keep = [
        "bin",
        "n_obs",
        "prop_good",
        "prop_n_obs",
        "n_good",
        "n_bad",
        "prop_n_good",
        "prop_n_bad",
        "WoE",
        "diff_prop_good",
        "diff_WoE",
        "IV_component",
        "IV",
    ]
    if "__order__" in pdf.columns:
        keep = ["__order__"] + keep
    return pdf[keep]


def plot_by_woe(
    woe_table: pd.DataFrame,
    *,
    x_col: str = "bin",
    y_col: str = "WoE",
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 4),
    rotate_xticks: int = 45,
):
    import matplotlib.pyplot as plt

    plot_df = woe_table.copy()
    plot_df[x_col] = plot_df[x_col].astype(str)

    plt.figure(figsize=figsize)
    plt.plot(plot_df[x_col], plot_df[y_col], marker="o")
    plt.xticks(rotation=rotate_xticks, ha="right")
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(title or "WoE by Category / Bin")
    plt.tight_layout()
    plt.show()

def add_bin_indicator_columns(
    df: DataFrame,
    grp_col: str,
    prefix: str | None = None,
    drop_grp_col: bool = False,
) -> DataFrame:
    """
    Expand one grouped column like 'addr_state_grp' into one indicator column per bin.

    Example output columns:
      addr_state_bin_ND_NE_IA_NV_FL_HI_AL
      addr_state_bin_NM_VA
      ...
    """
    prefix = prefix or grp_col.replace("_grp", "")
    labels = [r[0] for r in df.select(grp_col).distinct().collect()]
    labels = [x for x in labels if x is not None]

    out = df
    for label in labels:
        new_col = f"{prefix}_{_safe_str_col(label)}"
        out = out.withColumn(
            new_col,
            F.when(F.col(grp_col) == F.lit(label), F.lit(1)).otherwise(F.lit(0))
        )

    if drop_grp_col:
        out = out.drop(grp_col)

    return out

def add_all_bin_indicator_columns(
    df: DataFrame,
    pipelines: dict,
    drop_grp_cols: bool = False,
) -> DataFrame:
    out = df
    for feature in pipelines.keys():
        grp_col = f"{feature}_grp"
        if grp_col in out.columns:
            out = add_bin_indicator_columns(
                out,
                grp_col=grp_col,
                prefix=feature,
                drop_grp_col=drop_grp_cols,
            )
    return out



@dataclass
class CandidateReport:
    small_bins: List[str]
    close_pairs: List[Tuple[str, str, float]]
    monotonic_break_pairs: List[Tuple[str, str]]
    suggested_pair: Optional[Tuple[str, str]]
    stop_reason: str


class SingleFeatureWOEPipeline:

    def __init__(
        self,
        feature: str,
        target_col: str,
        *,
        feature_type: str = "unordered_discrete",
        good_value: Any = 1,
        n_initial_bins: int = 50,
        min_bin_frac: float = 0.05,
        min_prop_good: float = 0.05,
        min_prop_bad: float = 0.01,
        min_diff_woe: float = 0.005,
        min_diff_prop_good: float = 0.001,
        enforce_monotonic: bool = False,
        missing_label: str = "MISSING",
        max_iter: int = 50,
    ):
        self.feature = feature
        self.target_col = target_col
        self.feature_type = feature_type
        self.good_value = good_value
        self.n_initial_bins = n_initial_bins
        self.min_bin_frac = min_bin_frac
        self.min_prop_good = min_prop_good
        self.min_prop_bad = min_prop_bad
        self.min_diff_woe = min_diff_woe
        self.min_diff_prop_good = min_diff_prop_good
        self.enforce_monotonic = enforce_monotonic
        self.missing_label = missing_label
        self.max_iter = max_iter

        self.base_col_: Optional[str] = None
        self.group_col_: Optional[str] = None
        self.special_num_col_: Optional[str] = None
        self.splits_: Optional[List[float]] = None
        self.base_labels_: List[str] = []
        self.current_groups_: List[List[str]] = []
        self.fitted_: bool = False
        self.final_table_: Optional[pd.DataFrame] = None
        self.woe_map_: Dict[str, float] = {}
        self.group_map_: Dict[str, str] = {}
        self.history_: List[Dict[str, Any]] = []

    # -----------------------------
    # internals
    # -----------------------------
    def _is_unordered_discrete(self) -> bool:
        return self.feature_type == "unordered_discrete"

    def _is_continuous_like(self) -> bool:
        return self.feature_type == "continuous"
    
    def _parse_interval_left(self, label: str) -> str:
        m = re.match(r"^\((.*?),\s*(.*?)\]$", str(label).strip())
        if not m:
            return str(label)
        return m.group(1).strip()

    def _parse_interval_right(self, label: str) -> str:
        m = re.match(r"^\((.*?),\s*(.*?)\]$", str(label).strip())
        if not m:
            return str(label)
        return m.group(2).strip()

    def _merge_interval_labels(self, labels: Sequence[str]) -> str:
        """
        Merge labels like
        ['(6.08, 6.67]', '(6.67, 7.12]', '(7.12, 7.46]']
        into
        '(6.08, 7.46]'
        """
        labels = [str(x) for x in labels]

        if len(labels) == 1:
            return labels[0]

        left = self._parse_interval_left(labels[0])
        right = self._parse_interval_right(labels[-1])

        return f"({left}, {right}]"

    # -----------------------------
    # initial classing
    # -----------------------------
    def _prepare_base_groups(self, train_df: DataFrame) -> DataFrame:
        _validate_spark_df(train_df, "train_df")
        if self.feature not in train_df.columns:
            raise ValueError(f"Feature '{self.feature}' not found in train_df.")
        if self.target_col not in train_df.columns:
            raise ValueError(f"Target '{self.target_col}' not found in train_df.")

        base_col = f"{self.feature}__base_grp"
        self.base_col_ = base_col

        if self._is_unordered_discrete():
            prepared = train_df.withColumn(
                base_col,
                F.when(F.col(self.feature).isNull(), F.lit(self.missing_label))
                .otherwise(F.col(self.feature).cast("string")),
            )
            labels = [row[0] for row in prepared.select(base_col).distinct().collect()]
            labels = [str(x) for x in labels if x is not None]
            self.base_labels_ = labels

        elif self.feature_type == "continuous":
            numeric_col = f"{self.feature}__num_for_bin"
            with_num = train_df.withColumn(numeric_col, _numeric_from_any(self.feature))
            self.splits_ = _compute_numeric_splits(with_num, numeric_col, self.n_initial_bins)
            prepared = _bin_numeric_feature(with_num, numeric_col, self.splits_, base_col, self.missing_label).drop(numeric_col)
            self.base_labels_ = [_interval_label(i, self.splits_) for i in range(len(self.splits_) - 1)]
            if self.missing_label not in self.base_labels_:
                self.base_labels_.append(self.missing_label)

        else:
            raise ValueError("feature_type must be 'unordered_discrete' or 'continuous'.")

        self.current_groups_ = [[label] for label in self.base_labels_]
        return prepared

    def _group_name(self, labels: Sequence[str]) -> str:
        labels = [str(x) for x in labels]
        if len(labels) == 1:
            return labels[0]

        if self._is_continuous_like():
            return self._merge_interval_labels(labels)

        return " | ".join(labels)

    def _current_group_map(self) -> Dict[str, str]:
        mapping: Dict[str, str] = {}
        for group in self.current_groups_:
            merged_name = self._group_name(group)
            for base_label in group:
                mapping[str(base_label)] = merged_name
        return mapping

    def _ordered_labels(self) -> List[str]:
        return [self._group_name(group) for group in self.current_groups_]

    def _apply_current_grouping(self, df: DataFrame) -> DataFrame:
        if self.base_col_ is None:
            raise RuntimeError("Base grouping column is not initialized. Call fit() first.")
        mapping = self._current_group_map()
        map_expr = _create_map_expr(mapping)
        group_col = f"{self.feature}__grp"
        self.group_col_ = group_col
        return df.withColumn(group_col, map_expr[F.col(self.base_col_).cast("string")])

    def _prepare_for_apply(self, df: DataFrame) -> DataFrame:
        _validate_spark_df(df)
        if self.base_col_ is None:
            raise RuntimeError("Call fit() before transform().")

        if self._is_unordered_discrete():
            prepared = df.withColumn(
                self.base_col_,
                F.when(F.col(self.feature).isNull(), F.lit(self.missing_label))
                .otherwise(F.col(self.feature).cast("string")),
            )
            known = set(self.base_labels_)
            fallback = self.missing_label if self.missing_label in known else next(iter(known))
            prepared = prepared.withColumn(
                self.base_col_,
                F.when(F.col(self.base_col_).isin(list(known)), F.col(self.base_col_)).otherwise(F.lit(fallback)),
            )
            return prepared

        if self.feature_type == "continuous":
            numeric_col = f"{self.feature}__num_for_bin"
            with_num = df.withColumn(numeric_col, _numeric_from_any(self.feature))
            return _bin_numeric_feature(with_num, numeric_col, self.splits_, self.base_col_, self.missing_label).drop(numeric_col)

        raise ValueError("feature_type must be 'unordered_discrete' or 'continuous'.")

    # -----------------------------
    # build table
    # -----------------------------
    def _build_current_table(self, prepared_df: DataFrame) -> pd.DataFrame:
        grouped_df = self._apply_current_grouping(prepared_df)

        if self._is_continuous_like():
            table = build_woe_iv_table(
                grouped_df,
                self.group_col_,
                self.target_col,
                good_value=self.good_value,
                ordered_labels=self._ordered_labels(),
                sort_by="feature",
            )
        else:
            table = build_woe_iv_table(
                grouped_df,
                self.group_col_,
                self.target_col,
                good_value=self.good_value,
                ordered_labels=None,
                sort_by="woe",
            )
            table["__order__"] = range(len(table))
        return table

    # -----------------------------
    # candidate identification
    # -----------------------------
    
    def _bin_priority_tuple(self, row: pd.Series) -> Tuple[float, float, float]:
        """
        Smaller tuple = higher merge priority.
        Priority:
        - smaller n_obs first
        - smaller min(prop_n_good, prop_n_bad) first
        - more extreme |WoE| first
        """
        n_obs = float(row["n_obs"])
        prop_n_good = float(row["prop_n_good"])
        prop_n_bad = float(row["prop_n_bad"])
        abs_woe = abs(float(row["WoE"]))

        minority_prop = min(prop_n_good, prop_n_bad)

        return (n_obs, minority_prop, -abs_woe)


    def _choose_priority_merge(self, table: pd.DataFrame) -> Optional[Tuple[str, str]]:
        """
        Choose exactly one small bin to merge in this iteration:
        - first select the highest-priority small bin
        - then merge it with the adjacent neighbor having the smallest WoE difference
        """
        if len(table) <= 1:
            return None


        work = table.copy()
        work["bin"] = work["bin"].astype(str)

        if work.empty or len(work) <= 1:
            return None

        candidate_rows: List[Tuple[int, pd.Series]] = []

        for i, row in work.iterrows():
            is_small = (
                float(row["prop_n_obs"]) < self.min_bin_frac
                or float(row["prop_n_good"]) < self.min_prop_good
                or float(row["prop_n_bad"]) < self.min_prop_bad
            )
            if is_small:
                candidate_rows.append((i, row))

        if not candidate_rows:
            return None

        best_idx, best_row = min(candidate_rows, key=lambda x: self._bin_priority_tuple(x[1]))

        neighbors: List[Tuple[str, float]] = []

        if best_idx > 0:
            left_row = work.iloc[best_idx - 1]
            left_name = str(left_row["bin"])
            diff_left = abs(float(best_row["WoE"]) - float(left_row["WoE"]))
            neighbors.append((left_name, diff_left))

        if best_idx < len(work) - 1:
            right_row = work.iloc[best_idx + 1]
            right_name = str(right_row["bin"])
            diff_right = abs(float(best_row["WoE"]) - float(right_row["WoE"]))
            neighbors.append((right_name, diff_right))

        if not neighbors:
            return None

        best_neighbor, _ = min(neighbors, key=lambda x: x[1])
        return (str(best_row["bin"]), best_neighbor)
    
    def identify_candidates(self, table: pd.DataFrame) -> CandidateReport:
        small_bins: List[str] = []
        close_pairs: List[Tuple[str, str, float]] = []
        monotonic_break_pairs: List[Tuple[str, str]] = []


        for _, row in table.iterrows():
            bin_name = str(row["bin"])

            too_small = (
                float(row["prop_n_obs"]) < self.min_bin_frac
                or float(row["prop_n_good"]) < self.min_prop_good
                or float(row["prop_n_bad"]) < self.min_prop_bad
            )

            if too_small:
                small_bins.append(bin_name)

        for i in range(1, len(table)):
            left = str(table.iloc[i - 1]["bin"])
            right = str(table.iloc[i]["bin"])

            diff_woe = float(table.iloc[i]["diff_WoE"])
            diff_prop = float(table.iloc[i]["diff_prop_good"])

            if diff_woe < self.min_diff_woe or diff_prop < self.min_diff_prop_good:
                close_pairs.append((left, right, diff_woe))

        if self.enforce_monotonic and self._is_continuous_like() and len(table) >= 3:
            monotonic_table = table.copy()
            monotonic_table["bin"] = monotonic_table["bin"].astype(str)
            monotonic_table = monotonic_table.reset_index(drop=True)
            if len(monotonic_table) >= 3:
                woe = monotonic_table["WoE"].tolist()
                increasing = woe[-1] >= woe[0]

                for i in range(1, len(woe)):
                    if increasing and woe[i] < woe[i - 1]:
                        monotonic_break_pairs.append(
                            (str(monotonic_table.iloc[i - 1]["bin"]), str(monotonic_table.iloc[i]["bin"]))
                        )
                    if (not increasing) and woe[i] > woe[i - 1]:
                        monotonic_break_pairs.append(
                            (str(monotonic_table.iloc[i - 1]["bin"]), str(monotonic_table.iloc[i]["bin"]))
                        )

        suggested_pair: Optional[Tuple[str, str]] = None
        stop_reason = "No more merge candidates."

        if small_bins:
            suggested_pair = self._choose_priority_merge(table)
            stop_reason = "Small bin detected."
        elif close_pairs:
            left, right, _ = min(close_pairs, key=lambda x: x[2])
            suggested_pair = (left, right)
            stop_reason = "Adjacent bins are too similar."
        elif monotonic_break_pairs:
            suggested_pair = monotonic_break_pairs[0]
            stop_reason = "Monotonicity break detected."

        return CandidateReport(
            small_bins=small_bins,
            close_pairs=close_pairs,
            monotonic_break_pairs=monotonic_break_pairs,
            suggested_pair=suggested_pair,
            stop_reason=stop_reason,
        )

    def _choose_small_or_unstable_merge(self, table: pd.DataFrame, candidates: Sequence[str]) -> Optional[Tuple[str, str]]:
        if len(table) <= 1:
            return None


        name_to_idx = {str(table.iloc[i]["bin"]): i for i in range(len(table))}
        candidate = str(candidates[0])
        idx = name_to_idx.get(candidate)

        neighbors = []
        if idx > 0:
            left_bin = str(table.iloc[idx - 1]["bin"])
        if idx < len(table) - 1:
            right_bin = str(table.iloc[idx + 1]["bin"])

        if not neighbors:
            return None
        best_neighbor, _ = min(neighbors, key=lambda x: x[1])
        return (candidate, best_neighbor)

    # -----------------------------
    # merge
    # -----------------------------
    def _merge_groups_by_name(self, left_name: str, right_name: str) -> None:
        left_idx = None
        right_idx = None
        for i, group in enumerate(self.current_groups_):
            name = self._group_name(group)
            if name == left_name:
                left_idx = i
            if name == right_name:
                right_idx = i

        if left_idx is None or right_idx is None:
            raise ValueError(f"Cannot merge '{left_name}' and '{right_name}': one of them is missing.")
        if left_idx == right_idx:
            return

        if left_idx > right_idx:
            left_idx, right_idx = right_idx, left_idx

        merged_group = self.current_groups_[left_idx] + self.current_groups_[right_idx]
        self.current_groups_[left_idx] = merged_group
        self.current_groups_.pop(right_idx)

    # -----------------------------
    # fit / transform
    # -----------------------------
    def fit(self, train_df: DataFrame) -> "SingleFeatureWOEPipeline":
        prepared = self._prepare_base_groups(train_df)
        self.history_ = []

        for iteration in range(self.max_iter):
            table = self._build_current_table(prepared)
            candidates = self.identify_candidates(table)

            self.history_.append(
                {
                    "iter": iteration,
                    "table": table.copy(),
                    "small_bins": list(candidates.small_bins),
                    "close_pairs": list(candidates.close_pairs),
                    "monotonic_break_pairs": list(candidates.monotonic_break_pairs),
                    "suggested_pair": candidates.suggested_pair,
                    "stop_reason": candidates.stop_reason,
                }
            )

            # For unordered discrete variables, optionally merge exactly one rare category
            # into OTHER per iteration before general pairwise merge.

            if candidates.suggested_pair is None:
                self.final_table_ = table.copy()
                break

            left_name, right_name = candidates.suggested_pair
            self._merge_groups_by_name(left_name, right_name)
        else:
            self.final_table_ = self._build_current_table(prepared)

        if self.final_table_ is None:
            self.final_table_ = self._build_current_table(prepared)

        self.group_map_ = self._current_group_map()
        self.woe_map_ = {
            str(row["bin"]): float(row["WoE"])
            for _, row in self.final_table_[["bin", "WoE"]].iterrows()
        }
        self.fitted_ = True
        return self

    def report(self) -> pd.DataFrame:
        if not self.fitted_:
            raise RuntimeError("Call fit() first.")
        return self.final_table_.copy()

    def transform(
        self,
        df: DataFrame,
        *,
        output: str = "group",
        group_output_col: Optional[str] = None,
        woe_output_col: Optional[str] = None,
    ) -> DataFrame:
        if not self.fitted_:
            raise RuntimeError("Call fit() first.")

        prepared = self._prepare_for_apply(df)
        group_map_expr = _create_map_expr(self.group_map_)

        group_col = group_output_col or f"{self.feature}_grp"
        woe_col = woe_output_col or f"{self.feature}_woe"

        out = prepared.withColumn(group_col, group_map_expr[F.col(self.base_col_).cast("string")])

        if output == "group":
            return out
        if output == "woe":
            woe_expr = _create_map_expr({k: float(v) for k, v in self.woe_map_.items()})
            return out.withColumn(woe_col, woe_expr[F.col(group_col).cast("string")]).drop(group_col)
        if output == "both":
            woe_expr = _create_map_expr({k: float(v) for k, v in self.woe_map_.items()})
            return out.withColumn(woe_col, woe_expr[F.col(group_col).cast("string")])
        raise ValueError("output must be 'group', 'woe', or 'both'.")

    def fit_transform(
        self,
        train_df: DataFrame,
        *,
        output: str = "group",
        group_output_col: Optional[str] = None,
        woe_output_col: Optional[str] = None,
    ) -> DataFrame:
        return self.fit(train_df).transform(
            train_df,
            output=output,
            group_output_col=group_output_col,
            woe_output_col=woe_output_col,
        )

    def export_mapping(self) -> Dict[str, Any]:
        if not self.fitted_:
            raise RuntimeError("Call fit() first.")
        return {
            "feature": self.feature,
            "feature_type": self.feature_type,
            "splits_": self.splits_,
            "base_labels_": list(self.base_labels_),
            "group_map_": dict(self.group_map_),
            "woe_map_": dict(self.woe_map_),
            "ordered_groups_": [list(group) for group in self.current_groups_],
        }


# ============================================================
# Higher-level batch helper
# ============================================================

def fit_feature_pipelines(
    train_df: DataFrame,
    test_df: Optional[DataFrame],
    configs: Mapping[str, Mapping[str, Any]],
    *,
    target_col: str,
) -> Tuple[DataFrame, Optional[DataFrame], Dict[str, SingleFeatureWOEPipeline], Dict[str, pd.DataFrame]]:
    """
    Batch helper.

    Example
    -------
    configs = {
        "home_ownership": {
            "feature_type": "unordered_discrete",
            "merge_rare_to_other_first": True,
        },
        "dti": {
            "feature_type": "continuous",
            "n_initial_bins": 30,
            "enforce_monotonic": True,
        },
        "annual_inc": {
            "feature_type": "special_continuous",
            "n_initial_bins": 40,
            "enforce_monotonic": True,
        },
    }
    """
    _validate_spark_df(train_df, "train_df")
    if test_df is not None:
        _validate_spark_df(test_df, "test_df")

    train_out = train_df
    test_out = test_df
    pipelines: Dict[str, SingleFeatureWOEPipeline] = {}
    reports: Dict[str, pd.DataFrame] = {}

    for feature, cfg in configs.items():
        pipe = SingleFeatureWOEPipeline(feature=feature, target_col=target_col, **cfg)
        pipe.fit(train_out)
        train_out = pipe.transform(
            train_out,
            output="both",
            group_output_col=f"{feature}_grp",
            woe_output_col=f"{feature}_woe",
        )
        if test_out is not None:
            test_out = pipe.transform(
                test_out,
                output="both",
                group_output_col=f"{feature}_grp",
                woe_output_col=f"{feature}_woe",
            )
        pipelines[feature] = pipe
        reports[feature] = pipe.report()

    return train_out, test_out, pipelines, reports


def get_pd_coarse_classing_specs() -> list[dict[str, Any]]:
    """Default coarse-classing spec for PD features requested in the project."""
    return [
        # 2) Discrete variables
        {"name": "grade", "candidates": ["grade"], "feature_type": "unordered_discrete"},
        {"name": "home_ownership", "candidates": ["home_ownership"], "feature_type": "unordered_discrete"},
        {"name": "addr_state", "candidates": ["addr_state"], "feature_type": "unordered_discrete"},
        {"name": "verification_status", "candidates": ["verification_status"], "feature_type": "unordered_discrete"},
        {"name": "purpose", "candidates": ["purpose"], "feature_type": "unordered_discrete"},
        {"name": "initial_list_status", "candidates": ["initial_list_status"], "feature_type": "unordered_discrete"},
        # 3) Continuous variables
        {"name": "term_int", "candidates": ["term_int"], "feature_type": "continuous", "n_initial_bins": 8, "enforce_monotonic": False},
        {"name": "emp_length_int", "candidates": ["emp_length_int"], "feature_type": "continuous", "n_initial_bins": 11, "enforce_monotonic": False},
        {"name": "delinq_2yrs", "candidates": ["delinq_2yrs"], "feature_type": "continuous", "n_initial_bins": 10},
        {"name": "inq_last_6mths", "candidates": ["inq_last_6mths"], "feature_type": "continuous", "n_initial_bins": 10},
        {"name": "open_acc", "candidates": ["open_acc"], "feature_type": "continuous", "n_initial_bins": 20},
        {"name": "pub_rec", "candidates": ["pub_rec"], "feature_type": "continuous", "n_initial_bins": 10},
        {"name": "acc_now_delinq", "candidates": ["acc_now_delinq"], "feature_type": "continuous", "n_initial_bins": 6},
        {"name": "mths_since_issue_d", "candidates": ["mths_since_issue_d"], "feature_type": "continuous", "n_initial_bins": 30, "enforce_monotonic": True},
        {"name": "int_rate", "candidates": ["int_rate"], "feature_type": "continuous", "n_initial_bins": 30, "enforce_monotonic": True},
        {"name": "funded_amnt", "candidates": ["funded_amnt"], "feature_type": "continuous", "n_initial_bins": 30},
        {"name": "mths_since_earliest_cr_line", "candidates": ["mths_since_earliest_cr_line"], "feature_type": "continuous", "n_initial_bins": 30, "enforce_monotonic": True},
        {"name": "total_acc", "candidates": ["total_acc"], "feature_type": "continuous", "n_initial_bins": 25},
        {"name": "total_rev_hi_lim", "candidates": ["total_rev_hi_lim"], "feature_type": "continuous", "n_initial_bins": 30},
        {"name": "installment", "candidates": ["installment"], "feature_type": "continuous", "n_initial_bins": 30},
        # 4) Continuous variables with criteria
        {"name": "annual_inc", "candidates": ["annual_inc"], "feature_type": "continuous", "n_initial_bins": 40, "enforce_monotonic": True},
        {"name": "dti", "candidates": ["dti", "dti_factor"], "feature_type": "continuous", "n_initial_bins": 40, "enforce_monotonic": True},
        {"name": "mths_since_last_delinq", "candidates": ["mths_since_last_delinq"], "feature_type": "continuous", "n_initial_bins": 30},
        {"name": "mths_since_last_record", "candidates": ["mths_since_last_record"], "feature_type": "continuous", "n_initial_bins": 30},
    ]


def _resolve_feature_column(df: DataFrame, candidates: list[str]) -> str | None:
    """Resolve the first existing feature column from candidates."""
    for col_name in candidates:
        if col_name in df.columns:
            return col_name
    return None


def build_pd_coarse_configs(df: DataFrame, target_col: str = "good_bad") -> tuple[dict[str, dict[str, Any]], dict[str, str], list[str]]:
    """
    Build pipeline configs for available PD coarse-classing variables.

    Returns:
        configs: config dict keyed by resolved feature column
        name_mapping: business variable name -> resolved feature column
        missing: business variable names not found in df
    """
    specs = get_pd_coarse_classing_specs()
    configs: dict[str, dict[str, Any]] = {}
    name_mapping: dict[str, str] = {}
    missing: list[str] = []

    for spec in specs:
        resolved = _resolve_feature_column(df, spec["candidates"])
        if resolved is None:
            missing.append(spec["name"])
            continue
        cfg = {k: v for k, v in spec.items() if k not in {"name", "candidates"}}
        configs[resolved] = cfg
        name_mapping[spec["name"]] = resolved

    return configs, name_mapping, missing


def fit_pd_coarse_classing(
    train_df: DataFrame,
    test_df: DataFrame | None = None,
    *,
    target_col: str = "good_bad",
) -> tuple[DataFrame, DataFrame | None, dict[str, SingleFeatureWOEPipeline], dict[str, pd.DataFrame], dict[str, str], list[str]]:
    """
    Fit and apply coarse-classing pipelines for the PD variable set.

    Returns:
        train_out, test_out, pipelines, reports, name_mapping, missing_variables
    """
    configs, name_mapping, missing = build_pd_coarse_configs(train_df, target_col=target_col)
    train_out, test_out, pipelines, reports = fit_feature_pipelines(
        train_df,
        test_df,
        configs=configs,
        target_col=target_col,
    )
    return train_out, test_out, pipelines, reports, name_mapping, missing

DF_FEATURES = [
    # Discrete variables
    "grade",
    "home_ownership",
    "addr_state",
    "verification_status",
    "purpose",
    "initial_list_status",

    # Continuous / numeric variables
    "term_int",
    "emp_length_int",
    "delinq_2yrs",
    "inq_last_6mths",
    "open_acc",
    "pub_rec",
    "acc_now_delinq",
    "mths_since_issue_d",
    "int_rate",
    "funded_amnt",
    "mths_since_earliest_cr_line",
    "total_acc",
    "total_rev_hi_lim",
    "installment",
    "annual_inc",
    "dti",
    "mths_since_last_delinq",
    "mths_since_last_record",
]

__all__ = [
    "build_woe_iv_table",
    "plot_by_woe",
    "SingleFeatureWOEPipeline",
    "fit_feature_pipelines",
    "get_pd_coarse_classing_specs",
    "build_pd_coarse_configs",
    "fit_pd_coarse_classing",
]
