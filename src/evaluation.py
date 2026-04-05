"""Model evaluation metrics and visualization functions."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve


def compute_auc_roc(y_true, y_pred_proba):
    """Compute AUC-ROC score."""
    return roc_auc_score(y_true, y_pred_proba)


def compute_gini(auc):
    """Compute Gini coefficient: Gini = 2 * AUC - 1."""
    return 2 * auc - 1


def compute_ks_statistic(y_true, y_pred_proba):
    """
    Compute KS statistic: max separation between cumulative
    default and non-default distributions.

    Returns:
        (ks_statistic, threshold_at_max)
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    ks = np.max(tpr - fpr)
    idx = np.argmax(tpr - fpr)
    return ks, thresholds[idx]


def compute_lift_table(y_true, y_pred_proba, n_deciles=10):
    """
    Compute decile-based lift table.

    Returns:
        DataFrame with decile, count, default_rate, cumulative_default_rate, lift
    """
    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred_proba})
    df["decile"] = pd.qcut(df["y_pred"], q=n_deciles, labels=False, duplicates="drop")
    df["decile"] = n_deciles - df["decile"]  # decile 1 = highest risk

    lift = df.groupby("decile").agg(
        count=("y_true", "count"),
        defaults=("y_true", "sum"),
    ).reset_index()

    lift["default_rate"] = lift["defaults"] / lift["count"]
    overall_rate = df["y_true"].mean()
    lift["lift"] = lift["default_rate"] / overall_rate
    lift["cum_defaults"] = lift["defaults"].cumsum()
    lift["cum_default_rate"] = lift["cum_defaults"] / lift["defaults"].sum()

    return lift


def plot_roc_curves(results_dict, save_path=None):
    """
    Plot overlaid ROC curves for baseline vs hybrid.

    Args:
        results_dict: {"Baseline": (y_true, y_pred), "Hybrid": (y_true, y_pred)}
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    for name, (y_true, y_pred) in results_dict.items():
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        auc = roc_auc_score(y_true, y_pred)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.4f})")

    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve Comparison")
    ax.legend()
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_ks_chart(y_true, y_pred_proba, title="KS Chart", save_path=None):
    """Plot KS separation chart."""
    fig, ax = plt.subplots(figsize=(8, 6))

    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    ks = np.max(tpr - fpr)

    ax.plot(thresholds[1:], tpr[1:], label="TPR (Defaults)")
    ax.plot(thresholds[1:], fpr[1:], label="FPR (Non-Defaults)")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Cumulative Rate")
    ax.set_title(f"{title} (KS={ks:.4f})")
    ax.legend()
    ax.invert_xaxis()
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_lift_chart(lift_table, save_path=None):
    """Plot lift by decile."""
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.bar(lift_table["decile"], lift_table["lift"], color="steelblue")
    ax.axhline(y=1, color="red", linestyle="--", label="Baseline (lift=1)")
    ax.set_xlabel("Decile (1=highest risk)")
    ax.set_ylabel("Lift")
    ax.set_title("Lift Chart")
    ax.legend()
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_feature_importance(coef_df, top_n=20, save_path=None):
    """Plot horizontal bar chart of top feature coefficients."""
    fig, ax = plt.subplots(figsize=(10, 8))

    top = coef_df.head(top_n).sort_values("abs_coefficient")
    ax.barh(top["feature"], top["coefficient"], color="steelblue")
    ax.set_xlabel("Coefficient")
    ax.set_title(f"Top {top_n} Feature Importance")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def summary_comparison_table(baseline_metrics, hybrid_metrics):
    """
    Create a side-by-side comparison table.

    Args:
        baseline_metrics: dict with keys "AUC", "Gini", "KS"
        hybrid_metrics: dict with same keys

    Returns:
        pandas DataFrame
    """
    df = pd.DataFrame({
        "Metric": list(baseline_metrics.keys()),
        "Baseline": list(baseline_metrics.values()),
        "Hybrid": list(hybrid_metrics.values()),
    })
    df["Improvement"] = df["Hybrid"] - df["Baseline"]
    return df
