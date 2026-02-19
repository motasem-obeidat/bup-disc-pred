import pandas as pd
import numpy as np
import random
import os
import matplotlib
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted


matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    classification_report,
    precision_score,
    recall_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.calibration import calibration_curve, CalibratedClassifierCV

warnings.filterwarnings("ignore")

# -------------------------------------------------------------------
# Global Settings & Seed
# -------------------------------------------------------------------
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

RESULTS = []

# -------------------------------------------------------------------
# GPU Detection
# -------------------------------------------------------------------
try:
    import torch

    GPU_AVAILABLE = torch.cuda.is_available()
    if GPU_AVAILABLE:
        print("\n" + "=" * 60)
        print("GPU DETECTED (in model_utils) - Enabling GPU acceleration.")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("No GPU detected - Using CPU.")
        print("=" * 60)
except ImportError:
    GPU_AVAILABLE = False
    print("PyTorch not found - Assuming no GPU available.")

# -------------------------------------------------------------------
# Paths and Constants
# -------------------------------------------------------------------
TARGET_COL = "BUP_STATUS_NUM"
BASE_DIR = "/dataset"
OUTPUT_DIR = "/BUPResults"

DATASETS = {
    "MULTI": {
        "train": os.path.join(BASE_DIR, "train_multi.csv"),
        "val": os.path.join(BASE_DIR, "val_multi.csv"),
        "test": os.path.join(BASE_DIR, "test_multi.csv"),
    },
    "BINARY": {
        "train": os.path.join(BASE_DIR, "train_binary.csv"),
        "val": os.path.join(BASE_DIR, "val_binary.csv"),
        "test": os.path.join(BASE_DIR, "test_binary.csv"),
    },
}

RANKING_FILES = {
    ("MULTI", "NO_PDC"): os.path.join(
        BASE_DIR, "multi-class_without_pdc_shap_ranking.csv"
    ),
    ("MULTI", "WITH_PDC"): os.path.join(
        BASE_DIR, "multi-class_with_pdc_shap_ranking.csv"
    ),
    ("BINARY", "NO_PDC"): os.path.join(BASE_DIR, "binary_without_pdc_shap_ranking.csv"),
    ("BINARY", "WITH_PDC"): os.path.join(BASE_DIR, "binary_with_pdc_shap_ranking.csv"),
}

TOP_K_RANKED_FEATURES = 20
PDC_COLS = ["PDC_30", "PDC30_CAT"]

ALL_NUMERIC_COLS = [
    "CHARLSON_INDEX",
    "OUD_OFFSET_DAYS",
    "INITIAL_DAYSUPP",
    "PDC_30",
    "INPATIENT_VISIT_COUNT",
    "OUTPATIENT_VISIT_COUNT",
    "FACILITY_VISIT_COUNT",
]

LEGEND_SORT_ORDER = [
    "LogisticRegression",
    "DecisionTree",
    "RandomForest",
    "AdaBoost",
    "XGBoost",
    "LightGBM",
    "CatBoost",
    "TabICL",
    "TabPFN",
    "TabM",
]


# -------------------------------------------------------------------
# Shared Helper Functions
# -------------------------------------------------------------------
def roc_auc_class_k(estimator, X, y_true, class_label, return_curves=False):
    """
    Calculates ROC AUC for a specific class k vs rest.
    """
    if not hasattr(estimator, "predict_proba"):
        return np.nan if not return_curves else {}

    y_proba = estimator.predict_proba(X)
    classes = estimator.classes_

    if class_label not in classes:
        return np.nan if not return_curves else {}

    pos_idx = list(classes).index(class_label)
    y_bin = (y_true == class_label).astype(int)

    y_score = y_proba[:, pos_idx]

    if len(np.unique(y_bin)) < 2:
        return np.nan if not return_curves else {}

    try:
        score = roc_auc_score(y_bin, y_score)
    except ValueError:
        return np.nan if not return_curves else {}

    if not return_curves:
        return score

    fpr, tpr, _ = roc_curve(y_bin, y_score)
    return {"score": score, "fpr": fpr, "tpr": tpr}


def pr_auc_class_k(estimator, X, y_true, class_label, return_curves=False):
    """
    Calculates PR AUC for a specific class k vs rest.
    """
    if not hasattr(estimator, "predict_proba"):
        return np.nan if not return_curves else {}

    y_proba = estimator.predict_proba(X)
    classes = estimator.classes_

    if class_label not in classes:
        return np.nan if not return_curves else {}

    pos_idx = list(classes).index(class_label)
    y_bin = (y_true == class_label).astype(int)

    y_score = y_proba[:, pos_idx]

    if len(np.unique(y_bin)) < 2:
        return np.nan if not return_curves else {}

    try:
        score = average_precision_score(y_bin, y_score)
    except ValueError:
        return np.nan if not return_curves else {}

    if not return_curves:
        return score

    precision, recall, _ = precision_recall_curve(y_bin, y_score)
    return {"score": score, "precision": precision, "recall": recall}


def roc_auc_positive_class_1(estimator, X, y_true, return_curves=False):
    return roc_auc_class_k(
        estimator, X, y_true, class_label=1, return_curves=return_curves
    )


def pr_auc_positive_class_1(estimator, X, y_true, return_curves=False):
    return pr_auc_class_k(
        estimator, X, y_true, class_label=1, return_curves=return_curves
    )


def evaluate_model(name, model, X, y, return_dict=True):
    """
    Evaluates a model and returns a dictionary of metrics.
    """
    y_pred = model.predict(X)

    print(f"\n=== {name} ===")

    acc = accuracy_score(y, y_pred)
    print(f"Accuracy: {acc:.4f}")

    micro_precision = precision_score(y, y_pred, average="micro", zero_division=0)
    micro_recall = recall_score(y, y_pred, average="micro", zero_division=0)
    micro_f1 = f1_score(y, y_pred, average="micro")

    print(
        f"Micro -> Precision: {micro_precision:.4f}, "
        f"Recall: {micro_recall:.4f}, "
        f"F1: {micro_f1:.4f}"
    )

    macro_precision = precision_score(y, y_pred, average="macro", zero_division=0)
    macro_recall = recall_score(y, y_pred, average="macro", zero_division=0)
    macro_f1 = f1_score(y, y_pred, average="macro")

    print(
        f"Macro -> Precision: {macro_precision:.4f}, "
        f"Recall: {macro_recall:.4f}, "
        f"F1: {macro_f1:.4f}"
    )

    weighted_precision = precision_score(y, y_pred, average="weighted", zero_division=0)
    weighted_recall = recall_score(y, y_pred, average="weighted", zero_division=0)
    weighted_f1 = f1_score(y, y_pred, average="weighted")

    print(
        f"Weighted -> Precision: {weighted_precision:.4f}, "
        f"Recall: {weighted_recall:.4f}, "
        f"F1: {weighted_f1:.4f}"
    )

    metrics_dict = {
        "accuracy": acc,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1,
    }

    interesting_classes = [0, 1, 2]

    model_classes = getattr(model, "classes_", np.unique(y))

    for k in interesting_classes:
        if k in model_classes:
            f1_k = f1_score(y, y_pred, labels=[k], average=None, zero_division=0)[0]
            prec_k = precision_score(
                y, y_pred, labels=[k], average=None, zero_division=0
            )[0]
            rec_k = recall_score(y, y_pred, labels=[k], average=None, zero_division=0)[
                0
            ]

            print(
                f"Class {k} -> Precision: {prec_k:.4f}, Recall: {rec_k:.4f}, F1: {f1_k:.4f}"
            )

            metrics_dict[f"precision_class{k}"] = prec_k
            metrics_dict[f"recall_class{k}"] = rec_k
            metrics_dict[f"f1_class{k}"] = f1_k

            roc_val = np.nan
            pr_val = np.nan

            if hasattr(model, "predict_proba"):
                try:
                    roc_val = roc_auc_class_k(model, X, y, class_label=k)
                    pr_val = pr_auc_class_k(model, X, y, class_label=k)

                    if not np.isnan(roc_val):
                        print(f"ROC AUC (class {k} vs rest): {roc_val:.4f}")
                    if not np.isnan(pr_val):
                        print(f"PR AUC  (class {k} vs rest): {pr_val:.4f}")

                except Exception as e:
                    print(f"AUC calculation skipped for class {k}:", e)

            metrics_dict[f"roc_auc_class{k}"] = roc_val
            metrics_dict[f"pr_auc_class{k}"] = pr_val

    print("\nClassification report:")
    print(classification_report(y, y_pred, digits=4))

    if return_dict:
        return metrics_dict


def plot_performance_curves(
    curve_data_dict,
    title_prefix,
    save_dir=os.path.join(OUTPUT_DIR, "curve_plots"),
    baseline_prevalence=None,
):
    """
    Plots ROC and PR curves for multiple models.
    """

    os.makedirs(save_dir, exist_ok=True)

    def sort_key(item):
        k = item[0]
        if k in LEGEND_SORT_ORDER:
            return LEGEND_SORT_ORDER.index(k)
        return len(LEGEND_SORT_ORDER)

    sorted_items = sorted(curve_data_dict.items(), key=sort_key)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # =======================================================
    # PLOT 1: ROC Curve (Left Panel)
    # X-Axis: 1-Specificity (FPR) -- False Positive Rate
    # Y-Axis: Sensitivity (TPR) -- True Positive Rate
    # =======================================================
    ax_roc = axes[0]

    ax_roc.plot([0, 1], [0, 1], "k--", label="Chance", alpha=0.5)

    for name, metrics in sorted_items:
        if "fpr" in metrics:
            label_str = f"{name}, C-statistics={metrics['roc_score']:.2f}"
            ax_roc.plot(metrics["fpr"], metrics["tpr"], label=label_str, linewidth=2)

    ax_roc.set_title(f"ROC: {title_prefix}", fontsize=14)
    ax_roc.set_xlabel("1-Specificity", fontsize=12)
    ax_roc.set_ylabel("Sensitivity", fontsize=12)

    ax_roc.legend(loc="lower right", fontsize=10)
    ax_roc.grid(True, alpha=0.3)
    ax_roc.set_xlim([-0.02, 1.02])
    ax_roc.set_ylim([-0.02, 1.02])

    # =======================================================
    # PLOT 2: Precision-Recall Curve (Right Panel)
    # X-Axis: Recall
    # Y-Axis: Precision
    # =======================================================
    ax_pr = axes[1]

    if baseline_prevalence is not None:
        ax_pr.plot(
            [0, 1],
            [baseline_prevalence, baseline_prevalence],
            "k--",
            label=f"Chance ({baseline_prevalence:.2f})",
            alpha=0.6,
        )

    for name, metrics in sorted_items:
        if "precision" in metrics:
            label_str = f"{name} (AUC={metrics['pr_score']:.2f})"
            ax_pr.plot(
                metrics["recall"], metrics["precision"], label=label_str, linewidth=2
            )

    ax_pr.set_title(f"Precision-Recall: {title_prefix}", fontsize=14)
    ax_pr.set_xlabel("Recall", fontsize=12)
    ax_pr.set_ylabel("Precision", fontsize=12)

    ax_pr.legend(loc="upper right", fontsize=10)
    ax_pr.grid(True, alpha=0.3)
    ax_pr.set_xlim([-0.02, 1.02])
    ax_pr.set_ylim([-0.02, 1.02])

    plt.tight_layout()

    clean_name = title_prefix.replace(" ", "_").replace(",", "")
    save_path = os.path.join(save_dir, f"PAPER_STYLE_{clean_name}.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\nSaved paper-style plots to: {save_path}")


def make_calibration_plot(
    model,
    X_test,
    y_test,
    X_val,
    y_val,
    model_name,
    dataset,
    scenario,
    feature_set,
    save_dir=os.path.join(OUTPUT_DIR, "calibration_plots"),
):
    """
    Generates calibration plots (reliability diagrams) for a model.
    """

    os.makedirs(save_dir, exist_ok=True)

    # --- 1. Get Raw Probabilities (No Calibration) ---
    if hasattr(model, "predict_proba"):
        if 1 not in model.classes_:
            print(f"Skipping calibration for {model_name}: Class 1 not in base model.")
            return
        pos_idx_raw = list(model.classes_).index(1)
        prob_raw = model.predict_proba(X_test)[:, pos_idx_raw]
    else:
        print(f"Skipping calibration for {model_name}: No predict_proba.")
        return

    # --- 2. Get Isotonic Calibrated Probabilities ---
    prob_iso = None
    try:
        estimator_to_calibrate = model
        if model.__class__.__name__ == "ThresholdWrapper":
            estimator_to_calibrate = model.estimator

        iso_calibrator = CalibratedClassifierCV(
            estimator_to_calibrate, cv="prefit", method="isotonic"
        )
        iso_calibrator.fit(X_val, y_val)

        if 1 not in iso_calibrator.classes_:
            print(f"Class 1 not in calibrated classes for {model_name}")
        else:
            pos_idx_iso = list(iso_calibrator.classes_).index(1)
            prob_iso = iso_calibrator.predict_proba(X_test)[:, pos_idx_iso]

    except Exception as e:
        print(f"Isotonic calibration failed for {model_name}: {e}")

    # --- 3. Prepare Data for Plotting ---
    y_true = (y_test == 1).astype(int)

    prob_true_raw, prob_pred_raw = calibration_curve(
        y_true, prob_raw, n_bins=10, strategy="uniform"
    )

    if prob_iso is not None:
        prob_true_iso, prob_pred_iso = calibration_curve(
            y_true, prob_iso, n_bins=10, strategy="uniform"
        )

    # --- 4. Plot ---
    plt.figure(figsize=(8, 8))

    # Perfect Calibration (Dashed Black)
    plt.plot(
        [0, 1], [0, 1], linestyle="--", color="black", label="Perfectly Calibrated"
    )

    # No Calibration (crimson)
    plt.plot(
        prob_pred_raw,
        prob_true_raw,
        marker="x",
        linewidth=2,
        color="crimson",
        label=f"No Calibration ({model_name})",
    )

    # Isotonic Calibration (Green)
    if prob_iso is not None:
        plt.plot(
            prob_pred_iso,
            prob_true_iso,
            marker="x",
            linewidth=2,
            color="green",
            label=f"Isotonic Calibration ({model_name})",
        )

    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives (Observed)")
    plt.title(
        f"Calibration Curve: {model_name}\n({dataset} - {scenario} - {feature_set})"
    )
    plt.legend(loc="upper left")
    plt.grid(True, alpha=0.3)

    fname = f"Calib_{dataset}_{scenario}_{feature_set}_{model_name}.png".replace(
        " ", "_"
    )
    plt.savefig(os.path.join(save_dir, fname), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved calibration plot: {fname}")


def make_risk_stratification_plot(
    model,
    X_test,
    y_test,
    model_name,
    dataset,
    scenario,
    feature_set,
    save_dir=os.path.join(OUTPUT_DIR, "risk_plots"),
):
    """
    Generates risk stratification plots (decile analysis).
    """

    os.makedirs(save_dir, exist_ok=True)

    if hasattr(model, "predict_proba"):
        probas = model.predict_proba(X_test)
        classes = model.classes_
        if 1 not in classes:
            return
        pos_idx = list(classes).index(1)
        y_prob = probas[:, pos_idx]
    else:
        return

    df_risk = pd.DataFrame({"y_true": (y_test == 1).astype(int), "y_prob": y_prob})

    total_n = len(df_risk)

    try:
        df_risk["Decile"] = 10 - pd.qcut(
            df_risk["y_prob"], 10, labels=False, duplicates="drop"
        )
    except ValueError:
        print(
            f"Skipping risk stratification for {model_name}: Not enough unique probabilities for deciles."
        )
        return

    stats = (
        df_risk.groupby("Decile")
        .agg(
            Count=("y_true", "count"),
            Observed_Rate=("y_true", "mean"),
            Positive_Cases=("y_true", "sum"),
        )
        .sort_index(ascending=True)  # Decile 1 (High Risk) comes first
    )

    total_positives = stats["Positive_Cases"].sum()
    if total_positives == 0:
        print(
            f"Skipping risk stratification for {model_name}: No positive cases found."
        )
        return

    stats["Cumulative_Positives"] = stats["Positive_Cases"].cumsum()
    stats["Cumulative_Capture_Rate"] = (
        stats["Cumulative_Positives"] / total_positives
    ) * 100

    fig, ax1 = plt.subplots(figsize=(14, 7))

    x_labels = []
    for decile_idx, row in stats.iterrows():
        n = int(row["Count"])
        pct = (n / total_n) * 100
        label = f"Decile {decile_idx}\n(n={n}, {pct:.1f}%)"
        x_labels.append(label)

    x_pos = np.arange(len(x_labels))

    bars = ax1.bar(
        x_pos,
        stats["Observed_Rate"] * 100,
        color="firebrick",
        alpha=0.7,
        label="Actual Discontinuation Rate",
    )
    ax1.set_ylabel("Actual Discontinuation Rate (%)", color="firebrick", fontsize=12)
    ax1.tick_params(axis="y", labelcolor="firebrick")
    ax1.set_xlabel("Risk Subgroups (Highest Risk -> Lowest Risk)", fontsize=12)

    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(x_labels, rotation=0, fontsize=9)
    ax1.set_ylim(0, 100)

    for bar in bars:
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 1,
            f"{height:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    overall_prevalence = df_risk["y_true"].mean() * 100
    ax1.axhline(
        y=overall_prevalence,
        color="black",
        linestyle="--",
        linewidth=2,
        label=f"Avg Prevalence ({overall_prevalence:.1f}%)",
    )

    ax2 = ax1.twinx()
    ax2.plot(
        x_pos,
        stats["Cumulative_Capture_Rate"],
        color="green",
        marker="o",
        linewidth=2,
        label="Cumulative % Captured",
    )
    ax2.set_ylabel(
        "Cumulative % of Positive Cases Captured", color="green", fontsize=12
    )
    ax2.tick_params(axis="y", labelcolor="green")
    ax2.set_ylim(0, 110)

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="best")

    plt.title(
        f"Risk Stratification: {model_name} \n({dataset} - {scenario} - {feature_set})"
    )
    plt.tight_layout()

    fname = f"Risk_{dataset}_{scenario}_{feature_set}_{model_name}.png".replace(
        " ", "_"
    )
    plt.savefig(os.path.join(save_dir, fname), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved risk plot: {fname}")


class ThresholdWrapper(BaseEstimator, ClassifierMixin):
    """
    A wrapper that applies a custom threshold to class 1 predictions.
    """

    def __init__(self, estimator, threshold=0.5):
        self.estimator = estimator
        self.threshold = threshold
        self.classes_ = None

    def fit(self, X, y, **fit_params):
        self.estimator.fit(X, y, **fit_params)
        self.classes_ = self.estimator.classes_
        return self

    def predict(self, X):
        check_is_fitted(self.estimator)
        probas = self.estimator.predict_proba(X)

        class_to_idx = {c: i for i, c in enumerate(self.classes_)}

        if 1 not in class_to_idx:
            return self.estimator.predict(X)

        idx_1 = class_to_idx[1]

        p1 = probas[:, idx_1]

        mask_c1 = p1 >= self.threshold

        probas_no_c1 = probas.copy()
        probas_no_c1[:, idx_1] = -1.0

        alt_preds = self.estimator.classes_[np.argmax(probas_no_c1, axis=1)]

        final_preds = np.where(mask_c1, 1, alt_preds)

        return final_preds

    def predict_proba(self, X):
        check_is_fitted(self.estimator)
        return self.estimator.predict_proba(X)
