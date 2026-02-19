import os
import sys
import joblib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# -------------------------------------------------------------------
# Helper Functions
# -------------------------------------------------------------------
try:
    from model_utils import (
        DATASETS,
        evaluate_model,
        OUTPUT_DIR,
        RANKING_FILES,
        TOP_K_RANKED_FEATURES,
        PDC_COLS,
        TARGET_COL,
        RESULTS,
        ThresholdWrapper,
    )
    from training import append_to_results
    from preprocessing import (
        build_xgboost_preprocessor,
        build_lightgbm_preprocessor,
        build_catboost_preprocessor,
        prepare_feature_sets,
        get_ranked_features,
    )

    from xgboost_model import XGBClassifierAutoWeight
    from TabM import TabMClassifier, TabM

    try:
        from lightgbm import LGBMClassifier
    except ImportError:
        pass
    try:
        from catboost import CatBoostClassifier
    except ImportError:
        pass
    try:
        from tabicl import TabICLClassifier
    except ImportError:
        pass
    try:
        from tabpfn import TabPFNClassifier
    except ImportError:
        pass
    try:
        import umap
    except ImportError:
        pass

except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)


MODELS_DIR_BASE = os.path.join(OUTPUT_DIR, "saved_models")
MODELS_DIR_TABM = os.path.join(OUTPUT_DIR, "saved_models_tabm")
MODELS_DIR_TABICL = os.path.join(OUTPUT_DIR, "saved_models_tabicl")
MODELS_DIR_TABPFN = os.path.join(OUTPUT_DIR, "saved_models_tabpfn")

# Maps model name to its specific directory
MODEL_DIRS = {
    "XGBoost": MODELS_DIR_BASE,
    "LogisticRegression": MODELS_DIR_BASE,
    "DecisionTree": MODELS_DIR_BASE,
    "RandomForest": MODELS_DIR_BASE,
    "AdaBoost": MODELS_DIR_BASE,
    "LightGBM": MODELS_DIR_BASE,
    "CatBoost": MODELS_DIR_BASE,
    "TabICL": MODELS_DIR_TABICL,
    "TabPFN": MODELS_DIR_TABPFN,
    "TabM": MODELS_DIR_TABM,
}

MODEL_LIST = [
    "XGBoost",
    "LogisticRegression",
    "DecisionTree",
    "RandomForest",
    "AdaBoost",
    "LightGBM",
    "CatBoost",
    "TabICL",
    "TabPFN",
    "TabM",
]


def plot_tsne(X, df_results, output_path, title):
    """
    Computes t-SNE on the full X, but plots ONLY True Positives (TP)
    and False Positives (FP) to analyze classification errors.
    """
    try:
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        X_numeric = pd.get_dummies(X, drop_first=True)

        imputer = SimpleImputer(strategy="mean")
        X_imputed = imputer.fit_transform(X_numeric)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_imputed)

        tsne = TSNE(
            n_components=2, random_state=42, perplexity=min(30, len(X_scaled) - 1)
        )
        X_embedded = tsne.fit_transform(X_scaled)

        plot_df = pd.DataFrame(X_embedded, columns=["t-SNE 1", "t-SNE 2"])
        plot_df["Category"] = df_results["Category"].values

        plot_df = plot_df[plot_df["Category"].isin(["TP", "FP"])]

        category_map = {"TP": "True Positive (TP)", "FP": "False Positive (FP)"}
        plot_df["Category"] = plot_df["Category"].map(category_map)

        plt.figure(figsize=(10, 8))
        sns.scatterplot(
            data=plot_df,
            x="t-SNE 1",
            y="t-SNE 2",
            hue="Category",
            hue_order=["True Positive (TP)", "False Positive (FP)"],
            palette={"True Positive (TP)": "#2ca02c", "False Positive (FP)": "#d62728"},
            s=10,
            alpha=0.7,
        )
        plt.legend(title=None)
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        print(f"      Saved t-SNE plot: {os.path.basename(output_path)}")
    except Exception as e:
        print(f"      Error plotting t-SNE: {e}")


def plot_umap(X, df_results, output_path, title):
    """
    Computes UMAP on the full X, but plots ONLY True Positives (TP)
    and False Positives (FP) to analyze classification errors.
    """
    if "umap" not in sys.modules:
        return

    try:
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        X_numeric = pd.get_dummies(X, drop_first=True)

        imputer = SimpleImputer(strategy="mean")
        X_imputed = imputer.fit_transform(X_numeric)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_imputed)

        reducer = umap.UMAP(random_state=42)
        X_embedded = reducer.fit_transform(X_scaled)

        plot_df = pd.DataFrame(X_embedded, columns=["UMAP 1", "UMAP 2"])
        plot_df["Category"] = df_results["Category"].values

        plot_df = plot_df[plot_df["Category"].isin(["TP", "FP"])]

        category_map = {"TP": "True Positive (TP)", "FP": "False Positive (FP)"}
        plot_df["Category"] = plot_df["Category"].map(category_map)

        plt.figure(figsize=(10, 8))
        sns.scatterplot(
            data=plot_df,
            x="UMAP 1",
            y="UMAP 2",
            hue="Category",
            hue_order=["True Positive (TP)", "False Positive (FP)"],
            palette={"True Positive (TP)": "#2ca02c", "False Positive (FP)": "#d62728"},
            s=10,
            alpha=0.7,
        )
        plt.legend(title=None)
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        print(f"      Saved UMAP plot: {os.path.basename(output_path)}")
    except Exception as e:
        print(f"      Error plotting UMAP: {e}")


def load_and_evaluate():
    """
    Loads saved models, evaluates them, and generates detailed reports and plots.
    """
    print("=" * 80)
    print("REPRODUCING RESULTS FROM SAVED MODELS")
    print("=" * 80)

    for dataset_name, paths in DATASETS.items():
        print("\n" + "#" * 110)
        print("#" * 110)
        print(f"DATASET: {dataset_name}")
        print("#" * 110)
        print("#" * 110)

        try:
            df_train = pd.read_csv(paths["train"])
            df_val = pd.read_csv(paths["val"])
            df_test = pd.read_csv(paths["test"])
        except Exception as e:
            print(f"Error loading data for {dataset_name}: {e}")
            continue

        for include_pdc in [False, True]:
            scenario_name = "WITH_PDC" if include_pdc else "NO_PDC"
            print("\n" + "#" * 90)
            print(f"SCENARIO START: {dataset_name} | {scenario_name}")
            print("#" * 90 + "\n")

            X_train_base, X_val_base, X_test_base, y_train, y_val, y_test = (
                prepare_feature_sets(
                    df_train, df_val, df_test, TARGET_COL, PDC_COLS, include_pdc
                )
            )

            xgboost_preprocessor, _ = build_xgboost_preprocessor(X_train_base)
            xgboost_preprocessor.fit(X_train_base)

            lightgbm_preprocessor, _ = build_lightgbm_preprocessor(X_train_base)
            lightgbm_preprocessor.fit(X_train_base)

            catboost_preprocessor, _ = build_catboost_preprocessor(X_train_base)
            catboost_preprocessor.fit(X_train_base)

            feature_sets = ["BASELINE_FEATURES"]

            ranking_key = (dataset_name, scenario_name)
            ranking_path = RANKING_FILES.get(ranking_key, None)
            top_features_in_X = []

            if ranking_path and os.path.exists(ranking_path):
                top_features_in_X = get_ranked_features(
                    ranking_path, X_train_base, TOP_K_RANKED_FEATURES
                )
                if top_features_in_X:
                    feature_sets.append(f"RANKED_TOP_{TOP_K_RANKED_FEATURES}")

            for feature_set_name in feature_sets:
                print("\n" + "#" * 70)
                print(
                    f"SETTING START: {dataset_name} | {scenario_name} | {feature_set_name}"
                )
                print("#" * 70)

                if "RANKED" in feature_set_name:
                    X_test_curr = X_test_base[top_features_in_X].copy()
                    X_val_curr = X_val_base[top_features_in_X].copy()

                    X_train_curr = X_train_base[top_features_in_X].copy()

                    xgb_prep_curr, _ = build_xgboost_preprocessor(X_train_curr)
                    xgb_prep_curr.fit(X_train_curr)

                    lgbm_prep_curr, _ = build_lightgbm_preprocessor(X_train_curr)
                    lgbm_prep_curr.fit(X_train_curr)

                    cat_prep_curr, _ = build_catboost_preprocessor(X_train_curr)
                    cat_prep_curr.fit(X_train_curr)

                else:
                    X_test_curr = X_test_base.copy()
                    X_val_curr = X_val_base.copy()
                    X_train_curr = X_train_base.copy()
                    xgb_prep_curr = xgboost_preprocessor
                    lgbm_prep_curr = lightgbm_preprocessor
                    cat_prep_curr = catboost_preprocessor

                for model_name in MODEL_LIST:
                    fname = f"{dataset_name}_{scenario_name}_{feature_set_name}_{model_name}.joblib".replace(
                        " ", "_"
                    )
                    model_dir = MODEL_DIRS.get(model_name, MODELS_DIR_BASE)
                    model_path = os.path.join(model_dir, fname)

                    if not os.path.exists(model_path):
                        print(f"  [Skipping] Model file not found: {fname}")
                        continue

                    try:
                        model = joblib.load(model_path)

                        X_train_final = X_train_curr
                        X_val_final = X_val_curr
                        X_test_final = X_test_curr

                        if model_name == "XGBoost":
                            X_train_final = xgb_prep_curr.transform(X_train_curr)
                            X_val_final = xgb_prep_curr.transform(X_val_curr)
                            X_test_final = xgb_prep_curr.transform(X_test_curr)
                        elif model_name == "LightGBM":
                            X_train_final = lgbm_prep_curr.transform(X_train_curr)
                            X_val_final = lgbm_prep_curr.transform(X_val_curr)
                            X_test_final = lgbm_prep_curr.transform(X_test_curr)
                        elif model_name == "CatBoost":
                            X_train_final = cat_prep_curr.transform(X_train_curr)
                            X_val_final = cat_prep_curr.transform(X_val_curr)
                            X_test_final = cat_prep_curr.transform(X_test_curr)

                        threshold_val = 0.5
                        if isinstance(model, ThresholdWrapper):
                            threshold_val = model.threshold
                            print(
                                f"  -> Restored ThresholdWrapper with threshold={threshold_val:.4f}"
                            )
                        elif hasattr(model, "threshold"):
                            threshold_val = model.threshold
                            print(
                                f"  -> Found 'threshold' attribute: {threshold_val:.4f}"
                            )
                        else:
                            print("  -> No custom threshold found, using default 0.5")

                        train_m = evaluate_model(
                            f"{model_name} TRAIN ({dataset_name}, {scenario_name}, {feature_set_name})",
                            model,
                            X_train_final,
                            y_train,
                        )

                        val_m = evaluate_model(
                            f"{model_name} VAL ({dataset_name}, {scenario_name}, {feature_set_name})",
                            model,
                            X_val_final,
                            y_val,
                        )

                        test_m = evaluate_model(
                            f"{model_name} TEST ({dataset_name}, {scenario_name}, {feature_set_name})",
                            model,
                            X_test_final,
                            y_test,
                        )

                        append_to_results(
                            dataset_name,
                            scenario_name,
                            feature_set_name,
                            model_name,
                            train_m,
                            val_m,
                            test_m,
                            threshold=threshold_val,
                        )

                        DETAILED_RESULTS_DIR = os.path.join(
                            OUTPUT_DIR, "detailed_classification"
                        )
                        os.makedirs(DETAILED_RESULTS_DIR, exist_ok=True)

                        y_pred = model.predict(X_test_final)

                        if hasattr(model, "predict_proba"):
                            all_probs = model.predict_proba(X_test_final)
                            classes = getattr(model, "classes_", np.unique(y_test))

                            if 1 in classes:
                                pos_idx = list(classes).index(1)
                                y_prob = all_probs[:, pos_idx]
                            else:
                                y_prob = np.zeros(len(y_pred))

                            class_prob_cols = {}
                            for i, c in enumerate(classes):
                                class_prob_cols[f"Prob_Class_{c}"] = all_probs[:, i]

                        else:
                            y_prob = y_pred.astype(float)
                            class_prob_cols = {}

                        results_cv_df = X_test_base.copy()

                        results_cv_df["Data_Index"] = results_cv_df.index

                        if "RANKED" in feature_set_name:
                            pass

                        results_cv_df["Actual_Label"] = y_test.values
                        results_cv_df["Predicted_Label"] = y_pred
                        results_cv_df["Predicted_Prob"] = y_prob
                        results_cv_df["Threshold"] = threshold_val

                        for col_name, col_vals in class_prob_cols.items():
                            results_cv_df[col_name] = col_vals

                        base_cols = [
                            "Data_Index",
                            "Actual_Label",
                            "Predicted_Label",
                            "Predicted_Prob",
                            "Category",
                            "Threshold",
                        ]

                        prob_cols_sorted = sorted(class_prob_cols.keys())

                        existing_cols = [
                            c
                            for c in results_cv_df.columns
                            if c not in base_cols and c not in prob_cols_sorted
                        ]

                        actual_is_pos = results_cv_df["Actual_Label"] == 1
                        pred_is_pos = results_cv_df["Predicted_Label"] == 1

                        conditions = [
                            (actual_is_pos & pred_is_pos),  # TP
                            (
                                ~actual_is_pos & ~pred_is_pos
                            ),  # TN (Class 0 or 2, correctly ID'd as Not-1)
                            (~actual_is_pos & pred_is_pos),  # FP
                            (actual_is_pos & ~pred_is_pos),  # FN
                        ]
                        choices = ["TP", "TN", "FP", "FN"]
                        results_cv_df["Category"] = np.select(
                            conditions, choices, default="UNKNOWN"
                        )

                        final_cols = (
                            [c for c in base_cols if c in results_cv_df.columns]
                            + prob_cols_sorted
                            + existing_cols
                        )
                        results_cv_df = results_cv_df[final_cols]

                        df_fp = results_cv_df[
                            results_cv_df["Category"] == "FP"
                        ].sort_values("Predicted_Prob", ascending=False)
                        df_fn = results_cv_df[
                            results_cv_df["Category"] == "FN"
                        ].sort_values("Predicted_Prob", ascending=True)
                        df_tp = results_cv_df[
                            results_cv_df["Category"] == "TP"
                        ].sort_values("Predicted_Prob", ascending=True)
                        df_tn = results_cv_df[
                            results_cv_df["Category"] == "TN"
                        ].sort_values("Predicted_Prob", ascending=False)

                        out_fname = f"{dataset_name}_{scenario_name}_{feature_set_name}_{model_name}.xlsx".replace(
                            " ", "_"
                        )
                        out_path = os.path.join(DETAILED_RESULTS_DIR, out_fname)

                        with pd.ExcelWriter(out_path) as writer:
                            df_tp.to_excel(writer, sheet_name="TP", index=False)
                            df_tn.to_excel(writer, sheet_name="TN", index=False)
                            df_fp.to_excel(writer, sheet_name="FP", index=False)
                            df_fn.to_excel(writer, sheet_name="FN", index=False)

                        tsne_out_fname = out_fname.replace(".xlsx", "_tsne.png")
                        tsne_out_path = os.path.join(
                            DETAILED_RESULTS_DIR, tsne_out_fname
                        )
                        plot_tsne(
                            X_test_final,
                            results_cv_df,
                            tsne_out_path,
                            title=f"t-SNE: {dataset_name} {scenario_name} {model_name} ({feature_set_name})",
                        )

                        umap_out_fname = out_fname.replace(".xlsx", "_umap.png")
                        umap_out_path = os.path.join(
                            DETAILED_RESULTS_DIR, umap_out_fname
                        )
                        plot_umap(
                            X_test_final,
                            results_cv_df,
                            umap_out_path,
                            title=f"UMAP: {dataset_name} {scenario_name} {model_name} ({feature_set_name})",
                        )

                    except Exception as e:
                        print(f"Error evaluating {model_name}: {e}")

    # ==========================
    # SAVE ALL METRICS TO EXCEL
    # ==========================

    results_df = pd.DataFrame(RESULTS)
    excel_path = os.path.join(OUTPUT_DIR, "reproduced_model_results.xlsx")
    print(f"\nSaving reproduced results to: {excel_path} ...")

    try:
        results_df.to_excel(excel_path, index=False)
        print("Done! Excel file created successfully.")
    except Exception as e:
        print(f"Error saving Excel file: {e}")
        print("Make sure you have openpyxl installed: pip install openpyxl")


if __name__ == "__main__":
    load_and_evaluate()
