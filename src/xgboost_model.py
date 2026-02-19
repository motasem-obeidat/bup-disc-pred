import os
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_sample_weight

from model_utils import (
    GPU_AVAILABLE,
    OUTPUT_DIR,
    BASE_DIR,
)
from training import train_model, evaluate_and_save_model


class XGBClassifierAutoWeight(XGBClassifier):
    """
    XGBoost Classifier that automatically calculates scale_pos_weight or sample_weights.
    """

    _has_printed_weight = False

    def __init__(self, **kwargs):
        if kwargs.get("scale_pos_weight") == "auto":
            kwargs["scale_pos_weight"] = 1.0
            self._auto_scale = True
        else:
            self._auto_scale = False
        super().__init__(**kwargs)

    def fit(self, X, y, **fit_params):
        if self._auto_scale:
            unique_classes = np.unique(y)
            n_classes = len(unique_classes)

            # Case 1: Binary Classification (Use scale_pos_weight)
            if n_classes == 2:
                neg_count = (y == 0).sum()
                pos_count = (y == 1).sum()
                if pos_count > 0:
                    calculated_weight = neg_count / pos_count
                    self.set_params(scale_pos_weight=calculated_weight)

                    if not XGBClassifierAutoWeight._has_printed_weight:
                        print(
                            f"  -> Auto-calculated scale_pos_weight = {calculated_weight:.2f}"
                        )
                        XGBClassifierAutoWeight._has_printed_weight = True

            # Case 2: Multi-class Classification (Use sample_weight)
            else:
                # Calculate weights for every single sample
                weights = compute_sample_weight(class_weight="balanced", y=y)
                fit_params["sample_weight"] = weights

                if not XGBClassifierAutoWeight._has_printed_weight:
                    print(
                        "  -> Multi-class detected: Applied 'balanced' sample_weights."
                    )
                    XGBClassifierAutoWeight._has_printed_weight = True

        return super().fit(X, y, **fit_params)


def train_xgboost(
    X_train,
    X_val,
    X_test,
    y_train,
    y_val,
    y_test,
    xgboost_preprocessor,
    cat_cols,
    n_classes,
    dataset_name,
    scenario_name,
    feature_set_name,
    models_dir="saved_models",
):
    """
    Trains and evaluates an XGBoost model, including SHAP feature importance calculation.
    """

    print("-" * 50)

    print("\nPreprocessing data for XGBoost...")
    X_train_xgb = xgboost_preprocessor.fit_transform(X_train)
    X_val_xgb = xgboost_preprocessor.transform(X_val)
    X_test_xgb = xgboost_preprocessor.transform(X_test)

    if n_classes > 2:
        xgb_objective = "multi:softprob"
        xgb_eval_metric = "mlogloss"
    else:
        xgb_objective = "binary:logistic"
        xgb_eval_metric = "logloss"

    import xgboost as xgb
    from packaging import version

    xgb_version = version.parse(xgb.__version__)
    print(f"XGBoost Version: {xgb_version}")

    xgb_tree_method = "hist"
    xgb_device_params = {}

    if GPU_AVAILABLE:
        if xgb_version >= version.parse("2.0.0"):
            xgb_device_params["device"] = "cuda"
            xgb_tree_method = "hist"
        else:
            xgb_tree_method = "gpu_hist"
    else:
        xgb_tree_method = "hist"

    print(f"\n[XGBoost] Device Params: {xgb_device_params}")
    print(f"[XGBoost] Tree Method: {xgb_tree_method}")

    xgb_param_grid = {
        "objective": [xgb_objective],
        "eval_metric": [xgb_eval_metric],
        "tree_method": [xgb_tree_method],
        "n_jobs": [-1 if not GPU_AVAILABLE else 1],
        "early_stopping_rounds": [50],
        **{k: [v] for k, v in xgb_device_params.items()},
    }

    fit_params = {
        "eval_set": [(X_val_xgb, y_val)],
        "verbose": False,
    }

    best_xgb_model = train_model(
        model_class_or_instance=XGBClassifierAutoWeight,
        model_name="XGBoost",
        dataset_name=dataset_name,
        scenario_name=scenario_name,
        feature_set_name=feature_set_name,
        X_train=X_train_xgb,
        y_train=y_train,
        X_val=X_val_xgb,
        y_val=y_val,
        param_grid=xgb_param_grid,
        fit_params=fit_params,
        is_instance=False,
    )

    # ------------------------------------------------------------------------------------------------------------------
    # SHAP Feature Importance (Only compute SHAP for the BASELINE feature set to generate the global feature ranking)
    # ------------------------------------------------------------------------------------------------------------------
    plot_dir = os.path.join(OUTPUT_DIR, "plots_xgboost")
    os.makedirs(plot_dir, exist_ok=True)

    if feature_set_name == "BASELINE_FEATURES":
        print("\nComputing SHAP feature importance (CPU-only)...")

        try:
            booster = best_xgb_model.get_booster()

            print(
                "\n[XGBoost SHAP] Force-setting booster to CPU to avoid TreeExplainer GPU crashes."
            )
            print("[XGBoost SHAP] Starting SHAP calculation on CPU...")

            # Enable categorical support on CPU booster to handle categorical features correctly during SHAP
            booster.set_param({"device": "cpu", "enable_categorical": True})

            dtrain_shap = xgb.DMatrix(X_val_xgb, enable_categorical=True)

            explainer = shap.TreeExplainer(
                booster,
            )

            shap_values = explainer.shap_values(dtrain_shap)

            if isinstance(shap_values, list):
                # Case: multiclass, list of arrays (n_class, n_samples, n_features)
                print("SHAP Detected: list-of-arrays multiclass format")
                shap_arr = np.mean([np.abs(sv) for sv in shap_values], axis=0)

            elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
                # Case: multiclass, single 3D array (n_samples, n_features, n_classes)
                print("SHAP Detected: 3D array multiclass format")
                shap_arr = np.mean(np.abs(shap_values), axis=2)

            elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 2:
                # Case: binary or already reduced multiclass (n_samples, n_features)
                print("SHAP Detected: 2D array binary format")
                shap_arr = np.abs(shap_values)

            else:
                raise ValueError(
                    f"Unexpected SHAP output format: shape {np.array(shap_values).shape}"
                )

            shap_importance = np.mean(shap_arr, axis=0)

            importance_df = pd.Series(
                shap_importance, index=X_val_xgb.columns
            ).sort_values(ascending=False)

            file_subname = dataset_name.lower()
            if file_subname == "multi":
                file_subname = "multi-class"

            scen_subname = scenario_name.lower().replace("no_pdc", "without_pdc")

            filename = f"{file_subname}_{scen_subname}_shap_ranking.csv"
            save_path = os.path.join(BASE_DIR, filename)

            importance_df.to_csv(save_path, header=["mean_abs_shap"])
            print(f"Saved SHAP ranking -> {save_path}")

            top_n = 20
            to_plot = importance_df.head(top_n).sort_values(ascending=True)
            plt.figure(figsize=(10, 8))
            plt.barh(to_plot.index, to_plot.values)
            plt.xlabel("Mean |SHAP value|")
            plt.title(f"SHAP Feature Ranking - {dataset_name} {scenario_name}")
            plt.tight_layout()

            plot_path = os.path.join(
                plot_dir, f"SHAP_{filename.replace('.csv', '.png')}"
            )
            plt.savefig(plot_path)
            plt.close()
            print(f"Saved SHAP plot -> {plot_path}")

        except Exception as e:
            print(f"Error computing SHAP importance: {e}")
            import traceback

            traceback.print_exc()

    return evaluate_and_save_model(
        model=best_xgb_model,
        X_train=X_train_xgb,
        y_train=y_train,
        X_val=X_val_xgb,
        y_val=y_val,
        X_test=X_test_xgb,
        y_test=y_test,
        model_name="XGBoost",
        dataset_name=dataset_name,
        scenario_name=scenario_name,
        feature_set_name=feature_set_name,
        models_dir=models_dir,
        plots_dir=plot_dir,
    )
