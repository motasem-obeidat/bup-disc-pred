from lightgbm import LGBMClassifier, early_stopping

from model_utils import (
    GPU_AVAILABLE,
    OUTPUT_DIR,
)
import os
from training import train_model, evaluate_and_save_model


def train_lightgbm(
    X_train,
    X_val,
    X_test,
    y_train,
    y_val,
    y_test,
    lightgbm_preprocessor,
    cat_cols,
    n_classes,
    dataset_name,
    scenario_name,
    feature_set_name,
    models_dir="saved_models",
):
    """
    Trains and evaluates a LightGBM model.
    """

    X_train_lgbm = lightgbm_preprocessor.fit_transform(X_train)
    X_val_lgbm = lightgbm_preprocessor.transform(X_val)
    X_test_lgbm = lightgbm_preprocessor.transform(X_test)

    if n_classes > 2:
        lgbm_objective = "multiclass"
        lgbm_metric = "multi_logloss"
    else:
        lgbm_objective = "binary"
        lgbm_metric = "binary_logloss"

    lgbm_device = "cpu"  # Force CPU to avoid 'No OpenCL device found' error even if PyTorch detects a GPU
    print("\n" + "-" * 50)
    print(f"\n[LightGBM] Device set to: {lgbm_device}")

    lgbm_param_grid = {
        "objective": [lgbm_objective],
        "metric": [lgbm_metric],
        "device_type": [lgbm_device],
        "n_jobs": [-1 if not GPU_AVAILABLE else 1],
    }

    fit_params = {
        "categorical_feature": cat_cols,
        "eval_set": [(X_val_lgbm, y_val)],
        "callbacks": [early_stopping(stopping_rounds=50)],
    }

    best_lgbm_model = train_model(
        model_class_or_instance=LGBMClassifier,
        model_name="LightGBM",
        dataset_name=dataset_name,
        scenario_name=scenario_name,
        feature_set_name=feature_set_name,
        X_train=X_train_lgbm,
        y_train=y_train,
        X_val=X_val_lgbm,
        y_val=y_val,
        param_grid=lgbm_param_grid,
        fit_params=fit_params,
        is_instance=False,
    )

    return evaluate_and_save_model(
        model=best_lgbm_model,
        X_train=X_train_lgbm,
        y_train=y_train,
        X_val=X_val_lgbm,
        y_val=y_val,
        X_test=X_test_lgbm,
        y_test=y_test,
        model_name="LightGBM",
        dataset_name=dataset_name,
        scenario_name=scenario_name,
        feature_set_name=feature_set_name,
        models_dir=models_dir,
        plots_dir=os.path.join(OUTPUT_DIR, "plots_lightgbm"),
    )
