from catboost import CatBoostClassifier

from model_utils import (
    GPU_AVAILABLE,
    OUTPUT_DIR,
)
import os
from training import train_model, evaluate_and_save_model


def train_catboost(
    X_train,
    X_val,
    X_test,
    y_train,
    y_val,
    y_test,
    catboost_preprocessor,
    cat_cols,
    n_classes,
    dataset_name,
    scenario_name,
    feature_set_name,
    models_dir="saved_models",
):
    """
    Trains and evaluates a CatBoost model.
    """
    print("\nPreprocessing data for CatBoost...")
    X_train_cat = catboost_preprocessor.fit_transform(X_train)
    X_val_cat = catboost_preprocessor.transform(X_val)
    X_test_cat = catboost_preprocessor.transform(X_test)

    if n_classes > 2:
        cat_loss = "MultiClass"
        eval_metric = "MultiClass"
    else:
        cat_loss = "Logloss"
        eval_metric = "Logloss"

    catboost_info_dir = os.path.join(OUTPUT_DIR, "catboost_info")

    catboost_param_grid = {
        "loss_function": [cat_loss],
        "eval_metric": [eval_metric],
        "task_type": ["GPU" if GPU_AVAILABLE else "CPU"],
        "devices": ["0" if GPU_AVAILABLE else None],
        "train_dir": [catboost_info_dir],
        "cat_features": [cat_cols],
    }

    print("-" * 50)
    print(f"\n[CatBoost] Task Type: {catboost_param_grid['task_type'][0]}")

    fit_params = {
        "eval_set": (X_val_cat, y_val),
        "early_stopping_rounds": 50,
    }

    best_cat_model = train_model(
        model_class_or_instance=CatBoostClassifier,
        model_name="CatBoost",
        dataset_name=dataset_name,
        scenario_name=scenario_name,
        feature_set_name=feature_set_name,
        X_train=X_train_cat,
        y_train=y_train,
        X_val=X_val_cat,
        y_val=y_val,
        param_grid=catboost_param_grid,
        fit_params=fit_params,
        is_instance=False,
    )

    return evaluate_and_save_model(
        model=best_cat_model,
        X_train=X_train_cat,
        y_train=y_train,
        X_val=X_val_cat,
        y_val=y_val,
        X_test=X_test_cat,
        y_test=y_test,
        model_name="CatBoost",
        dataset_name=dataset_name,
        scenario_name=scenario_name,
        feature_set_name=feature_set_name,
        models_dir=models_dir,
        plots_dir=os.path.join(OUTPUT_DIR, "plots_catboost"),
    )
