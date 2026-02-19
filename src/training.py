import os
import joblib
import numpy as np
from sklearn.metrics import f1_score

from model_utils import (
    evaluate_model,
    roc_auc_positive_class_1,
    pr_auc_positive_class_1,
    make_calibration_plot,
    make_risk_stratification_plot,
    RESULTS,
    ThresholdWrapper,
)


def save_model(
    model, models_dir, dataset_name, scenario_name, feature_set_name, model_name
):
    """
    Saves the trained model to the specified directory.
    """

    os.makedirs(models_dir, exist_ok=True)

    fname = f"{dataset_name}_{scenario_name}_{feature_set_name}_{model_name}.joblib"
    fname = fname.replace(" ", "_")
    path = os.path.join(models_dir, fname)

    joblib.dump(model, path)
    print(f"Saved model: {model_name} -> {path}")
    return path


def append_to_results(
    dataset,
    scenario,
    feature_set,
    model_name,
    train_m,
    val_m,
    test_m,
    threshold=None,
):
    """
    Appends model evaluation results to the global results list.
    """

    row = {
        "Dataset": dataset,
        "Scenario": scenario,
        "Feature_Set": feature_set,
        "Model": model_name,
        "Best_Threshold": threshold if threshold is not None else "N/A",
    }

    for k, v in train_m.items():
        row[f"TRAIN_{k}"] = v

    for k, v in val_m.items():
        row[f"VAL_{k}"] = v

    for k, v in test_m.items():
        row[f"TEST_{k}"] = v

    RESULTS.append(row)


def evaluate_and_record(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    model_name,
    dataset_name,
    scenario_name,
    feature_set_name,
    threshold=None,
):
    """
    Evaluates a model on train, val, and test sets and records the results.
    """
    train_m = evaluate_model(
        f"{model_name} TRAIN ({dataset_name}, {scenario_name}, {feature_set_name})",
        model,
        X_train,
        y_train,
    )

    val_m = evaluate_model(
        f"{model_name} VAL ({dataset_name}, {scenario_name}, {feature_set_name})",
        model,
        X_val,
        y_val,
    )

    test_m = evaluate_model(
        f"{model_name} TEST ({dataset_name}, {scenario_name}, {feature_set_name})",
        model,
        X_test,
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
        threshold=threshold,
    )

    return train_m, val_m, test_m


def optimize_threshold_class1(model, X_val, y_val):
    """
    Optimizes the decision threshold for class 1 to maximize F1 score.
    """
    if not hasattr(model, "predict_proba"):
        return 0.5

    classes = model.classes_
    if 1 not in classes:
        return 0.5

    idx_1 = list(classes).index(1)
    probs = model.predict_proba(X_val)[:, idx_1]

    best_thresh = 0.5
    best_f1_c1 = -1.0

    thresholds = np.arange(0.05, 0.96, 0.01)

    all_probs = model.predict_proba(X_val)
    all_probs_no_c1 = all_probs.copy()
    all_probs_no_c1[:, idx_1] = -1.0
    fallback_preds = classes[np.argmax(all_probs_no_c1, axis=1)]

    for thresh in thresholds:
        preds = np.where(probs >= thresh, 1, fallback_preds)

        current_f1_c1 = f1_score(
            y_val, preds, labels=[1], average=None, zero_division=0
        )[0]

        if current_f1_c1 > best_f1_c1:
            best_f1_c1 = current_f1_c1
            best_thresh = thresh

    print(
        f"\n[Threshold Optimization] Best Threshold: {best_thresh:.2f} "
        f"(Class 1 F1 Score: {best_f1_c1:.4f})"
    )
    return best_thresh


def evaluate_and_save_model(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    model_name,
    dataset_name,
    scenario_name,
    feature_set_name,
    models_dir,
    plots_dir=None,
):
    """
    Evaluates a trained model, optimizes threshold, saves it, and generates plots.
    """
    best_threshold = 0.5
    print("\nOptimizing decision threshold for Class 1 (F1 maximization)...")
    best_threshold = optimize_threshold_class1(model, X_val, y_val)

    model = ThresholdWrapper(model, threshold=best_threshold)

    model.classes_ = model.estimator.classes_

    train_m, val_m, test_m = evaluate_and_record(
        model,
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        model_name,
        dataset_name,
        scenario_name,
        feature_set_name,
        threshold=best_threshold,
    )

    if plots_dir:
        os.makedirs(plots_dir, exist_ok=True)
        try:
            make_calibration_plot(
                model,
                X_test,
                y_test,
                X_val,
                y_val,
                model_name,
                dataset_name,
                scenario_name,
                feature_set_name,
                save_dir=os.path.join(plots_dir, "calibration"),
            )
        except Exception as e:
            print(f"Error generating calibration plot for {model_name}: {e}")

        try:
            make_risk_stratification_plot(
                model,
                X_test,
                y_test,
                model_name,
                dataset_name,
                scenario_name,
                feature_set_name,
                save_dir=os.path.join(plots_dir, "risk"),
            )
        except Exception as e:
            print(f"Error generating risk plot for {model_name}: {e}")

    save_model(
        model,
        models_dir,
        dataset_name,
        scenario_name,
        feature_set_name,
        model_name,
    )

    curves_data = None
    try:
        roc_data = roc_auc_positive_class_1(model, X_test, y_test, return_curves=True)
        pr_data = pr_auc_positive_class_1(model, X_test, y_test, return_curves=True)

        if isinstance(roc_data, dict) and isinstance(pr_data, dict):
            curves_data = {
                "roc_score": roc_data["score"],
                "fpr": roc_data["fpr"],
                "tpr": roc_data["tpr"],
                "pr_score": pr_data["score"],
                "precision": pr_data["precision"],
                "recall": pr_data["recall"],
            }
    except Exception as e:
        print(f"Skipping curves for {model_name}: {e}")

    return model, curves_data


def train_model(
    model_class_or_instance,
    model_name,
    dataset_name,
    scenario_name,
    feature_set_name,
    X_train,
    y_train,
    X_val,
    y_val,
    param_grid,
    fit_params=None,
    is_instance=False,
    verbose=False,
):
    """
    Trains a model using Optuna for hyperparameter optimization.
    """
    from config import OPTUNA_N_TRIALS, OPTUNA_TIMEOUT
    from optuna_optimizer import optimize_with_optuna

    fixed_params = {}
    if isinstance(param_grid, dict):
        for k, v in param_grid.items():
            if isinstance(v, list) and len(v) == 1:
                fixed_params[k] = v[0]

    return optimize_with_optuna(
        model_class_or_instance=model_class_or_instance,
        model_name=model_name,
        dataset_name=dataset_name,
        scenario_name=scenario_name,
        feature_set_name=feature_set_name,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        fit_params=fit_params,
        fixed_params=fixed_params,
        is_instance=is_instance,
        n_trials=OPTUNA_N_TRIALS,
        timeout=OPTUNA_TIMEOUT,
        verbose=verbose,
    )
