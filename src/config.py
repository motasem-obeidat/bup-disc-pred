import os
import torch
from model_utils import SEED

# ==========================================
# Security / Credentials
# ==========================================

os.environ["HF_TOKEN"] = ""

HF_TOKEN = os.environ.get("HF_TOKEN")

if HF_TOKEN is None:
    import warnings

    warnings.warn(
        "HF_TOKEN environment variable is not set.",
        UserWarning,
    )

# ==========================================
# Optuna Hyperparameter Optimization Settings
# ==========================================
OPTUNA_N_TRIALS = 100
OPTUNA_TIMEOUT = 7200
OPTUNA_N_JOBS = 1
OPTUNA_SHOW_PROGRESS = False


# ==========================================
# Optuna Search Space Definitions
# ==========================================


def create_xgboost_search_space(trial):
    """Define XGBoost hyperparameter search space for Optuna."""
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 2, 10),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-5, 100.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 100.0, log=True),
        "gamma": trial.suggest_float("gamma", 0.1, 20.0, log=True),
        "max_delta_step": trial.suggest_int("max_delta_step", 0, 5),
        "scale_pos_weight": "auto",
        "n_estimators": 5000,
        "verbosity": 0,
        "enable_categorical": True,
        "random_state": SEED,
    }


def create_lightgbm_search_space(trial):
    """Define LightGBM hyperparameter search space for Optuna."""
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 20, 256),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-5, 100.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 100.0, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 100),
        "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 0.5),
        "max_depth": -1,
        "class_weight": "balanced",
        "n_estimators": 5000,
        "verbosity": -1,
        "random_state": SEED,
    }


def create_catboost_search_space(trial):
    """Define CatBoost hyperparameter search space for Optuna."""
    return {
        "depth": trial.suggest_int("depth", 2, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
        "l2_leaf_reg": trial.suggest_int("l2_leaf_reg", 1, 10),
        "random_strength": trial.suggest_float("random_strength", 1, 10.0, log=True),
        "border_count": trial.suggest_int("border_count", 128, 254),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 5.0),
        "auto_class_weights": trial.suggest_categorical(
            "auto_class_weights", ["Balanced", "SqrtBalanced"]
        ),
        "n_estimators": 5000,
        "verbose": 0,
        "use_best_model": True,
        "random_state": SEED,
    }


def create_sklearn_logistic_search_space(trial):
    """Define Logistic Regression search space for Optuna."""
    solver = trial.suggest_categorical("clf__solver", ["newton-cg", "lbfgs", "saga"])

    if solver == "saga":
        penalty = trial.suggest_categorical("clf__penalty", ["l1", "l2", "elasticnet"])
    else:
        penalty = "l2"

    params = {
        "clf__solver": solver,
        "clf__penalty": penalty,
        "clf__C": trial.suggest_float("clf__C", 1e-4, 1e4, log=True),
        "clf__class_weight": "balanced",
        "clf__max_iter": 1000,
    }

    if penalty == "elasticnet":
        params["clf__l1_ratio"] = trial.suggest_float("clf__l1_ratio", 0.0, 1.0)

    return params


def create_sklearn_tree_search_space(trial):
    """Define Decision Tree hyperparameter search space for Optuna."""
    return {
        "clf__criterion": trial.suggest_categorical(
            "clf__criterion", ["gini", "entropy"]
        ),
        "clf__min_samples_leaf": trial.suggest_int("clf__min_samples_leaf", 1, 50),
        "clf__min_samples_split": trial.suggest_int("clf__min_samples_split", 2, 150),
        "clf__max_depth": trial.suggest_int("clf__max_depth", 2, 15),
        "clf__class_weight": "balanced",
    }


def create_random_forest_search_space(trial):
    """Define Random Forest hyperparameter search space for Optuna."""
    return {
        "clf__n_estimators": trial.suggest_int("clf__n_estimators", 50, 500),
        "clf__min_samples_leaf": trial.suggest_int("clf__min_samples_leaf", 1, 50),
        "clf__min_samples_split": trial.suggest_int("clf__min_samples_split", 2, 150),
        "clf__max_depth": trial.suggest_int("clf__max_depth", 2, 50),
        "clf__max_features": trial.suggest_categorical(
            "clf__max_features", ["sqrt", "log2", None]
        ),
        "clf__bootstrap": True,
        "clf__class_weight": trial.suggest_categorical(
            "clf__class_weight", ["balanced", "balanced_subsample"]
        ),
    }


def create_adaboost_search_space(trial):
    """Define AdaBoost search space for Optuna."""
    return {
        "clf__n_estimators": trial.suggest_int("clf__n_estimators", 50, 500),
        "clf__learning_rate": trial.suggest_float(
            "clf__learning_rate", 0.005, 0.3, log=True
        ),
        "clf__estimator__max_depth": trial.suggest_int(
            "clf__estimator__max_depth", 1, 10
        ),
        "clf__estimator__class_weight": "balanced",
    }


def create_tabm_search_space(trial):
    """Define TabM hyperparameter search space for Optuna."""
    emb_type = trial.suggest_categorical(
        "emb_type", ["piecewise", "linear", "periodic"]
    )

    params = {
        "scaler_type": trial.suggest_categorical(
            "scaler_type", ["standard", "quantile"]
        ),
        "n_blocks": trial.suggest_int("n_blocks", 1, 4),
        "d_block": trial.suggest_categorical("d_block", [64, 128, 256, 512, 768, 1024]),
        "lr": trial.suggest_float("lr", 1e-4, 5e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-4, 1e-1, log=True),
        "dropout": trial.suggest_float("dropout", 0.0, 0.5),
        "emb_type": emb_type,
        "d_embedding": trial.suggest_categorical("d_embedding", [8, 16, 32, 64]),
        "k": 32,
        "epochs": 200,
        "batch_size": 256,
        "patience": 16,
        "debug_preprocessing": False,
        "class_weight": "balanced",
    }

    if emb_type == "piecewise":
        params["n_bins"] = trial.suggest_int("n_bins", 2, 128)
    elif emb_type == "periodic":
        params["periodic_lite"] = False
        params["periodic_n_frequencies"] = trial.suggest_categorical(
            "periodic_n_frequencies", [16, 24]
        )

    return params


def create_tabicl_search_space(trial):
    """Define TabICLv2 hyperparameter search space for Optuna."""
    return {
        "n_estimators": trial.suggest_categorical("n_estimators", [8, 16, 32, 48, 64]),
        "norm_methods": trial.suggest_categorical("norm_methods", ["none", "power"]),
        "feat_shuffle_method": "latin",
        "class_shuffle_method": "shift",
        "outlier_threshold": 4.0,
        "softmax_temperature": 0.9,
        "average_logits": True,
        "support_many_classes": True,
        "batch_size": 8,
        "allow_auto_download": True,
        "checkpoint_version": "tabicl-classifier-v2-20260212.ckpt",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "use_amp": "auto",
        "use_fa3": "auto",
        "offload_mode": "auto",
        "random_state": SEED,
        "verbose": False,
    }


def create_tabpfn_search_space(trial):
    """Define TabPFN hyperparameter search space for Optuna."""
    return {
        "n_estimators": trial.suggest_categorical("n_estimators", [8, 16, 32, 48, 64]),
        "softmax_temperature": 0.9,
        "average_before_softmax": True,
        "balance_probabilities": True,
        "device": "auto",
        "random_state": SEED,
    }


def get_search_space_for_model(model_name):
    """Return the appropriate search space function for a given model name."""
    search_spaces = {
        "XGBoost": create_xgboost_search_space,
        "LightGBM": create_lightgbm_search_space,
        "CatBoost": create_catboost_search_space,
        "LogisticRegression": create_sklearn_logistic_search_space,
        "DecisionTree": create_sklearn_tree_search_space,
        "RandomForest": create_random_forest_search_space,
        "AdaBoost": create_adaboost_search_space,
        "TabM": create_tabm_search_space,
        "TabICL": create_tabicl_search_space,
        "TabPFN": create_tabpfn_search_space,
    }

    return search_spaces.get(model_name, lambda trial: {})
