import optuna
import numpy as np
from sklearn.base import clone
from sklearn.metrics import f1_score
from config import (
    get_search_space_for_model,
    OPTUNA_SHOW_PROGRESS,
    OPTUNA_N_JOBS,
)
from model_utils import roc_auc_positive_class_1, SEED

optuna.logging.set_verbosity(optuna.logging.WARNING)


def optimize_with_optuna(
    model_class_or_instance,
    model_name,
    dataset_name,
    scenario_name,
    feature_set_name,
    X_train,
    y_train,
    X_val,
    y_val,
    search_space_func=None,
    fit_params=None,
    fixed_params=None,
    is_instance=False,
    n_trials=100,
    timeout=3600,
    verbose=False,
):
    """
    Optimizes a model's hyperparameters using Optuna.
    """

    print("\n" + "-" * 50)
    print(f"OPTUNA OPTIMIZATION: {model_name}")
    print("-" * 50)
    print(
        f"\nStarting Bayesian hyperparameter search for {model_name} "
        f"({dataset_name}, {scenario_name}, {feature_set_name})..."
    )
    print(f"Trials: {n_trials} | Timeout: {timeout}s")

    if fit_params is None:
        fit_params = {}

    if fixed_params is None:
        fixed_params = {}

    if search_space_func is None:
        search_space_func = get_search_space_for_model(model_name)

    best_val_score = -np.inf
    best_model = None
    best_params = None

    def objective(trial):
        """Optuna objective function to maximize."""
        nonlocal best_val_score, best_model, best_params

        try:
            params = search_space_func(trial)

            for k, v in fixed_params.items():
                params[k] = v

            if is_instance:
                model = clone(model_class_or_instance)
                model.set_params(**params)
            else:
                model = model_class_or_instance(**params)

            model.fit(X_train, y_train, **fit_params)

            val_score = roc_auc_positive_class_1(model, X_val, y_val)

            if verbose:
                print(f"Trial {trial.number}: {val_score:.4f}")

            if val_score > best_val_score:
                best_val_score = val_score
                best_model = model
                best_params = params

            return val_score

        except Exception as e:
            print(f"Trial {trial.number} failed: {e}")
            raise optuna.TrialPruned()

    sampler = optuna.samplers.TPESampler(
        seed=SEED, multivariate=True, warn_independent_sampling=False
    )

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=optuna.pruners.NopPruner(),
        study_name=f"{model_name}_{dataset_name}_{scenario_name}_{feature_set_name}",
    )

    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        n_jobs=OPTUNA_N_JOBS,
        show_progress_bar=OPTUNA_SHOW_PROGRESS,
        catch=(Exception,),
    )

    print("\n" + "=" * 50)
    print("Optimization Completed!")
    print("=" * 50)
    print(f"Best {model_name} VAL: {best_val_score:.4f}")
    print(f"Best params: {best_params}")
    print(f"Total trials: {len(study.trials)}")
    print(f"Best trial: {study.best_trial.number}")

    if best_model is None:
        raise RuntimeError(
            f"Failed to train {model_name}: All trials failed. "
            f"Check the error messages above for details."
        )

    return best_model
