import os
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from model_utils import (
    SEED,
    OUTPUT_DIR,
)
from training import (
    train_model,
    evaluate_and_save_model,
)


def train_sklearn_models(
    X_train,
    X_val,
    X_test,
    y_train,
    y_val,
    y_test,
    linear_preprocessor,
    tree_preprocessor,
    dataset_name,
    scenario_name,
    feature_set_name,
    models_dir="saved_models",
):
    """
    Trains and evaluates standard scikit-learn models (LogReg, DT, RF, AdaBoost).
    """

    curves_for_plotting = {}

    log_reg_clf = LogisticRegression(multi_class="auto", n_jobs=-1, random_state=SEED)

    dt_clf = DecisionTreeClassifier(random_state=SEED)

    rf_clf = RandomForestClassifier(random_state=SEED, n_jobs=-1)

    try:
        ada_clf = AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=1, random_state=SEED),
            algorithm="SAMME",
            random_state=SEED,
        )
    except TypeError:
        ada_clf = AdaBoostClassifier(
            base_estimator=DecisionTreeClassifier(max_depth=1, random_state=SEED),
            algorithm="SAMME",
            random_state=SEED,
        )

    pipelines = {
        "LogisticRegression": Pipeline(
            [("preprocess", linear_preprocessor), ("clf", log_reg_clf)]
        ),
        "DecisionTree": Pipeline([("preprocess", tree_preprocessor), ("clf", dt_clf)]),
        "RandomForest": Pipeline([("preprocess", tree_preprocessor), ("clf", rf_clf)]),
        "AdaBoost": Pipeline([("preprocess", tree_preprocessor), ("clf", ada_clf)]),
    }

    print("\n" + "-" * 50)
    print(
        "\n[Sklearn Models] All standard sklearn models (LogisticRegression, DecisionTree, RandomForest, AdaBoost) run on CPU."
    )

    for name, base_pipe in pipelines.items():
        param_grid = {}

        best_model = train_model(
            model_class_or_instance=base_pipe,
            model_name=name,
            dataset_name=dataset_name,
            scenario_name=scenario_name,
            feature_set_name=feature_set_name,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            param_grid=param_grid,
            is_instance=True,
        )

        _, curves = evaluate_and_save_model(
            model=best_model,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            model_name=name,
            dataset_name=dataset_name,
            scenario_name=scenario_name,
            feature_set_name=feature_set_name,
            models_dir=models_dir,
            plots_dir=os.path.join(OUTPUT_DIR, "plots_sklearn"),
        )

        if curves:
            curves_for_plotting[name] = curves

    return curves_for_plotting
