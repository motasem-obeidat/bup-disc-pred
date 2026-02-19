from model_utils import *
from model_utils import OUTPUT_DIR

from preprocessing import (
    build_linear_preprocessor,
    build_tree_preprocessor,
    build_xgboost_preprocessor,
    build_lightgbm_preprocessor,
    build_catboost_preprocessor,
    prepare_feature_sets,
    get_ranked_features,
)

from sklearn_models import train_sklearn_models

from xgboost_model import train_xgboost
from lightgbm_model import train_lightgbm
from catboost_model import train_catboost
from TabICL import train_tabicl
from TabPFN import train_tabpfn
from TabM import train_tabm

MODELS_DIR = os.path.join(OUTPUT_DIR, "saved_models")
os.makedirs(MODELS_DIR, exist_ok=True)


def run_models_for_feature_set(
    feature_set_name,
    X_train,
    X_val,
    X_test,
    y_train,
    y_val,
    y_test,
    n_classes,
    dataset_name,
    scenario_name,
):
    """
    Runs the full model training loop for a specific feature set.
    """

    print("\n" + "#" * 70)
    print(f"SETTING START: {dataset_name} | {scenario_name} | {feature_set_name}")
    print("#" * 70)

    curves_for_plotting = {}

    # -----------------------------------------------------------
    # 1. Build model-specific preprocessing pipelines
    # -----------------------------------------------------------
    print("\n" + "=" * 50)
    print("Building model-specific preprocessors...")
    print("=" * 50)

    linear_preprocessor, _ = build_linear_preprocessor(X_train)
    tree_preprocessor, _ = build_tree_preprocessor(X_train)
    xgboost_preprocessor, xgb_cat_cols = build_xgboost_preprocessor(X_train)
    lightgbm_preprocessor, lgbm_cat_cols = build_lightgbm_preprocessor(X_train)
    catboost_preprocessor, cat_cols = build_catboost_preprocessor(X_train)

    # -------------------------------------------------------------------
    # 2. XGBoost (PRIORITY: Runs first to generate feature rankings)
    # -------------------------------------------------------------------

    best_xgb_model, xgb_curves = train_xgboost(
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        xgboost_preprocessor,
        xgb_cat_cols,
        n_classes,
        dataset_name,
        scenario_name,
        feature_set_name,
        MODELS_DIR,
    )

    if xgb_curves is not None:
        curves_for_plotting["XGBoost"] = xgb_curves

    # -----------------------------------------------------------
    # 3. Train sklearn models (LogReg, DT, RF, AdaBoost)
    # -----------------------------------------------------------
    sklearn_curves = train_sklearn_models(
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
        MODELS_DIR,
    )
    curves_for_plotting.update(sklearn_curves)

    # -------------------------------------------------------------------
    # 4. LightGBM
    # -------------------------------------------------------------------

    best_lgbm_model, lgbm_curves = train_lightgbm(
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        lightgbm_preprocessor,
        lgbm_cat_cols,
        n_classes,
        dataset_name,
        scenario_name,
        feature_set_name,
        MODELS_DIR,
    )

    if lgbm_curves is not None:
        curves_for_plotting["LightGBM"] = lgbm_curves

    # -------------------------------------------------------------------
    # 5. CatBoost
    # -------------------------------------------------------------------

    best_cat_model, cat_curves = train_catboost(
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
        MODELS_DIR,
    )

    if cat_curves is not None:
        curves_for_plotting["CatBoost"] = cat_curves

    # -------------------------------------------------------------------
    # 6. TabICL
    # -------------------------------------------------------------------

    best_tabicl_model, tabicl_curves = train_tabicl(
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        dataset_name,
        scenario_name,
        feature_set_name,
    )

    if tabicl_curves is not None:
        curves_for_plotting["TabICL"] = tabicl_curves

    # -------------------------------------------------------------------
    # 7. TabPFN
    # -------------------------------------------------------------------

    best_tabpfn_model, tabpfn_curves = train_tabpfn(
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        dataset_name,
        scenario_name,
        feature_set_name,
    )

    if tabpfn_curves is not None:
        curves_for_plotting["TabPFN"] = tabpfn_curves

    # -------------------------------------------------------------------
    # 8. TabM
    # -------------------------------------------------------------------
    best_tabm_model, tabm_curves = train_tabm(
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        dataset_name,
        scenario_name,
        feature_set_name,
    )

    if tabm_curves is not None:
        curves_for_plotting["TabM"] = tabm_curves

    # -------------------------------------------------------------------
    # 9. GENERATE PLOTS FOR THIS SETTING
    # -------------------------------------------------------------------

    plot_title = f"{dataset_name} - {scenario_name} - {feature_set_name}"

    print("\n" + "-" * 50)
    print(f"Generating ROC and PR curves for: {plot_title}...")
    print("-" * 50)

    # Calculate baseline prevalence (fraction of positives in test set)
    prevalence = y_test.mean()

    plot_performance_curves(
        curves_for_plotting,
        title_prefix=plot_title,
        baseline_prevalence=prevalence,
    )


# -------------------------------------------------------------------
# Loop over datasets (BINARY + MULTI) and PDC scenarios
# -------------------------------------------------------------------
for dataset_name, paths in DATASETS.items():
    print("\n" + "#" * 110)
    print("#" * 110)
    print(f"DATASET: {dataset_name}")
    print("#" * 110)
    print("#" * 110)

    df_train = pd.read_csv(paths["train"])
    df_val = pd.read_csv(paths["val"])
    df_test = pd.read_csv(paths["test"])

    for split_name, df_split in zip(
        ["train", "val", "test"], [df_train, df_val, df_test]
    ):
        assert (
            TARGET_COL in df_split.columns
        ), f"{TARGET_COL} not in {split_name} columns: {df_split.columns}"

    y_train = df_train[TARGET_COL]
    y_val = df_val[TARGET_COL]
    y_test = df_test[TARGET_COL]

    y_all = pd.concat([y_train, y_val, y_test], axis=0)
    print("\nClass distribution (train+val+test):\n", y_all.value_counts().sort_index())
    n_classes = y_all.nunique()
    print(f"\nDetected {n_classes} classes in {TARGET_COL} for {dataset_name}.")

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

        run_models_for_feature_set(
            feature_set_name="BASELINE_FEATURES",
            X_train=X_train_base,
            X_val=X_val_base,
            X_test=X_test_base,
            y_train=y_train,
            y_val=y_val,
            y_test=y_test,
            n_classes=n_classes,
            dataset_name=dataset_name,
            scenario_name=scenario_name,
        )

        ranking_key = (dataset_name, scenario_name)
        ranking_path = RANKING_FILES.get(ranking_key, None)

        if ranking_path is None:
            print(
                f"\nNo ranking file configured for {dataset_name}, {scenario_name}; skipping ranked feature set."
            )
            continue

        top_features_in_X = get_ranked_features(
            ranking_path, X_train_base, TOP_K_RANKED_FEATURES
        )

        if not top_features_in_X:
            print(f"No ranked features found. Skipping ranked models.")
            continue

        X_train_ranked = X_train_base[top_features_in_X].copy()
        X_val_ranked = X_val_base[top_features_in_X].copy()
        X_test_ranked = X_test_base[top_features_in_X].copy()

        run_models_for_feature_set(
            feature_set_name=f"RANKED_TOP_{TOP_K_RANKED_FEATURES}",
            X_train=X_train_ranked,
            X_val=X_val_ranked,
            X_test=X_test_ranked,
            y_train=y_train,
            y_val=y_val,
            y_test=y_test,
            n_classes=n_classes,
            dataset_name=dataset_name,
            scenario_name=scenario_name,
        )

print("\nAll datasets, scenarios, and feature sets completed.")

# ==========================
# SAVE ALL METRICS TO EXCEL
# ==========================
results_df = pd.DataFrame(RESULTS)

model_order = [
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
results_df["Model"] = pd.Categorical(
    results_df["Model"], categories=model_order, ordered=True
)

excel_path = os.path.join(OUTPUT_DIR, "complete_model_results.xlsx")
print(f"\nSaving complete results to: {excel_path} ...")

try:
    results_df.to_excel(excel_path, index=False)
    print("Done! Excel file created successfully.")
except Exception as e:
    print(f"Error saving Excel file: {e}")
    print("Make sure you have openpyxl installed: pip install openpyxl")
