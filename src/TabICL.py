import os
from model_utils import OUTPUT_DIR


from training import train_model, evaluate_and_save_model

try:
    from tabicl import TabICLClassifier

    TABICL_AVAILABLE = True
except ImportError:
    print("WARNING: TabICL not installed. Install with: pip install tabicl")
    TABICL_AVAILABLE = False

MODELS_DIR = os.path.join(OUTPUT_DIR, "saved_models_tabicl")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots_tabicl")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)


def train_tabicl(
    X_train,
    X_val,
    X_test,
    y_train,
    y_val,
    y_test,
    dataset_name,
    scenario_name,
    feature_set_name,
    models_dir=MODELS_DIR,
    plots_dir=PLOTS_DIR,
):
    """
    Trains and evaluates a TabICL model.
    """

    if not TABICL_AVAILABLE:
        print("TabICL not available. Skipping...")
        return None, None

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    try:
        best_model = train_model(
            model_class_or_instance=TabICLClassifier,
            model_name="TabICL",
            dataset_name=dataset_name,
            scenario_name=scenario_name,
            feature_set_name=feature_set_name,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            param_grid={},
            fit_params={},
            is_instance=False,
        )

        print(
            f"\n[TabICL] Device from config: {best_model.device if hasattr(best_model, 'device') else 'unknown'}"
        )

        if best_model is None:
            print("ERROR: No valid TabICL model found during search.")
            return None, None

        return evaluate_and_save_model(
            model=best_model,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            model_name="TabICL",
            dataset_name=dataset_name,
            scenario_name=scenario_name,
            feature_set_name=feature_set_name,
            models_dir=models_dir,
            plots_dir=plots_dir,
        )

    except Exception as e:
        print(f"\n[ERROR] Error running TabICL: {e}")
        import traceback

        traceback.print_exc()
        return None, None
