import os
import numpy as np
from config import HF_TOKEN

from training import train_model, evaluate_and_save_model
from model_utils import GPU_AVAILABLE, OUTPUT_DIR
from tabpfn import TabPFNClassifier
from tabpfn.constants import ModelVersion

try:
    from huggingface_hub import login

    if HF_TOKEN:
        login(token=HF_TOKEN)
        print("\n" + "=" * 60)
        print("Successfully logged in to Hugging Face Hub")
        print("=" * 60 + "\n")
    else:
        print("\n" + "=" * 60)
        print("No HF_TOKEN found in config or environment.")
        print("=" * 60 + "\n")
except ImportError:
    print(
        "WARNING: huggingface_hub not installed. Install with: pip install huggingface-hub"
    )
    print("Continuing without HF authentication...")
except Exception as e:
    print(f"WARNING: Failed to login to Hugging Face: {e}")
    print("Continuing without HF authentication...")


MODELS_DIR = os.path.join(OUTPUT_DIR, "saved_models_tabpfn")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots_tabpfn")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)


def validate_tabpfn_limits(X_train, y_train, ignore_limits=False):
    """
    Validates that the dataset fits within TabPFN's limits (samples, features, classes).
    """

    n_samples, n_features = X_train.shape
    n_classes = len(np.unique(y_train))

    issues = []

    if n_samples > 50000:
        msg = f"Training samples ({n_samples:,}) exceed TabPFN's pretraining limit of 50,000"
        issues.append(("warning", msg))

    if n_features > 2000:
        msg = f"Features ({n_features}) exceed 2,000. TabPFN will subsample 500 features per estimator."
        issues.append(("warning", msg))

    if n_classes > 10:
        msg = f"Classes ({n_classes}) exceed TabPFN's hard limit of 10"
        issues.append(("error", msg))

    for level, message in issues:
        if level == "error":
            print(f"❌ ERROR: {message}")
        else:
            print(f"⚠️  WARNING: {message}")

    has_errors = any(level == "error" for level, _ in issues)
    if has_errors and not ignore_limits:
        raise ValueError("TabPFN cannot handle more than 10 classes")

    return not has_errors


def create_tabpfn_classifier(config):
    """
    Creates and configures a TabPFNClassifier instance.
    """

    version = "v2.5"
    device = config.get("device", "auto")

    if device == "cuda" and not GPU_AVAILABLE:
        print("WARNING: CUDA not available. Using auto device selection.")
        device = "auto"

    clf_config = {k: v for k, v in config.items() if k != "version"}
    clf_config["device"] = device
    print(f"\n[TabPFN] Initializing with device: {device}")

    if version == "v2":
        return TabPFNClassifier.create_default_for_version(
            ModelVersion.V2, **clf_config
        )
    else:
        return TabPFNClassifier(**clf_config)


def train_tabpfn(
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
    Trains and evaluates a TabPFN model.
    """

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    try:
        validate_tabpfn_limits(
            X_train,
            y_train,
            ignore_limits=False,
        )

        class TabPFNWrapper:
            """Wrapper to allow easy instantiation via grid search utilities."""

            def __new__(cls, **kwargs):
                return create_tabpfn_classifier(kwargs)

        best_model = train_model(
            model_class_or_instance=TabPFNWrapper,
            model_name="TabPFN",
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

        if best_model is None:
            print("ERROR: No valid TabPFN model found during search.")
            return None, None

        return evaluate_and_save_model(
            model=best_model,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            model_name="TabPFN",
            dataset_name=dataset_name,
            scenario_name=scenario_name,
            feature_set_name=feature_set_name,
            models_dir=models_dir,
            plots_dir=plots_dir,
        )

    except Exception as e:
        print(f"\n[ERROR] Error running TabPFN: {e}")
        import traceback

        traceback.print_exc()
        return None, None
