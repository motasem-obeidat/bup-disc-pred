from model_utils import (
    SEED,
    ALL_NUMERIC_COLS,
    OUTPUT_DIR,
)

from training import evaluate_and_save_model, train_model

import copy
import os
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import (
    StandardScaler,
    QuantileTransformer,
    OrdinalEncoder,
    LabelEncoder,
)
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.class_weight import compute_class_weight

from tabm import TabM
from rtdl_num_embeddings import (
    PiecewiseLinearEmbeddings,
    PeriodicEmbeddings,
    LinearReLUEmbeddings,
    compute_bins,
)

torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[TabM] Global Device: {DEVICE}")

MODELS_DIR = os.path.join(OUTPUT_DIR, "saved_models_tabm")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots_tabm")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)


class TabMClassifier(ClassifierMixin, BaseEstimator):
    """
    A sklearn-compatible wrapper for the TabM model.
    """

    _estimator_type = "classifier"

    def __init__(
        self,
        n_blocks=3,
        d_block=512,
        lr=0.002,
        weight_decay=0.0003,
        emb_type="piecewise",
        d_embedding=24,
        n_bins=32,
        k=32,
        epochs=50,
        batch_size=256,
        patience=5,
        dropout=0.1,
        periodic_n_frequencies=16,
        periodic_lite=False,
        debug_preprocessing=False,
        scaler_type="standard",
        class_weight=None,
    ):
        """
        Initializes the TabMClassifier.
        """

        self.n_blocks = n_blocks
        self.d_block = d_block
        self.lr = lr
        self.weight_decay = weight_decay
        self.emb_type = emb_type
        self.d_embedding = d_embedding
        self.n_bins = n_bins
        self.k = k
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.dropout = dropout

        # fitted attrs
        self.classes_ = None
        self.label_encoder_ = None
        self.model_ = None
        self.num_imputer_ = None
        self.num_scaler_ = None
        self.cat_encoder_ = None
        self.bins_ = None
        self.num_cols_idx_ = None
        self.cat_cols_idx_ = None
        self.cat_cardinalities_ = None

        self.periodic_n_frequencies = periodic_n_frequencies
        self.periodic_lite = periodic_lite

        self.debug_preprocessing = debug_preprocessing
        self.scaler_type = scaler_type
        self.class_weight = class_weight

    def _get_embedding_module(self, n_num_features):
        """
        Returns the appropriate embedding module based on configuration.
        """
        if self.emb_type == "piecewise":
            return PiecewiseLinearEmbeddings(
                bins=self.bins_,
                d_embedding=self.d_embedding,
                activation=True,
                version="B",
            )
        elif self.emb_type == "periodic":
            return PeriodicEmbeddings(
                n_features=n_num_features,
                d_embedding=self.d_embedding,
                lite=self.periodic_lite,
                n_frequencies=self.periodic_n_frequencies,
            )
        elif self.emb_type == "linear":
            return LinearReLUEmbeddings(
                n_features=n_num_features, d_embedding=self.d_embedding
            )
        return None

    def _prep_num(self, X, fit=False):
        """
        Preprocesses numeric features (scaling, imputation).
        """
        X_num_raw = X.iloc[:, self.num_cols_idx_]

        if fit:
            is_constant = X_num_raw.nunique() <= 1
            self.constant_cols_idx_ = is_constant[is_constant].index.tolist()
            if self.constant_cols_idx_:
                print(f"Dropping constant numeric columns: {self.constant_cols_idx_}")
                X_num_raw = X_num_raw.drop(columns=self.constant_cols_idx_)
        else:
            if hasattr(self, "constant_cols_idx_") and self.constant_cols_idx_:
                X_num_raw = X_num_raw.drop(columns=self.constant_cols_idx_)

        if self.debug_preprocessing:
            print("\n[NUMERIC PREPROCESSING]")
            print(f"Fit mode: {fit}")
            print("Columns:", list(X_num_raw.columns))
            print("Raw sample (first 5 rows):")
            print(X_num_raw.head())

        X_num_raw_values = X_num_raw.values.astype(np.float32)

        if fit:
            self.num_imputer_ = SimpleImputer(strategy="median")
            X_num = self.num_imputer_.fit_transform(X_num_raw_values).astype(np.float32)
            if self.debug_preprocessing:
                print("Fitted median imputer")
                print(f"Imputer statistics (medians): {self.num_imputer_.statistics_}")
        else:
            X_num = self.num_imputer_.transform(X_num_raw_values).astype(np.float32)

        if fit:
            if self.scaler_type == "standard":
                self.num_scaler_ = StandardScaler()
                self.num_scaler_.fit(X_num)
            elif self.scaler_type == "quantile":
                n_samples = X_num.shape[0]
                n_quantiles = max(min(n_samples // 30, 1000), 10)

                self.num_scaler_ = QuantileTransformer(
                    n_quantiles=n_quantiles,
                    output_distribution="normal",
                    subsample=1_000_000_000,
                    random_state=SEED,
                )

                X_fit = X_num + np.random.RandomState(SEED).normal(
                    0.0, 1e-5, X_num.shape
                ).astype(X_num.dtype)
                self.num_scaler_.fit(X_fit)
            else:
                raise ValueError(f"Unknown scaler_type: {self.scaler_type}")

            X_num = self.num_scaler_.transform(X_num)
            if self.debug_preprocessing:
                print(f"Using scaler: {self.scaler_type}")
                if self.scaler_type == "standard":
                    print("Scaler mean:", self.num_scaler_.mean_)
                    print("Scaler scale:", self.num_scaler_.scale_)
                elif self.scaler_type == "quantile":
                    print(f"N quantiles: {n_quantiles}")
        else:
            X_num = self.num_scaler_.transform(X_num)

        if self.debug_preprocessing:
            print("Transformed sample (first 5 rows):")
            print(X_num[:5])
            print("Shape:", X_num.shape)

        return X_num

    def _prep_cat(self, X, fit=False):
        """
        Preprocesses categorical features (ordinal encoding).
        """
        if self.cat_cols_idx_ is None or len(self.cat_cols_idx_) == 0:
            if self.debug_preprocessing:
                print("\n[CATEGORICAL PREPROCESSING]")
                print("No categorical columns detected.")
            return None

        X_cat_raw = X.iloc[:, self.cat_cols_idx_].astype(str)

        if self.debug_preprocessing:
            print("\n[CATEGORICAL PREPROCESSING]")
            print(f"Fit mode: {fit}")
            print("Columns:", list(X_cat_raw.columns))
            print("Raw sample (first 5 rows):")
            print(X_cat_raw.head())

        if fit:
            self.cat_encoder_ = OrdinalEncoder(
                handle_unknown="use_encoded_value", unknown_value=-1
            )
            X_cat_enc = self.cat_encoder_.fit_transform(X_cat_raw.values).astype(
                np.int64
            )

            if self.debug_preprocessing:
                print("Learned categories:")
                for col, cats in zip(X_cat_raw.columns, self.cat_encoder_.categories_):
                    print(f"  {col}: {cats}")

            for i in range(X_cat_enc.shape[1]):
                n_cats = len(self.cat_encoder_.categories_[i])
                X_cat_enc[X_cat_enc[:, i] == -1, i] = n_cats
            X_cat = X_cat_enc

            self.cat_cardinalities_ = [
                len(cats) + 1 for cats in self.cat_encoder_.categories_
            ]

            if self.debug_preprocessing:
                print(
                    "Cardinalities (including unknown slot):", self.cat_cardinalities_
                )

        else:
            X_cat_enc = self.cat_encoder_.transform(X_cat_raw.values).astype(np.int64)
            for i in range(X_cat_enc.shape[1]):
                n_cats = len(self.cat_encoder_.categories_[i])
                X_cat_enc[X_cat_enc[:, i] == -1, i] = n_cats
            X_cat = X_cat_enc

        if self.debug_preprocessing:
            print("Encoded sample (first 5 rows):")
            print(X_cat[:5])
            print(
                "Min value:",
                X_cat.min(),
                "(should be 0 for known categories)",
            )
            print("Max value:", X_cat.max())
            print("Shape:", X_cat.shape)

        return X_cat

    def fit(self, X, y, X_val=None, y_val=None):
        """
        Fits the TabM model to the training data.
        """

        self.label_encoder_ = LabelEncoder()
        y_enc = self.label_encoder_.fit_transform(np.asarray(y))
        self.classes_ = self.label_encoder_.classes_
        d_out = len(self.classes_)

        numeric_names = [c for c in ALL_NUMERIC_COLS if c in X.columns]
        cat_names = [c for c in X.columns if c not in numeric_names]

        self.num_cols_idx_ = [X.columns.get_loc(c) for c in numeric_names]
        self.cat_cols_idx_ = [X.columns.get_loc(c) for c in cat_names]

        self.feature_names_in_ = X.columns.tolist()

        X_num = self._prep_num(X, fit=True)

        if self.emb_type == "piecewise":
            self.bins_ = compute_bins(torch.as_tensor(X_num), n_bins=self.n_bins)

        X_cat = self._prep_cat(X, fit=True)

        val_loader = None
        if X_val is not None and y_val is not None:
            X_val_num = self._prep_num(X_val, fit=False)
            X_val_cat = self._prep_cat(X_val, fit=False)

            y_val_enc = self.label_encoder_.transform(np.asarray(y_val))

            if X_val_cat is None:
                val_dataset = TensorDataset(
                    torch.tensor(X_val_num, dtype=torch.float32),
                    torch.tensor(y_val_enc, dtype=torch.long),
                )
                val_loader = DataLoader(
                    val_dataset, batch_size=self.batch_size, shuffle=False
                )
            else:
                val_dataset = TensorDataset(
                    torch.tensor(X_val_num, dtype=torch.float32),
                    torch.tensor(X_val_cat, dtype=torch.long),
                    torch.tensor(y_val_enc, dtype=torch.long),
                )
                val_loader = DataLoader(
                    val_dataset, batch_size=self.batch_size, shuffle=False
                )

        emb_module = self._get_embedding_module(len(self.num_cols_idx_))

        cat_cardinalities = None if (X_cat is None) else self.cat_cardinalities_

        self.model_ = TabM.make(
            n_num_features=len(self.num_cols_idx_),
            cat_cardinalities=cat_cardinalities,
            d_out=d_out,
            num_embeddings=emb_module,
            k=self.k,
            n_blocks=self.n_blocks,
            d_block=self.d_block,
            dropout=self.dropout,
        ).to(DEVICE)

        optimizer = torch.optim.AdamW(
            self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        weight_tensor = None
        if self.class_weight == "balanced":
            classes = np.unique(y_enc)
            weights = compute_class_weight(
                class_weight="balanced", classes=classes, y=y_enc
            )
            weight_tensor = torch.tensor(weights, dtype=torch.float32, device=DEVICE)
            if self.debug_preprocessing:
                print(f"[TabM] Class weights: {weights}")

        loss_fn = nn.CrossEntropyLoss(weight=weight_tensor)

        if X_cat is None:
            dataset = TensorDataset(
                torch.tensor(X_num, dtype=torch.float32),
                torch.tensor(y_enc, dtype=torch.long),
            )
            loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        else:
            dataset = TensorDataset(
                torch.tensor(X_num, dtype=torch.float32),
                torch.tensor(X_cat, dtype=torch.long),
                torch.tensor(y_enc, dtype=torch.long),
            )
            loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        best_val_score = -1.0
        best_state = None
        no_improve = 0

        for epoch in range(self.epochs):
            self.model_.train()

            if X_cat is None:
                for xb_num, yb in loader:
                    xb_num, yb = xb_num.to(DEVICE), yb.to(DEVICE)
                    optimizer.zero_grad()

                    pred = self.model_(xb_num)

                    k = self.model_.k
                    y_k = yb.unsqueeze(1).expand(-1, k).reshape(-1)
                    loss = loss_fn(pred.reshape(-1, d_out), y_k)

                    loss.backward()
                    optimizer.step()
            else:
                for xb_num, xb_cat, yb in loader:
                    xb_num, xb_cat, yb = (
                        xb_num.to(DEVICE),
                        xb_cat.to(DEVICE),
                        yb.to(DEVICE),
                    )
                    optimizer.zero_grad()

                    pred = self.model_(xb_num, xb_cat)

                    k = self.model_.k
                    y_k = yb.unsqueeze(1).expand(-1, k).reshape(-1)
                    loss = loss_fn(pred.reshape(-1, d_out), y_k)

                    loss.backward()
                    optimizer.step()

            if val_loader is not None:
                self.model_.eval()
                val_probs = []
                val_targets_enc = []
                with torch.no_grad():
                    if X_cat is None:
                        for xb_num, yb in val_loader:
                            xb_num = xb_num.to(DEVICE)
                            pred = self.model_(xb_num)
                            probs = F.softmax(pred, dim=-1).mean(dim=1)
                            val_probs.append(probs.cpu())
                            val_targets_enc.append(yb)
                    else:
                        for xb_num, xb_cat, yb in val_loader:
                            xb_num, xb_cat = xb_num.to(DEVICE), xb_cat.to(DEVICE)
                            pred = self.model_(xb_num, xb_cat)
                            probs = F.softmax(pred, dim=-1).mean(dim=1)
                            val_probs.append(probs.cpu())
                            val_targets_enc.append(yb)

                val_probs = torch.cat(val_probs).numpy()
                val_targets_enc = torch.cat(val_targets_enc).numpy()
                val_targets_orig = self.label_encoder_.inverse_transform(
                    val_targets_enc
                )

                if 1 in self.classes_:
                    pos_idx = list(self.classes_).index(1)
                    y_bin = (val_targets_orig == 1).astype(int)
                    current_score = roc_auc_score(y_bin, val_probs[:, pos_idx])
                else:
                    current_score = 0.0

                if current_score > best_val_score:
                    best_val_score = current_score
                    best_state = copy.deepcopy(self.model_.state_dict())
                    self.best_epoch_ = epoch + 1
                    no_improve = 0
                else:
                    no_improve += 1

                if no_improve >= self.patience:
                    print(
                        f"Early stopping at epoch {epoch + 1} (Best: {self.best_epoch_})"
                    )
                    break

        if best_state is not None:
            self.model_.load_state_dict(best_state)

        return self

    def predict_proba(self, X):
        """
        Predicts class probabilities for the input data.
        """

        self.model_.eval()

        if not isinstance(X, pd.DataFrame):
            if hasattr(self, "feature_names_in_"):
                X = pd.DataFrame(X, columns=self.feature_names_in_)
            else:
                pass

        X_num = self._prep_num(X, fit=False)
        X_cat = self._prep_cat(X, fit=False)

        all_probs = []

        with torch.no_grad():
            if X_cat is None:
                dataset = TensorDataset(torch.tensor(X_num, dtype=torch.float32))
                loader = DataLoader(
                    dataset, batch_size=self.batch_size * 2, shuffle=False
                )
                for (xb_num,) in loader:
                    xb_num = xb_num.to(DEVICE)
                    pred = self.model_(xb_num)
                    probs = F.softmax(pred, dim=-1).mean(dim=1)
                    all_probs.append(probs.cpu())
            else:
                dataset = TensorDataset(
                    torch.tensor(X_num, dtype=torch.float32),
                    torch.tensor(X_cat, dtype=torch.long),
                )
                loader = DataLoader(
                    dataset, batch_size=self.batch_size * 2, shuffle=False
                )
                for xb_num, xb_cat in loader:
                    xb_num, xb_cat = xb_num.to(DEVICE), xb_cat.to(DEVICE)
                    pred = self.model_(xb_num, xb_cat)
                    probs = F.softmax(pred, dim=-1).mean(dim=1)
                    all_probs.append(probs.cpu())

        return torch.cat(all_probs).numpy()

    def predict(self, X):
        """
        Predicts class labels for the input data.
        """

        probs = self.predict_proba(X)
        pred_idx = np.argmax(probs, axis=1)
        return self.label_encoder_.inverse_transform(pred_idx)


def train_tabm(
    X_train_base,
    X_val_base,
    X_test_base,
    y_train,
    y_val,
    y_test,
    dataset_name,
    scenario_name,
    feature_set_main_name="BASELINE_FEATURES",
    top_features=None,
    models_dir=MODELS_DIR,
    plots_dir=PLOTS_DIR,
):
    """
    Trains and evaluates a TabM model.
    """

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    fit_params = {
        "X_val": X_val_base,
        "y_val": y_val,
    }

    best_tabm_model = train_model(
        model_class_or_instance=TabMClassifier,
        model_name="TabM",
        dataset_name=dataset_name,
        scenario_name=scenario_name,
        feature_set_name=feature_set_main_name,
        X_train=X_train_base,
        y_train=y_train,
        X_val=X_val_base,
        y_val=y_val,
        param_grid={},
        fit_params=fit_params,
        is_instance=False,
    )

    if best_tabm_model is None:
        print("TabM failed to produce a valid model.")
        return None, None

    return evaluate_and_save_model(
        model=best_tabm_model,
        X_train=X_train_base,
        y_train=y_train,
        X_val=X_val_base,
        y_val=y_val,
        X_test=X_test_base,
        y_test=y_test,
        model_name="TabM",
        dataset_name=dataset_name,
        scenario_name=scenario_name,
        feature_set_name=feature_set_main_name,
        models_dir=models_dir,
        plots_dir=plots_dir,
    )
