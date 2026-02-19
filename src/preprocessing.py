from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd


from model_utils import ALL_NUMERIC_COLS


def get_feature_types(X_train):
    """
    Identifies numeric and categorical features in the training data.
    """
    numeric_cols = [c for c in ALL_NUMERIC_COLS if c in X_train.columns]
    categorical_cols = [c for c in X_train.columns if c not in numeric_cols]
    return numeric_cols, categorical_cols


def build_linear_preprocessor(X_train):
    """
    Builds a preprocessor for linear models (scaling + one-hot encoding).
    """

    numeric_cols = [c for c in ALL_NUMERIC_COLS if c in X_train.columns]
    categorical_cols = [c for c in X_train.columns if c not in numeric_cols]

    print(
        f"\n[Linear Preprocessor] Numeric: {len(numeric_cols)}, Categorical: {len(categorical_cols)}"
    )

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    transformers = []
    if numeric_cols:
        transformers.append(("num", numeric_transformer, numeric_cols))
    if categorical_cols:
        transformers.append(("cat", categorical_transformer, categorical_cols))

    preprocessor = ColumnTransformer(transformers=transformers)
    return preprocessor, categorical_cols


def build_tree_preprocessor(X_train):
    """
    Builds a preprocessor for tree models (no scaling + one-hot encoding).
    """

    numeric_cols = [c for c in ALL_NUMERIC_COLS if c in X_train.columns]
    categorical_cols = [c for c in X_train.columns if c not in numeric_cols]

    print(
        f"\n[Tree Preprocessor] Numeric: {len(numeric_cols)}, Categorical: {len(categorical_cols)}"
    )

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    transformers = []
    if numeric_cols:
        transformers.append(("num", numeric_transformer, numeric_cols))
    if categorical_cols:
        transformers.append(("cat", categorical_transformer, categorical_cols))

    preprocessor = ColumnTransformer(transformers=transformers)
    return preprocessor, categorical_cols


class PandasPreservingTransformer(BaseEstimator, TransformerMixin):
    """
    A transformer that ensures pandas DataFrame output.
    """

    def __init__(self, numeric_cols=None, categorical_cols=None):
        self.numeric_cols = numeric_cols if numeric_cols is not None else []
        self.categorical_cols = categorical_cols if categorical_cols is not None else []
        self.columns_ = None

    def fit(self, X, y=None):
        self.columns_ = X.columns
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            return X.copy()
        return pd.DataFrame(
            X, columns=self.columns_, index=X.index if hasattr(X, "index") else None
        )


class CatBoostCategoricalTransformer(PandasPreservingTransformer):
    def transform(self, X):
        """
        Transforms categorical columns for CatBoost (converts to category dtype).
        """
        X = super().transform(X)
        if self.categorical_cols:
            for col in self.categorical_cols:
                if col in X.columns:
                    X[col] = (
                        X[col]
                        .astype(object)
                        .fillna("Missing")
                        .astype(str)
                        .astype("category")
                    )
        return X


class ConsistentCategoricalTransformer(PandasPreservingTransformer):
    def __init__(self, numeric_cols=None, categorical_cols=None):
        super().__init__(numeric_cols, categorical_cols)
        self.cat_dtypes_ = {}

    def fit(self, X, y=None):
        super().fit(X, y)
        if self.categorical_cols:
            for col in self.categorical_cols:
                if col in X.columns:
                    unique_cats = [x for x in X[col].unique() if pd.notna(x)]
                    self.cat_dtypes_[col] = pd.api.types.CategoricalDtype(
                        categories=unique_cats, ordered=False
                    )
        return self

    def transform(self, X):
        X = super().transform(X)
        if self.categorical_cols:
            for col in self.categorical_cols:
                if col in X.columns and col in self.cat_dtypes_:
                    X[col] = X[col].astype(self.cat_dtypes_[col])
        return X


def build_catboost_preprocessor(X_train):
    """
    Builds a preprocessor for CatBoost models.
    """
    numeric_cols = [c for c in ALL_NUMERIC_COLS if c in X_train.columns]
    categorical_cols = [c for c in X_train.columns if c not in numeric_cols]

    print(
        f"\n[CatBoost Preprocessor] Numeric: {len(numeric_cols)}, Categorical: {len(categorical_cols)}"
    )

    preprocessor = CatBoostCategoricalTransformer(numeric_cols, categorical_cols)

    return preprocessor, categorical_cols


def build_xgboost_preprocessor(X_train):
    """
    Builds a preprocessor for XGBoost models.
    """
    numeric_cols = [c for c in ALL_NUMERIC_COLS if c in X_train.columns]
    categorical_cols = [c for c in X_train.columns if c not in numeric_cols]

    print(
        f"\n[XGBoost Preprocessor] Numeric: {len(numeric_cols)}, Categorical: {len(categorical_cols)}"
    )

    preprocessor = ConsistentCategoricalTransformer(numeric_cols, categorical_cols)

    return preprocessor, categorical_cols


def build_lightgbm_preprocessor(X_train):
    """
    Builds a preprocessor for LightGBM models.
    """
    numeric_cols = [c for c in ALL_NUMERIC_COLS if c in X_train.columns]
    categorical_cols = [c for c in X_train.columns if c not in numeric_cols]

    print(
        f"\n[LightGBM Preprocessor] Numeric: {len(numeric_cols)}, Categorical: {len(categorical_cols)}"
    )

    preprocessor = ConsistentCategoricalTransformer(numeric_cols, categorical_cols)

    return preprocessor, categorical_cols


def ensure_categorical_strings(df, numeric_cols):
    """
    Ensures that categorical columns are valid strings.
    """
    cat_cols = [c for c in df.columns if c not in numeric_cols]

    if not cat_cols:
        return df

    for col in cat_cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            s = df[col].astype(object)
            mask = s.notna()
            s[mask] = s[mask].astype(str)
            df[col] = s

    return df


def prepare_feature_sets(
    df_train, df_val, df_test, target_col, pdc_cols, include_pdc=True
):
    """
    Prepares training, validation, and test datasets, optionally dropping PDC columns.
    """

    y_train = df_train[target_col]
    y_val = df_val[target_col]
    y_test = df_test[target_col]

    if include_pdc:
        X_train = df_train.drop(columns=[target_col])
        X_val = df_val.drop(columns=[target_col])
        X_test = df_test.drop(columns=[target_col])
    else:
        drop_cols_train = [c for c in pdc_cols if c in df_train.columns] + [target_col]
        drop_cols_val = [c for c in pdc_cols if c in df_val.columns] + [target_col]
        drop_cols_test = [c for c in pdc_cols if c in df_test.columns] + [target_col]

        if drop_cols_train or drop_cols_val or drop_cols_test:
            all_drop = sorted(
                set(drop_cols_train + drop_cols_val + drop_cols_test) - {target_col}
            )
            if all_drop:
                print(f"Dropped PDC columns: {all_drop}")

        X_train = df_train.drop(columns=drop_cols_train)
        X_val = df_val.drop(columns=drop_cols_val)
        X_test = df_test.drop(columns=drop_cols_test)

    X_train = ensure_categorical_strings(X_train, ALL_NUMERIC_COLS)
    X_val = ensure_categorical_strings(X_val, ALL_NUMERIC_COLS)
    X_test = ensure_categorical_strings(X_test, ALL_NUMERIC_COLS)

    return X_train, X_val, X_test, y_train, y_val, y_test


def get_ranked_features(ranking_path, X_train, top_k=20):
    """
    Loads top-k ranked features from a file.
    """

    import pandas as pd

    print("\n" + "-" * 50)
    print(f"Loading SHAP ranking from: {ranking_path}")
    print("-" * 50)

    ranking_df = pd.read_csv(ranking_path)

    feature_name_col = ranking_df.columns[0]
    ranked_features = ranking_df[feature_name_col].tolist()

    top_features = ranked_features[:top_k]
    top_features_in_X = [f for f in top_features if f in X_train.columns]

    if not top_features_in_X:
        print("Warning: No ranked features found in X_train")
        return []

    if len(top_features_in_X) < top_k:
        print(
            f"Warning: Only {len(top_features_in_X)} of top-{top_k} ranked features are present"
        )

    print(f"\nUsing ranked top-{len(top_features_in_X)} features:")
    print(top_features_in_X)

    return top_features_in_X
