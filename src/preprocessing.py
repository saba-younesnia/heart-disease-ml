"""
Data preprocessing pipeline for the heart disease prediction project.

This module provides reusable preprocessing utilities based on scikit-learn.
The preprocessing logic is designed to be fitted only on training data and
then reused consistently on validation, test, and inference data.
"""

from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "data.csv"


# Columns that should not be used as predictive features
ID_COLUMNS = ["id"]

# Target column
TARGET_COLUMN = "num"


def load_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """
    Load the raw dataset from disk.

    Parameters
    ----------
    path : Path
        Path to the raw CSV dataset.

    Returns
    -------
    pd.DataFrame
        Loaded dataset.
    """
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    return pd.read_csv(path)


def split_features_target(
    df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Separate input features from the target variable.

    Identifier columns are removed because they do not represent
    meaningful predictive features.
    """
    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' was not found in the dataset."
        )

    X = df.drop(columns=[target_column] + ID_COLUMNS, errors="ignore")
    y = df[target_column]

    return X, y


def build_preprocessing_pipeline(
    X: pd.DataFrame,
) -> ColumnTransformer:
    """
    Build the preprocessing pipeline for numerical and categorical features.

    Numerical features:
        - Median imputation
        - Standard scaling

    Categorical features:
        - Most-frequent imputation
        - One-hot encoding

    The transformer is fitted later on training data only.
    """
    numerical_features = X.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=["object", "string", "category", "bool"]
    ).columns.tolist()

    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numerical", numerical_pipeline, numerical_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )

    return preprocessor


def build_full_preprocessing_pipeline(
    X: pd.DataFrame,
) -> Pipeline:
    """
    Build the complete preprocessing pipeline.

    This wrapper allows preprocessing to be integrated directly
    into a machine-learning training pipeline.
    """
    preprocessor = build_preprocessing_pipeline(X)

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
        ]
    )


if __name__ == "__main__":
    df = load_data()

    X, y = split_features_target(df)

    pipeline = build_full_preprocessing_pipeline(X)

    X_transformed = pipeline.fit_transform(X)

    print("Preprocessing completed successfully.")
    print(f"Original feature count: {X.shape[1]}")
    print(f"Transformed feature count: {X_transformed.shape[1]}")
    print(f"Number of samples: {X_transformed.shape[0]}")