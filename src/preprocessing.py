"""
Data preprocessing pipeline for the heart disease prediction project.

This module provides reusable preprocessing utilities based on scikit-learn.
The preprocessing logic is designed to be fitted only on training data and
then reused consistently on validation, test, and inference data.
"""

import logging

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer

from src.config import (
    RAW_DATA_PATH,
    TARGET_COLUMN,
    ID_COLUMNS,
    ZERO_AS_MISSING_COLUMNS,
)

logger = logging.getLogger(__name__)


def load_data(path=RAW_DATA_PATH) -> pd.DataFrame:
    """
    Load the raw dataset from disk.
    """
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)
    logger.info("Loaded dataset with shape %s from %s", df.shape, path)
    return df


def split_features_target(
    df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Separate input features from the target variable.

    Identifier columns are removed here because they are not
    meaningful predictive features. This is the ONLY place in the
    project where id columns should be dropped — all downstream
    code (data_split, modeling) must route through this function.
    """
    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' was not found in the dataset."
        )

    X = df.drop(columns=[target_column] + ID_COLUMNS, errors="ignore")
    y = df[target_column]

    return X, y


def clean_domain_issues(X: pd.DataFrame) -> pd.DataFrame:
    """
    Fix values that are not real missing values (NaN) but are still
    invalid from a clinical standpoint. Converting them to NaN allows
    the imputers downstream to handle them consistently.

    - trestbps == 0 / chol == 0: physiologically impossible, treated
      as a missing-value encoding.
    - oldpeak < 0: not a valid clinical measurement.
    """
    X = X.copy()

    for col in ZERO_AS_MISSING_COLUMNS:
        if col in X.columns:
            n_bad = (X[col] == 0).sum()
            if n_bad > 0:
                logger.info("Converting %d zero values in '%s' to NaN", n_bad, col)
            X.loc[X[col] == 0, col] = pd.NA

    if "oldpeak" in X.columns:
        n_bad = (X["oldpeak"] < 0).sum()
        if n_bad > 0:
            logger.info("Converting %d negative values in 'oldpeak' to NaN", n_bad)
        X.loc[X["oldpeak"] < 0, "oldpeak"] = pd.NA

    return X


def build_preprocessing_pipeline(X: pd.DataFrame) -> ColumnTransformer:
    """
    Build the preprocessing pipeline for numerical and categorical features.

    Numerical features:
        - Median imputation, with a missing-value indicator flag
          (missingness itself may carry information, e.g. `ca`/`thal`
          being unmeasured in certain source hospitals).
        - Standard scaling.

    Categorical features:
        - Most-frequent imputation.
        - One-hot encoding.
    """
    numerical_features = X.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=["object", "string", "category", "bool"]
    ).columns.tolist()

    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
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


def build_full_preprocessing_pipeline(X: pd.DataFrame) -> Pipeline:
    """
    Build the complete preprocessing pipeline, including domain cleaning.

    This wrapper allows preprocessing to be integrated directly
    into a machine-learning training pipeline.
    """
    preprocessor = build_preprocessing_pipeline(X)

    return Pipeline(
        steps=[
            ("domain_cleaning", FunctionTransformer(clean_domain_issues)),
            ("preprocessor", preprocessor),
        ]
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    df = load_data()
    X, y = split_features_target(df)

    pipeline = build_full_preprocessing_pipeline(X)
    X_transformed = pipeline.fit_transform(X)

    logger.info("Preprocessing completed successfully.")
    logger.info("Original feature count: %d", X.shape[1])
    logger.info("Transformed feature count: %d", X_transformed.shape[1])
    logger.info("Number of samples: %d", X_transformed.shape[0])