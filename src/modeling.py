"""
Machine learning pipeline construction for the heart disease project.

This module connects data preprocessing with a machine learning model
using a scikit-learn Pipeline.
"""

import logging

import joblib
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from src.config import MODEL_OUTPUT_PATH, RANDOM_STATE
from src.preprocessing import build_preprocessing_pipeline
from src.preprocessing import clean_domain_issues
from sklearn.preprocessing import FunctionTransformer

logger = logging.getLogger(__name__)


def build_model_pipeline(X_train: pd.DataFrame) -> Pipeline:
    """
    Build a complete preprocessing and model pipeline.

    The preprocessing component (including domain cleaning) is fitted
    only on training data when the returned pipeline is trained.
    """
    preprocessor = build_preprocessing_pipeline(X_train)

    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )

    pipeline = Pipeline(
        steps=[
            ("domain_cleaning", FunctionTransformer(clean_domain_issues)),
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    return pipeline


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    """
    Train the complete machine learning pipeline.
    """
    pipeline = build_model_pipeline(X_train)
    pipeline.fit(X_train, y_train)
    logger.info("Model trained on %d samples.", len(X_train))
    return pipeline


def save_model(pipeline: Pipeline, path=MODEL_OUTPUT_PATH) -> None:
    """
    Save the trained pipeline to disk.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
    logger.info("Model saved to: %s", path)


if __name__ == "__main__":
    from src.preprocessing import load_data
    from src.data_split import split_data

    logging.basicConfig(level=logging.INFO)

    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)

    pipeline = train_model(X_train, y_train)
    save_model(pipeline)