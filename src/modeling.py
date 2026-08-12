"""
Machine learning pipeline construction for the heart disease project.

This module connects data preprocessing with a machine learning model
using a scikit-learn Pipeline.
"""

from pathlib import Path

import joblib
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from src.preprocessing import (
    load_data,
    split_features_target,
    build_preprocessing_pipeline,
)
from src.data_split import split_data


MODEL_OUTPUT_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "processed"
    / "baseline_model.joblib"
)


def build_model_pipeline(X_train: pd.DataFrame) -> Pipeline:
    """
    Build a complete preprocessing and model pipeline.

    The preprocessing component is fitted only on training data
    when the returned pipeline is trained.
    """
    preprocessor = build_preprocessing_pipeline(X_train)

    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    return pipeline


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Pipeline:
    """
    Train the complete machine learning pipeline.
    """
    pipeline = build_model_pipeline(X_train)

    pipeline.fit(X_train, y_train)

    return pipeline


def save_model(
    pipeline: Pipeline,
    path: Path = MODEL_OUTPUT_PATH,
) -> None:
    """
    Save the trained pipeline to disk.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipeline, path)

    print(f"Model saved to: {path}")


if __name__ == "__main__":
    df = load_data()

    X_train, X_test, y_train, y_test = split_data(df)

    pipeline = train_model(X_train, y_train)

    print("Model training completed successfully.")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")

    save_model(pipeline)