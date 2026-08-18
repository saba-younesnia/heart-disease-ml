"""
Utilities for splitting datasets into training and test sets.

The split is performed before fitting any preprocessing component
to prevent data leakage from the test set.
"""

import logging

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import TARGET_COLUMN, TEST_SIZE, RANDOM_STATE
from src.preprocessing import split_features_target, load_data

logger = logging.getLogger(__name__)


def split_data(
    df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split the dataset into stratified training and test sets.

    Feature/target separation (including dropping id columns) is
    delegated to `split_features_target` so this logic exists in
    exactly one place in the codebase.
    """
    X, y = split_features_target(df, target_column=target_column)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    logger.info(
        "Split data into %d train / %d test samples (stratified).",
        len(X_train),
        len(X_test),
    )

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)

    logger.info("Data split completed successfully.")
    logger.info("Training target distribution:\n%s", y_train.value_counts().sort_index())
    logger.info("Test target distribution:\n%s", y_test.value_counts().sort_index())