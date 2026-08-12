"""
Utilities for splitting datasets into training and test sets.

The split is performed before fitting any preprocessing component
to prevent data leakage from the test set.
"""

import pandas as pd

from sklearn.model_selection import train_test_split


TARGET_COLUMN = "num"
TEST_SIZE = 0.20
RANDOM_STATE = 42


def split_data(
    df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split the dataset into stratified training and test sets.

    Parameters
    ----------
    df : pd.DataFrame
        Complete input dataset.

    target_column : str
        Name of the target column.

    test_size : float
        Proportion of the dataset assigned to the test set.

    random_state : int
        Seed used to make the split reproducible.

    Returns
    -------
    X_train : pd.DataFrame
        Training features.

    X_test : pd.DataFrame
        Test features.

    y_train : pd.Series
        Training target.

    y_test : pd.Series
        Test target.
    """
    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' was not found."
        )

    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    from src.preprocessing import load_data

    df = load_data()

    X_train, X_test, y_train, y_test = split_data(df)

    print("Data split completed successfully.")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Training target distribution:\n{y_train.value_counts().sort_index()}")
    print(f"Test target distribution:\n{y_test.value_counts().sort_index()}")