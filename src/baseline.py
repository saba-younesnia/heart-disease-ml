"""
Baseline model evaluation using stratified cross-validation.

This module evaluates the baseline machine learning pipeline
without using the held-out test set for model selection.
"""

from sklearn.model_selection import StratifiedKFold, cross_validate

from src.preprocessing import load_data, split_features_target
from src.modeling import build_model_pipeline


RANDOM_STATE = 42
N_SPLITS = 5


def run_cross_validation():
    """
    Evaluate the baseline model using stratified cross-validation.
    """

    # Load the complete dataset.
    df = load_data()

    # Separate features and target.
    X, y = split_features_target(df)

    # Create a fresh model pipeline.
    model = build_model_pipeline(X)

    # Preserve the target class distribution in each fold.
    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    scoring = {
        "accuracy": "accuracy",
        "precision_macro": "precision_macro",
        "recall_macro": "recall_macro",
        "f1_macro": "f1_macro",
    }

    results = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=scoring,
        return_train_score=False,
    )

    print("\n=== Baseline Model: 5-Fold Cross-Validation ===")

    for metric_name in scoring:
        scores = results[f"test_{metric_name}"]

        print(
            f"{metric_name}: "
            f"{scores.mean():.4f} "
            f"+/- {scores.std():.4f}"
        )

    return results


if __name__ == "__main__":
    run_cross_validation()