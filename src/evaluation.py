from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
)


def evaluate_model(model, X_test, y_test):
    """
    Evaluate a trained classification model on the test set.

    Returns:
        dict: Main evaluation metrics.
    """

    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_macro": precision_score(
            y_test,
            y_pred,
            average="macro",
            zero_division=0,
        ),
        "recall_macro": recall_score(
            y_test,
            y_pred,
            average="macro",
            zero_division=0,
        ),
        "f1_macro": f1_score(
            y_test,
            y_pred,
            average="macro",
            zero_division=0,
        ),
    }

    print("\n=== Model Evaluation ===")

    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")

    print("\n=== Classification Report ===")

    print(
        classification_report(
            y_test,
            y_pred,
            zero_division=0,
        )
    )

    return metrics, y_pred


def plot_confusion_matrix(y_test, y_pred, output_path=None):
    """
    Plot and optionally save the confusion matrix.
    """

    cm = confusion_matrix(y_test, y_pred)

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm
    )

    display.plot(cmap="Blues")

    plt.title("Confusion Matrix")
    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150)

    plt.show()