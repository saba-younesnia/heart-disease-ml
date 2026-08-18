"""
End-to-end training entry point.

Running this module performs the full pipeline:
load data -> split -> train -> evaluate on the held-out test set ->
save the trained model and evaluation artifacts to disk.
"""

import logging

from src.config import REPORTS_DIR
from src.preprocessing import load_data
from src.data_split import split_data
from src.modeling import train_model, save_model
from src.evaluation import evaluate_model, plot_confusion_matrix

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    logger.info("Starting end-to-end training pipeline.")

    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)

    pipeline = train_model(X_train, y_train)

    metrics, y_pred = evaluate_model(pipeline, X_test, y_test)

    confusion_matrix_path = REPORTS_DIR / "confusion_matrix.png"
    plot_confusion_matrix(y_test, y_pred, output_path=confusion_matrix_path)

    save_model(pipeline)

    logger.info("Pipeline finished successfully.")
    logger.info("Final test F1-macro: %.4f", metrics["f1_macro"])


if __name__ == "__main__":
    main()