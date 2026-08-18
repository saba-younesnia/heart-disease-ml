"""
Centralized configuration for the heart disease prediction project.

Every module in this project should import constants from here instead
of redefining them locally. This is the single source of truth for
paths, column names, and reproducibility settings.
"""

from pathlib import Path

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "data.csv"
MODEL_OUTPUT_PATH = PROJECT_ROOT / "models" / "baseline_model.joblib"
REPORTS_DIR = PROJECT_ROOT / "reports"

# ---------------------------------------------------------------------
# Column definitions
# ---------------------------------------------------------------------
TARGET_COLUMN = "num"
ID_COLUMNS = ["id"]

# Columns where a value of 0 is not physiologically valid and should
# be treated as a missing-value encoding instead of a real measurement.
ZERO_AS_MISSING_COLUMNS = ["trestbps", "chol"]

# ---------------------------------------------------------------------
# Reproducibility / experiment settings
# ---------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.20
N_SPLITS = 5