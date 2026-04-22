from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
ARCHIVE_DIR = PROJECT_ROOT / "archive"

RAE2822_CASE_DIR = DATA_DIR / "rae2822" / "angvar_sa"
TRADEOFF_DATA_PATH = (
    ARCHIVE_DIR / "tradestudies" / "n2412" / "n2412_optimisation.csv"
)

MODELS_DIR = PROJECT_ROOT / "models"
CP_MODEL_DIR = MODELS_DIR / "cp"
TRADEOFF_MODEL_DIR = MODELS_DIR / "tradeoff"

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_DIR = REPORTS_DIR / "metrics"
