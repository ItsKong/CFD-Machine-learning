from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"

RAE2822_CASE_DIR = DATA_DIR / "rae2822" / "angvar_sa"
DENSE_RAE2822_CASE_DIR = DATA_DIR / "rae2822_dense_su2" / "angvar_sa"
AOA_MACH_RAE2822_CASE_DIR = DATA_DIR / "rae2822_aoa_mach_su2" / "extended_sa"

MODELS_DIR = PROJECT_ROOT / "models"

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_DIR = REPORTS_DIR / "metrics"
