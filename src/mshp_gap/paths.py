"""Centralized path management for the MSHP coverage gap project."""
from pathlib import Path

# Project root (two levels up from this file)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MANUAL_DIR = DATA_DIR / "manual"
GEO_DIR = DATA_DIR / "geo"

# Other directories
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
CONFIGS_DIR = PROJECT_ROOT / "configs"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
INTERACTIVE_DIR = OUTPUTS_DIR / "interactive"
TABLES_DIR = OUTPUTS_DIR / "tables"
LOGS_DIR = PROJECT_ROOT / "logs"
DOCS_DIR = PROJECT_ROOT / "docs"


def ensure_dir(path: Path) -> Path:
    """Create directory if it doesn't exist and return the path."""
    path.mkdir(parents=True, exist_ok=True)
    return path


# Ensure directories exist on import
for d in [RAW_DIR, PROCESSED_DIR, MANUAL_DIR, GEO_DIR,
          FIGURES_DIR, INTERACTIVE_DIR, TABLES_DIR, LOGS_DIR]:
    ensure_dir(d)

