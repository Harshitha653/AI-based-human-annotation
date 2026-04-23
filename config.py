"""Project paths and environment loading (call as early as possible in each entry script)."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent

# Load `.env` from project root so GROQ_API_KEY works without Windows UI setup.
load_dotenv(PROJECT_ROOT / ".env")

ECGEN_XML_DIR = PROJECT_ROOT / "ecgen-radiology"
PROCESSED_CSV = PROJECT_ROOT / "processed_reports.csv"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
AUTO_LABELED_CSV = OUTPUTS_DIR / "auto_labeled.csv"
REVIEW_QUEUE_CSV = OUTPUTS_DIR / "review_queue.csv"
KEYWORDS_JSON = OUTPUTS_DIR / "chexpert_keywords_aggregate.json"
DATA_DIR = PROJECT_ROOT / "data"
HUMAN_GOLD_CSV = DATA_DIR / "human_gold.csv"
GOLD_SUBSET_CSV = DATA_DIR / "gold_subset.csv"

# Optional external dataset roots (set in `.env`)
# Example:
#   CHEXPERT_ROOT="D:/datasets/CheXpert-v1.0-small"
CHEXPERT_ROOT = Path(os.environ.get("CHEXPERT_ROOT", "")).expanduser().resolve() if os.environ.get("CHEXPERT_ROOT") else None

# Convenience output when ingesting CheXpert dataset into our unified format
CHEXPERT_PROCESSED_CSV = DATA_DIR / "chexpert_processed_reports.csv"


def ensure_outputs_dir() -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def get_groq_api_key() -> str:
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "Missing GROQ_API_KEY. Add it to a `.env` file in the project root "
            "(see `.env.example`) or set the environment variable, then restart the terminal."
        )
    return key
