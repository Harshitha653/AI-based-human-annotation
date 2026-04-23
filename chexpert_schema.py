"""CheXpert 14 label schema aligned with Stanford chexpert-labeler phrase filenames."""

from __future__ import annotations

# Official CheXpert label order (subset of 14 observation classes used in labeling).
SLUG_ORDER: list[str] = [
    "no_finding",
    "enlarged_cardiomediastinum",
    "cardiomegaly",
    "lung_lesion",
    "lung_opacity",
    "edema",
    "consolidation",
    "pneumonia",
    "atelectasis",
    "pneumothorax",
    "pleural_effusion",
    "pleural_other",
    "fracture",
    "support_devices",
]

CHEXPERT_TXT_FILES: list[str] = [
    "no_finding.txt",
    "enlarged_cardiomediastinum.txt",
    "cardiomegaly.txt",
    "lung_lesion.txt",
    "lung_opacity.txt",
    "edema.txt",
    "consolidation.txt",
    "pneumonia.txt",
    "atelectasis.txt",
    "pneumothorax.txt",
    "pleural_effusion.txt",
    "pleural_other.txt",
    "fracture.txt",
    "support_devices.txt",
]


def _title_from_filename(filename: str) -> str:
    stem = filename.replace(".txt", "")
    return stem.replace("_", " ").title()


# Display title (as used in RAG-LLM vocab keys) -> canonical slug
DISPLAY_TITLE_TO_SLUG: dict[str, str] = {
    _title_from_filename(f): f.replace(".txt", "") for f in CHEXPERT_TXT_FILES
}

SLUG_SET = set(SLUG_ORDER)

# Extra strings LLMs sometimes return instead of our display titles
ALIASES_TO_SLUG: dict[str, str] = {
    "airspace opacity": "lung_opacity",
    "lung opacity": "lung_opacity",
    "opacity": "lung_opacity",
    "pleural effusions": "pleural_effusion",
    "enlarged cardiomediastinum": "enlarged_cardiomediastinum",
    "support device": "support_devices",
    "no findings": "no_finding",
    "normal": "no_finding",
}


def label_column(slug: str) -> str:
    return f"chex_{slug}"


def conf_column(slug: str) -> str:
    return f"chex_{slug}_conf"


CHEX_COLUMNS: list[str] = [label_column(s) for s in SLUG_ORDER]
CONF_COLUMNS: list[str] = [conf_column(s) for s in SLUG_ORDER]


def disease_key_to_slug(key: str) -> str | None:
    """Map a disease header from the LLM JSON to a CheXpert slug."""
    raw = " ".join(str(key).strip().split())
    if not raw:
        return None
    low = raw.lower()
    if low in ALIASES_TO_SLUG:
        return ALIASES_TO_SLUG[low]
    for title, slug in DISPLAY_TITLE_TO_SLUG.items():
        if title.lower() == low:
            return slug
    underscored = low.replace(" ", "_")
    if underscored in SLUG_SET:
        return underscored
    return None


def zero_label_dict() -> dict[str, int]:
    return {label_column(s): 0 for s in SLUG_ORDER}


def zero_conf_dict() -> dict[str, float]:
    return {conf_column(s): 0.0 for s in SLUG_ORDER}
