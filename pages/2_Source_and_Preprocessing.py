from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

import config


st.title("Source and preprocessing")

st.markdown(
    """
### Source
- **ECGEN XML** files live in `ecgen-radiology/`.

### Preprocessing goal
Convert XML reports into a single CSV (`processed_reports.csv`) with:
- `report_id`, `pmc_id`
- sections like `findings`, `impression`
- a single model input field: **`input_text = impression if available else findings`**
"""
)


st.subheader("Current paths")
st.code(
    "\n".join(
        [
            f"ECGEN_XML_DIR = {config.ECGEN_XML_DIR}",
            f"PROCESSED_CSV = {config.PROCESSED_CSV}",
        ]
    )
)


st.subheader("Preview processed CSV")
csv_path: Path = config.PROCESSED_CSV
if not csv_path.exists():
    st.warning(
        f"Missing `{csv_path}`. Run `python main.py` to generate it from the XML files."
    )
else:
    df = pd.read_csv(csv_path)
    st.write(f"Rows: **{len(df)}**  Columns: **{len(df.columns)}**")
    show_cols = [c for c in ["report_id", "pmc_id", "findings", "impression", "input_text"] if c in df.columns]
    st.dataframe(df[show_cols].head(25), use_container_width=True)

