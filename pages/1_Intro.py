from __future__ import annotations

import streamlit as st


st.title("Intro")

st.markdown(
    """
This project builds a **weak-labeling pipeline** for radiology reports using:

- **Source reports**: ECGEN radiology XML files
- **Preprocessing**: XML → cleaned CSV with a single `input_text` field
- **Retrieval (RAG)**: retrieve CheXpert mention phrases relevant to a report
- **LLM labeling**: a Groq-hosted LLM decides which phrases are **present vs negated**
- **Routing**: high-confidence predictions go to **auto**, the rest go to **human review**

Use the sidebar pages to walk through the pipeline and try the interactive prototype.
"""
)

