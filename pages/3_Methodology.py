from __future__ import annotations

import streamlit as st


st.title("Methodology")

st.markdown(
    """
### Retrieval (RAG)
- We fetch the **CheXpert “mention” phrase lists** (Stanford chexpert-labeler).
- We embed phrases using **SentenceTransformers** and index them in **Chroma**.
- For each report, we retrieve the top‑10 most similar phrases and group them by disease.

### LLM labeling
- A Groq-hosted LLM is prompted with:
  - the radiology report text
  - the retrieved candidate phrase buckets
- The LLM returns JSON splitting phrase matches into:
  - **present**: truly present / implied findings
  - **negated**: explicitly absent findings (e.g. “no edema”)

### Self-generated labels
- “Present” phrases are mapped to **CheXpert-14** binary labels (`chex_*`).
- Each positive label gets a **confidence score** derived from retrieval similarity.
- Rows are routed to:
  - **auto**: all positive labels meet a confidence threshold
  - **review**: anything else (parse issues, low confidence, no positives, etc.)
"""
)

st.markdown(
    "**RAG note**: In practice, the retrieved top‑k phrase buckets are usually highly relevant to the report, "
    "but may include a few near-misses—hence the confidence threshold + human review routing."
)

