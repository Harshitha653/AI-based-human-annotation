from __future__ import annotations

import streamlit as st


def main() -> None:
    st.set_page_config(
        page_title="CheXpert RAG Labeling",
        page_icon="🩻",
        layout="wide",
    )
    st.title("CheXpert RAG + LLM Labeling")
    st.caption(
        "Use the pages in the left sidebar to navigate: Intro → Source/Preprocessing → "
        "Methodology → Prototype."
    )
    st.info("Open a page from the sidebar to get started.")


if __name__ == "__main__":
    main()

