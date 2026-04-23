from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

import chexpert_schema as cx
import config
import rag_labeler


@st.cache_resource
def _get_collection() -> Any:
    vocab = rag_labeler.fetch_chexpert_vocab()
    return rag_labeler.build_collection(vocab)


@st.cache_resource
def _get_groq_client() -> Any:
    from groq import Groq

    return Groq(api_key=config.get_groq_api_key())


def _append_decision(row: dict) -> None:
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.DATA_DIR / "prototype_decisions.csv"
    df = pd.DataFrame([row])
    header = not out_path.exists() or out_path.stat().st_size == 0
    df.to_csv(out_path, mode="a", index=False, header=header)


@st.cache_data
def _load_predictions() -> pd.DataFrame:
    """
    Fast path: load previously generated predictions so Prototype can be instant.
    """
    path = config.OUTPUTS_DIR / "review_queue.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


st.title("Prototype")
st.caption(
    "Select a report from `processed_reports.csv`, generate labels with RAG+Groq, "
    "then accept/reject (and optionally edit) the auto-filled labels."
)

st.info(
    "Fast mode: this page can instantly load cached predictions from `outputs/review_queue.csv`. "
    "Use **Regenerate with Groq** only if you want fresh predictions for the selected row."
)

csv_path = config.PROCESSED_CSV
if not csv_path.exists():
    st.error(f"Missing `{csv_path}`. Generate it first (run `python main.py`).")
    st.stop()

df = pd.read_csv(csv_path)
if df.empty:
    st.error("`processed_reports.csv` is empty.")
    st.stop()

top = st.columns([1, 1, 1, 1, 1])
with top[0]:
    idx = st.number_input("Row index", min_value=0, max_value=len(df) - 1, value=0, step=1)
with top[1]:
    rag_n = st.number_input("RAG top‑k", min_value=5, max_value=200, value=40, step=5)
with top[2]:
    threshold = st.slider("Auto threshold", min_value=0.5, max_value=0.99, value=0.8, step=0.01)
with top[3]:
    model = st.text_input("Groq model", value="llama-3.3-70b-versatile")
with top[4]:
    sleep_s = st.number_input("Sleep (s)", min_value=0.0, max_value=2.0, value=0.0, step=0.05)

row = df.iloc[int(idx)]
report_id = row.get("report_id", "")
pmc_id = row.get("pmc_id", "")
findings = row.get("findings", "")
impression = row.get("impression", "")
input_text = row.get("input_text", impression if pd.notna(impression) and str(impression).strip() else findings)
input_text = "" if (input_text is None or (isinstance(input_text, float) and pd.isna(input_text))) else str(input_text)

st.subheader(f"Report: {report_id}")
cols = st.columns(2)
with cols[0]:
    st.write("**findings**")
    st.text_area("Findings", value=str(findings) if findings is not None else "", height=220, disabled=True, label_visibility="collapsed")
with cols[1]:
    st.write("**impression**")
    st.text_area("Impression", value=str(impression) if impression is not None else "", height=220, disabled=True, label_visibility="collapsed")

st.write("**Model input (input_text)**")
st.text_area("Input text", value=input_text, height=200, disabled=True, label_visibility="collapsed")


if "proto_result" not in st.session_state:
    st.session_state.proto_result = None

pred_df = _load_predictions()
cached_row = None
if not pred_df.empty and "report_id" in pred_df.columns and str(report_id).strip():
    m = pred_df[pred_df["report_id"].astype(str) == str(report_id)]
    if len(m):
        cached_row = m.iloc[0]

btns = st.columns([1, 1, 4])
use_cached = btns[0].button("Load cached output", type="primary", disabled=cached_row is None)
regen = btns[1].button("Regenerate with Groq", help="Slower: runs retrieval + LLM call.")

if cached_row is None and not pred_df.empty:
    st.warning("No cached predictions found for this `report_id` in `outputs/review_queue.csv`.")
elif pred_df.empty:
    st.warning("No `outputs/review_queue.csv` found yet. Run `python RAG-LLM.py` first.")


if use_cached and cached_row is not None:
    preds = {cx.label_column(s): int(cached_row.get(cx.label_column(s), 0) or 0) for s in cx.SLUG_ORDER}
    confs = {cx.conf_column(s): float(cached_row.get(cx.conf_column(s), 0.0) or 0.0) for s in cx.SLUG_ORDER}
    st.session_state.proto_result = {
        "present": json.loads(cached_row.get("present_json", "{}") or "{}")
        if isinstance(cached_row.get("present_json", "{}"), str)
        else {},
        "negated": json.loads(cached_row.get("negated_json", "{}") or "{}")
        if isinstance(cached_row.get("negated_json", "{}"), str)
        else {},
        "preds": preds,
        "confs": confs,
        "rationale": str(cached_row.get("rationale", "") or ""),
        "route": str(cached_row.get("route", "") or ""),
        "review_reason": str(cached_row.get("review_reason", "") or ""),
        "retrieved_context": json.loads(cached_row.get("rag_top_json", "{}") or "{}")
        if isinstance(cached_row.get("rag_top_json", "{}"), str)
        else {},
    }


if regen:
    with st.spinner("Running retrieval + LLM…"):
        collection = _get_collection()
        groq_client = _get_groq_client()

        art = rag_labeler.retrieve_artifacts(collection, input_text, int(rag_n))
        retrieved_context = art["retrieved_context"]
        phrase_sim = art["phrase_sim"]
        disease_top_sim = art["disease_top_sim"]

        present: dict[str, Any] = {}
        negated: dict[str, Any] = {}
        rationale = ""
        parse_ok = False
        try:
            result = rag_labeler.call_groq_json(
                groq_client, input_text, retrieved_context, model=str(model)
            )
            present = result.get("present") or result.get("matched") or {}
            negated = result.get("negated") or {}
            if not isinstance(present, dict):
                present = {}
            if not isinstance(negated, dict):
                negated = {}
            rationale = str(result.get("rationale", "") or "")
            parse_ok = True
        except Exception as e:
            rationale = f"error:{e}"

        # Safety guardrail: drop explicit negations from present
        for dk, plist in list(present.items()):
            if not isinstance(plist, list):
                continue
            affirmed = [p for p in plist if not rag_labeler.is_negated_phrase(str(p))]
            moved = [p for p in plist if rag_labeler.is_negated_phrase(str(p))]
            if moved:
                negated.setdefault(dk, [])
                if isinstance(negated[dk], list):
                    negated[dk].extend(moved)
            present[dk] = affirmed

        preds, confs = rag_labeler.labels_from_match(present, phrase_sim, disease_top_sim)
        auto_ok, route_reason = rag_labeler.decide_route(
            preds, confs, float(threshold), bool(parse_ok), present
        )

        st.session_state.proto_result = {
            "present": present,
            "negated": negated,
            "preds": preds,
            "confs": confs,
            "rationale": rationale,
            "route": "auto" if auto_ok else "review",
            "review_reason": route_reason,
            "retrieved_context": retrieved_context,
        }

    st.success("Done.")


res = st.session_state.proto_result
if not res:
    st.info("Click **Load cached output** (fast) or **Regenerate with Groq** (slow).")
    st.stop()

st.subheader("Prediction summary")
st.write(
    {
        "route": res["route"],
        "review_reason": res["review_reason"],
    }
)

with st.expander("Rationale"):
    st.write(res.get("rationale", ""))

with st.expander("Retrieved phrase buckets (RAG)"):
    st.code(json.dumps(res.get("retrieved_context", {}), indent=2)[:12000])

with st.expander("LLM phrase decisions (present vs negated)"):
    st.write("**present**")
    st.code(json.dumps(res.get("present", {}), indent=2)[:12000])
    st.write("**negated**")
    st.code(json.dumps(res.get("negated", {}), indent=2)[:12000])


st.subheader("Auto-filled labels (editable)")
positive = []
for s in cx.SLUG_ORDER:
    if int(res["preds"].get(cx.label_column(s), 0) or 0) == 1:
        positive.append(s)

if positive:
    top_conf = max(float(res["confs"].get(cx.conf_column(s), 0.0) or 0.0) for s in positive)
else:
    top_conf = 0.0

default_label = ", ".join(positive) if positive else "no_finding"

st.write("**Auto-filled label**")
label_text = st.text_input(
    "Label",
    value=default_label,
    help="Comma-separated CheXpert label slugs (editable).",
)
st.caption(f"Top label confidence: **{top_conf*100:.1f}%**")

note = st.text_input("Reviewer note (optional)", value="")
accept = st.button("Accept annotation", type="primary")

if accept:
    out = {
        "decision": "accept",
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "report_id": report_id,
        "pmc_id": pmc_id,
        "row_index": int(idx),
        "label_text": label_text.strip(),
        "top_confidence": float(top_conf),
        "input_text": input_text[:5000],
        "reviewer_note": note,
        "route": res.get("route", ""),
        "review_reason": res.get("review_reason", ""),
        "present_json": json.dumps(res.get("present", {}), ensure_ascii=False)[:20000],
        "negated_json": json.dumps(res.get("negated", {}), ensure_ascii=False)[:20000],
    }
    _append_decision(out)
    st.success(f"Saved annotation → `{config.DATA_DIR / 'prototype_decisions.csv'}`")

with st.expander("Advanced: per-label checkboxes and confidences"):
    defaults: dict[str, bool] = {}
    for s in cx.SLUG_ORDER:
        defaults[cx.label_column(s)] = bool(int(res["preds"].get(cx.label_column(s), 0) or 0))

    cols = st.columns(4)
    edited: dict[str, int] = {}
    edited_conf: dict[str, float] = {}
    for i, s in enumerate(cx.SLUG_ORDER):
        c = cols[i % 4]
        lc = cx.label_column(s)
        cc = cx.conf_column(s)
        conf_val = float(res["confs"].get(cc, 0.0) or 0.0)
        edited[lc] = 1 if c.checkbox(
            f"{lc} ({conf_val*100:.1f}%)",
            value=defaults[lc],
            key=f"proto_{idx}_{lc}",
        ) else 0
        edited_conf[cc] = conf_val

    if st.button("Save advanced annotation"):
        out = {
            "decision": "accept",
            "saved_at_utc": datetime.now(timezone.utc).isoformat(),
            "report_id": report_id,
            "pmc_id": pmc_id,
            "row_index": int(idx),
            "label_text": label_text.strip(),
            "top_confidence": float(top_conf),
            "input_text": input_text[:5000],
            "reviewer_note": note,
            "route": res.get("route", ""),
            "review_reason": res.get("review_reason", ""),
            "present_json": json.dumps(res.get("present", {}), ensure_ascii=False)[:20000],
            "negated_json": json.dumps(res.get("negated", {}), ensure_ascii=False)[:20000],
        }
        out.update({cx.label_column(s): int(edited[cx.label_column(s)]) for s in cx.SLUG_ORDER})
        out.update({cx.conf_column(s): float(edited_conf[cx.conf_column(s)]) for s in cx.SLUG_ORDER})
        _append_decision(out)
        st.success(f"Saved annotation → `{config.DATA_DIR / 'prototype_decisions.csv'}`")

