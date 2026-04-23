"""Fetch CheXpert phrases, RAG retrieve, LLM match, CheXpert-14 columns, confidence, routing."""

from __future__ import annotations

import json
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import chromadb
import pandas as pd
import requests
from chromadb.utils import embedding_functions
from groq import Groq

import chexpert_schema as cx
from config import (
    KEYWORDS_JSON,
    OUTPUTS_DIR,
    PROCESSED_CSV,
    ensure_outputs_dir,
    get_groq_api_key,
)


GITHUB_API = (
    "https://api.github.com/repos/stanfordmlgroup/chexpert-labeler/contents/phrases/mention"
)
RAW_BASE = (
    "https://raw.githubusercontent.com/stanfordmlgroup/chexpert-labeler/master/phrases/mention"
)


def fetch_chexpert_vocab() -> dict[str, list[str]]:
    ensure_outputs_dir()
    cache_path = OUTPUTS_DIR / "chexpert_vocab_cache.json"
    if cache_path.exists():
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                vocab = json.load(f)
            if isinstance(vocab, dict) and vocab:
                print(f"Loaded cached CheXpert vocab → {cache_path}")
                return {str(k): list(v) for k, v in vocab.items()}
        except Exception:
            pass

    print("Fetching CheXpert phrases from GitHub...")
    resp = requests.get(GITHUB_API, timeout=60)
    resp.raise_for_status()
    files = resp.json()
    vocab: dict[str, list[str]] = {}
    for file_info in files:
        filename = file_info["name"]
        if not str(filename).endswith(".txt"):
            continue
        disease_name = filename.replace(".txt", "").replace("_", " ").title()
        raw_url = f"{RAW_BASE}/{filename}"
        raw = requests.get(raw_url, timeout=60).text
        phrases = [line.strip() for line in raw.splitlines() if line.strip()]
        vocab[disease_name] = phrases
        print(f"  ✓ {disease_name}: {len(phrases)} phrases")

    try:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(vocab, f)
        print(f"Saved vocab cache → {cache_path}")
    except Exception:
        pass
    return vocab


def build_collection(vocab: dict[str, list[str]]) -> Any:
    ensure_outputs_dir()
    persist_dir = OUTPUTS_DIR / "chroma_chexpert_phrases"
    print(f"\nBuilding/loading vector store (persist={persist_dir})...")
    chroma_client = chromadb.PersistentClient(path=str(persist_dir))
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    collection = chroma_client.get_or_create_collection(
        name="chexpert_phrases",
        embedding_function=embedding_fn,
    )
    try:
        if int(collection.count()) > 0:
            print(f"Loaded existing index: {collection.count()} phrases\n")
            return collection
    except Exception:
        pass
    documents, metadatas, ids = [], [], []
    for disease, phrases in vocab.items():
        for i, phrase in enumerate(phrases):
            documents.append(phrase)
            metadatas.append({"disease": disease})
            ids.append(f"{disease}_{i}")
    if documents:
        collection.add(documents=documents, metadatas=metadatas, ids=ids)
    print(f"Indexed {len(documents)} phrases across {len(vocab)} diseases\n")
    return collection


def norm_phrase(s: str) -> str:
    return " ".join(str(s).strip().lower().split())


_NEGATION_PREFIX_RE = re.compile(
    r"^(?:"
    r"no\b|"
    r"without\b|"
    r"absence of\b|"
    r"absent\b|"
    r"free of\b|"
    r"negative for\b|"
    r"rule out\b|"
    r"ruled out\b|"
    r"denies\b|"
    r"denied\b|"
    r"not\b"
    r")\s+",
    flags=re.IGNORECASE,
)


def is_negated_phrase(phrase: str) -> bool:
    """
    CheXpert mention vocab includes explicit negations (e.g. "no edema").
    If the LLM returns those under a disease key, treat them as NEGATED evidence,
    not as a positive label trigger.
    """
    p = " ".join(str(phrase).strip().split())
    if not p:
        return False
    if _NEGATION_PREFIX_RE.search(p):
        return True
    low = p.lower()
    if "no evidence of" in low or "without evidence of" in low:
        return True
    return False


def distance_to_similarity(distance: float) -> float:
    """Chroma cosine space: distance = 1 - cosine_similarity for normalized embeddings."""
    sim = 1.0 - float(distance)
    return max(0.0, min(1.0, sim))


def _max_n_results(collection: Any, requested: int) -> int:
    try:
        n = int(collection.count())
    except Exception:
        n = requested
    return max(1, min(requested, n))


def retrieve_artifacts(collection: Any, report: str, n_results: int) -> dict[str, Any]:
    k = _max_n_results(collection, n_results)
    retrieved = collection.query(
        query_texts=[report],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )
    docs = retrieved["documents"][0]
    metas = retrieved["metadatas"][0]
    dists = retrieved["distances"][0]

    phrase_sim: dict[str, float] = {}
    disease_top_sim: dict[str, float] = defaultdict(float)
    retrieved_context: dict[str, list[str]] = {}

    for doc, dist, meta in zip(docs, dists, metas):
        disease_title = meta.get("disease", "")
        slug = cx.disease_key_to_slug(disease_title)
        sim = distance_to_similarity(dist)
        nk = norm_phrase(doc)
        phrase_sim[nk] = max(phrase_sim.get(nk, 0.0), sim)
        retrieved_context.setdefault(disease_title, []).append(doc)
        if slug:
            disease_top_sim[slug] = max(disease_top_sim[slug], sim)

    return {
        "retrieved_context": retrieved_context,
        "phrase_sim": phrase_sim,
        "disease_top_sim": dict(disease_top_sim),
    }


def confidence_for_slug(
    slug: str,
    matched_phrases: list[str],
    phrase_sim: dict[str, float],
    disease_top_sim: dict[str, float],
) -> float:
    sims: list[float] = []
    for p in matched_phrases:
        sims.append(phrase_sim.get(norm_phrase(p), disease_top_sim.get(slug, 0.0)))
    if sims:
        return float(max(sims))
    return float(disease_top_sim.get(slug, 0.0))


def call_groq_json(
    groq_client: Groq,
    report: str,
    retrieved_context: dict[str, list[str]],
    *,
    model: str,
) -> dict[str, Any]:
    prompt = (
        f"Radiology report:\n{report}\n\n"
        f"Retrieved candidate phrases (from CheXpert vocabulary):\n"
        f"{json.dumps(retrieved_context, indent=2)}\n\n"
        "Decide which of these phrases are PRESENT vs explicitly NEGATED in the report.\n"
        "Important: phrases like 'no edema' or 'without pneumothorax' are NEGATED, not present.\n"
        "Return ONLY valid JSON (no extra text, no markdown fences) in this shape:\n"
        '{'
        '"present": {"Disease Name": ["phrase1", "phrase2"]}, '
        '"negated": {"Disease Name": ["no phrase", "without ..."]}, '
        '"unmatched_terms": ["term1"], '
        '"rationale": "1-3 short sentences citing evidence"'
        '}'
    )
    response = groq_client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a medical NLP specialist. Given a radiology report and "
                    "relevant clinical phrases, identify which phrases actually appear "
                    "or are clearly implied vs explicitly negated. "
                    "Use only disease headers that appear in the retrieved candidate JSON keys. "
                    "Return valid JSON only, no markdown fences."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    )
    raw = response.choices[0].message.content.strip()
    if raw.startswith("```json"):
        raw = raw[7:]
    elif raw.startswith("```"):
        raw = raw[3:]
    if raw.endswith("```"):
        raw = raw[:-3]
    raw = raw.strip()
    return json.loads(raw)


def labels_from_match(
    matched: dict[str, Any],
    phrase_sim: dict[str, float],
    disease_top_sim: dict[str, float],
) -> tuple[dict[str, int], dict[str, float]]:
    preds = cx.zero_label_dict()
    confs = cx.zero_conf_dict()
    for disease_key, phrases in matched.items():
        slug = cx.disease_key_to_slug(disease_key)
        if not slug or not isinstance(phrases, list):
            continue
        affirmed = [str(p) for p in phrases if not is_negated_phrase(str(p))]
        if not affirmed:
            continue
        c = confidence_for_slug(slug, affirmed, phrase_sim, disease_top_sim)
        lc, cc = cx.label_column(slug), cx.conf_column(slug)
        preds[lc] = 1
        confs[cc] = max(float(confs[cc]), float(c))
    return preds, confs


def decide_route(
    preds: dict[str, int],
    confs: dict[str, float],
    threshold: float,
    parse_ok: bool,
    matched_raw: dict[str, Any],
) -> tuple[bool, str]:
    if not parse_ok:
        return False, "parse_error"
    if not matched_raw:
        return False, "empty_llm_match"
    positive_slugs = [s for s in cx.SLUG_ORDER if preds[cx.label_column(s)] == 1]
    if not positive_slugs:
        return False, "no_positive_labels"
    for s in positive_slugs:
        if confs[cx.conf_column(s)] < float(threshold):
            return False, f"below_threshold:{s}"
    return True, "ok"


def run_pipeline(
    *,
    processed_csv: Path | str | None = None,
    limit: int | None = 100,
    confidence_threshold: float = 0.8,
    rag_n_results: int = 40,
    sleep_s: float = 0.25,
    groq_model: str = "llama-3.3-70b-versatile",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ensure_outputs_dir()
    csv_path = Path(processed_csv or PROCESSED_CSV)
    df = pd.read_csv(csv_path)
    if limit is not None:
        df = df.head(int(limit)).copy()
    if "input_text" not in df.columns:
        raise ValueError("processed_reports.csv must contain an input_text column")

    vocab = fetch_chexpert_vocab()
    collection = build_collection(vocab)
    groq_client = Groq(api_key=get_groq_api_key())

    auto_rows: list[dict[str, Any]] = []
    review_rows: list[dict[str, Any]] = []
    keyword_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for n, (_, row) in enumerate(df.iterrows(), start=1):
        report = row.get("input_text")
        if report is None or (isinstance(report, float) and pd.isna(report)):
            continue
        report = str(report)
        report_id = row.get("report_id", "")
        pmc_id = row.get("pmc_id", "")
        print(f"Processing {n}/{len(df)} — {report_id!s}")

        art = retrieve_artifacts(collection, report, rag_n_results)
        retrieved_context = art["retrieved_context"]
        phrase_sim = art["phrase_sim"]
        disease_top_sim = art["disease_top_sim"]

        parse_ok = False
        matched: dict[str, Any] = {}
        negated: dict[str, Any] = {}
        rationale = ""
        try:
            result = call_groq_json(groq_client, report, retrieved_context, model=groq_model)
            # Backward compatible: older prompt returned {"matched": {...}}.
            matched = result.get("present") or result.get("matched") or {}
            negated = result.get("negated") or {}
            if not isinstance(matched, dict):
                matched = {}
            if not isinstance(negated, dict):
                negated = {}
            rationale = str(result.get("rationale", "") or "")
            parse_ok = True
        except json.JSONDecodeError:
            rationale = "json_decode_error"
        except Exception as e:
            rationale = f"groq_error:{e}"

        # Guardrail: ensure explicit negations never count as positive labels.
        for dk, plist in list(matched.items()):
            if not isinstance(plist, list):
                continue
            affirmed = [p for p in plist if not is_negated_phrase(str(p))]
            moved = [p for p in plist if is_negated_phrase(str(p))]
            if moved:
                negated.setdefault(dk, [])
                if isinstance(negated[dk], list):
                    negated[dk].extend(moved)
            matched[dk] = affirmed

        preds, confs = labels_from_match(matched, phrase_sim, disease_top_sim)
        auto_ok, reason = decide_route(
            preds, confs, confidence_threshold, parse_ok, matched
        )

        for dk, plist in matched.items():
            if not isinstance(plist, list):
                continue
            slug = cx.disease_key_to_slug(dk)
            if not slug:
                continue
            for p in plist:
                keyword_counts[slug][str(p)] += 1

        base = {
            "report_id": report_id,
            "pmc_id": pmc_id,
            "input_text": report,
            "rationale": rationale,
            "review_reason": reason,
            "confidence_threshold": confidence_threshold,
            "rag_top_json": json.dumps(retrieved_context, ensure_ascii=False)[:20000],
            "present_json": json.dumps(matched, ensure_ascii=False)[:20000],
            "negated_json": json.dumps(negated, ensure_ascii=False)[:20000],
        }
        base.update(preds)
        base.update(confs)

        if auto_ok:
            base["route"] = "auto"
            auto_rows.append(base)
        else:
            base["route"] = "review"
            review_rows.append(base)

        time.sleep(max(0.0, float(sleep_s)))

    auto_df = pd.DataFrame(auto_rows)
    review_df = pd.DataFrame(review_rows)

    final_keywords = {
        slug: [p for p, c in phrases.items() if c >= 1]
        for slug, phrases in keyword_counts.items()
        if phrases
    }
    with open(KEYWORDS_JSON, "w", encoding="utf-8") as f:
        json.dump(final_keywords, f, indent=2)
    print(f"\n✓ Saved keyword aggregate → {KEYWORDS_JSON}")

    return auto_df, review_df


def save_outputs(auto_df: pd.DataFrame, review_df: pd.DataFrame) -> None:
    ensure_outputs_dir()
    from config import AUTO_LABELED_CSV, REVIEW_QUEUE_CSV

    auto_df.to_csv(AUTO_LABELED_CSV, index=False)
    review_df.to_csv(REVIEW_QUEUE_CSV, index=False)
    print(f"✓ Wrote {len(auto_df)} auto rows → {AUTO_LABELED_CSV}")
    print(f"✓ Wrote {len(review_df)} review rows → {REVIEW_QUEUE_CSV}")


if __name__ == "__main__":
    a, r = run_pipeline(limit=5, sleep_s=0.0)
    save_outputs(a, r)
