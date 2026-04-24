# CheXpert RAG + LLM weak labeling (CSE-881)

Weak-labeling pipeline for chest X-ray report text: retrieve **CheXpert “mention” phrases** (from the [Stanford chexpert-labeler](https://github.com/stanfordmlgroup/chexpert-labeler) vocabulary), use **embedding retrieval (Chroma + SentenceTransformers)** and a **Groq-hosted LLM** to separate **present** vs **negated** findings, then map to **CheXpert-14** binary labels with similarity-based confidence and **auto vs human-review** routing.

## What’s in the repo

| Piece | Role |
|--------|------|
| `main.py` | Parse **ECGEN** XML from `ecgen-radiology/` → `processed_reports.csv` (`input_text` = impression if present, else findings). |
| `Pre-processing_csv.py` | Extra cleaning / checks on `processed_reports.csv`. |
| `rag_labeler.py` | Core RAG index, Groq JSON labeling, CheXpert columns, routing logic. |
| `RAG-LLM.py` | CLI entry: runs the full pipeline and writes CSV/JSON under `outputs/`. |
| `streamlit_app.py` + `pages/` | Multipage demo: intro, data paths, methodology, interactive prototype. |
| `review_app.py` | Streamlit UI to review `outputs/review_queue.csv` and append rows to `data/human_gold.csv`. |
| `evaluate_labels.py` | Compare predictions to a small gold CSV (`data/gold_subset.csv`); optional threshold sweep (`--calibrate`). |
| `chexpert_schema.py` | CheXpert-14 slug/column helpers shared across scripts. |
| `config.py` | Paths, `.env` loading, `GROQ_API_KEY` helper. |

## Requirements

- Python 3.10+ recommended  
- Dependencies: `requirements.txt` (pandas, beautifulsoup4, requests, chromadb, sentence-transformers, groq, python-dotenv, streamlit)

Install:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Configuration

Create a `.env` file in the project root (same folder as `config.py`):

```env
GROQ_API_KEY=your_groq_api_key
```

Optional:

```env
CHEXPERT_ROOT=D:/path/to/CheXpert-v1.0-small
```

Used when ingesting CheXpert into a unified CSV (`config.CHEXPERT_PROCESSED_CSV`).

## Data layout

- **`ecgen-radiology/`** — ECGEN radiology XML files (input for `main.py`).
- **`processed_reports.csv`** — Generated at project root by `main.py`.
- **`data/`** — Gold subset for evaluation (`gold_subset.csv`, `gold_subset.example.csv`), human annotations (`human_gold.csv` when using the review app).
- **`outputs/`** — Pipeline artifacts: `auto_labeled.csv`, `review_queue.csv`, Chroma persist dir, vocab cache, keyword aggregates (created when you run labeling).

## Typical workflow

1. **Preprocess XML → CSV**

   ```bash
   python main.py
   ```

   Optionally run `python Pre-processing_csv.py` if you use that step in your workflow.

2. **Run RAG + LLM labeling**

   ```bash
   python RAG-LLM.py --limit 100 --threshold 0.8 --rag-n 40 --sleep 0.25
   ```

   Useful flags: `--limit 0` for all rows, `--processed-csv` for a custom CSV path, `--model` for another Groq model id.

   For a quick smoke test of the library code only, `python rag_labeler.py` runs a tiny built-in sample (`limit=5`).

3. **Explore the project (UI)**

   ```bash
   streamlit run streamlit_app.py
   ```

   Use the sidebar: Intro → Source and preprocessing → Methodology → Prototype.

4. **Human review queue** (after labeling)

   ```bash
   streamlit run review_app.py
   ```

5. **Evaluate vs gold**

   ```bash
   python evaluate_labels.py
   python evaluate_labels.py --calibrate
   ```

   Gold format: `report_id` plus `chex_*` columns (0/1); see `data/gold_subset.example.csv`.

## Outputs

- **`outputs/auto_labeled.csv`** — Rows where all positive CheXpert labels meet the confidence threshold and parsing succeeded.
- **`outputs/review_queue.csv`** — Everything else (low confidence, parse issues, empty positives, etc.).
- **`outputs/chexpert_keywords_aggregate.json`** — Aggregated matched phrases for analysis.

## Notes

- First run downloads/embeds CheXpert mention phrases and may download the SentenceTransformer model; Chroma data is persisted under `outputs/`.
- Rate limits: adjust `--sleep` on `RAG-LLM.py` if Groq returns throttling errors.
- Do not commit secrets: `.env` is listed in `.gitignore`.

## References

- [CheXpert dataset and label schema](https://stanfordmlgroup.github.io/competitions/chexpert/)  
- [chexpert-labeler phrase lists](https://github.com/stanfordmlgroup/chexpert-labeler/tree/master/phrases/mention)  
- [Groq API](https://console.groq.com/)
