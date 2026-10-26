# Method and Data

For how to run the code, see [README.md](../README.md). This document describes the
experimental setup, the proposed method and the data conventions.

## 1. Experimental setup

Queries are voice-phishing call transcripts. The documents to search come from the other
impersonation domain (finance ↔ government). One gold document (the needle) from the
query's own domain is mixed in. We measure whether the needle appears in the top 20, as
recall@20.

| Direction | Query / needle category | Queries | Distractors |
|---|---|---|---|
| `fin_gov` | finance | 224 | 914 government |
| `gov_fin` | government | 447 | 2,100 finance |

### needle=1 instances

The number of needles is fixed to 1. Each query's gold documents are listed in `qrels.json`.
A query with k gold documents is expanded into k instances, each using one of them as the
needle, and all instances are evaluated. The search corpus of an instance is all distractors
plus that one needle. The query's other gold documents are left out.

| Direction | Queries with ≥ 1 gold document | Evaluated instances |
|---|---|---|
| `fin_gov` | 222 | 1,951 |
| `gov_fin` | 446 | 4,006 |

recall@20 is the mean over instances of whether the needle is ranked in the top 20.

## 2. Proposed method (`ours_rrf_bm25_kure`)

Three retrieval results are fused with weighted Reciprocal Rank Fusion (RRF).

| Component | Query | Weight |
|---|---|---|
| BM25 (regex tokenizer) | Original call transcript | 0.495 |
| KURE-v1 (`nlpai-lab/KURE-v1`) | Original call transcript | 0.01 |
| KURE-v1 | GPT-5-compressed query | 0.495 |

- The top 40 documents (= 2 × top-k) from each retriever are taken as candidates.
- A document's score is `Σ weight × 1 / (60 + rank)` (RRF constant 60, rank starting at 1).
  Documents are re-ranked by this score and the top 20 are evaluated.
- KURE ranks documents by cosine similarity of L2-normalised query and document embeddings.
- The weights were chosen by grid search.

### GPT query compression

Call transcripts are long and full of noise such as greetings, repetition and hesitation.
GPT-5 (`gpt-5`) rewrites each transcript as a compressed query: a list of short phrases that
capture the key situations and expressions. The prompt asks the model to keep the following
verbatim:

- Institution, bank and product names, and proper nouns such as carriers and card companies
- Concrete amounts, limits and interest rates
- Specific threatening or persuasive phrasing

Identifiers that are randomly generated each time, such as case numbers and phone numbers,
are replaced by their type. The prompt is in Korean, since the transcripts are Korean. It is
`COMPRESSION_PROMPT_TEMPLATE` in `ex/rag_official.py`, with an English summary in the
comment above it.

Compressed queries are cached per query ID in
`ex/outputs/compression_cache_{direction}_full.json`. Each entry stores:

- The compressed text
- API response metadata (request id, creation time, model version, `system_fingerprint`)
- A SHA-256 hash of the whole entry

The hashes are recomputed every time the cache is loaded, and the run stops if any of them
differ.

## 3. Results

| | Target Finance (`fin_gov`) | Target Government (`gov_fin`) | Avg. |
|---|---|---|---|
| Recall@20 | 0.7940 (1549/1951) | 0.8567 (3432/4006) | 0.8254 |

GPT can return different output for the same input on each call. The compression cache is
not included in this repository, so rerunning the code produces new compressed queries, and
recall@20 may differ slightly from the values above.

## 4. Data

The dataset is not included in this repository. The required files and folder layout are
listed in [README.md](../README.md#data-preparation).

### gold_final.csv

3,014 relabelled call transcripts (2,100 finance / 914 government), the source of all
documents to search. 3,121 transcripts were cross-checked by two annotators. Disagreements
were resolved by LLM majority vote, and the remaining cases were adjudicated manually.

| Column | Description |
|---|---|
| `review_id` | Unique key in `cat_review_00001` format |
| `set` | Annotator split (set1 1,532 / set2 1,482) |
| `gold_label` | Impersonated domain: `finance` / `government` |
| `gold_source` | How the label was decided: `agree_domain` 2,726 (annotators agreed), `conflict_llm_majority` 258 (LLM majority vote), `human_review` 30 (manual adjudication) |
| `text` | Call transcript |

The code uses `review_id`, `gold_label` and `text`.

### Query sets (`*_ood_challenge.csv`)

These are the OOD challenge sets from prior work, used as is, with no overlap with
`gold_final.csv`. Columns are `date`, `text`, `category` and `label`. Every row is a phishing
call, so `label` is always 1.

### corpus.jsonl (for verification)

The delivered distractor corpus for each direction: 914 government documents in
`finance_eval/`, and 2,100 finance documents in `government_eval/`.

```json
{"doc_id": "gold_finance_00001", "text": "...", "category": "finance",
 "label": 0.0, "review_id": "cat_review_00003"}
```

On every run the distractors are rebuilt from `gold_final.csv` and compared with these files.
The run stops if even one `doc_id` or `text` differs. Adding or removing a single row in
`gold_final.csv` would shift every later document ID, so `qrels.json` would point to the
wrong documents. This check keeps that from going unnoticed.

### ID conventions

`qrels.json` and the compression cache use the IDs below, so the rows of the data files must
not be reordered.

- **Document ID:** `gold_final.csv` is read in file order and numbered within each category as
  `gold_{category}_{number:05d}` (e.g. `gold_finance_00001`).
- **Query ID:** query csv rows are numbered in order as `{direction}_q{number:04d}`
  (e.g. `fin_gov_q0001`).
- **Instance ID:** in the output files, `{query ID}::{needle document ID}`.

`work/combined/{finance,government}/full/qrels.json` has the form
`{query ID: [gold document ID, ...]}`.

## 5. Code structure

| File | Role |
|---|---|
| `ex/run_official_once.sh` | Entry point. Runs both directions concurrently on different GPUs, then aggregates |
| `ex/rag_official.py` | Runs one direction: build instances → GPT compression → BM25 (CPU) and KURE (GPU) in parallel processes → RRF → recall@20 |
| `ex/aggregate_ours_results.py` | Combines per-direction results (`*.entries.jsonl`) into the recall@20 table and confusion matrix |
| `config.py` | Data path (`VISHING_DATA_ROOT`) and direction definitions |
| `build_corpus.py` | Loads `gold_final.csv` and checks it against the delivered `corpus.jsonl` |
| `ood_retrieval/data.py` | Query csv loader |
| `evaluate.py` | sentence-transformers embedding wrapper |

Each instance has its own corpus (distractors plus that instance's needle), so a new BM25
index is built per instance. This work is split across `--bm25-workers` CPU processes. When
`--extra-devices` adds GPUs, KURE document encoding is split across them in one thread per
GPU.
