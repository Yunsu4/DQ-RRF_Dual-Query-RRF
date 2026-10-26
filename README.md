# Voice-Phishing Call OOD Retrieval — Weighted RRF of BM25 + KURE

Experiment code for retrieving voice-phishing call transcripts across impersonation domains.
Each query is a call transcript. The task is to find the one gold document from the same
domain, hidden among documents from the other domain (finance ↔ government). The proposed
method fuses BM25 and KURE-v1 (original query and GPT-5-compressed query) with weighted RRF,
and is evaluated with recall@20.

For details on the method and data, see [docs/METHOD.md](docs/METHOD.md).

## Quick start

```bash
pip install -r requirements.txt

export VISHING_DATA_ROOT=/path/to/data   # data folder (see "Data preparation")
export OPENAI_API_KEY=sk-...             # for GPT query compression

bash ex/run_official_once.sh
```

Results are written to `ex/outputs/aggregate_once.table.md`.

## Requirements

- Python 3.10+
- PyTorch with CUDA support
- GPUs: the default configuration uses 4. See [GPU settings](#gpu-settings) to run on 1–2.
- An OpenAI API key (for GPT-5 calls)

```bash
pip install -r requirements.txt
```

## Data preparation

The dataset is not included in this repository. Put the files below in one folder with this
exact layout, and point `VISHING_DATA_ROOT` to that folder. The code only reads from it.

```
$VISHING_DATA_ROOT/
├── gold_final.csv
├── finance_ood_challenge.csv
├── government_ood_challenge.csv
├── finance_eval/corpus.jsonl
└── government_eval/corpus.jsonl
```

```bash
export VISHING_DATA_ROOT=/path/to/data   # a relative path also works
```

| File | Purpose | Required columns / fields |
|---|---|---|
| `gold_final.csv` | Documents to search (distractors + gold documents), 3,014 transcripts | `review_id`, `gold_label` (`finance`/`government`), `text` |
| `finance_ood_challenge.csv` | 224 queries for the `fin_gov` direction | `date`, `text`, `category`, `label` |
| `government_ood_challenge.csv` | 447 queries for the `gov_fin` direction | `date`, `text`, `category`, `label` |
| `finance_eval/corpus.jsonl` | Delivered `fin_gov` distractor corpus (for verification) | `doc_id`, `text` |
| `government_eval/corpus.jsonl` | Delivered `gov_fin` distractor corpus (for verification) | `doc_id`, `text` |

- **Do not reorder rows in any file.** Document and query IDs are assigned by row order, and
  the gold labels in `work/combined/*/full/qrels.json` refer to those IDs.
- At startup the code checks that all five files exist and that `gold_final.csv` matches the
  `corpus.jsonl` files. If either check fails, it stops immediately.

See [docs/METHOD.md](docs/METHOD.md#4-data) for a description of each file and the ID rules.

## OpenAI API key

`OPENAI_API_KEY` is required to compress queries with GPT-5 (`gpt-5`).

```bash
export OPENAI_API_KEY=sk-...
```

- The first run calls GPT once per query (222 for `fin_gov`, 446 for `gov_fin`). The queries
  are long call transcripts, so this takes time and incurs API cost.
- Compressed queries are saved to `ex/outputs/compression_cache_{direction}_full.json` after
  every call. If a run is interrupted, rerunning resumes from where it stopped, and queries
  that are already compressed are not sent again.
- Once the cache covers every query, later runs no longer need the API key.

## Running

```bash
bash ex/run_official_once.sh
```

This runs the `fin_gov` and `gov_fin` directions concurrently on different GPUs and
aggregates the results when both finish. The script locates the project root from its own
path, so it can be run from any directory.

### GPU settings

By default `fin_gov` uses `cuda:0` and `gov_fin` uses `cuda:1`. `gov_fin` has more documents,
so it also uses `cuda:2,cuda:3` for document encoding.

```bash
# 2 GPUs
EXTRA_DEVICES_GOV_FIN= bash ex/run_official_once.sh

# 1 GPU (both directions share it)
DEVICE_GOV_FIN=cuda:0 EXTRA_DEVICES_GOV_FIN= bash ex/run_official_once.sh
```

### Environment variables

| Variable | Default | Description |
|---|---|---|
| `VISHING_DATA_ROOT` | (required) | Data folder |
| `OPENAI_API_KEY` | (required) | For GPT query compression. Not needed once the compression cache covers every query |
| `DEVICE_FIN_GOV` / `DEVICE_GOV_FIN` | `cuda:0` / `cuda:1` | Primary GPU per direction |
| `EXTRA_DEVICES_FIN_GOV` / `EXTRA_DEVICES_GOV_FIN` | (none) / `cuda:2,cuda:3` | Extra GPUs for document encoding (comma-separated) |
| `FP16` | `1` | Load KURE in bf16/fp16. Set to empty for fp32 |
| `BATCH_SIZE` | `64` | KURE encoding batch size |
| `BM25_WORKERS` | `32` | BM25 CPU processes per direction |
| `TOKENIZER` | `regex` | BM25 tokenizer. `kiwi` requires `pip install kiwipiepy` |

### Running a single direction

```bash
python ex/rag_official.py --direction fin_gov \
    --qrels work/combined/finance/full/qrels.json --device cuda:0 --fp16

python ex/rag_official.py --direction gov_fin \
    --qrels work/combined/government/full/qrels.json \
    --device cuda:1 --extra-devices cuda:2,cuda:3 --fp16
```

Run `python ex/rag_official.py --help` for all options.

## Outputs

Written under `ex/outputs/`:

| File | Contents |
|---|---|
| `aggregate_once.table.md` | recall@20 per direction |
| `aggregate_once.confusion.md` | hits/misses per direction |
| `aggregate_once.json` | Both of the above as JSON |
| `rag_ours_only_{direction}_full_n1_regex.json` | Per-direction summary |
| `rag_ours_only_{direction}_full_n1_regex.entries.jsonl` | Per instance: query, compressed query, gold document, predicted rank |
| `compression_cache_{direction}_full.json` | GPT query compression cache |
| `logs/{fin_gov,gov_fin}.log` | Per-direction run logs |

### Reference results

| | Target Finance (`fin_gov`) | Target Government (`gov_fin`) | Avg. |
|---|---|---|---|
| Recall@20 | 0.7940 (1549/1951) | 0.8567 (3432/4006) | 0.8254 |

> **Your numbers may not match these exactly.** The GPT compression cache is not included in
> this repository, so running the code calls GPT again. GPT can return different output for
> the same input on each call, so the compressed queries, and therefore recall@20, will vary
> slightly.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `VISHING_DATA_ROOT is not set` / `Required files not found in the data folder` | Make sure `VISHING_DATA_ROOT` points to the folder containing the five files above |
| `OPENAI_API_KEY is not set` / OpenAI authentication error | Check `export OPENAI_API_KEY=...` |
| `... differ from the delivered corpus` / `Data and code versions do not match` | A data file was modified or reordered. Obtain the original data again |
| `Cache integrity check failed` | A compression cache file changed after it was created. Delete that `compression_cache_*.json` and rerun (GPT will be called again) |
| `CUDA out of memory` | Lower `BATCH_SIZE` or add GPUs with `EXTRA_DEVICES_*` |
| Only one direction fails | Check `ex/outputs/logs/{direction}.log`. Aggregation still runs on the successful direction |
