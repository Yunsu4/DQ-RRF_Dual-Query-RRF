#!/usr/bin/env bash
# Runs rag_official.py exactly once for fin_gov and once for gov_fin (no tag, no repeats),
# then aggregates the results with aggregate_ours_results.py to show recall@20 and the
# confusion matrix in one go. fin_gov and gov_fin run concurrently on two GPUs (default
# cuda:0 and cuda:1). gov_fin has the larger corpus, so by default it gets two more GPUs
# (cuda:2, cuda:3) to split document encoding (--extra-devices). The default setup therefore
# uses 4 GPUs. fp16, batch_size=64 and bm25-workers=32 are already the defaults, so just run:
#
#   bash ex/run_official_once.sh
#
# To change paths, GPUs or core counts, edit the "Settings" block below or override with
# environment variables (e.g. raise BM25_WORKERS to 40 when you have the server to yourself):
#   BM25_WORKERS=40 bash ex/run_official_once.sh

set -uo pipefail

# ============================== Settings ==============================
# The parent of this script's folder (ex/) is the project root, wherever it is run from.
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
# All paths below are relative to PROJECT_ROOT (the script cd's there before running).
SCRIPT="ex/rag_official.py"
AGGREGATE_SCRIPT="ex/aggregate_ours_results.py"
OUT_DIR="ex/outputs"
LOG_DIR="$OUT_DIR/logs"

# File name prefix of the entries.jsonl written by rag_official.py (must match file_stem in main())
OUTPUT_PREFIX="${OUTPUT_PREFIX:-rag_ours_only}"

QRELS_FIN_GOV="${QRELS_FIN_GOV:-work/combined/finance/full/qrels.json}"
QRELS_GOV_FIN="${QRELS_GOV_FIN:-work/combined/government/full/qrels.json}"

QUERY_SET="${QUERY_SET:-full}"
TOKENIZER="${TOKENIZER:-regex}"

DEVICE_FIN_GOV="${DEVICE_FIN_GOV:-cuda:0}"
DEVICE_GOV_FIN="${DEVICE_GOV_FIN:-cuda:1}"

# Extra GPUs used only for corpus (document) encoding, comma-separated. Leave empty to use
# only the DEVICE above. gov_fin has the larger corpus, so it gets cuda:2 and cuda:3 by
# default. fin_gov uses only DEVICE_FIN_GOV by default.
EXTRA_DEVICES_FIN_GOV="${EXTRA_DEVICES_FIN_GOV:-}"
EXTRA_DEVICES_GOV_FIN="${EXTRA_DEVICES_GOV_FIN:-cuda:2,cuda:3}"

# Load KURE in fp16 (bf16 if supported). "1" enables it; empty means fp32.
FP16="${FP16:-1}"

# KURE encoding batch size (larger is faster as long as it does not run out of memory).
BATCH_SIZE="${BATCH_SIZE:-64}"

# Number of CPU processes for BM25. Each entry has a different corpus, so a new BM25Okapi
# index is built per entry, which is expensive and is spread over several cores. fin_gov and
# gov_fin run at the same time, so the default of 32 each uses 64 of 80 cores (on our server)
# and leaves 16 for other users. If you have the machine to yourself you can raise it to 40.
BM25_WORKERS="${BM25_WORKERS:-32}"
# ======================================================================

FP16_FLAG=()
if [ -n "$FP16" ]; then
  FP16_FLAG=(--fp16)
fi

BM25_WORKERS_FLAG=()
if [ -n "$BM25_WORKERS" ]; then
  BM25_WORKERS_FLAG=(--bm25-workers "$BM25_WORKERS")
fi

extra_devices_args() {
  # Emit --extra-devices arguments if the comma-separated GPU list is not empty.
  local devices="$1"
  if [ -n "$devices" ]; then
    printf -- '--extra-devices\n%s\n' "$devices"
  fi
}

if [ -z "${VISHING_DATA_ROOT:-}" ]; then
  echo "VISHING_DATA_ROOT is not set. Point it to the data folder (the one containing gold_final.csv):" >&2
  echo "  export VISHING_DATA_ROOT=/path/to/data" >&2
  exit 1
fi
# We cd to PROJECT_ROOT below, so turn a relative path into an absolute one first.
data_root_abs="$(cd "$VISHING_DATA_ROOT" 2>/dev/null && pwd)" \
  || { echo "VISHING_DATA_ROOT folder does not exist: $VISHING_DATA_ROOT" >&2; exit 1; }
export VISHING_DATA_ROOT="$data_root_abs"

# If the GPT query compression caches already exist for both directions, no API key is needed.
# If either is missing, the key is required. (A cache that covers only some queries will still
# call GPT for the rest, which also needs the key.)
if [ -z "${OPENAI_API_KEY:-}" ]; then
  for d in fin_gov gov_fin; do
    if [ ! -f "$PROJECT_ROOT/$OUT_DIR/compression_cache_${d}_${QUERY_SET}.json" ]; then
      echo "OPENAI_API_KEY is not set. It is required for GPT query compression:" >&2
      echo "  export OPENAI_API_KEY=sk-..." >&2
      exit 1
    fi
  done
fi

cd "$PROJECT_ROOT" || { echo "Failed to cd to PROJECT_ROOT: $PROJECT_ROOT" >&2; exit 1; }
mkdir -p "$OUT_DIR" "$LOG_DIR"

mapfile -t fin_gov_extra_args < <(extra_devices_args "$EXTRA_DEVICES_FIN_GOV")
mapfile -t gov_fin_extra_args < <(extra_devices_args "$EXTRA_DEVICES_GOV_FIN")

echo "Settings:"
echo "  SCRIPT       = $SCRIPT"
echo "  DATA_ROOT    = $VISHING_DATA_ROOT"
echo "  fin_gov: qrels=$QRELS_FIN_GOV, device=$DEVICE_FIN_GOV, extra_devices=${EXTRA_DEVICES_FIN_GOV:-(none)}"
echo "  gov_fin: qrels=$QRELS_GOV_FIN, device=$DEVICE_GOV_FIN, extra_devices=${EXTRA_DEVICES_GOV_FIN:-(none)}"
echo "  QUERY_SET/TOKENIZER = $QUERY_SET / $TOKENIZER, batch_size=$BATCH_SIZE, fp16=${FP16:-off}, bm25_workers=${BM25_WORKERS:-(all cores per direction)}"
echo "  logs = $LOG_DIR/fin_gov.log, $LOG_DIR/gov_fin.log"
echo

python "$SCRIPT" --direction fin_gov --qrels "$QRELS_FIN_GOV" \
    --query-set "$QUERY_SET" --tokenizer "$TOKENIZER" \
    --device "$DEVICE_FIN_GOV" --batch-size "$BATCH_SIZE" \
    "${fin_gov_extra_args[@]}" "${FP16_FLAG[@]}" "${BM25_WORKERS_FLAG[@]}" \
    > "$LOG_DIR/fin_gov.log" 2>&1 &
pid_fin_gov=$!
echo "Started: fin_gov @ $DEVICE_FIN_GOV (pid=$pid_fin_gov)"

python "$SCRIPT" --direction gov_fin --qrels "$QRELS_GOV_FIN" \
    --query-set "$QUERY_SET" --tokenizer "$TOKENIZER" \
    --device "$DEVICE_GOV_FIN" --batch-size "$BATCH_SIZE" \
    "${gov_fin_extra_args[@]}" "${FP16_FLAG[@]}" "${BM25_WORKERS_FLAG[@]}" \
    > "$LOG_DIR/gov_fin.log" 2>&1 &
pid_gov_fin=$!
echo "Started: gov_fin @ $DEVICE_GOV_FIN (extra=${EXTRA_DEVICES_GOV_FIN:-none}) (pid=$pid_gov_fin)"

echo
echo "Both runs in progress... (waiting for them to finish)"
echo

fail=0
if wait "$pid_fin_gov"; then
  echo "Done: fin_gov"
else
  echo "Failed: fin_gov (see $LOG_DIR/fin_gov.log)" >&2
  fail=1
fi

if wait "$pid_gov_fin"; then
  echo "Done: gov_fin"
else
  echo "Failed: gov_fin (see $LOG_DIR/gov_fin.log)" >&2
  fail=1
fi

echo
if [ "$fail" -ne 0 ]; then
  echo "Some directions failed. Aggregating the successful outputs anyway." >&2
fi

echo "===== Aggregating ====="
# No tag was given, so file names have no _tag suffix (e.g. rag_ours_only_fin_gov_full_n1_regex.entries.jsonl)
python "$AGGREGATE_SCRIPT" \
    "$OUT_DIR/${OUTPUT_PREFIX}_fin_gov_${QUERY_SET}_n1_${TOKENIZER}.entries.jsonl" \
    "$OUT_DIR/${OUTPUT_PREFIX}_gov_fin_${QUERY_SET}_n1_${TOKENIZER}.entries.jsonl" \
    --out "$OUT_DIR/aggregate_once.json"

echo
if [ "$fail" -ne 0 ]; then
  echo "Finished, but some directions failed. Check the output above and the logs in $LOG_DIR."
  exit 1
fi
echo "All succeeded. Final results: $OUT_DIR/aggregate_once.json (+ .table.md, .confusion.md)"
