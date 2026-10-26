"""Experiment settings — data paths and directions live here.

There are two directions:
  fin_gov : search the government corpus with finance queries. Needles are finance documents.
  gov_fin : search the finance corpus with government queries. Needles are government documents.

Corpus documents come from the relabelled gold set (gold_final.csv), and its gold_label is
taken as the document category.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parent

# --- Paths ---
# The data is not included in the repository. The data folder (the one containing
# gold_final.csv) must be set via the VISHING_DATA_ROOT environment variable; otherwise
# check_data_root() stops. Nothing is ever written to the data folder.
DATA_ROOT_ENV = os.environ.get("VISHING_DATA_ROOT", "")
DATA_ROOT = Path(DATA_ROOT_ENV).resolve()

GOLD_FINAL_CSV = DATA_ROOT / "gold_final.csv"


@dataclass(frozen=True)
class Direction:
    key: str                 # fin_gov / gov_fin
    query_category: str      # category of the queries (and needles)
    corpus_category: str     # category of the distractors
    query_full_csv: Path
    delivered_corpus_path: Path   # distractor corpus as originally delivered

    def query_csv(self, query_set: str) -> Path:
        if query_set == "full":
            return self.query_full_csv
        raise ValueError(f"Only query_set='full' is supported: {query_set!r}")


DIRECTIONS = {
    "fin_gov": Direction(
        key="fin_gov",
        query_category="finance",
        corpus_category="government",
        query_full_csv=DATA_ROOT / "finance_ood_challenge.csv",
        delivered_corpus_path=DATA_ROOT / "finance_eval" / "corpus.jsonl",
    ),
    "gov_fin": Direction(
        key="gov_fin",
        query_category="government",
        corpus_category="finance",
        query_full_csv=DATA_ROOT / "government_ood_challenge.csv",
        delivered_corpus_path=DATA_ROOT / "government_eval" / "corpus.jsonl",
    ),
}


def check_data_root(query_set: str = "full") -> None:
    """Stop here if the data path is wrong. Only the query csv for query_set is checked."""
    if not DATA_ROOT_ENV:
        raise SystemExit(
            "VISHING_DATA_ROOT is not set. Point it to the folder containing gold_final.csv.\n"
            "  export VISHING_DATA_ROOT=/path/to/data")
    needed = [GOLD_FINAL_CSV]
    for direction in DIRECTIONS.values():
        needed += [direction.query_csv(query_set), direction.delivered_corpus_path]
    missing = [p for p in needed if not p.exists()]
    if missing:
        print(f"Required files not found in the data folder: {DATA_ROOT}")
        for path in missing:
            print(f"  missing: {path}")
        raise SystemExit(
            "Set VISHING_DATA_ROOT to the folder containing gold_final.csv.\n"
            "  export VISHING_DATA_ROOT=/path/to/data")
