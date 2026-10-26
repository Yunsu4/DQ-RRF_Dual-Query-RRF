"""Load per-direction corpus documents from gold_final.csv and check them against the
delivered corpus.jsonl files.

  1. Read gold_final.csv, split it by category and assign doc_ids.
  2. Rebuild the distractors (the opposite category) and compare them with the delivered
     corpus.jsonl.

No document is dropped or filtered. Counts are printed at each step.
"""
from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass

import config
from config import DIRECTIONS, GOLD_FINAL_CSV

# Call transcripts are long and exceed the default csv field size limit.
csv.field_size_limit(sys.maxsize)


@dataclass(frozen=True)
class GoldDoc:
    doc_id: str        # gold_finance_00001 format
    review_id: str     # for tracing back to gold_final.csv
    category: str      # gold_label: finance / government
    text: str


def load_gold_docs() -> dict[str, list[GoldDoc]]:
    """Read gold_final.csv split by category.

    doc_ids are assigned within each category in file order as gold_<category>_<number>.
    The delivered corpus.jsonl was built with the same rule, so the numbers must match.
    """
    by_category: dict[str, list[GoldDoc]] = {"finance": [], "government": []}
    total = 0
    with GOLD_FINAL_CSV.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            category = row["gold_label"].strip()
            if category not in by_category:
                raise ValueError(f"Unknown gold_label {category!r} ({row['review_id']})")
            seq = len(by_category[category]) + 1
            by_category[category].append(GoldDoc(
                doc_id=f"gold_{category}_{seq:05d}",
                review_id=row["review_id"],
                category=category,
                text=row["text"],
            ))
            total += 1

    counts = ", ".join(f"{cat} {len(docs)}" for cat, docs in by_category.items())
    print(f"gold_final.csv: {total} documents ({counts})", flush=True)
    return by_category


def verify_against_delivered(by_category: dict[str, list[GoldDoc]]) -> None:
    """Check that the rebuilt distractors are identical to the delivered corpus.jsonl.

    Adding or removing even one row in gold_final.csv shifts every doc_id after it. A saved
    qrels.json would then silently point to different documents and produce wrong numbers,
    so this stops the run instead of warning.
    """
    for direction in DIRECTIONS.values():
        rebuilt = by_category[direction.corpus_category]
        path = direction.delivered_corpus_path

        delivered = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                delivered.append((rec["doc_id"], rec.get("text", "")))

        if len(delivered) != len(rebuilt):
            raise SystemExit(
                f"{direction.key}: the delivered corpus has {len(delivered)} documents but "
                f"gold_final.csv yields {len(rebuilt)} {direction.corpus_category} documents. "
                f"Data and code versions do not match.")

        diff = [(i + 1, delivered_id, doc.doc_id)
                for i, ((delivered_id, delivered_text), doc)
                in enumerate(zip(delivered, rebuilt))
                if delivered_id != doc.doc_id or delivered_text != doc.text]
        if diff:
            print(f"{direction.key}: {len(diff)} documents differ from the delivered corpus. First 3:")
            for line_no, delivered_id, rebuilt_id in diff[:3]:
                print(f"  line {line_no} — delivered {delivered_id} / rebuilt {rebuilt_id}")
            raise SystemExit("Do not run the evaluation until this is resolved.")

        print(f"{direction.key}: {len(delivered)} distractors match the delivered corpus", flush=True)


if __name__ == "__main__":
    # Running this file directly only checks the data path and compares with the delivered corpus.
    config.check_data_root()
    print(f"Data: {config.DATA_ROOT}")
    verify_against_delivered(load_gold_docs())
