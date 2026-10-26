"""Collect the per-entry logs (entries.jsonl) from one or more rag_official.py runs and compute
the combined confusion matrix and overall recall@20 (hits / total instances). It also reads the
direction (fin_gov/gov_fin) and tag (tag1/tag2/tag3...) from each file name and builds a
"Try index × direction" recall@20 table (Markdown).

recall@20 is computed per file (run) first, and those values are averaged with equal weight
(per-file macro average): each file counts once, even if files have different numbers of
entries. The confusion matrix (TP/FN) is the total over all entries from all files.

Example: running both fin_gov and gov_fin with tag1/tag2/tag3 produces 6 entries.jsonl files.
All 6 are aggregated as above, and a table like this is produced:

    | Try index | Recall@20 |  |  |
    | --- | --- | --- | --- |
    |  | Target Finance | Target Government | Avg. |
    | 1 | 0.7970 | 0.8557 | 0.8264 |
    | 2 | 0.7924 | 0.8552 | 0.8238 |
    | 3 | 0.7924 | 0.8537 | 0.8231 |
    | Avg. | 0.7951 | 0.8545 | 0.8249 |

    python ex/aggregate_ours_results.py \
        ex/outputs/rag_ours_only_fin_gov_full_n1_regex_tag1.entries.jsonl \
        ex/outputs/rag_ours_only_fin_gov_full_n1_regex_tag2.entries.jsonl \
        ex/outputs/rag_ours_only_fin_gov_full_n1_regex_tag3.entries.jsonl \
        ex/outputs/rag_ours_only_gov_fin_full_n1_regex_tag1.entries.jsonl \
        ex/outputs/rag_ours_only_gov_fin_full_n1_regex_tag2.entries.jsonl \
        ex/outputs/rag_ours_only_gov_fin_full_n1_regex_tag3.entries.jsonl \
        --out ex/outputs/aggregate_summary.json

    # Files can also be selected with a glob pattern instead of listing them
    python ex/aggregate_ours_results.py \
        --glob "ex/outputs/rag_ours_only_*_full_n1_regex_tag*.entries.jsonl" \
        --out ex/outputs/aggregate_summary.json
"""
from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

# File names produced by rag_official.py: rag_ours_only_{direction}_{query_set}_n{N}_{tokenizer}[_{tag}].entries.jsonl
FNAME_RE = re.compile(
    r"^rag_ours_only_(?P<direction>.+?)_(?P<query_set>pilot|full)_n(?P<n>\d+)"
    r"_(?P<tokenizer>regex|kiwi)(?:_(?P<tag>.+))?\.entries\.jsonl$"
)

# Direction key -> label used in tables. Directions not listed here use the key itself.
DIRECTION_LABELS = {
    "fin_gov": "Target Finance",
    "gov_fin": "Target Government",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("entries_files", nargs="*",
                    help="Paths of entries.jsonl files to aggregate")
    p.add_argument("--glob", dest="glob_pattern", default=None,
                    help="Glob pattern selecting entries.jsonl files "
                         "(e.g. 'ex/outputs/rag_ours_only_*_full_n1_regex_tag*.entries.jsonl'). "
                         "Combined with any positional files.")
    p.add_argument("--out", default=None,
                    help="JSON path to save the aggregate results. If omitted, nothing is saved.")
    args = p.parse_args()
    if not args.entries_files and not args.glob_pattern:
        p.error("Specify at least one entries.jsonl file (positional or --glob).")
    return args


def load_entries(path: Path) -> list[dict]:
    entries = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entries.append(json.loads(line))
    return entries


def parse_filename(path: Path) -> tuple[str | None, str | None]:
    """Extract (direction, tag) from a file name. Returns (None, None) if the name does not
    follow the rag_official.py format (such files are left out of the tables but still
    included in the overall totals)."""
    m = FNAME_RE.match(path.name)
    if not m:
        return None, None
    return m.group("direction"), m.group("tag")


def build_recall_table_markdown(table: dict[str, dict[str, float]]) -> str:
    """Turn {tag: {direction: recall@20}} into a "Try index x direction" Markdown table.
    Rows = tags (try index); columns = recall per direction + the row average (Avg.); the last
    row holds each column's average over tags and the grand average."""
    if not table:
        return ""

    seen_directions: list[str] = []
    for row in table.values():
        for d in row:
            if d not in seen_directions:
                seen_directions.append(d)
    known_order = [d for d in ("fin_gov", "gov_fin") if d in seen_directions]
    rest = [d for d in seen_directions if d not in known_order]
    directions = known_order + rest
    labels = [DIRECTION_LABELS.get(d, d) for d in directions]

    def tag_sort_key(t: str):
        m = re.search(r"\d+", t)
        return (0, int(m.group())) if m else (1, t)
    tags = sorted(table.keys(), key=tag_sort_key)

    def tag_label(t: str) -> str:
        # For the common "tag1", "tag2" pattern, show only the number as the try index (1, 2, 3).
        m = re.fullmatch(r"(?i:tag)(\d+)", t)
        return m.group(1) if m else t

    n_data_cols = len(directions) + 1  # + Avg. column
    lines = []
    lines.append("| Try index | Recall@20 |" + " |" * (n_data_cols - 1))
    lines.append("| --- " * (n_data_cols + 1) + "|")
    lines.append("|  | " + " | ".join(labels) + " | Avg. |")

    col_sums = [0.0] * len(directions)
    col_counts = [0] * len(directions)
    for tag in tags:
        row_vals = [table[tag].get(d) for d in directions]
        present = [v for v in row_vals if v is not None]
        row_avg = sum(present) / len(present) if present else None
        cells = []
        for j, v in enumerate(row_vals):
            if v is None:
                cells.append("-")
            else:
                cells.append(f"{v:.4f}")
                col_sums[j] += v
                col_counts[j] += 1
        row_avg_str = f"{row_avg:.4f}" if row_avg is not None else "-"
        lines.append(f"| {tag_label(tag)} | " + " | ".join(cells) + f" | {row_avg_str} |")

    col_avgs = [col_sums[j] / col_counts[j] if col_counts[j] else None for j in range(len(directions))]
    present_col_avgs = [a for a in col_avgs if a is not None]
    grand_avg = sum(present_col_avgs) / len(present_col_avgs) if present_col_avgs else None
    avg_cells = [f"{a:.4f}" if a is not None else "-" for a in col_avgs]
    grand_avg_str = f"{grand_avg:.4f}" if grand_avg is not None else "-"
    lines.append("| Avg. | " + " | ".join(avg_cells) + f" | {grand_avg_str} |")

    return "\n".join(lines)


def build_confusion_matrix_markdown(confusion_by_direction: dict[str, dict[str, int]],
                                    total: dict[str, int]) -> str:
    """Build a Markdown table of the per-direction confusion matrix (TP/FN/TN/FP, number of
    entries, recall@20) with a total row at the end. Every entry has a gold document in this
    recall evaluation, so TN/FP are always 0."""
    if not confusion_by_direction:
        return ""

    known_order = [d for d in ("fin_gov", "gov_fin") if d in confusion_by_direction]
    rest = [d for d in confusion_by_direction if d not in known_order]
    directions = known_order + rest

    lines = []
    lines.append("| | TP (Hit) | FN (Miss) | TN | FP | Total | Recall@20 |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for d in directions:
        c = confusion_by_direction[d]
        n = c["TP"] + c["FN"]
        recall = c["TP"] / n if n else 0.0
        label = DIRECTION_LABELS.get(d, d)
        lines.append(f"| {label} | {c['TP']} | {c['FN']} | {c['TN']} | {c['FP']} | "
                     f"{n} | {recall:.4f} |")

    n_total = total["TP"] + total["FN"]
    recall_total = total["TP"] / n_total if n_total else 0.0
    lines.append(f"| Total | {total['TP']} | {total['FN']} | "
                 f"{total['TN']} | {total['FP']} | {n_total} | "
                 f"{recall_total:.4f} |")

    return "\n".join(lines)


def main() -> None:
    args = parse_args()

    files = [Path(p) for p in args.entries_files]
    if args.glob_pattern:
        files += [Path(p) for p in sorted(glob.glob(args.glob_pattern))]

    # Deduplicate files matched by both the positional list and --glob (order preserved).
    seen = set()
    unique_files = []
    for f in files:
        rf = f.resolve()
        if rf not in seen:
            seen.add(rf)
            unique_files.append(f)
    files = unique_files

    if not files:
        raise SystemExit("No entries.jsonl files to aggregate.")

    print(f"Aggregating {len(files)} files:")
    for f in files:
        print(f"  - {f}")

    all_entries = []
    per_file_stats = []
    for f in files:
        if not f.exists():
            raise SystemExit(f"File not found: {f}")
        entries = load_entries(f)
        tp = sum(1 for e in entries if e["hit_at_20"])
        fn = len(entries) - tp
        per_file_stats.append({
            "file": str(f), "n": len(entries), "TP": tp, "FN": fn,
            "recall@20": round(tp / len(entries), 4) if entries else 0.0,
        })
        all_entries.extend(entries)

    print()
    print("Per-file statistics:")
    for s in per_file_stats:
        print(f"  {s['file']}: n={s['n']}, TP={s['TP']}, FN={s['FN']}, "
              f"recall@20={s['recall@20']:.4f}")

    # Read direction/tag from file names to build the "Try index x direction" recall@20 table.
    recall_table: dict[str, dict[str, float]] = {}
    # Also build a per-direction confusion matrix (TP/FN totals pooled over all tags).
    confusion_by_direction: dict[str, dict[str, int]] = {}
    for f, s in zip(files, per_file_stats):
        direction, tag = parse_filename(f)
        if direction is None:
            continue
        tag = tag or "(no tag)"
        recall_table.setdefault(tag, {})[direction] = s["recall@20"]
        cd = confusion_by_direction.setdefault(direction, {"TP": 0, "FN": 0, "TN": 0, "FP": 0})
        cd["TP"] += s["TP"]
        cd["FN"] += s["FN"]
    table_md = build_recall_table_markdown(recall_table)

    tp_total = sum(1 for e in all_entries if e["hit_at_20"])
    n_total = len(all_entries)
    fn_total = n_total - tp_total
    confusion = {"TP": tp_total, "FN": fn_total, "TN": 0, "FP": 0}

    # Per-file average: compute recall@20 per file (run) first, then take the plain mean
    # (each file counts once, even if files have different numbers of entries).
    recall_macro = sum(s["recall@20"] for s in per_file_stats) / len(per_file_stats)

    confusion_md = build_confusion_matrix_markdown(confusion_by_direction, confusion)

    print()
    print(f"=== Overall ({len(files)} files, {n_total} entries) ===")
    print("Confusion Matrix (at recall@20; every entry has a gold document, so TN/FP=0)")
    if confusion_md:
        print(confusion_md)
    else:
        print(f"  TP(Hit)  = {confusion['TP']}")
        print(f"  FN(Miss) = {confusion['FN']}")
        print(f"  TN       = {confusion['TN']}")
        print(f"  FP       = {confusion['FP']}")
    print(f"  Overall recall@20 (per-file average) = {recall_macro:.4f} ({tp_total}/{n_total})")

    if table_md:
        print()
        print("Try index x direction recall@20 table:")
        print(table_md)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({
            "files": [str(f) for f in files],
            "per_file": per_file_stats,
            "n_total": n_total,
            "confusion_matrix": confusion,
            "confusion_matrix_by_direction": confusion_by_direction,
            "confusion_matrix_markdown": confusion_md,
            "recall@20_macro_avg_over_files": round(recall_macro, 4),
            "hits_over_total": f"{tp_total}/{n_total}",
            "recall_table": recall_table,
            "recall_table_markdown": table_md,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print()
        print(f"Aggregate results saved: {out_path}")

        if table_md:
            md_path = out_path.with_suffix(".table.md")
            md_path.write_text(table_md + "\n", encoding="utf-8")
            print(f"Recall table saved: {md_path}")

        if confusion_md:
            cm_path = out_path.with_suffix(".confusion.md")
            cm_path.write_text(confusion_md + "\n", encoding="utf-8")
            print(f"Confusion matrix table saved: {cm_path}")


if __name__ == "__main__":
    main()
