"""Query data structure and file loader."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Query:
    qid: str
    text: str
    category: str
    label: int | float | str
    date: str


def _parse_label(value: str | int | float | None) -> int | float | str:
    """Convert a label to a number. Non-numeric values are returned unchanged."""
    if value is None:
        return ""
    if isinstance(value, (int, float)):
        return value
    try:
        number = float(value)
    except ValueError:
        return value
    return int(number) if number.is_integer() else number


def load_queries(path: Path, domain: str) -> list[Query]:
    """Read an OOD challenge csv. qids are assigned in file order."""
    queries: list[Query] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"date", "text", "category", "label"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
        for row_idx, row in enumerate(reader, start=1):
            queries.append(Query(
                qid=f"{domain}_q{row_idx:04d}",
                text=row["text"],
                category=row["category"],
                label=_parse_label(row.get("label")),
                date=row["date"],
            ))
    return queries
