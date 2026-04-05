from __future__ import annotations

import csv
from pathlib import Path

FIELDNAMES = [
    "source_file",
    "captured_at",
    "row_index",
    "item_name_raw",
    "item_name_normalized",
    "quantity_raw",
    "quantity_normalized",
    "price_raw",
    "price_normalized",
    "price_suspect",
]


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
