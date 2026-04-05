from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RowSlice:
    row_index: int
    image: np.ndarray
    y1: int
    y2: int
    ocr_input: np.ndarray | None = None


def split_rows(roi_image: np.ndarray, row_count: int) -> list[RowSlice]:
    if row_count < 1:
        raise ValueError("row_count must be >= 1")
    height = roi_image.shape[0]
    boundaries = [round(height * index / row_count) for index in range(row_count + 1)]
    rows: list[RowSlice] = []
    for index in range(row_count):
        y1 = boundaries[index]
        y2 = boundaries[index + 1]
        rows.append(RowSlice(row_index=index + 1, image=roi_image[y1:y2, :].copy(), y1=y1, y2=y2))
    return rows
