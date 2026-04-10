from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def natural_sort_key(path: Path) -> list[object]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]


@dataclass
class LoadedImage:
    path: Path
    image: object
    width: int
    height: int
    captured_at: datetime


def list_images(folder: Path) -> list[Path]:
    files = [path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS]
    return sorted(files, key=natural_sort_key)


def get_captured_at(path: Path) -> datetime:
    stat = path.stat()
    captured_ts = getattr(stat, "st_ctime", stat.st_mtime)
    return datetime.fromtimestamp(captured_ts)


def load_image(path: Path) -> LoadedImage:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unable to read image: {path}")
    height, width = image.shape[:2]
    captured_at = get_captured_at(path)
    return LoadedImage(
        path=path,
        image=image,
        width=width,
        height=height,
        captured_at=captured_at,
    )
