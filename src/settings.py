from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REGION_ORDER = ("item_name", "quantity", "price")
REGION_COLORS = {
    "item_name": (255, 0, 0),
    "quantity": (0, 200, 0),
    "price": (0, 0, 255),
}


@dataclass
class Region:
    x: int
    y: int
    w: int
    h: int

    def to_xyxy(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.x + self.w, self.y + self.h


@dataclass
class ImageSize:
    width: int
    height: int


@dataclass
class PreprocessSettings:
    enabled: bool = False
    grayscale: bool = True
    threshold_enabled: bool = False
    threshold_value: int = 180
    scale: float = 2.0
    median_blur: int = 0


@dataclass
class OCRSettings:
    mode: str = "accuracy"
    device: str = "auto"


@dataclass
class AppConfig:
    image_size: ImageSize
    row_count: int
    regions: dict[str, Region | None]
    preprocess: PreprocessSettings
    ocr: OCRSettings
    debug_save_enabled: bool = False

    @classmethod
    def create_default(cls, width: int, height: int, row_count: int = 10) -> "AppConfig":
        return cls(
            image_size=ImageSize(width=width, height=height),
            row_count=row_count,
            regions={name: None for name in REGION_ORDER},
            preprocess=PreprocessSettings(),
            ocr=OCRSettings(),
            debug_save_enabled=False,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AppConfig":
        image_size = ImageSize(**data["image_size"])
        preprocess = PreprocessSettings(**data.get("preprocess", {}))
        ocr = OCRSettings(**data.get("ocr", {}))
        regions: dict[str, Region | None] = {}
        region_data = data.get("regions", {})
        for name in REGION_ORDER:
            value = region_data.get(name)
            regions[name] = Region(**value) if value else None
        return cls(
            image_size=image_size,
            row_count=int(data["row_count"]),
            regions=regions,
            preprocess=preprocess,
            ocr=ocr,
            debug_save_enabled=bool(data.get("debug_save_enabled", False)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_size": asdict(self.image_size),
            "row_count": self.row_count,
            "regions": {
                name: asdict(region) if region else None
                for name, region in self.regions.items()
            },
            "preprocess": asdict(self.preprocess),
            "ocr": asdict(self.ocr),
            "debug_save_enabled": self.debug_save_enabled,
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> "AppConfig":
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))
