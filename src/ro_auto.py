from __future__ import annotations

import hashlib
import json
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from src.image_loader import SUPPORTED_EXTENSIONS, get_captured_at, natural_sort_key
from src.settings import AppConfig

DEFAULT_RO_SOURCE_DIR = Path(r"C:\Gravity\Ragnarok\ScreenShot")
DEFAULT_FILENAME_REGEX = r"^screenNoatun\d+\.(jpg|jpeg|png)$"
VALID_OCR_MODES = {"accuracy", "speed"}


@dataclass(frozen=True)
class ROSourceImage:
    path: Path
    captured_at: datetime
    size: int


@dataclass(frozen=True)
class ROBatch:
    batch_id: str
    signature: str
    captured_from: datetime
    captured_to: datetime
    images: list[ROSourceImage]


@dataclass(frozen=True)
class RORunPlan:
    batch: ROBatch
    staging_dir: Path
    output_dir: Path
    debug_dir: Path

    @property
    def log_path(self) -> Path:
        return self.output_dir / "ro_auto.log"

    @property
    def manifest_path(self) -> Path:
        return self.staging_dir / "batch_manifest.json"


@dataclass
class ROAutoConfig:
    source_dir: Path
    filename_regex: str
    batch_window_seconds: int
    staging_root: Path
    output_root: Path
    debug_root: Path
    ocr_mode: str
    debug_save_enabled: bool
    skip_already_processed_batch: bool

    @classmethod
    def defaults(cls, project_root: Path) -> "ROAutoConfig":
        return cls(
            source_dir=DEFAULT_RO_SOURCE_DIR,
            filename_regex=DEFAULT_FILENAME_REGEX,
            batch_window_seconds=60,
            staging_root=project_root / "input" / "ro_auto",
            output_root=project_root / "output" / "ro_auto",
            debug_root=project_root / "debug" / "ro_auto",
            ocr_mode="accuracy",
            debug_save_enabled=False,
            skip_already_processed_batch=True,
        )

    @classmethod
    def load(cls, path: Path, project_root: Path) -> "ROAutoConfig":
        defaults = cls.defaults(project_root)
        if not path.exists():
            defaults.validate()
            return defaults

        data = json.loads(path.read_text(encoding="utf-8"))
        config = cls(
            source_dir=_resolve_path(data.get("source_dir"), project_root, defaults.source_dir),
            filename_regex=str(data.get("filename_regex", defaults.filename_regex)),
            batch_window_seconds=int(data.get("batch_window_seconds", defaults.batch_window_seconds)),
            staging_root=_resolve_path(data.get("staging_root"), project_root, defaults.staging_root),
            output_root=_resolve_path(data.get("output_root"), project_root, defaults.output_root),
            debug_root=_resolve_path(data.get("debug_root"), project_root, defaults.debug_root),
            ocr_mode=str(data.get("ocr_mode", defaults.ocr_mode)),
            debug_save_enabled=bool(data.get("debug_save_enabled", defaults.debug_save_enabled)),
            skip_already_processed_batch=bool(
                data.get("skip_already_processed_batch", defaults.skip_already_processed_batch)
            ),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.batch_window_seconds <= 0:
            raise ValueError("ro_auto batch_window_seconds must be greater than 0.")
        if self.ocr_mode not in VALID_OCR_MODES:
            raise ValueError("ro_auto ocr_mode must be either 'accuracy' or 'speed'.")
        re.compile(self.filename_regex, re.IGNORECASE)


@dataclass
class ROAutoState:
    last_processed_signature: str | None = None
    last_batch_id: str | None = None
    last_output_dir: str | None = None
    last_processed_at: str | None = None

    @classmethod
    def load(cls, path: Path) -> "ROAutoState":
        if not path.exists():
            return cls()
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            last_processed_signature=data.get("last_processed_signature"),
            last_batch_id=data.get("last_batch_id"),
            last_output_dir=data.get("last_output_dir"),
            last_processed_at=data.get("last_processed_at"),
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "last_processed_signature": self.last_processed_signature,
            "last_batch_id": self.last_batch_id,
            "last_output_dir": self.last_output_dir,
            "last_processed_at": self.last_processed_at,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_ro_pipeline_config(config_path: Path, auto_config: ROAutoConfig) -> AppConfig:
    if not config_path.exists():
        raise FileNotFoundError("ROI config not found. Launch the normal mode once and save config/region.json first.")

    config = AppConfig.load(config_path)
    missing_regions = [name for name, region in config.regions.items() if region is None]
    if missing_regions:
        joined = ", ".join(missing_regions)
        raise ValueError(f"Saved ROI config is incomplete. Missing regions: {joined}")

    config.ocr.mode = auto_config.ocr_mode
    config.debug_save_enabled = auto_config.debug_save_enabled
    return config


def discover_latest_batch(config: ROAutoConfig) -> ROBatch | None:
    if not config.source_dir.exists():
        raise FileNotFoundError(f"Screenshot folder not found: {config.source_dir}")
    if not config.source_dir.is_dir():
        raise NotADirectoryError(f"Screenshot source is not a directory: {config.source_dir}")

    pattern = re.compile(config.filename_regex, re.IGNORECASE)
    candidates: list[ROSourceImage] = []

    for path in config.source_dir.iterdir():
        if not path.is_file():
            continue
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        if not pattern.match(path.name):
            continue
        candidates.append(
            ROSourceImage(
                path=path,
                captured_at=get_captured_at(path),
                size=path.stat().st_size,
            )
        )

    if not candidates:
        return None

    candidates.sort(key=lambda item: (item.captured_at, natural_sort_key(item.path)))
    latest_captured_at = candidates[-1].captured_at
    window_seconds = float(config.batch_window_seconds)
    selected = [
        item for item in candidates if 0 <= (latest_captured_at - item.captured_at).total_seconds() <= window_seconds
    ]
    selected.sort(key=lambda item: natural_sort_key(item.path))

    captured_from = min(item.captured_at for item in selected)
    captured_to = max(item.captured_at for item in selected)
    signature = _build_signature(selected, config.batch_window_seconds)
    batch_id = f"capture_{captured_to.strftime('%Y%m%d_%H%M%S')}_{len(selected):02d}shots"
    return ROBatch(
        batch_id=batch_id,
        signature=signature,
        captured_from=captured_from,
        captured_to=captured_to,
        images=selected,
    )


def is_already_processed(batch: ROBatch, state: ROAutoState, config: ROAutoConfig, *, force: bool) -> bool:
    if force:
        return False
    if not config.skip_already_processed_batch:
        return False
    return state.last_processed_signature == batch.signature


def create_run_plan(config: ROAutoConfig, batch: ROBatch) -> RORunPlan:
    run_id = _ensure_unique_run_id(
        batch.batch_id,
        config.staging_root,
        config.output_root,
        config.debug_root,
    )
    return RORunPlan(
        batch=batch,
        staging_dir=config.staging_root / run_id,
        output_dir=config.output_root / run_id,
        debug_dir=config.debug_root / run_id,
    )


def stage_batch(plan: RORunPlan, source_dir: Path) -> None:
    plan.staging_dir.mkdir(parents=True, exist_ok=False)

    for image in plan.batch.images:
        destination = plan.staging_dir / image.path.name
        shutil.copy2(image.path, destination)

    manifest = {
        "batch_id": plan.batch.batch_id,
        "signature": plan.batch.signature,
        "source_dir": str(source_dir),
        "captured_from": plan.batch.captured_from.isoformat(timespec="seconds"),
        "captured_to": plan.batch.captured_to.isoformat(timespec="seconds"),
        "images": [
            {
                "name": image.path.name,
                "source_path": str(image.path),
                "captured_at": image.captured_at.isoformat(timespec="seconds"),
                "size": image.size,
            }
            for image in plan.batch.images
        ],
    }
    plan.manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def record_processed_batch(state_path: Path, plan: RORunPlan) -> None:
    state = ROAutoState(
        last_processed_signature=plan.batch.signature,
        last_batch_id=plan.batch.batch_id,
        last_output_dir=str(plan.output_dir),
        last_processed_at=datetime.now().isoformat(timespec="seconds"),
    )
    state.save(state_path)


def build_batch_summary(batch: ROBatch) -> str:
    return (
        f"{batch.batch_id} | images={len(batch.images)} | "
        f"captured_from={batch.captured_from.strftime('%Y-%m-%d %H:%M:%S')} | "
        f"captured_to={batch.captured_to.strftime('%Y-%m-%d %H:%M:%S')}"
    )


def _resolve_path(value: Any, project_root: Path, fallback: Path) -> Path:
    if value in (None, ""):
        return fallback
    path = Path(str(value))
    if path.is_absolute():
        return path
    return project_root / path


def _build_signature(images: list[ROSourceImage], batch_window_seconds: int) -> str:
    payload = {
        "batch_window_seconds": batch_window_seconds,
        "images": [
            {
                "name": image.path.name,
                "path": str(image.path),
                "captured_at": image.captured_at.isoformat(timespec="seconds"),
                "size": image.size,
            }
            for image in images
        ],
    }
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _ensure_unique_run_id(batch_id: str, *roots: Path) -> str:
    candidate = batch_id
    suffix = 2
    while any((root / candidate).exists() for root in roots):
        candidate = f"{batch_id}_r{suffix:02d}"
        suffix += 1
    return candidate
