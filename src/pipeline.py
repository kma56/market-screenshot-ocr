from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2

from src.csv_writer import write_csv
from src.image_loader import LoadedImage, list_images, load_image
from src.normalize import normalize_item_name, normalize_price, normalize_quantity
from src.ocr_engine import ItemNameOCREngine, NumericOCREngine
from src.preprocess import preprocess_price_roi, preprocess_roi
from src.row_splitter import split_rows
from src.settings import AppConfig
from src.ui_helpers import crop_region, draw_regions, save_debug_image


@dataclass
class ProcessResult:
    csv_path: Path
    processed_images: int
    total_rows: int
    errors: list[str]
    warnings: list[str]


class OCRPipeline:
    def __init__(self, config: AppConfig, input_dir: Path, output_dir: Path, debug_dir: Path, logger: logging.Logger) -> None:
        self.config = config
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.debug_dir = debug_dir
        self.logger = logger
        self.numeric_ocr_engine = NumericOCREngine(config.ocr)
        self.item_name_ocr_engine = ItemNameOCREngine(config.ocr)

    def run(self) -> ProcessResult:
        image_paths = list_images(self.input_dir)
        if not image_paths:
            raise ValueError("No .jpg/.png images found in the selected folder.")

        csv_rows: list[dict[str, str]] = []
        errors: list[str] = []
        warnings: list[str] = []
        processed_images = 0

        for image_path in image_paths:
            try:
                loaded = load_image(image_path)
                if not self._matches_resolution(loaded):
                    message = f"Skipped {image_path.name}: resolution {loaded.width}x{loaded.height} does not match config."
                    warnings.append(message)
                    self.logger.warning(message)
                    continue
                csv_rows.extend(self._process_image(loaded))
                processed_images += 1
            except Exception as exc:
                message = f"Failed to process {image_path.name}: {exc}"
                errors.append(message)
                self.logger.exception(message)

        csv_path = self.output_dir / f"ocr_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        write_csv(csv_path, csv_rows)
        return ProcessResult(
            csv_path=csv_path,
            processed_images=processed_images,
            total_rows=len(csv_rows),
            errors=errors,
            warnings=warnings,
        )

    def _matches_resolution(self, loaded: LoadedImage) -> bool:
        return (
            loaded.width == self.config.image_size.width
            and loaded.height == self.config.image_size.height
        )

    def _process_image(self, loaded: LoadedImage) -> list[dict[str, str]]:
        item_name_region = self.config.regions["item_name"]
        quantity_region = self.config.regions["quantity"]
        price_region = self.config.regions["price"]
        if item_name_region is None:
            raise ValueError("Item name ROI is not configured.")
        if quantity_region is None:
            raise ValueError("Quantity ROI is not configured.")
        if price_region is None:
            raise ValueError("Price ROI is not configured.")

        item_name_rows = split_rows(crop_region(loaded.image, item_name_region), self.config.row_count)
        quantity_rows = split_rows(crop_region(loaded.image, quantity_region), self.config.row_count)
        price_rows = split_rows(crop_region(loaded.image, price_region), self.config.row_count)
        output_rows: list[dict[str, str]] = []

        if self.config.debug_save_enabled:
            self._save_debug_overview(loaded)

        self._prepare_rows(item_name_rows, region_name="item_name")
        self._prepare_rows(quantity_rows, region_name="quantity")
        self._prepare_rows(price_rows, region_name="price")

        item_name_raw_values = self.item_name_ocr_engine.recognize_batch([row.ocr_input for row in item_name_rows])
        quantity_raw_values = self.numeric_ocr_engine.recognize_batch([row.ocr_input for row in quantity_rows])
        price_raw_values = self.numeric_ocr_engine.recognize_batch([row.ocr_input for row in price_rows])
        price_suspects = self._detect_price_suspects(price_raw_values)

        for item_name_row, quantity_row, price_row, item_name_raw, quantity_raw, price_raw, price_suspect in zip(
            item_name_rows,
            quantity_rows,
            price_rows,
            item_name_raw_values,
            quantity_raw_values,
            price_raw_values,
            price_suspects,
        ):
            item_name_normalized = normalize_item_name(item_name_raw)
            quantity_normalized = normalize_quantity(quantity_raw)
            price_normalized = normalize_price(price_raw)

            output_rows.append(
                {
                    "source_file": loaded.path.name,
                    "captured_at": loaded.captured_at.strftime("%Y-%m-%d %H:%M:%S"),
                    "row_index": str(quantity_row.row_index),
                    "item_name_raw": item_name_raw,
                    "item_name_normalized": item_name_normalized,
                    "quantity_raw": quantity_raw,
                    "quantity_normalized": quantity_normalized,
                    "price_raw": price_raw,
                    "price_normalized": price_normalized,
                    "price_suspect": "1" if price_suspect else "",
                }
            )

            if self.config.debug_save_enabled:
                stem_dir = self.debug_dir / loaded.path.stem
                save_debug_image(stem_dir / "item_name" / f"row_{item_name_row.row_index:02d}_raw.png", item_name_row.image)
                save_debug_image(stem_dir / "item_name" / f"row_{item_name_row.row_index:02d}_processed.png", item_name_row.ocr_input)
                save_debug_image(stem_dir / "quantity" / f"row_{quantity_row.row_index:02d}_raw.png", quantity_row.image)
                save_debug_image(stem_dir / "quantity" / f"row_{quantity_row.row_index:02d}_processed.png", quantity_row.ocr_input)
                save_debug_image(stem_dir / "price" / f"row_{price_row.row_index:02d}_raw.png", price_row.image)
                save_debug_image(stem_dir / "price" / f"row_{price_row.row_index:02d}_processed.png", price_row.ocr_input)

        self._mark_same_item_price_outliers(output_rows)
        return self._trim_trailing_empty_rows(output_rows)

    def _prepare_rows(self, rows, region_name: str) -> None:
        for row in rows:
            raw_row_image = row.image
            if region_name == "price":
                row.ocr_input = preprocess_price_roi(raw_row_image)
            else:
                row.ocr_input = preprocess_roi(raw_row_image, self.config.preprocess) if self.config.preprocess.enabled else raw_row_image

    def _save_debug_overview(self, loaded: LoadedImage) -> None:
        stem_dir = self.debug_dir / loaded.path.stem
        overview = draw_regions(loaded.image, self.config.regions)
        save_debug_image(stem_dir / "overview.png", overview)
        for region_name in ("item_name", "quantity", "price"):
            region = self.config.regions[region_name]
            if region:
                roi = crop_region(loaded.image, region)
                save_debug_image(stem_dir / f"{region_name}_roi.png", roi)
                lined = roi.copy()
                for row in split_rows(roi, self.config.row_count)[:-1]:
                    cv2.line(lined, (0, row.y2), (lined.shape[1], row.y2), (0, 255, 255), 1)
                save_debug_image(stem_dir / f"{region_name}_roi_rows.png", lined)

    def _trim_trailing_empty_rows(self, output_rows):
        trimmed_rows = list(output_rows)
        while len(trimmed_rows) > 1:
            index = len(trimmed_rows) - 1
            row = trimmed_rows[index]
            if not self._is_effectively_empty_row(row):
                break
            trimmed_rows.pop()
        return trimmed_rows

    def _is_effectively_empty_row(self, row_data: dict[str, str]) -> bool:
        if row_data["item_name_normalized"]:
            return False
        if row_data["quantity_normalized"]:
            return False
        if row_data["price_normalized"]:
            return False

        item_name_has_text = self._has_meaningful_item_name_text(row_data["item_name_raw"])
        quantity_has_text = self._has_meaningful_numeric_text(row_data["quantity_raw"])
        price_has_text = self._has_meaningful_numeric_text(row_data["price_raw"])
        return not any([item_name_has_text, quantity_has_text, price_has_text])

    def _has_meaningful_item_name_text(self, value: str) -> bool:
        if not value:
            return False
        cleaned = re.sub(r"[\s\-_.,'\"`~|/\\()\[\]{}]+", "", value)
        return len(cleaned) >= 2

    def _has_meaningful_numeric_text(self, value: str) -> bool:
        if not value:
            return False
        return bool(re.search(r"\d", value))

    def _detect_price_suspects(self, price_raw_values: list[str]) -> list[bool]:
        normalized_values = [normalize_price(value) for value in price_raw_values]
        parsed_values: list[int | None] = [int(value) if value else None for value in normalized_values]
        suspects = [False] * len(price_raw_values)

        for index, current in enumerate(parsed_values):
            if current is None:
                continue
            prev_value = self._nearest_numeric_value(parsed_values, index, -1)
            next_value = self._nearest_numeric_value(parsed_values, index, 1)

            if prev_value is not None and next_value is not None:
                if self._is_clear_local_outlier(prev_value, current, next_value):
                    suspects[index] = True

            raw_value = price_raw_values[index].strip()
            if re.search(r"(?i)(z2|2z)$", raw_value.replace(" ", "")):
                suspects[index] = True

        return suspects

    def _is_clear_local_outlier(self, prev_value: int, current: int, next_value: int) -> bool:
        if prev_value <= current <= next_value:
            return False

        # When only one side is inconsistent, prefer not to flag the current row
        # if it still agrees with the other side and the opposite neighbor looks
        # like the true outlier.
        if prev_value <= current and current > next_value:
            return not self._looks_like_neighbor_outlier(anchor=current, other=next_value)

        if current < prev_value and current <= next_value:
            return not self._looks_like_neighbor_outlier(anchor=current, other=prev_value)

        # If the current value breaks ordering on both sides, it is much more
        # likely that the current row itself is wrong.
        return True

    def _looks_like_neighbor_outlier(self, anchor: int, other: int) -> bool:
        if anchor <= 0 or other <= 0:
            return False
        ratio = min(anchor, other) / max(anchor, other)
        return ratio <= 0.5

    def _nearest_numeric_value(self, values: list[int | None], start_index: int, step: int) -> int | None:
        index = start_index + step
        while 0 <= index < len(values):
            if values[index] is not None:
                return values[index]
            index += step
        return None

    def _mark_same_item_price_outliers(self, output_rows: list[dict[str, str]]) -> None:
        grouped_rows: dict[str, list[tuple[int, int]]] = {}

        for index, row in enumerate(output_rows):
            item_name = row["item_name_normalized"].strip()
            price_value = row["price_normalized"].strip()
            if not item_name or not price_value:
                continue
            try:
                parsed_price = int(price_value)
            except ValueError:
                continue
            grouped_rows.setdefault(item_name, []).append((index, parsed_price))

        for entries in grouped_rows.values():
            if len(entries) < 4:
                continue

            valid_entries = [(row_index, price) for row_index, price in entries if price > 0]
            log_values = [math.log10(price) for _, price in valid_entries]
            if len(log_values) < 4:
                continue

            median_log = self._median(log_values)
            deviations = [abs(value - median_log) for value in log_values]
            mad = self._median(deviations)
            if mad <= 0:
                continue

            for (row_index, price), log_value in zip(valid_entries, log_values):
                modified_z_score = 0.6745 * abs(log_value - median_log) / mad
                if modified_z_score >= 3.5:
                    if not self._has_local_price_support(output_rows, row_index, price):
                        output_rows[row_index]["price_suspect"] = "1"

    def _median(self, values: list[float]) -> float:
        ordered = sorted(values)
        count = len(ordered)
        midpoint = count // 2
        if count % 2 == 1:
            return ordered[midpoint]
        return (ordered[midpoint - 1] + ordered[midpoint]) / 2.0

    def _has_local_price_support(self, output_rows: list[dict[str, str]], row_index: int, current_price: int) -> bool:
        current_item_name = output_rows[row_index]["item_name_normalized"].strip()
        if not current_item_name or current_price <= 0:
            return False

        for neighbor_index in (row_index - 1, row_index + 1):
            if not (0 <= neighbor_index < len(output_rows)):
                continue
            neighbor_row = output_rows[neighbor_index]
            if neighbor_row["item_name_normalized"].strip() != current_item_name:
                continue
            neighbor_price_text = neighbor_row["price_normalized"].strip()
            if not neighbor_price_text:
                continue
            try:
                neighbor_price = int(neighbor_price_text)
            except ValueError:
                continue
            if neighbor_price <= 0:
                continue

            ratio = min(current_price, neighbor_price) / max(current_price, neighbor_price)
            if ratio >= 0.85:
                return True

        return False
