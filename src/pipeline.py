from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

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
    graph_path: Path
    summary_txt_path: Path
    processed_images: int
    total_rows: int
    errors: list[str]
    warnings: list[str]
    has_price_suspects: bool


class OCRPipeline:
    def __init__(
        self, config: AppConfig, input_dir: Path, output_dir: Path, debug_dir: Path, logger: logging.Logger
    ) -> None:
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
                    message = (
                        f"Skipped {image_path.name}: resolution {loaded.width}x{loaded.height} does not match config."
                    )
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
        graph_path = csv_path.with_name(f"{csv_path.stem}_price_quantity_chart.png")
        summary_txt_path = csv_path.with_name(f"{csv_path.stem}_summary.txt")
        self._write_outputs(csv_rows, graph_path, summary_txt_path)
        return ProcessResult(
            csv_path=csv_path,
            graph_path=graph_path,
            summary_txt_path=summary_txt_path,
            processed_images=processed_images,
            total_rows=len(csv_rows),
            errors=errors,
            warnings=warnings,
            has_price_suspects=any(row["price_suspect"].strip() for row in csv_rows),
        )

    def _matches_resolution(self, loaded: LoadedImage) -> bool:
        return loaded.width == self.config.image_size.width and loaded.height == self.config.image_size.height

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
                save_debug_image(
                    stem_dir / "item_name" / f"row_{item_name_row.row_index:02d}_raw.png", item_name_row.image
                )
                save_debug_image(
                    stem_dir / "item_name" / f"row_{item_name_row.row_index:02d}_processed.png", item_name_row.ocr_input
                )
                save_debug_image(
                    stem_dir / "quantity" / f"row_{quantity_row.row_index:02d}_raw.png", quantity_row.image
                )
                save_debug_image(
                    stem_dir / "quantity" / f"row_{quantity_row.row_index:02d}_processed.png", quantity_row.ocr_input
                )
                save_debug_image(stem_dir / "price" / f"row_{price_row.row_index:02d}_raw.png", price_row.image)
                save_debug_image(
                    stem_dir / "price" / f"row_{price_row.row_index:02d}_processed.png", price_row.ocr_input
                )

        self._mark_same_item_price_outliers(output_rows)
        return self._trim_trailing_empty_rows(output_rows)

    def _prepare_rows(self, rows, region_name: str) -> None:
        for row in rows:
            raw_row_image = row.image
            if region_name == "price":
                row.ocr_input = preprocess_price_roi(raw_row_image)
            else:
                row.ocr_input = (
                    preprocess_roi(raw_row_image, self.config.preprocess)
                    if self.config.preprocess.enabled
                    else raw_row_image
                )

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

    def _write_outputs(self, csv_rows: list[dict[str, str]], graph_path: Path, summary_txt_path: Path) -> None:
        aggregated_points = self._aggregate_price_quantities(csv_rows)
        title = self._build_graph_title(csv_rows)
        self._write_price_quantity_chart(graph_path, aggregated_points, title)
        self._write_quantity_summary(summary_txt_path, aggregated_points)

    def _aggregate_price_quantities(self, csv_rows: list[dict[str, str]]) -> list[tuple[int, int]]:
        aggregated: dict[int, int] = {}
        for row in csv_rows:
            price_text = row["price_normalized"].strip()
            quantity_text = row["quantity_normalized"].strip()
            if not price_text or not quantity_text:
                continue
            try:
                price = int(price_text)
                quantity = int(quantity_text)
            except ValueError:
                continue
            if price <= 0 or quantity <= 0:
                continue
            aggregated[price] = aggregated.get(price, 0) + quantity
        return sorted(aggregated.items())

    def _build_graph_title(self, csv_rows: list[dict[str, str]]) -> str:
        if not csv_rows:
            return "OCR Result"

        first_row = csv_rows[0]
        item_name = first_row["item_name_normalized"].strip() or first_row["item_name_raw"].strip() or "不明アイテム"
        captured_at = first_row["captured_at"].strip() or "unknown"
        return f"{item_name}＠{captured_at}"

    def _write_quantity_summary(self, summary_txt_path: Path, aggregated_points: list[tuple[int, int]]) -> None:
        total_quantity = sum(quantity for _, quantity in aggregated_points)
        first_band_limit = 0
        first_band_quantity = 0

        if aggregated_points:
            minimum_price = aggregated_points[0][0]
            raw_limit = minimum_price * 1.5
            first_band_limit = math.floor(raw_limit)
            eligible_prices = [price for price, _ in aggregated_points if price <= raw_limit]
            if eligible_prices:
                first_band_limit = max(eligible_prices)
            first_band_quantity = sum(quantity for price, quantity in aggregated_points if price <= raw_limit)

        lines = [
            f"・全価格帯の合計数量：{total_quantity}",
            f"・{first_band_limit}zまでの合計数量：{first_band_quantity}",
        ]
        summary_txt_path.parent.mkdir(parents=True, exist_ok=True)
        summary_txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _write_price_quantity_chart(
        self, graph_path: Path, aggregated_points: list[tuple[int, int]], title: str
    ) -> None:
        width = 1600
        height = 900
        background = np.full((height, width, 3), 250, dtype=np.uint8)

        frame_color = (190, 190, 190)
        cv2.rectangle(background, (14, 14), (width - 14, height - 14), frame_color, 1)

        title_image = self._render_text_image(title, font_scale=0.62, thickness=1, text_color=(110, 110, 110))
        title_height = title_image.shape[0]
        title_x = 48
        title_y = 42
        self._paste_image(background, title_image, title_x, title_y)

        chart_left = 150
        chart_top = title_y + title_height + 32
        chart_right = width - 70
        chart_bottom = height - 130
        chart_width = chart_right - chart_left
        chart_height = chart_bottom - chart_top

        plot_background = (252, 252, 252)
        axis_color = (165, 165, 165)
        grid_color = (225, 225, 225)
        bar_color = (236, 127, 42)
        label_color = (236, 127, 42)

        cv2.rectangle(background, (chart_left, chart_top), (chart_right, chart_bottom), plot_background, -1)
        cv2.line(background, (chart_left, chart_bottom), (chart_right, chart_bottom), axis_color, 1)
        cv2.line(background, (chart_left, chart_top), (chart_left, chart_bottom), axis_color, 1)

        axis_x_label = self._render_text_image("価格", font_scale=0.6, thickness=1, text_color=(40, 40, 40))
        axis_x_y = chart_bottom + 52
        self._paste_image(background, axis_x_label, chart_left + (chart_width - axis_x_label.shape[1]) // 2, axis_x_y)

        axis_y_label = self._render_text_image(
            "数量", font_scale=0.6, thickness=1, rotate_ccw=True, text_color=(40, 40, 40)
        )
        axis_y_x = 42
        axis_y_y = chart_top + (chart_height - axis_y_label.shape[0]) // 2
        self._paste_image(background, axis_y_label, axis_y_x, axis_y_y)

        if not aggregated_points:
            empty_label = self._render_text_image("No valid data", font_scale=0.9, thickness=2)
            self._paste_image(
                background,
                empty_label,
                chart_left + (chart_width - empty_label.shape[1]) // 2,
                chart_top + (chart_height - empty_label.shape[0]) // 2,
            )
            graph_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(graph_path), background)
            return

        max_quantity = max(quantity for _, quantity in aggregated_points)
        y_tick_step = self._choose_y_tick_step(max_quantity)
        y_axis_max = max(y_tick_step, int(math.ceil(max_quantity / y_tick_step) * y_tick_step))
        tick_values = list(range(0, y_axis_max + y_tick_step, y_tick_step))

        for tick_value in tick_values:
            ratio = 0 if y_axis_max <= 0 else tick_value / y_axis_max
            y = int(chart_bottom - chart_height * ratio)
            cv2.line(background, (chart_left, y), (chart_right, y), grid_color, 1)
            tick_label = self._render_text_image(str(tick_value), font_scale=0.42, thickness=1, text_color=(70, 70, 70))
            self._paste_image(
                background, tick_label, chart_left - tick_label.shape[1] - 18, y - tick_label.shape[0] // 2
            )

        bar_count = len(aggregated_points)
        slot_width = chart_width / max(bar_count, 1)
        bar_width = max(10, int(slot_width * 0.62))

        for index, (price, quantity) in enumerate(aggregated_points):
            center_x = chart_left + int((index + 0.5) * chart_width / bar_count)
            half_width = bar_width // 2
            x1 = max(chart_left + 1, center_x - half_width)
            x2 = min(chart_right - 1, center_x + half_width)
            bar_height = 0 if y_axis_max <= 0 else int(chart_height * (quantity / y_axis_max))
            y1 = max(chart_top + 1, chart_bottom - bar_height)
            cv2.rectangle(background, (x1, y1), (x2, chart_bottom - 1), bar_color, -1)

            quantity_label = self._render_text_image(
                str(quantity), font_scale=0.36, thickness=1, text_color=label_color
            )
            quantity_x = center_x - quantity_label.shape[1] // 2
            quantity_y = max(chart_top + 4, y1 - quantity_label.shape[0] - 6)
            self._paste_image(background, quantity_label, quantity_x, quantity_y)

            price_label = self._render_text_image(
                f"{price}z", font_scale=0.34, thickness=1, rotate_ccw=True, text_color=(70, 70, 70)
            )
            label_x = center_x - price_label.shape[1] // 2
            label_y = chart_bottom + 10
            self._paste_image(background, price_label, label_x, label_y)

        graph_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(graph_path), background)

    def _render_text_image(
        self,
        text: str,
        *,
        font_scale: float,
        thickness: int,
        rotate_ccw: bool = False,
        text_color: tuple[int, int, int] = (20, 20, 20),
    ) -> np.ndarray[Any, Any]:
        font_size = max(14, int(20 * font_scale * 2.2))
        font = ImageFont.truetype(str(self._find_font_path(text)), font_size)

        dummy_image = Image.new("RGB", (1, 1), "white")
        dummy_draw = ImageDraw.Draw(dummy_image)
        bbox = dummy_draw.textbbox((0, 0), text, font=font, stroke_width=max(0, thickness - 1))
        text_width = max(1, bbox[2] - bbox[0])
        text_height = max(1, bbox[3] - bbox[1])

        image = Image.new("RGB", (text_width + 8, text_height + 8), "white")
        draw = ImageDraw.Draw(image)
        draw.text(
            (4 - bbox[0], 4 - bbox[1]),
            text,
            fill=text_color,
            font=font,
            stroke_width=max(0, thickness - 1),
        )

        rendered = np.array(image)
        if rotate_ccw:
            return cv2.rotate(rendered, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return rendered

    def _paste_image(self, base: np.ndarray[Any, Any], overlay: np.ndarray[Any, Any], x: int, y: int) -> None:
        x = max(0, x)
        y = max(0, y)
        if x >= base.shape[1] or y >= base.shape[0]:
            return

        height = min(overlay.shape[0], base.shape[0] - y)
        width = min(overlay.shape[1], base.shape[1] - x)
        if height <= 0 or width <= 0:
            return

        base[y : y + height, x : x + width] = overlay[:height, :width]

    def _find_font_path(self, text: str) -> Path:
        ascii_candidate_paths = [
            Path(r"C:\Windows\Fonts\arial.ttf"),
            Path(r"C:\Windows\Fonts\arialbd.ttf"),
        ]
        japanese_candidate_paths = [
            Path(r"C:\Windows\Fonts\meiryo.ttc"),
            Path(r"C:\Windows\Fonts\YuGothR.ttc"),
            Path(r"C:\Windows\Fonts\msgothic.ttc"),
        ]
        candidate_paths = (
            japanese_candidate_paths
            if self._contains_non_ascii(text)
            else ascii_candidate_paths + japanese_candidate_paths
        )
        for path in candidate_paths:
            if path.exists():
                return path
        raise FileNotFoundError("No suitable Japanese font was found.")

    def _contains_non_ascii(self, text: str) -> bool:
        return any(ord(char) > 127 for char in text)

    def _choose_y_tick_step(self, max_quantity: int) -> int:
        if max_quantity <= 0:
            return 1

        magnitude = 10 ** int(math.floor(math.log10(max_quantity)))
        normalized = max_quantity / magnitude

        if normalized <= 1:
            multiplier = 0.1
        elif normalized <= 2:
            multiplier = 0.2
        elif normalized <= 5:
            multiplier = 0.5
        else:
            multiplier = 1.0

        return max(1, int(magnitude * multiplier))
