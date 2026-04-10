from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from src.preprocess import preprocess_roi
from src.settings import REGION_COLORS, AppConfig, Region
from src.ui_helpers import build_preview_panel, crop_region

WINDOW_NAME = "Market Screenshot OCR - ROI Selector"


@dataclass
class ViewState:
    zoom: float = 1.0
    offset_x: int = 0
    offset_y: int = 0
    selecting: bool = False
    dragging_view: bool = False
    drag_start_screen: tuple[int, int] | None = None
    drag_start_offset: tuple[int, int] | None = None
    selection_start_image: tuple[int, int] | None = None
    selection_current_image: tuple[int, int] | None = None


class ROISelector:
    def __init__(
        self,
        image: np.ndarray,
        config: AppConfig,
        config_path: Path,
        output_dir: Path,
        debug_dir: Path,
    ) -> None:
        self.image = image
        self.config = config
        self.config_path = config_path
        self.output_dir = output_dir
        self.debug_dir = debug_dir
        self.selectable_region_names = ["item_name", "quantity", "price"]
        self.active_region_name = self.selectable_region_names[0]
        self.review_mode = False
        self.view = ViewState()
        self.base_width = 960
        self.base_height = 720
        self.display_rect = (0, 0, self.base_width, self.base_height)
        self.status_message = (
            f"移動モードです。ドラッグで表示位置を動かし、V キーで {self.active_region_name} の範囲選択に切り替えます。"
        )

    def run(self) -> AppConfig | None:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, 1640, 900)
        cv2.setMouseCallback(WINDOW_NAME, self._on_mouse)

        while True:
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                cv2.destroyAllWindows()
                return None

            panel = self._build_panel()
            cv2.imshow(WINDOW_NAME, panel)
            key = cv2.waitKeyEx(16)

            if key == -1:
                continue
            if key == 27:
                cv2.destroyWindow(WINDOW_NAME)
                return None
            if key in (13, 10):
                if self.review_mode:
                    cv2.destroyWindow(WINDOW_NAME)
                    return self.config
                if self.config.regions[self.active_region_name] is None:
                    self.status_message = (
                        f"{self.active_region_name} ROI が未設定です。V キーで範囲選択に切り替えてください。"
                    )
                    continue
                next_region = self._next_region_name()
                if next_region is not None:
                    self.active_region_name = next_region
                    self.view.selecting = False
                    self.view.selection_start_image = None
                    self.view.selection_current_image = None
                    self.status_message = f"{self.active_region_name} の確認に移動しました。必要なら再選択して、Enter で次へ進んでください。"
                    continue
                self.review_mode = True
                self.view.selecting = False
                self.view.selection_start_image = None
                self.view.selection_current_image = None
                self.status_message = "最終確認です。両方の ROI を確認して、Enter で OCR を実行します。"
                continue
            if key == 9 or key in (ord("n"), ord("N")):
                if self.review_mode:
                    continue
                self._cycle_region()
                continue
            if key in (ord("v"), ord("V")):
                if self.review_mode:
                    self.status_message = "最終確認中です。V は使えません。Enter で OCR 実行、Esc でキャンセルです。"
                    continue
                self.view.selecting = not self.view.selecting
                self.view.selection_start_image = None
                self.view.selection_current_image = None
                self.status_message = (
                    f"範囲選択モードです。画像上をドラッグして {self.active_region_name} の範囲を指定します。"
                    if self.view.selecting
                    else f"移動モードです。ドラッグで表示位置を動かせます。現在の対象は {self.active_region_name} です。"
                )
            if key in (ord("r"), ord("R")):
                if self.review_mode:
                    self.review_mode = False
                    self.active_region_name = self.selectable_region_names[0]
                    self.status_message = "最終確認を抜けて item_name に戻りました。必要なら再選択してください。"
                    continue
                self.config.regions[self.active_region_name] = None
                self.status_message = f"{self.active_region_name} の範囲をクリアしました。"
            if key in (ord("p"), ord("P")):
                self.config.preprocess.enabled = not self.config.preprocess.enabled
                self.status_message = (
                    "前処理プレビューを ON にしました。OCR 実行時も同じ前処理を使います。"
                    if self.config.preprocess.enabled
                    else "前処理プレビューを OFF にしました。OCR 実行時は元画像を使います。"
                )
            if key in (ord("s"), ord("S")):
                self.config.save(self.config_path)
                self.status_message = f"設定を {self.config_path.name} に保存しました。"
            if key in (ord("c"), ord("C")):
                removed_count = self._clear_generated_files()
                self.status_message = f"生成物を掃除しました。削除数: {removed_count}"
            if key in (ord("0"),):
                self._reset_view()
                self.status_message = "拡大率と表示位置をリセットしました。"
            if key in (2424832, ord("a"), ord("A")):
                self.view.offset_x -= 40
            if key in (2555904, ord("d"), ord("D")):
                self.view.offset_x += 40
            if key in (2490368, ord("w"), ord("W")):
                self.view.offset_y -= 40
            if key in (2621440, ord("x"), ord("X")):
                self.view.offset_y += 40

    def _build_panel(self) -> np.ndarray:
        left_view = self._build_zoomed_view()
        region = None if self.review_mode else self.config.regions.get(self.active_region_name)
        roi_image = crop_region(self.image, region) if region else None
        processed = preprocess_roi(roi_image, self.config.preprocess) if roi_image is not None else None
        return build_preview_panel(
            image=left_view,
            regions={},
            active_region_name="review" if self.review_mode else self.active_region_name,
            roi_image=roi_image,
            processed_image=processed if self.config.preprocess.enabled else roi_image,
            preview_enabled=self.config.preprocess.enabled,
            status_lines=self._status_lines(),
        )

    def _build_zoomed_view(self) -> np.ndarray:
        height, width = self.image.shape[:2]
        zoom = max(1.0, self.view.zoom)
        crop_w = max(1, min(width, int(width / zoom)))
        crop_h = max(1, min(height, int(height / zoom)))
        max_x = max(0, width - crop_w)
        max_y = max(0, height - crop_h)
        self.view.offset_x = max(0, min(self.view.offset_x, max_x))
        self.view.offset_y = max(0, min(self.view.offset_y, max_y))

        x1 = self.view.offset_x
        y1 = self.view.offset_y
        x2 = x1 + crop_w
        y2 = y1 + crop_h
        cropped = self.image[y1:y2, x1:x2].copy()
        scale = min(self.base_width / crop_w, self.base_height / crop_h)
        draw_w = max(1, int(crop_w * scale))
        draw_h = max(1, int(crop_h * scale))
        resized_crop = cv2.resize(cropped, (draw_w, draw_h), interpolation=cv2.INTER_CUBIC)
        resized = np.zeros((self.base_height, self.base_width, 3), dtype=np.uint8)
        pad_x = (self.base_width - draw_w) // 2
        pad_y = (self.base_height - draw_h) // 2
        resized[pad_y : pad_y + draw_h, pad_x : pad_x + draw_w] = resized_crop

        region_names = self.selectable_region_names if self.review_mode else [self.active_region_name]
        for region_name in region_names:
            region = self.config.regions.get(region_name)
            if not region:
                continue
            color = REGION_COLORS[region_name]
            rx1, ry1 = self._image_to_view(region.x, region.y)
            rx2, ry2 = self._image_to_view(region.x + region.w, region.y + region.h)
            cv2.rectangle(resized, (rx1, ry1), (rx2, ry2), color, 2)
            cv2.putText(
                resized,
                region_name,
                (rx1, max(24, ry1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2,
                cv2.LINE_AA,
            )

        if not self.review_mode and self.view.selection_start_image and self.view.selection_current_image:
            color = REGION_COLORS[self.active_region_name]
            sx1, sy1 = self._image_to_view(*self.view.selection_start_image)
            sx2, sy2 = self._image_to_view(*self.view.selection_current_image)
            cv2.rectangle(resized, (sx1, sy1), (sx2, sy2), color, 2)

        self.display_rect = (pad_x, pad_y, pad_x + draw_w, pad_y + draw_h)
        return resized

    def _on_mouse(self, event: int, x: int, y: int, flags: int, _param: object) -> None:
        if not self._is_in_image_view(x, y):
            return

        image_point = self._view_to_image(x, y)

        if event == cv2.EVENT_MOUSEWHEEL:
            wheel_delta = self._mouse_wheel_delta(flags)
            delta = 1.15 if wheel_delta > 0 else (1 / 1.15)
            self._zoom_at(x, y, delta)
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            if self.view.selecting:
                self.view.selection_start_image = image_point
                self.view.selection_current_image = image_point
            else:
                self.view.dragging_view = True
                self.view.drag_start_screen = (x, y)
                self.view.drag_start_offset = (self.view.offset_x, self.view.offset_y)
            return

        if event == cv2.EVENT_MOUSEMOVE:
            if self.view.selecting and self.view.selection_start_image is not None:
                self.view.selection_current_image = image_point
            elif self.view.dragging_view and self.view.drag_start_screen and self.view.drag_start_offset:
                start_x, start_y = self.view.drag_start_screen
                offset_x, offset_y = self.view.drag_start_offset
                scale_x = self._crop_width() / self.base_width
                scale_y = self._crop_height() / self.base_height
                self.view.offset_x = int(offset_x - (x - start_x) * scale_x)
                self.view.offset_y = int(offset_y - (y - start_y) * scale_y)
            return

        if event == cv2.EVENT_LBUTTONUP:
            if self.view.selecting and self.view.selection_start_image and self.view.selection_current_image:
                self._commit_selection()
            self.view.dragging_view = False
            self.view.drag_start_screen = None
            self.view.drag_start_offset = None

    def _commit_selection(self) -> None:
        x1, y1 = self.view.selection_start_image
        x2, y2 = self.view.selection_current_image
        left = max(0, min(x1, x2))
        top = max(0, min(y1, y2))
        right = min(self.image.shape[1], max(x1, x2))
        bottom = min(self.image.shape[0], max(y1, y2))
        if right - left >= 4 and bottom - top >= 4:
            self.config.regions[self.active_region_name] = Region(
                x=int(left),
                y=int(top),
                w=int(right - left),
                h=int(bottom - top),
            )
        self.view.selection_start_image = None
        self.view.selection_current_image = None

    def _zoom_at(self, view_x: int, view_y: int, factor: float) -> None:
        old_zoom = self.view.zoom
        new_zoom = max(1.0, min(8.0, old_zoom * factor))
        if abs(new_zoom - old_zoom) < 1e-6:
            return

        image_x, image_y = self._view_to_image(view_x, view_y)
        self.view.zoom = new_zoom
        new_crop_w = self._crop_width()
        new_crop_h = self._crop_height()
        left, top, right, bottom = self.display_rect
        rel_x = 0.5 if right == left else (view_x - left) / max(1, right - left)
        rel_y = 0.5 if bottom == top else (view_y - top) / max(1, bottom - top)
        self.view.offset_x = int(image_x - rel_x * new_crop_w)
        self.view.offset_y = int(image_y - rel_y * new_crop_h)

    def _reset_view(self) -> None:
        self.view.zoom = 1.0
        self.view.offset_x = 0
        self.view.offset_y = 0
        self.view.dragging_view = False
        self.view.drag_start_screen = None
        self.view.drag_start_offset = None
        self.view.selection_start_image = None
        self.view.selection_current_image = None

    def _crop_width(self) -> int:
        return max(1, min(self.image.shape[1], int(self.image.shape[1] / max(1.0, self.view.zoom))))

    def _crop_height(self) -> int:
        return max(1, min(self.image.shape[0], int(self.image.shape[0] / max(1.0, self.view.zoom))))

    def _is_in_image_view(self, x: int, y: int) -> bool:
        left, top, right, bottom = self.display_rect
        return left <= x < right and top <= y < bottom

    def _view_to_image(self, x: int, y: int) -> tuple[int, int]:
        crop_w = self._crop_width()
        crop_h = self._crop_height()
        left, top, right, bottom = self.display_rect
        rel_x = np.clip((x - left) / max(1, right - left), 0.0, 1.0)
        rel_y = np.clip((y - top) / max(1, bottom - top), 0.0, 1.0)
        img_x = self.view.offset_x + int(rel_x * crop_w)
        img_y = self.view.offset_y + int(rel_y * crop_h)
        img_x = min(self.image.shape[1] - 1, max(0, img_x))
        img_y = min(self.image.shape[0] - 1, max(0, img_y))
        return img_x, img_y

    def _image_to_view(self, x: int, y: int) -> tuple[int, int]:
        crop_w = self._crop_width()
        crop_h = self._crop_height()
        rel_x = (x - self.view.offset_x) / max(1, crop_w)
        rel_y = (y - self.view.offset_y) / max(1, crop_h)
        left, top, right, bottom = self.display_rect
        view_x = left + int(rel_x * max(1, right - left))
        view_y = top + int(rel_y * max(1, bottom - top))
        return view_x, view_y

    def _mouse_wheel_delta(self, flags: int) -> int:
        value = flags >> 16
        if value >= 0x8000:
            value -= 0x10000
        return value

    def _status_lines(self) -> list[str]:
        if self.review_mode:
            return [
                "対象: 最終確認 | item_name・quantity・price の 3 つを重ねて表示しています",
                "Enter: OCR 実行 | R: item_name から確認し直す | Esc: キャンセル",
                "ホイール: 拡大縮小 | ドラッグ: 表示移動 | C: 生成物掃除 | S: 設定保存",
                f"状態: {self.status_message}",
            ]

        mode_label = "範囲選択" if self.view.selecting else "移動"
        return [
            f"対象: {self.active_region_name} | モード: {mode_label} | 拡大率: {self.view.zoom:.2f}x | ホイール: 拡大縮小",
            "V: モード切替 | N/Tab: 対象切替 | R: 現在の範囲クリア | P: 前処理プレビュー切替 | S: 設定保存 | C: 生成物掃除",
            "0: 拡大率と表示位置をリセット | Enter: 次へ / OCR 実行 | Esc: キャンセル",
            f"状態: {self.status_message}",
        ]

    def _clear_generated_files(self) -> int:
        removed_count = 0
        if self.output_dir.exists():
            for path in self.output_dir.iterdir():
                if path.is_file() and path.name.startswith("ocr_result_") and path.suffix.lower() == ".csv":
                    path.unlink(missing_ok=True)
                    removed_count += 1
                elif path.is_file() and path.name == "ocr_tool.log":
                    path.unlink(missing_ok=True)
                    removed_count += 1

        if self.debug_dir.exists():
            for path in self.debug_dir.iterdir():
                if path.is_dir():
                    shutil.rmtree(path, ignore_errors=True)
                    removed_count += 1
                elif path.is_file():
                    path.unlink(missing_ok=True)
                    removed_count += 1
        return removed_count

    def _next_region_name(self) -> str | None:
        current_index = self.selectable_region_names.index(self.active_region_name)
        next_index = current_index + 1
        if next_index >= len(self.selectable_region_names):
            return None
        return self.selectable_region_names[next_index]

    def _cycle_region(self) -> None:
        current_index = self.selectable_region_names.index(self.active_region_name)
        next_index = (current_index + 1) % len(self.selectable_region_names)
        self.active_region_name = self.selectable_region_names[next_index]
        self.view.selecting = False
        self.view.selection_start_image = None
        self.view.selection_current_image = None
        self.status_message = f"対象を {self.active_region_name} に切り替えました。"


def select_regions(
    image: np.ndarray,
    config: AppConfig,
    config_path: Path,
    output_dir: Path,
    debug_dir: Path,
) -> AppConfig | None:
    selector = ROISelector(
        image=image,
        config=config,
        config_path=config_path,
        output_dir=output_dir,
        debug_dir=debug_dir,
    )
    return selector.run()
