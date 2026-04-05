from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.preprocess import ensure_bgr
from src.settings import REGION_COLORS, Region

WINDOWS_FONT_CANDIDATES = [
    Path(r"C:\Windows\Fonts\meiryo.ttc"),
    Path(r"C:\Windows\Fonts\msgothic.ttc"),
    Path(r"C:\Windows\Fonts\YuGothM.ttc"),
]


def crop_region(image: np.ndarray, region: Region) -> np.ndarray:
    return image[region.y : region.y + region.h, region.x : region.x + region.w].copy()


def draw_regions(image: np.ndarray, regions: dict[str, Region | None]) -> np.ndarray:
    canvas = image.copy()
    for name, region in regions.items():
        if not region:
            continue
        color = REGION_COLORS[name]
        x1, y1, x2, y2 = region.to_xyxy()
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            canvas,
            name,
            (x1, max(24, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )
    return canvas


def build_preview_panel(
    image: np.ndarray,
    regions: dict[str, Region | None],
    active_region_name: str,
    roi_image: np.ndarray | None,
    processed_image: np.ndarray | None,
    preview_enabled: bool,
    status_lines: list[str] | None = None,
) -> np.ndarray:
    left = draw_regions(image, regions)
    left = pad_to_size(left, 960, 720)
    panel_width = 560

    roi_panel = np.zeros((360, panel_width, 3), dtype=np.uint8)
    processed_panel = np.zeros((360, panel_width, 3), dtype=np.uint8)

    if roi_image is not None and roi_image.size:
        roi_panel = resize_to_fit(ensure_bgr(roi_image), panel_width, 360)
    if processed_image is not None and processed_image.size:
        processed_panel = resize_to_fit(ensure_bgr(processed_image), panel_width, 360)

    roi_panel = pad_to_size(roi_panel, panel_width, 360)
    roi_panel = put_panel_label(roi_panel, f"{active_region_name} ROI")
    processed_title = f"{active_region_name} Preprocess {'ON' if preview_enabled else 'OFF'}"
    processed_panel = pad_to_size(processed_panel, panel_width, 360)
    processed_panel = put_panel_label(processed_panel, processed_title)
    right = np.vstack([roi_panel, processed_panel])
    left = pad_to_size(left, 960, right.shape[0])
    top = np.hstack([left, right])
    footer = build_footer(top.shape[1], status_lines or [])
    return np.vstack([top, footer])


def resize_to_fit(image: np.ndarray, max_width: int, max_height: int) -> np.ndarray:
    height, width = image.shape[:2]
    if width == 0 or height == 0:
        return np.zeros((max_height, max_width, 3), dtype=np.uint8)
    scale = min(max_width / width, max_height / height)
    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)


def pad_to_size(image: np.ndarray, width: int, height: int) -> np.ndarray:
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    src_h, src_w = image.shape[:2]
    offset_x = max(0, (width - src_w) // 2)
    offset_y = max(0, (height - src_h) // 2)
    canvas[offset_y : offset_y + src_h, offset_x : offset_x + src_w] = image[: min(src_h, height), : min(src_w, width)]
    return canvas


def put_panel_label(image: np.ndarray, label: str) -> np.ndarray:
    canvas = image.copy()
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 30), (30, 30, 30), -1)
    return draw_text_fit(canvas, label, (10, 4), max_width=canvas.shape[1] - 20, start_font_size=22, color=(240, 240, 240))


def save_debug_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)


def build_footer(width: int, lines: list[str]) -> np.ndarray:
    footer_height = 132
    footer = np.zeros((footer_height, width, 3), dtype=np.uint8)
    cv2.rectangle(footer, (0, 0), (width, footer_height), (18, 18, 18), -1)
    for index, line in enumerate(lines[:4]):
        footer = draw_text(footer, line, (16, 12 + index * 24), font_size=22, color=(235, 235, 235))
    return footer


def draw_text(image: np.ndarray, text: str, position: tuple[int, int], font_size: int, color: tuple[int, int, int]) -> np.ndarray:
    font = load_font(font_size)
    if font is None:
        cv2.putText(
            image,
            text,
            (position[0], position[1] + font_size),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            1,
            cv2.LINE_AA,
        )
        return image

    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)
    draw.text(position, text, font=font, fill=(color[2], color[1], color[0]))
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


def draw_text_fit(
    image: np.ndarray,
    text: str,
    position: tuple[int, int],
    max_width: int,
    start_font_size: int,
    color: tuple[int, int, int],
) -> np.ndarray:
    for font_size in range(start_font_size, 11, -1):
        font = load_font(font_size)
        if font is None:
            break
        if font.getlength(text) <= max_width:
            return draw_text(image, text, position, font_size, color)
    return draw_text(image, text, position, 16, color)


def load_font(font_size: int) -> ImageFont.FreeTypeFont | None:
    for candidate in WINDOWS_FONT_CANDIDATES:
        if candidate.exists():
            try:
                return ImageFont.truetype(str(candidate), font_size)
            except OSError:
                continue
    return None
