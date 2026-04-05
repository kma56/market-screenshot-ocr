from __future__ import annotations

import ctypes
import logging
import sys
from pathlib import Path
from tkinter import Tk, filedialog, messagebox, simpledialog

import cv2

from src.image_loader import list_images, load_image
from src.pipeline import OCRPipeline
from src.region_selector import select_regions
from src.settings import AppConfig

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "region.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output"
DEFAULT_DEBUG_DIR = PROJECT_ROOT / "debug"
DEFAULT_LOG_PATH = PROJECT_ROOT / "output" / "ocr_tool.log"


def main() -> int:
    enable_windows_dpi_awareness()
    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    root.update()

    input_dir = choose_input_folder(root)
    if input_dir is None:
        return 1

    image_paths = list_images(input_dir)
    if not image_paths:
        messagebox.showerror("No Images", "選択フォルダ内に .jpg / .png が見つかりません。")
        return 1

    first_image = load_image(image_paths[0])
    config = prepare_config(root, first_image.width, first_image.height)
    if config is None:
        return 1

    selected = select_regions(
        first_image.image,
        config,
        DEFAULT_CONFIG_PATH,
        DEFAULT_OUTPUT_DIR,
        DEFAULT_DEBUG_DIR,
    )
    cv2.destroyAllWindows()
    if selected is None:
        return 1

    save_after_selection(root, selected)
    logger = configure_logging()

    pipeline = OCRPipeline(
        config=selected,
        input_dir=input_dir,
        output_dir=DEFAULT_OUTPUT_DIR,
        debug_dir=DEFAULT_DEBUG_DIR,
        logger=logger,
    )
    try:
        result = pipeline.run()
    except Exception as exc:
        logger.exception("Pipeline failed: %s", exc)
        messagebox.showerror("OCR Failed", str(exc))
        return 1

    summary = (
        f"CSV: {result.csv_path}\n"
        f"Processed images: {result.processed_images}\n"
        f"Output rows: {result.total_rows}\n"
        f"Warnings: {len(result.warnings)}\n"
        f"Errors: {len(result.errors)}"
    )
    messagebox.showinfo("Completed", summary)
    return 0


def choose_input_folder(root: Tk) -> Path | None:
    root.deiconify()
    root.lift()
    root.focus_force()
    selected = filedialog.askdirectory(title="OCR対象フォルダを選択してください", parent=root)
    root.withdraw()
    if not selected:
        return None
    return Path(selected)


def prepare_config(root: Tk, width: int, height: int) -> AppConfig | None:
    config = None
    if DEFAULT_CONFIG_PATH.exists():
        reuse = messagebox.askyesno("設定再利用", "保存済み設定を再利用しますか？")
        if reuse:
            config = AppConfig.load(DEFAULT_CONFIG_PATH)

    if config is None:
        row_count = simpledialog.askinteger(
            "行数",
            "row_count を入力してください (1-10)",
            initialvalue=10,
            minvalue=1,
            maxvalue=10,
            parent=root,
        )
        if row_count is None:
            return None
        config = AppConfig.create_default(width=width, height=height, row_count=row_count)
        config.debug_save_enabled = ask_skip_debug_images(root)
    else:
        if config.image_size.width != width or config.image_size.height != height:
            messagebox.showwarning(
                "解像度不一致",
                "保存済み設定の解像度と先頭画像の解像度が一致しません。ROIを再選択してください。",
            )
            config = AppConfig.create_default(width=width, height=height, row_count=config.row_count)
        config.debug_save_enabled = ask_skip_debug_images(root)
    config.ocr.mode = choose_ocr_mode(root, config.ocr.mode)
    return config


def choose_ocr_mode(root: Tk, current_mode: str) -> str:
    use_accuracy_mode = messagebox.askyesno(
        "OCRモード",
        f"現在の設定: {'精度優先' if current_mode == 'accuracy' else '速度優先'}\n\n"
        "精度優先モードを使いますか？\n\n"
        "はい: 精度優先\n"
        "いいえ: 速度優先",
        parent=root,
    )
    return "accuracy" if use_accuracy_mode else "speed"


def ask_skip_debug_images(root: Tk) -> bool:
    skip_debug = messagebox.askyesno(
        "デバッグ画像",
        "デバッグ画像の保存をスキップしますか？\n\nはい: 保存しない\nいいえ: 保存する",
        parent=root,
    )
    return not skip_debug


def save_after_selection(root: Tk, config: AppConfig) -> None:
    should_save = messagebox.askyesno("設定保存", "現在の設定を保存しますか？")
    if should_save:
        config.save(DEFAULT_CONFIG_PATH)


def configure_logging() -> logging.Logger:
    DEFAULT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("market_screenshot_ocr")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    file_handler = logging.FileHandler(DEFAULT_LOG_PATH, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def enable_windows_dpi_awareness() -> None:
    if sys.platform != "win32":
        return

    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
        return
    except Exception:
        pass

    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass
