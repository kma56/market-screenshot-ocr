from __future__ import annotations

import argparse
import ctypes
import logging
import sys
from pathlib import Path
from tkinter import Tk, filedialog, messagebox, simpledialog
from typing import Sequence

import cv2

from src.image_loader import list_images, load_image
from src.ro_auto import (
    ROAutoConfig,
    ROAutoState,
    build_batch_summary,
    create_run_plan,
    discover_latest_batch,
    is_already_processed,
    load_ro_pipeline_config,
    record_processed_batch,
    stage_batch,
)
from src.settings import AppConfig

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "region.json"
DEFAULT_RO_AUTO_CONFIG_PATH = PROJECT_ROOT / "config" / "ro_auto.json"
DEFAULT_RO_AUTO_STATE_PATH = PROJECT_ROOT / "config" / "ro_auto_state.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output"
DEFAULT_DEBUG_DIR = PROJECT_ROOT / "debug"
DEFAULT_LOG_PATH = PROJECT_ROOT / "output" / "ocr_tool.log"


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.ro_auto:
        return run_ro_auto(force=args.ro_auto_force)
    return run_interactive()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Market Screenshot OCR Tool")
    parser.add_argument(
        "--ro-auto",
        action="store_true",
        help="Use saved ROI config and automatically process the latest Ragnarok Online screenshot batch.",
    )
    parser.add_argument(
        "--ro-auto-force",
        action="store_true",
        help="Process the latest Ragnarok Online screenshot batch even if it matches the last auto-run.",
    )
    return parser.parse_args(argv)


def run_interactive() -> int:
    enable_windows_dpi_awareness()
    from src.pipeline import OCRPipeline
    from src.region_selector import select_regions

    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    root.update()

    input_dir = choose_input_folder(root)
    if input_dir is None:
        root.destroy()
        return 1

    image_paths = list_images(input_dir)
    if not image_paths:
        messagebox.showerror("No Images", "選択フォルダ内に .jpg / .png が見つかりません。")
        root.destroy()
        return 1

    first_image = load_image(image_paths[0])
    config = prepare_config(root, first_image.width, first_image.height)
    if config is None:
        root.destroy()
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
        root.destroy()
        return 1

    save_after_selection(root, selected)
    logger = configure_logging(DEFAULT_LOG_PATH)

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
        root.destroy()
        return 1

    summary = (
        f"CSV: {result.csv_path}\n"
        f"Graph: {result.graph_path}\n"
        f"Summary TXT: {result.summary_txt_path}\n"
        f"Processed images: {result.processed_images}\n"
        f"Output rows: {result.total_rows}\n"
        f"Warnings: {len(result.warnings)}\n"
        f"Errors: {len(result.errors)}"
    )
    messagebox.showinfo("Completed", summary)
    if result.has_price_suspects:
        messagebox.showwarning(
            "Price Warning",
            "price_suspect が付いた行があります。\nCSV を確認してください。",
        )
    root.destroy()
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


def run_ro_auto(force: bool) -> int:
    enable_windows_dpi_awareness()
    from src.pipeline import OCRPipeline

    try:
        auto_config = ROAutoConfig.load(DEFAULT_RO_AUTO_CONFIG_PATH, PROJECT_ROOT)
        pipeline_config = load_ro_pipeline_config(DEFAULT_CONFIG_PATH, auto_config)
        state = ROAutoState.load(DEFAULT_RO_AUTO_STATE_PATH)
        batch = discover_latest_batch(auto_config)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if batch is None:
        print(
            f"No matching screenshots were found in {auto_config.source_dir}.",
            file=sys.stderr,
        )
        return 1

    if is_already_processed(batch, state, auto_config, force=force):
        message = f"Latest batch already processed: {build_batch_summary(batch)}"
        if state.last_output_dir:
            message += f"\nLast output: {state.last_output_dir}"
        print(message)
        return 0

    plan = create_run_plan(auto_config, batch)
    logger = configure_logging(plan.log_path)
    logger.info("RO auto mode started.")
    logger.info("Selected batch: %s", build_batch_summary(batch))
    logger.info("Staging screenshots from %s", auto_config.source_dir)

    try:
        stage_batch(plan, auto_config.source_dir)
        logger.info("Staged screenshots to %s", plan.staging_dir)
    except Exception as exc:
        logger.exception("Failed to stage latest screenshot batch: %s", exc)
        return 1

    pipeline = OCRPipeline(
        config=pipeline_config,
        input_dir=plan.staging_dir,
        output_dir=plan.output_dir,
        debug_dir=plan.debug_dir,
        logger=logger,
    )
    try:
        result = pipeline.run()
    except Exception as exc:
        logger.exception("RO auto pipeline failed: %s", exc)
        return 1

    record_processed_batch(DEFAULT_RO_AUTO_STATE_PATH, plan)
    logger.info("CSV: %s", result.csv_path)
    logger.info("Graph: %s", result.graph_path)
    logger.info("Summary TXT: %s", result.summary_txt_path)
    logger.info("Processed images: %s", result.processed_images)
    logger.info("Output rows: %s", result.total_rows)
    logger.info("Warnings: %s", len(result.warnings))
    logger.info("Errors: %s", len(result.errors))
    if result.has_price_suspects:
        logger.warning("price_suspect rows were detected. Please review the CSV.")
    return 0


def configure_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("market_screenshot_ocr")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
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
