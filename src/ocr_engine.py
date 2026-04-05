from __future__ import annotations

import os
import re
import warnings

os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
warnings.filterwarnings("ignore", message="No ccache found.*")

import cv2
import numpy as np
import paddle
from paddleocr import TextRecognition

from src.settings import OCRSettings


class BaseTextRecognitionEngine:
    def __init__(self, settings: OCRSettings, model_name: str, score_mode: str = "digits") -> None:
        self.settings = settings
        self.score_mode = score_mode
        self.engine = TextRecognition(
            model_name=model_name,
            device=self._resolve_device(settings.device),
        )

    def recognize(self, image: np.ndarray) -> str:
        return self.recognize_batch([image])[0]

    def recognize_batch(self, images: list[np.ndarray]) -> list[str]:
        if not images:
            return []

        candidate_lists = [self._variants(image) if image.size else [image] for image in images]
        variant_count = max(len(candidates) for candidates in candidate_lists)
        best_texts = [""] * len(images)
        best_scores: list[tuple[int, float]] = [(-1, 0.0)] * len(images)

        for variant_index in range(variant_count):
            batch_images: list[np.ndarray] = []
            batch_mapping: list[int] = []
            for image_index, candidates in enumerate(candidate_lists):
                if variant_index < len(candidates):
                    batch_images.append(candidates[variant_index])
                    batch_mapping.append(image_index)

            batch_results = self._recognize_batch_single(batch_images)
            for image_index, (text, confidence) in zip(batch_mapping, batch_results):
                score = self._score_text(text, confidence)
                if score > best_scores[image_index]:
                    best_scores[image_index] = score
                    best_texts[image_index] = text

        return best_texts

    def _recognize_batch_single(self, images: list[np.ndarray]) -> list[tuple[str, float]]:
        if not images:
            return []

        prepared = []
        for image in images:
            if len(image.shape) == 2:
                prepared.append(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR))
            else:
                prepared.append(image)
        try:
            result = list(self.engine.predict(input=prepared, batch_size=len(prepared)))
            outputs: list[tuple[str, float]] = []
            for item in result:
                text = str(item.get("rec_text", "")).strip()
                confidence = float(item.get("rec_score", 0.0) or 0.0)
                outputs.append((text, confidence))
            return outputs
        except Exception:
            return [("", 0.0) for _ in prepared]

    def _variants(self, image: np.ndarray) -> list[np.ndarray]:
        base = image
        gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY) if len(base.shape) == 3 else base.copy()
        scale = 2.0 if self.settings.mode == "speed" else 3.0
        enlarged = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        if self.settings.mode == "speed":
            return [enlarged]
        _, otsu = cv2.threshold(enlarged, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return [base, enlarged, otsu]

    def _resolve_device(self, device: str) -> str:
        if device == "gpu" and paddle.device.is_compiled_with_cuda():
            return "gpu:0"
        if device == "auto" and paddle.device.is_compiled_with_cuda():
            return "gpu:0"
        return "cpu"

    def _score_text(self, text: str, confidence: float) -> tuple[int, float]:
        if self.score_mode == "text":
            return (len(text.strip()), confidence)
        digit_score = len(re.sub(r"\D", "", text))
        return (digit_score, confidence)


class NumericOCREngine(BaseTextRecognitionEngine):
    def __init__(self, settings: OCRSettings) -> None:
        super().__init__(settings=settings, model_name="PP-OCRv5_server_rec", score_mode="digits")


class ItemNameOCREngine(BaseTextRecognitionEngine):
    def __init__(self, settings: OCRSettings) -> None:
        super().__init__(settings=settings, model_name="japan_PP-OCRv3_mobile_rec", score_mode="text")

    def _variants(self, image: np.ndarray) -> list[np.ndarray]:
        base = image
        gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY) if len(base.shape) == 3 else base.copy()
        scale = 1.8 if self.settings.mode == "speed" else 2.5
        enlarged = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        if self.settings.mode == "speed":
            return [enlarged]
        blur = cv2.GaussianBlur(enlarged, (3, 3), 0)
        return [base, enlarged, blur]
