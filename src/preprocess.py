from __future__ import annotations

import cv2
import numpy as np

from src.settings import PreprocessSettings


def preprocess_roi(image: np.ndarray, settings: PreprocessSettings) -> np.ndarray:
    if image.size == 0:
        return image

    output = image.copy()
    if settings.grayscale:
        output = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
    if settings.median_blur and settings.median_blur >= 3 and settings.median_blur % 2 == 1:
        output = cv2.medianBlur(output, settings.median_blur)
    if settings.threshold_enabled:
        if len(output.shape) == 3:
            output = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
        _, output = cv2.threshold(output, settings.threshold_value, 255, cv2.THRESH_BINARY)
    if settings.scale and settings.scale != 1.0:
        output = cv2.resize(output, None, fx=settings.scale, fy=settings.scale, interpolation=cv2.INTER_CUBIC)
    return output


def preprocess_price_roi(image: np.ndarray) -> np.ndarray:
    if image.size == 0:
        return image

    output = image.copy()
    if len(output.shape) == 3:
        output = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)

    output = cv2.copyMakeBorder(output, 8, 8, 10, 10, cv2.BORDER_CONSTANT, value=255)
    output = cv2.resize(output, None, fx=3.2, fy=3.2, interpolation=cv2.INTER_CUBIC)
    output = cv2.GaussianBlur(output, (3, 3), 0)
    _, output = cv2.threshold(output, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return output


def ensure_bgr(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image
