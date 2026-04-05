from __future__ import annotations

import re
import unicodedata


def normalize_quantity(value: str) -> str:
    if not value:
        return ""
    normalized = unicodedata.normalize("NFKC", value)
    normalized = re.sub(r"\D", "", normalized)
    return normalized


def normalize_price(value: str) -> str:
    if not value:
        return ""
    normalized = unicodedata.normalize("NFKC", value)
    normalized = normalized.replace(" ", "")
    normalized = re.sub(r"(?i)z$", "", normalized)
    normalized = re.sub(r"(?i)z2$", "", normalized)
    normalized = re.sub(r"(?i)2z$", "", normalized)
    normalized = re.sub(r"[.,]", "", normalized)
    normalized = re.sub(r"(?<=\d)2$", "", normalized)
    normalized = re.sub(r"\D", "", normalized)
    return normalized


def normalize_item_name(value: str) -> str:
    if not value:
        return ""
    normalized = unicodedata.normalize("NFKC", value)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    meaningful = re.sub(r"[\s\-_.,'\"`~|/\\()\[\]{}:;+*]+", "", normalized)
    if not meaningful:
        return ""
    return normalized
