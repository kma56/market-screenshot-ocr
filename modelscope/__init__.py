"""Minimal local stub for ModelScope.

This project uses locally cached Paddle model files and does not rely on
ModelScope downloads at runtime. PaddleX imports `modelscope` eagerly, so this
stub avoids importing a broken global torch installation through the real
ModelScope package on Windows.
"""

from __future__ import annotations


def snapshot_download(*args, **kwargs):
    raise RuntimeError(
        "ModelScope download is not available in this local stub. "
        "Use locally cached Paddle models or another model hoster."
    )
