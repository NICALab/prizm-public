"""Helpers for writing human-viewable PRIZM frame outputs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
import tifffile


VISUALIZATION_FORMATS = ("jpg", "tif")


def normalize_visualization_format(value: str | None) -> str:
    """Return the canonical extension for a visualization output format."""
    normalized = str(value or "jpg").strip().lower().lstrip(".")
    if normalized == "jpeg":
        normalized = "jpg"
    if normalized in {"tiff"}:
        normalized = "tif"
    if normalized not in VISUALIZATION_FORMATS:
        valid = ", ".join(VISUALIZATION_FORMATS)
        raise ValueError(
            f"Unsupported visualization format {value!r}; expected one of: {valid}"
        )
    return normalized


def to_uint8(array: np.ndarray) -> np.ndarray:
    """Convert a visualization to uint8 without frame-specific integer stretching.

    Integer inputs use their dtype's fixed range so brightness remains comparable
    between frames. Floating-point inputs in [0, 1] are scaled to [0, 255]; other
    floating-point inputs are clipped directly to [0, 255].
    """
    values = np.asarray(array)
    if values.dtype == np.uint8:
        return values
    if np.issubdtype(values.dtype, np.bool_):
        return values.astype(np.uint8) * 255
    if np.issubdtype(values.dtype, np.integer):
        info = np.iinfo(values.dtype)
        if info.min < 0:
            denominator = max(1, int(info.max) - int(info.min))
            scaled = (values.astype(np.float64) - float(info.min)) * (
                255.0 / denominator
            )
        else:
            scaled = values.astype(np.float64) * (255.0 / max(1, int(info.max)))
        return np.clip(scaled, 0, 255).astype(np.uint8)

    floating = values.astype(np.float64, copy=False)
    floating = np.nan_to_num(floating, nan=0.0, posinf=255.0, neginf=0.0)
    if floating.size and float(np.min(floating)) >= 0.0 and float(np.max(floating)) <= 1.0:
        floating = floating * 255.0
    return np.clip(floating, 0, 255).astype(np.uint8)


def visualization_name(prefix: str, source_name: str, output_format: str) -> str:
    """Build a visualization filename independent of the source extension."""
    extension = normalize_visualization_format(output_format)
    return f"{prefix}_{Path(str(source_name)).stem}.{extension}"


def save_visualization(path: str | Path, array: np.ndarray, output_format: str) -> None:
    """Save one visualization, with no duplicate in the alternate format."""
    output_format = normalize_visualization_format(output_format)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if output_format == "jpg":
        values = to_uint8(array)
        Image.fromarray(values).save(
            destination,
            format="JPEG",
            quality=100,
        )
        return
    # TIFF is the lossless alternative and therefore retains the source dtype.
    tifffile.imwrite(destination, np.asarray(array))


def segmentation_overlay(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return a human-viewable RGB overlay for labels 1 and 2."""
    base = to_uint8(image)
    if base.ndim == 2:
        rgb = np.repeat(base[..., None], 3, axis=-1)
    elif base.ndim == 3 and base.shape[-1] == 1:
        rgb = np.repeat(base, 3, axis=-1)
    elif base.ndim == 3 and base.shape[-1] >= 3:
        rgb = base[..., :3].copy()
    else:
        raise ValueError(f"Unsupported visualization shape: {base.shape}")

    labels = np.asarray(mask)
    if labels.shape != rgb.shape[:2]:
        raise ValueError(
            f"Mask shape {labels.shape} does not match image shape {rgb.shape[:2]}"
        )
    output = rgb.astype(np.float32)
    # Match the established August 2026 PRIZM output convention exactly:
    # a faint blue ventricle and teal-green atrium overlay.
    alpha = 0.15
    for label_value, color in (
        (1, np.asarray([0, 0, 240], dtype=np.float32)),
        (2, np.asarray([0, 220, 130], dtype=np.float32)),
    ):
        selected = labels == label_value
        output[selected] = (1.0 - alpha) * output[selected] + alpha * color
    return np.clip(output, 0, 255).astype(np.uint8)
