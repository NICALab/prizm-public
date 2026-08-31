"""Series-level quality-control checks for PRIZM batch outputs.

The checks in this module report review flags only.  They do not modify input
frames, segmentation masks, or downstream cardiac measurements.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np


QC_SAMPLE_SIZE = 30
LOW_SIGNAL_MEAN_FRAME_MAX_THRESHOLD = 75.0
FRAME_INTERVAL_FLAG_THRESHOLD_SECONDS = 0.120
TENENGRAD_FLAG_THRESHOLD = 800.0
SEGMENTATION_COMPONENT_MIN_AREA = 300

QC_FLAG_COLUMNS = (
    "LowSignalQCFlag",
    "SegmentationErrorQCFlag",
    "InsufficientFramesQCFlag",
    "OutOfFocusQCFlag",
)


def evenly_spaced_indices(length: int, sample_size: int = QC_SAMPLE_SIZE) -> np.ndarray:
    """Return up to ``sample_size`` indices spanning an entire series."""
    length = int(length)
    sample_size = int(sample_size)
    if length <= 0 or sample_size <= 0:
        return np.empty(0, dtype=int)
    count = min(length, sample_size)
    return np.linspace(0, length - 1, num=count, dtype=int)


def _gray_8bit_scale(frame: np.ndarray) -> np.ndarray:
    """Convert a raw frame to grayscale on an absolute 0--255 scale."""
    arr = np.asarray(frame)
    source_dtype = arr.dtype
    if arr.ndim == 3:
        if arr.shape[2] >= 3:
            arr = (
                0.2989 * arr[..., 0]
                + 0.5870 * arr[..., 1]
                + 0.1140 * arr[..., 2]
            )
        else:
            arr = arr[..., 0]

    gray = np.asarray(arr, dtype=np.float64)
    gray[~np.isfinite(gray)] = 0.0

    if np.issubdtype(source_dtype, np.bool_):
        gray *= 255.0
    elif np.issubdtype(source_dtype, np.integer):
        dtype_max = float(np.iinfo(source_dtype).max)
        if dtype_max > 0.0 and dtype_max != 255.0:
            gray *= 255.0 / dtype_max
    elif gray.size and float(np.nanmax(gray)) <= 1.0:
        gray *= 255.0

    return np.clip(gray, 0.0, 255.0)


def check_low_signal_qc(
    frames: Sequence[np.ndarray],
    sample_size: int = QC_SAMPLE_SIZE,
    threshold: float = LOW_SIGNAL_MEAN_FRAME_MAX_THRESHOLD,
) -> dict[str, float | int | bool]:
    """Flag low raw-image signal using the mean sampled-frame maximum.

    The threshold is applied before PRIZM's adaptive contrast preprocessing.
    A value of 75 on the 8-bit grayscale scale was calibrated using the
    20260803_PRIZM_val data: all 10 ``DsRed_low-signal`` series were positive,
    3/10 of the brighter paired DsRed series were positive, and all 210 other
    normal-control series were negative.
    """
    indices = evenly_spaced_indices(len(frames), sample_size)
    maxima = []
    means = []
    p99_values = []
    for index in indices:
        gray = _gray_8bit_scale(frames[int(index)])
        if gray.size == 0:
            continue
        maxima.append(float(np.max(gray)))
        means.append(float(np.mean(gray)))
        p99_values.append(float(np.percentile(gray, 99)))

    mean_frame_maximum = float(np.mean(maxima)) if maxima else np.nan
    mean_gray_value = float(np.mean(means)) if means else np.nan
    mean_frame_p99 = float(np.mean(p99_values)) if p99_values else np.nan
    return {
        "mean_frame_maximum_gray": mean_frame_maximum,
        "mean_gray": mean_gray_value,
        "mean_frame_p99_gray": mean_frame_p99,
        "n_sampled": int(len(maxima)),
        "flag": bool(
            np.isfinite(mean_frame_maximum)
            and mean_frame_maximum <= float(threshold)
        ),
    }


def check_focus_qc(
    preprocessing_dir: str | Path,
    sample_size: int = QC_SAMPLE_SIZE,
    threshold: float = TENENGRAD_FLAG_THRESHOLD,
) -> dict[str, float | int | bool]:
    """Flag an out-of-focus preprocessed series using mean Tenengrad.

    Up to 30 frames are sampled at evenly spaced indices.  For every grayscale
    frame, Tenengrad is ``mean(Sobel_x**2 + Sobel_y**2)`` and the sample values
    are averaged.  The threshold of 800 was calibrated on the
    20260723_PRIZM_val normal/outfocusing dataset: it produced 0/14 false
    positives in the in-focus group (minimum mean 838) and detected 11/14
    out-of-focus series.
    """
    directory = Path(preprocessing_dir)
    valid_extensions = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    paths = sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in valid_extensions
    ) if directory.is_dir() else []

    tenengrad_values = []
    for index in evenly_spaced_indices(len(paths), sample_size):
        gray = cv2.imread(str(paths[int(index)]), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        gray_float = gray.astype(np.float64, copy=False)
        gx = cv2.Sobel(gray_float, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray_float, cv2.CV_64F, 0, 1, ksize=3)
        tenengrad_values.append(float(np.mean(gx * gx + gy * gy)))

    mean_tenengrad = (
        float(np.mean(tenengrad_values)) if tenengrad_values else np.nan
    )
    return {
        "mean_tenengrad": mean_tenengrad,
        "n_sampled": int(len(tenengrad_values)),
        "flag": bool(
            np.isfinite(mean_tenengrad)
            and mean_tenengrad <= float(threshold)
        ),
    }


def _component_count(mask: np.ndarray, min_area: int) -> int:
    binary = np.asarray(mask, dtype=np.uint8)
    if binary.size == 0 or not np.any(binary):
        return 0
    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    return int(
        sum(
            int(stats[label, cv2.CC_STAT_AREA]) >= int(min_area)
            for label in range(1, n_labels)
        )
    )


def check_segmentation_regions_qc(
    cleaned_masks: np.ndarray,
    min_area: int = SEGMENTATION_COMPONENT_MIN_AREA,
) -> dict[str, int | bool]:
    """Flag a series when a cleaned class mask has multiple valid regions."""
    masks = np.asarray(cleaned_masks)
    if masks.ndim == 2:
        masks = masks[np.newaxis, ...]
    if masks.ndim != 3:
        raise ValueError(
            "cleaned_masks must have shape (frames, height, width) or "
            f"(height, width); got {masks.shape}"
        )

    frames_with_multiple_regions = 0
    max_ventricle_components = 0
    max_atrium_components = 0
    for labels in masks:
        ventricle_components = _component_count(labels == 1, min_area)
        atrium_components = _component_count(labels == 2, min_area)
        max_ventricle_components = max(
            max_ventricle_components, ventricle_components
        )
        max_atrium_components = max(max_atrium_components, atrium_components)
        if ventricle_components > 1 or atrium_components > 1:
            frames_with_multiple_regions += 1

    return {
        "frames_with_multiple_regions": int(frames_with_multiple_regions),
        "max_ventricle_components": int(max_ventricle_components),
        "max_atrium_components": int(max_atrium_components),
        "flag": bool(frames_with_multiple_regions > 0),
    }


def check_frame_interval_qc(
    relative_times: Sequence[float] | np.ndarray | None,
    frame_interval: float | None = None,
    threshold_seconds: float = FRAME_INTERVAL_FLAG_THRESHOLD_SECONDS,
) -> dict[str, float | bool]:
    """Flag series acquired at intervals of 120 ms or longer."""
    interval_seconds = np.nan
    if relative_times is not None:
        times = np.asarray(relative_times, dtype=float).reshape(-1)
        finite_times = times[np.isfinite(times)]
        if finite_times.size >= 2:
            differences = np.diff(finite_times)
            differences = differences[np.isfinite(differences) & (differences > 0)]
            if differences.size:
                interval_seconds = float(np.median(differences))

    if not np.isfinite(interval_seconds) and frame_interval is not None:
        try:
            interval_seconds = float(frame_interval)
        except (TypeError, ValueError):
            interval_seconds = np.nan

    return {
        "frame_interval_seconds": interval_seconds,
        "flag": bool(
            np.isfinite(interval_seconds)
            and interval_seconds >= float(threshold_seconds) - 1e-9
        ),
    }
