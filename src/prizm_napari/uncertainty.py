"""Segmentation uncertainty metrics used by PRIZM batch analysis."""

from __future__ import annotations

import math

import numpy as np


EPSILON = 1e-7

# Linear fit from the full annotated PRIZM dataset:
# raw mean atrial Dice = intercept + slope * mean atrial segment entropy.
# The QC cutoff is the entropy value where the fitted raw Dice equals 0.4.
ATRIUM_ENTROPY_QC_DICE_CUTOFF = 0.4
ATRIUM_ENTROPY_RAW_DICE_FIT_SLOPE = -3.0059063866503086
ATRIUM_ENTROPY_RAW_DICE_FIT_INTERCEPT = 1.000781681138433
ATRIUM_SEGMENT_ENTROPY_QC_THRESHOLD_NATS = (
    ATRIUM_ENTROPY_QC_DICE_CUTOFF - ATRIUM_ENTROPY_RAW_DICE_FIT_INTERCEPT
) / ATRIUM_ENTROPY_RAW_DICE_FIT_SLOPE


def binary_entropy(probability: np.ndarray) -> np.ndarray:
    """Return binary Shannon entropy in nats (Mehrtash et al., Equation 7)."""
    probability = np.clip(
        np.asarray(probability, dtype=np.float64),
        EPSILON,
        1.0 - EPSILON,
    )
    return -probability * np.log(probability) - (1.0 - probability) * np.log(
        1.0 - probability
    )


def atrium_segment_entropy_qc_flag(segment_entropy_mean: float) -> bool | float:
    """Flag finite atrial segment entropy at or above the fitted QC cutoff."""
    value = float(segment_entropy_mean)
    if not math.isfinite(value):
        return math.nan
    return bool(value >= ATRIUM_SEGMENT_ENTROPY_QC_THRESHOLD_NATS)


def cleaned_atrium_segment_entropy(
    atrium_probability: np.ndarray,
    cleaned_labels: np.ndarray,
    *,
    atrium_label: int = 2,
) -> dict[str, float | bool | int]:
    """Calculate mean atrial entropy over the final cleaned atrial mask."""
    probability = np.asarray(atrium_probability, dtype=np.float64)
    labels = np.asarray(cleaned_labels)
    if probability.ndim != 2:
        raise ValueError(
            "atrium_probability must have shape (height, width); "
            f"received {probability.shape}"
        )
    if labels.shape != probability.shape:
        raise ValueError(
            f"cleaned label shape {labels.shape} does not match probability shape "
            f"{probability.shape}"
        )
    if not np.all(np.isfinite(probability)):
        raise ValueError("atrium_probability contains non-finite values")
    if np.any(probability < -1e-6) or np.any(probability > 1.0 + 1e-6):
        raise ValueError("atrium_probability values must lie between zero and one")

    atrium_mask = labels == int(atrium_label)
    pixels = int(np.count_nonzero(atrium_mask))
    if pixels == 0:
        return {
            "atrium_prediction_present": False,
            "segment_pixels": 0,
            "segment_entropy_mean": math.nan,
            "segment_entropy_qc_flag": math.nan,
        }
    entropy = binary_entropy(probability)
    segment_entropy_mean = float(np.mean(entropy[atrium_mask]))
    return {
        "atrium_prediction_present": True,
        "segment_pixels": pixels,
        "segment_entropy_mean": segment_entropy_mean,
        "segment_entropy_qc_flag": atrium_segment_entropy_qc_flag(
            segment_entropy_mean
        ),
    }
