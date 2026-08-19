"""Segmentation uncertainty metrics used by PRIZM batch analysis."""

from __future__ import annotations

import math

import numpy as np


EPSILON = 1e-7


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
        }
    entropy = binary_entropy(probability)
    return {
        "atrium_prediction_present": True,
        "segment_pixels": pixels,
        "segment_entropy_mean": float(np.mean(entropy[atrium_mask])),
    }
