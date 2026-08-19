import math

import numpy as np
import pandas as pd
import torch
from PIL import Image

from prizm_napari.analysis import compute_segmentation_statistics
from prizm_napari.infer import PRIZMInference
from prizm_napari.uncertainty import (
    binary_entropy,
    cleaned_atrium_segment_entropy,
)


def test_binary_entropy_matches_mehrtash_equation_7_in_nats():
    probabilities = np.array([0.5, 0.8], dtype=float)
    measured = binary_entropy(probabilities)
    expected = -probabilities * np.log(probabilities) - (
        1.0 - probabilities
    ) * np.log(1.0 - probabilities)
    np.testing.assert_allclose(measured, expected)
    assert measured[0] == math.log(2.0)


def test_entropy_is_measured_only_inside_final_cleaned_atrium():
    probability = np.array([[0.5, 0.8], [0.1, 0.9]], dtype=float)
    cleaned_labels = np.array([[0, 2], [2, 1]], dtype=np.uint8)
    result = cleaned_atrium_segment_entropy(probability, cleaned_labels)
    expected = np.mean(binary_entropy(np.array([0.8, 0.1])))
    assert result["atrium_prediction_present"] is True
    assert result["segment_pixels"] == 2
    np.testing.assert_allclose(result["segment_entropy_mean"], expected)


def test_no_cleaned_atrium_returns_nan():
    result = cleaned_atrium_segment_entropy(
        np.full((3, 3), 0.5),
        np.zeros((3, 3), dtype=np.uint8),
    )
    assert result["atrium_prediction_present"] is False
    assert result["segment_pixels"] == 0
    assert np.isnan(result["segment_entropy_mean"])


def test_model_logits_are_converted_to_probabilities():
    logits = np.array([[[[0.0]], [[1.0]], [[2.0]]]], dtype=np.float32)
    probabilities = PRIZMInference._class_probabilities(logits)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
    assert probabilities[0, 2, 0, 0] > probabilities[0, 1, 0, 0]


def test_onnx_probability_channels_follow_label_remap():
    inference = object.__new__(PRIZMInference)
    inference.onnx_label_remap = np.array([0, 2, 1], dtype=np.uint16)
    raw = np.array([[[[0.1]], [[0.7]], [[0.2]]]], dtype=np.float32)
    standardized = inference._standardize_probability_channels(raw)
    np.testing.assert_allclose(standardized[:, 2], raw[:, 1])
    np.testing.assert_allclose(standardized[:, 1], raw[:, 2])


def test_postprocess_brightness_threshold_uses_unit_scale():
    inference = object.__new__(PRIZMInference)
    inference.onnx_input_scale = 255.0
    image = torch.zeros((1, 64, 64), dtype=torch.float32)
    mask = np.zeros((64, 64), dtype=np.uint16)

    mask[2:22, 2:22] = 2
    image[:, 2:22, 2:22] = 25.5  # 0.10 after scaling: reject as dark
    mask[40:60, 40:60] = 2
    image[:, 40:60, 40:60] = 51.0  # 0.20 after scaling: retain

    cleaned = inference.postprocess_image(image, mask)
    assert np.count_nonzero(cleaned[2:22, 2:22]) == 0
    assert np.all(cleaned[40:60, 40:60] == 2)


def test_frame_statistics_receive_entropy_and_keep_label_stack_tiff_contract(tmp_path):
    frames = np.zeros((2, 96, 96), dtype=np.uint8)
    masks = np.zeros((2, 96, 96), dtype=np.uint8)
    masks[:, 20:45, 18:43] = 1
    masks[0, 48:73, 50:75] = 2
    frames[masks > 0] = 220
    atrium_probability = np.full((2, 96, 96), 0.05, dtype=np.float32)
    atrium_probability[0, 48:73, 50:75] = 0.8

    frame_stats, _fs_frames, cleaned_masks = compute_segmentation_statistics(
        frames,
        masks,
        "sample",
        str(tmp_path),
        meta_info={"len_per_px": 0.921, "frame_interval": 0.062},
        frame_filenames=["sample_t0.tif", "sample_t1.tif"],
        return_cleaned_masks=True,
        apply_mask_cleanup=False,
        atrium_probabilities=atrium_probability,
        visualization_format="jpg",
    )

    expected = float(binary_entropy(np.array([0.8]))[0])
    np.testing.assert_allclose(
        frame_stats.loc[0, "AtriumSegmentEntropyMean_nats"], expected
    )
    assert np.isnan(frame_stats.loc[1, "AtriumSegmentEntropyMean_nats"])
    assert frame_stats.loc[0, "AtriumPredictionPresent"]
    assert not frame_stats.loc[1, "AtriumPredictionPresent"]
    assert not any("Threshold" in column for column in frame_stats.columns)
    assert not any("Flag" in column for column in frame_stats.columns)
    saved_frame_stats = pd.read_excel(tmp_path / "sample.xlsx")
    assert "AtriumSegmentEntropyMean_nats" in saved_frame_stats.columns
    assert not any("Threshold" in column for column in saved_frame_stats.columns)
    assert not any("Flag" in column for column in saved_frame_stats.columns)
    np.testing.assert_array_equal(cleaned_masks, masks)
    assert (tmp_path / "FS" / "FS_sample_t0.jpg").exists()
    assert not list((tmp_path / "FS").glob("*.tif"))
    with Image.open(tmp_path / "sample_FS.gif") as saved:
        assert saved.n_frames == 2
        assert saved.info["loop"] == 0
        for frame_index in range(saved.n_frames):
            saved.seek(frame_index)
            assert saved.info["duration"] == 60
