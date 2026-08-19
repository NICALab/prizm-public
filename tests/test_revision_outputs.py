from pathlib import Path

import numpy as np
from PIL import Image, JpegImagePlugin
import pytest
import tifffile

from prizm_napari.analysis import (
    MINOR_AXIS_FRAC_OFFSETS,
    _burn_in_timestamp_and_scalebar,
    _matlab_preprocess_pdouble,
    _select_minor_axis_candidate,
    save_gif_with_relative_times,
    summarize_segmentation_uncertainty,
)
from prizm_napari.batch_segmentation_core import (
    BASE_CROP_PX,
    BASE_DETECT_PX,
    _crop_resize_center,
    _estimate_center_2stage,
    _save_input_visualizations,
    _save_masked_gif,
    _scaled_crop_sizes,
    _to_gray_unit,
)
from prizm_napari.output_utils import (
    normalize_visualization_format,
    save_visualization,
    segmentation_overlay,
    to_uint8,
    visualization_name,
)


def test_onnx_repair_is_idempotent_after_fixed_model_is_renamed(tmp_path: Path):
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper

    from prizm_napari.infer import PRIZMInference

    shape_output = "dec_crop1_InputShape"
    cast_output = f"{shape_output}_float"
    graph = helper.make_graph(
        [
            helper.make_node(
                "Shape",
                inputs=["input"],
                outputs=[shape_output],
                name="dec_crop1_Shape",
            ),
            helper.make_node(
                "Cast",
                inputs=[shape_output],
                outputs=[cast_output],
                name="dec_crop1_Shape_CastFloat",
                to=TensorProto.FLOAT,
            ),
            helper.make_node(
                "Identity",
                inputs=[cast_output],
                outputs=["output"],
                name="dec_crop1_Identity",
            ),
        ],
        "already-repaired-model",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 4, 4])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [4])],
    )
    model = helper.make_model(graph)
    renamed_model = tmp_path / "renamed.onnx"
    onnx.save(model, renamed_model)

    inference = object.__new__(PRIZMInference)
    prepared_path = Path(inference._prepare_onnx_model(str(renamed_model)))
    prepared = onnx.load(prepared_path)
    all_outputs = [name for node in prepared.graph.node for name in node.output]

    onnx.checker.check_model(prepared)
    assert all_outputs.count(cast_output) == 1


def test_uint16_normalization_uses_full_dynamic_range_once():
    image = np.array([[1000, 2000], [3000, 5000]], dtype=np.uint16)
    normalized = _to_gray_unit(image)
    assert normalized.min() == 0.0
    assert normalized.max() == 1.0
    np.testing.assert_allclose(normalized[0, 1], 0.25)


def test_crop_sizes_preserve_baseline_physical_field_of_view():
    assert _scaled_crop_sizes(0.9210) == (BASE_CROP_PX, BASE_DETECT_PX)
    assert _scaled_crop_sizes(0.4605) == (600, 800)
    assert _scaled_crop_sizes(None) == (BASE_CROP_PX, BASE_DETECT_PX)


def test_scaled_crop_is_resized_to_model_input_shape():
    image = np.arange(700 * 700, dtype=np.uint32).reshape(700, 700).astype(np.uint16)
    crop = _crop_resize_center(image, (350, 350), out_size=300, crop_size=600)
    assert crop.shape == (300, 300)


def test_adaptive_centering_ignores_broad_dim_haze():
    yy, xx = np.mgrid[:600, :800]
    haze = 0.42 * np.exp(-(((xx - 180) / 220) ** 2 + ((yy - 310) / 250) ** 2))
    heart_radius = ((xx - 610) / 65) ** 2 + ((yy - 245) / 50) ** 2
    heart = np.where(heart_radius <= 1.0, 0.65 + 0.35 * (1.0 - heart_radius), 0.0)
    image = np.clip(haze + heart, 0, 1)
    image = np.rint(image * np.iinfo(np.uint16).max).astype(np.uint16)

    center_x, center_y = _estimate_center_2stage(image, crop_size=400)
    assert abs(center_x - 610) < 20
    assert abs(center_y - 245) < 20


def test_visualization_format_and_fixed_integer_scaling(tmp_path: Path):
    assert normalize_visualization_format("jpeg") == "jpg"
    assert normalize_visualization_format("tiff") == "tif"
    assert visualization_name("cropped", "frame_001.tif", "jpg") == "cropped_frame_001.jpg"
    assert int(to_uint8(np.array([32768], dtype=np.uint16))[0]) == 127

    destination = tmp_path / "cropped_frame_001.jpg"
    save_visualization(destination, np.full((16, 16), 32768, dtype=np.uint16), "jpg")
    assert destination.exists()
    assert not destination.with_suffix(".tif").exists()
    with Image.open(destination) as saved:
        assert saved.format == "JPEG"
        assert saved.mode == "L"

    rgb_destination = tmp_path / "labeled_frame_001.jpg"
    save_visualization(
        rgb_destination,
        np.zeros((16, 16, 3), dtype=np.uint8),
        "jpg",
    )
    with Image.open(rgb_destination) as saved:
        assert saved.mode == "RGB"
        assert JpegImagePlugin.get_sampling(saved) == 2

    tif_destination = tmp_path / "cropped_frame_001.tif"
    source_tif = np.full((16, 16), 32768, dtype=np.uint16)
    save_visualization(tif_destination, source_tif, "tif")
    assert tifffile.imread(tif_destination).dtype == np.uint16


def test_segmentation_overlay_preserves_integer_labels_as_colors():
    image = np.zeros((4, 4), dtype=np.uint8)
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[1, 1] = 1
    mask[2, 2] = 2
    overlay = segmentation_overlay(image, mask)
    assert overlay.shape == (4, 4, 3)
    np.testing.assert_array_equal(overlay[1, 1], [0, 0, 36])
    np.testing.assert_array_equal(overlay[2, 2], [0, 33, 19])


def test_preprocessing_visualization_uses_matlab_rgb_to_gray(tmp_path: Path):
    horizontal = np.tile(np.arange(16, dtype=np.uint8), (16, 1))
    vertical = horizontal.T
    cropped = np.stack([horizontal, vertical, np.zeros_like(horizontal)], axis=-1)
    _save_input_visualizations(
        str(tmp_path),
        ["frame.tif"],
        [cropped],
        np.zeros((1, 16, 16), dtype=np.uint8),
        "tif",
    )
    observed = tifffile.imread(tmp_path / "preprocessing" / "preprocessing_frame.tif")
    expected = to_uint8(_matlab_preprocess_pdouble(cropped))
    np.testing.assert_array_equal(observed, expected)


def test_gif_burn_in_uses_copy_and_scale():
    frame = np.zeros((80, 120, 3), dtype=np.uint8)
    annotated = _burn_in_timestamp_and_scalebar(frame, 1.25, 1.0, 50.0)
    assert not np.shares_memory(frame, annotated)
    assert np.count_nonzero(frame) == 0
    assert np.count_nonzero(annotated) > 0


def test_scale_bar_label_is_centered_and_matches_timestamp_font(monkeypatch):
    import prizm_napari.analysis as analysis_module

    text_calls = []
    line_calls = []

    def capture_text(image, text, origin, font, font_scale, color, thickness, line_type):
        text_calls.append((text, origin, font, font_scale, thickness))
        return image

    def capture_line(image, start, end, color, thickness, line_type):
        line_calls.append((start, end, thickness))
        return image

    monkeypatch.setattr(analysis_module.cv2, "putText", capture_text)
    monkeypatch.setattr(analysis_module.cv2, "line", capture_line)

    _burn_in_timestamp_and_scalebar(
        np.zeros((80, 120, 3), dtype=np.uint8),
        1.25,
        1.0,
        50.0,
    )

    assert [call[0] for call in text_calls] == ["1.25s", "50 um"]
    assert text_calls[0][3] == text_calls[1][3] == 0.4
    bar_start, bar_end, _bar_thickness = line_calls[0]
    (text_width, _text_height), _baseline = analysis_module.cv2.getTextSize(
        "50 um",
        text_calls[1][2],
        text_calls[1][3],
        text_calls[1][4],
    )
    expected_x = int(round((bar_start[0] + bar_end[0] - text_width) / 2.0))
    assert text_calls[1][1][0] == expected_x


def test_gif_writer_preserves_frame_timing_loop_and_inputs(tmp_path: Path):
    frames = []
    for index in range(3):
        frame = np.zeros((80, 120, 3), dtype=np.uint8)
        frame[10:30, 10 + 20 * index : 30 + 20 * index] = 80 * (index + 1)
        frames.append(frame)
    originals = [frame.copy() for frame in frames]
    destination = tmp_path / "timed.gif"

    save_gif_with_relative_times(
        frames,
        [0.0, 0.06, 0.12],
        destination,
        um_per_px=1.0,
    )

    with Image.open(destination) as saved:
        assert saved.n_frames == 3
        assert saved.info["loop"] == 0
        durations = []
        for frame_index in range(saved.n_frames):
            saved.seek(frame_index)
            durations.append(saved.info["duration"])
    assert durations == [60, 60, 60]
    for frame, original in zip(frames, originals, strict=True):
        np.testing.assert_array_equal(frame, original)


def test_masked_gif_is_written_with_expected_name_and_metadata(tmp_path: Path):
    cropped = [
        np.full((80, 120, 3), 100 + index * 20, dtype=np.uint8)
        for index in range(2)
    ]
    masks = np.zeros((2, 80, 120), dtype=np.uint8)
    masks[:, 20:50, 25:55] = 1
    masks[:, 25:55, 65:95] = 2

    output = _save_masked_gif(
        str(tmp_path),
        "sample",
        cropped,
        masks,
        [0.0, 0.06],
        1.0,
    )

    assert Path(output) == tmp_path / "sample_masked.gif"
    with Image.open(output) as saved:
        assert saved.n_frames == 2
        assert saved.info["loop"] == 0
        for frame_index in range(saved.n_frames):
            saved.seek(frame_index)
            assert saved.info["duration"] == 60


def test_minor_axis_selection_uses_best_signed_area_correlation():
    area = np.arange(1.0, 7.0)
    candidates = np.ones((area.size, len(MINOR_AXIS_FRAC_OFFSETS)), dtype=float)
    candidates[:, 1] = area[::-1]
    candidates[:, 7] = 3.0 * area + 2.0
    winner, correlations = _select_minor_axis_candidate(candidates, area)
    assert winner == 7
    assert correlations[7] == 1.0
    assert correlations[1] == -1.0


def test_entropy_summary_omits_threshold_and_flag_fields():
    import pandas as pd

    summary = summarize_segmentation_uncertainty(
        pd.DataFrame(
            {
                "AtriumPredictionPresent": [True, False, True],
                "AtriumSegmentEntropyMean_nats": [0.2, np.nan, 0.6],
            }
        )
    )
    assert summary["AtriumSegmentEntropyMean_nats"] == 0.4
    assert summary["AtriumSegmentEntropyMax_nats"] == 0.6
    assert summary["AtriumSegmentEntropyP95_nats"] == 0.58
    assert summary["AtriumSegmentEntropyValidFrames"] == 2
    assert summary["AtriumPredictionMissingFrames"] == 1
    assert not any("Threshold" in key for key in summary)
    assert not any("Flag" in key for key in summary)
