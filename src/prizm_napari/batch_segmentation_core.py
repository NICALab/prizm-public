"""
Core batch segmentation function used by both GUI and CLI.
This module contains no GUI dependencies.
"""

import os
import re
from typing import Optional
from time import perf_counter
import numpy as np
import tifffile
import dask
import pandas as pd
import skimage.io as skio
import cv2
from datetime import datetime
from tqdm import tqdm

from prizm_napari.infer import PRIZMInference
from prizm_napari.analysis import (
    _preprocess_refinement_frame,
    compute_segmentation_statistics,
    compute_functional_statistics,
    compute_synchronize_analysis,
    combine_results,
    derive_series_key,
    perfish_export_dataframe,
    parse_xml_times_and_scale,
    save_gif_with_relative_times,
)
from prizm_napari.output_utils import (
    normalize_visualization_format,
    save_visualization,
    segmentation_overlay,
    to_uint8,
    visualization_name,
)


VALID_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
BASE_UM_PER_PX = 0.9210
BASE_CROP_PX = 300
BASE_DETECT_PX = 400


def _read_frame_fast(path: str, as_gray: bool = False) -> np.ndarray:
    """
    Faster image loader than skimage.io.imread for large batch runs.
    Falls back to skimage for uncommon formats/edge cases.
    """
    if as_gray:
        arr = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if arr is not None:
            return arr
        return skio.imread(path, as_gray=True)

    arr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if arr is None:
        return skio.imread(path, as_gray=False)

    if arr.ndim == 3:
        # OpenCV loads color as BGR; convert to RGB for consistency.
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    return arr


def _extract_t_index(fname: str) -> Optional[int]:
    m = re.search(r"_t(\d+)", str(fname))
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _sorted_frame_names(sample_path: str, valid_exts: set[str]) -> list[str]:
    files = [
        fn for fn in os.listdir(sample_path)
        if os.path.splitext(fn)[1].lower() in valid_exts
    ]
    def _key(fn: str):
        t_idx = _extract_t_index(fn)
        if t_idx is None:
            return (1, fn.lower())
        return (0, int(t_idx), fn.lower())
    return sorted(files, key=_key)


def _extract_date_yyyymmdd(text: str) -> Optional[str]:
    m = re.search(r"(20\d{6})", str(text))
    if not m:
        return None
    return m.group(1)


def _discover_sample_dirs(chem_conc_path: str, valid_exts: Optional[set[str]] = None) -> list[str]:
    """
    Discover per-sample subdirectories under a {chem}_{conc} directory.

    Naming of sample subdirectories is intentionally unconstrained.
    Any unique subdirectory that contains frame-like image files is treated
    as a sample directory.
    """
    if valid_exts is None:
        valid_exts = VALID_EXTS

    if not os.path.isdir(chem_conc_path):
        return []

    ignore_names = {
        "metadata",
        "meta",
        "results",
        "result",
        "synchronize",
        "sync",
        "figures_300dpi",
        "lda_report",
        "panel_heatmap",
        "__pycache__",
    }

    all_subdirs = sorted(
        d
        for d in os.listdir(chem_conc_path)
        if os.path.isdir(os.path.join(chem_conc_path, d))
        and not d.startswith(".")
        and d.lower() not in ignore_names
    )
    if not all_subdirs:
        return []

    # Primary rule: any subdir that has frame-like images at top-level.
    has_frames = []
    for d in all_subdirs:
        p = os.path.join(chem_conc_path, d)
        frame_count = sum(
            1
            for fn in os.listdir(p)
            if os.path.isfile(os.path.join(p, fn))
            and os.path.splitext(fn)[1].lower() in valid_exts
        )
        if frame_count > 0:
            has_frames.append(d)
    if has_frames:
        return sorted(has_frames)

    # Last-resort fallback when frame discovery fails.
    return sorted(all_subdirs)


def _find_metadata_xml(
    sample_path: str,
    sample_dir: str,
    sample_id: str,
    chem_conc_path: str,
    metadata_file: Optional[str] = None,
) -> Optional[str]:
    if metadata_file:
        return metadata_file if os.path.exists(metadata_file) else None

    search_dirs = [
        sample_path,
        os.path.join(sample_path, "metadata"),
        os.path.join(sample_path, "Metadata"),
        os.path.join(sample_path, "MetaData"),
        os.path.join(sample_path, "METADATA"),
        os.path.join(chem_conc_path, "metadata"),
        os.path.join(chem_conc_path, "Metadata"),
        os.path.join(chem_conc_path, "MetaData"),
        os.path.join(chem_conc_path, "METADATA"),
    ]

    seen = set()
    uniq_dirs = []
    for d in search_dirs:
        if d in seen:
            continue
        seen.add(d)
        if os.path.isdir(d):
            uniq_dirs.append(d)

    exact_names = [
        f"{sample_id}_Properties.xml",
        f"{sample_dir}_Properties.xml",
    ]
    if sample_dir.startswith("sample_"):
        sample_suffix = sample_dir[len("sample_") :]
        if sample_suffix:
            exact_names.append(f"{sample_suffix}_Properties.xml")

    for d in uniq_dirs:
        for nm in exact_names:
            p = os.path.join(d, nm)
            if os.path.exists(p):
                return p

    for d in uniq_dirs:
        props = sorted(
            f
            for f in os.listdir(d)
            if f.lower().endswith("_properties.xml") and os.path.isfile(os.path.join(d, f))
        )
        if props:
            return os.path.join(d, props[0])

    for d in uniq_dirs:
        xmls = sorted(
            f
            for f in os.listdir(d)
            if f.lower().endswith(".xml") and os.path.isfile(os.path.join(d, f))
        )
        if xmls:
            return os.path.join(d, xmls[0])

    return None


def _to_gray_unit(img: np.ndarray, prefer_green: bool = True) -> np.ndarray:
    arr = np.asarray(img)
    if arr.ndim == 3:
        if prefer_green and arr.shape[2] >= 3:
            arr = arr[..., 1]
        else:
            # Weighted RGB-to-gray conversion during the fine-center stage.
            if arr.shape[2] >= 3:
                arr = (
                    0.2989 * arr[..., 0]
                    + 0.5870 * arr[..., 1]
                    + 0.1140 * arr[..., 2]
                )
            else:
                arr = arr[..., 0]
    arr = np.asarray(arr, dtype=float)
    arr[~np.isfinite(arr)] = 0.0
    mx = float(np.max(arr)) if arr.size else 0.0
    mn = float(np.min(arr)) if arr.size else 0.0
    if mx > 1.0 or mn < 0.0:
        if mx > mn:
            arr = (arr - mn) / (mx - mn)
        else:
            arr = np.zeros_like(arr, dtype=float)
    return np.clip(arr, 0.0, 1.0)


def _largest_component_mask(mask: np.ndarray) -> np.ndarray:
    m = np.asarray(mask, dtype=np.uint8)
    if m.size == 0:
        return np.zeros_like(m, dtype=bool)
    n_lbl, lbls, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n_lbl <= 1:
        return np.zeros_like(m, dtype=bool)
    areas = stats[1:, cv2.CC_STAT_AREA]
    if areas.size == 0:
        return np.zeros_like(m, dtype=bool)
    lab = int(np.argmax(areas) + 1)
    return lbls == lab


def _binary_centroid(mask: np.ndarray) -> Optional[tuple[float, float]]:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    return float(np.mean(xs)), float(np.mean(ys))


def _estimate_center_2stage(
    frame: np.ndarray,
    crop_size: int = 400,
    coarse_thresh: float = 0.10,
    fine_thresh: float = 0.10,
) -> tuple[float, float]:
    """
    Two-stage center estimation:
    1) coarse threshold on green/full frame
    2) crop around coarse centroid
    3) stretch-like adjustment + fine threshold in ROI
    """
    arr = np.asarray(frame)
    h, w = arr.shape[:2]
    coarse_src = _to_gray_unit(arr, prefer_green=True)
    adaptive_thresh = max(
        float(coarse_thresh),
        float(np.nanpercentile(coarse_src, 98.0)),
    )
    bw = _largest_component_mask(coarse_src > adaptive_thresh)
    c0 = _binary_centroid(bw)
    if c0 is None:
        coarse_center = (w / 2.0, h / 2.0)
    else:
        coarse_center = c0

    fine_roi_size = max(60, int(round(float(crop_size) * 0.375)))
    half = int(round(fine_roi_size / 2.0))
    x1 = max(0, int(round(coarse_center[0])) - half)
    y1 = max(0, int(round(coarse_center[1])) - half)
    x2 = min(w, x1 + fine_roi_size)
    y2 = min(h, y1 + fine_roi_size)
    x1 = max(0, x2 - fine_roi_size)
    y1 = max(0, y2 - fine_roi_size)

    roi = arr[y1:y2, x1:x2]
    if roi.size == 0:
        return coarse_center

    gray = _to_gray_unit(roi, prefer_green=False)
    j_lo = float(np.nanpercentile(gray, 1.0))
    j_hi = float(np.nanpercentile(gray, 99.0))
    j_hi = min(1.0, 0.90 * j_hi)
    if not np.isfinite(j_lo):
        j_lo = 0.0
    if not np.isfinite(j_hi):
        j_hi = 1.0
    if j_hi <= j_lo:
        prep = np.zeros_like(gray, dtype=float)
    else:
        prep = np.clip((gray - j_lo) / (j_hi - j_lo), 0.0, 1.0)

    adaptive_fine_thresh = max(
        float(fine_thresh),
        float(np.nanpercentile(prep, 98.0)),
    )
    bw2 = _largest_component_mask(prep > adaptive_fine_thresh)
    c_local = _binary_centroid(bw2)
    if c_local is None:
        return coarse_center
    return (float(x1 + c_local[0]), float(y1 + c_local[1]))


def _crop_resize_center(
    frame: np.ndarray,
    center_xy: tuple[float, float],
    out_size: int = BASE_CROP_PX,
    crop_size: Optional[int] = None,
) -> np.ndarray:
    """Crop around a fixed center, then resize to ``out_size`` square pixels."""
    arr = np.asarray(frame)
    h, w = arr.shape[:2]
    cx, cy = float(center_xy[0]), float(center_xy[1])
    crop_size = max(1, int(crop_size if crop_size is not None else out_size))
    x1 = int(round(cx - crop_size / 2.0))
    y1 = int(round(cy - crop_size / 2.0))
    x2 = x1 + crop_size
    y2 = y1 + crop_size
    x1c, y1c = max(0, x1), max(0, y1)
    x2c, y2c = min(w, x2), min(h, y2)
    roi = arr[y1c:y2c, x1c:x2c]
    if roi.size == 0:
        roi = arr
    return cv2.resize(roi, (int(out_size), int(out_size)), interpolation=cv2.INTER_LINEAR)


def _valid_um_per_px(value) -> Optional[float]:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(parsed) or parsed <= 0:
        return None
    return parsed


def _scaled_crop_sizes(um_per_px: Optional[float]) -> tuple[int, int]:
    """Return crop and detection sizes preserving the baseline physical FOV."""
    valid_scale = _valid_um_per_px(um_per_px)
    if valid_scale is None:
        return BASE_CROP_PX, BASE_DETECT_PX
    scale = BASE_UM_PER_PX / valid_scale
    crop_px = max(1, int(round(BASE_CROP_PX * scale)))
    detect_px = max(60, int(round(BASE_DETECT_PX * scale)))
    return crop_px, detect_px


def _save_input_visualizations(
    sample_out: str,
    frame_names: list[str],
    cropped_frames: list[np.ndarray],
    masks: np.ndarray,
    output_format: str,
) -> None:
    """Write the document-specified cropped, preprocessing, and labeled frames."""
    for frame_name, cropped, mask in zip(frame_names, cropped_frames, masks):
        # Use the established weighted grayscale conversion on the complete
        # RGB crop. This visualization is deliberately independent of the
        # model's selected segmentation channel.
        preprocessed = to_uint8(_preprocess_refinement_frame(cropped))
        outputs = (
            ("cropped", "cropped", cropped),
            ("preprocessing", "preprocessing", preprocessed),
            ("masked", "labeled", segmentation_overlay(preprocessed, mask)),
        )
        for directory, prefix, image in outputs:
            filename = visualization_name(prefix, frame_name, output_format)
            save_visualization(
                os.path.join(sample_out, directory, filename),
                image,
                output_format,
            )


def _relative_times_for_frames(
    relative_times,
    frame_count: int,
    frame_interval: Optional[float],
) -> np.ndarray:
    """Resolve one timestamp per frame, falling back to the manual interval."""
    if relative_times is not None:
        resolved = np.asarray(relative_times, dtype=float).reshape(-1)
        if resolved.size == int(frame_count):
            return resolved

    interval = _valid_um_per_px(frame_interval)
    if interval is None:
        interval = 0.062
    return np.arange(int(frame_count), dtype=float) * float(interval)


def _save_masked_gif(
    sample_out: str,
    video_name: str,
    cropped_frames: list[np.ndarray],
    masks: np.ndarray,
    relative_times,
    um_per_px: float,
) -> str:
    """Write the document-specified annotated masked GIF."""
    masked_frames = (
        segmentation_overlay(
            to_uint8(_preprocess_refinement_frame(cropped)),
            mask,
        )
        for cropped, mask in zip(cropped_frames, masks, strict=False)
    )
    return save_gif_with_relative_times(
        masked_frames,
        relative_times,
        os.path.join(sample_out, f"{video_name}_masked.gif"),
        um_per_px=um_per_px,
        scale_bar_um=50.0,
    )


def run_batch_segmentation_core(
    root_dir: str,
    out_dir: str,
    model_path: str,
    model_type: str,
    channel: int,
    grayscale: bool,
    backbone: str,
    encoder_depth: int,
    decoder_channels: int,
    encoder_output_stride: int,
    atrous_rates: list,
    meta_manual: bool,
    input_channels: int = 1,
    resize_scale: float = None,
    frame_interval: float = None,
    metadata_file: str = None,
    load_to_viewer: bool = False,
    save_analysis_vis: bool = False,
    infer_postprocess: bool = False,
    infer_batch_size: int = 1,
    use_amp: bool = True,
    visualization_format: str = "jpg",
    progress_callback=None,
):
    """
    Core batch segmentation function used by both GUI and CLI.
    
    This is the exact same code that the GUI uses, extracted into a shared function.
    """
    visualization_format = normalize_visualization_format(visualization_format)
    infer = PRIZMInference(
        model_path,
        model_type=model_type,
        num_classes=3,
        backbone=backbone,
        encoder_depth=encoder_depth,
        decoder_channels=decoder_channels,
        encoder_output_stride=encoder_output_stride,
        decoder_atrous_rates=atrous_rates,
        input_channels=input_channels,
        enable_postprocess=infer_postprocess,
        infer_batch_size=infer_batch_size,
        use_amp=use_amp,
    )

    def emit_log(message: str):
        if progress_callback:
            progress_callback(str(message))

    if infer.model_backend == "onnx":
        providers_text = f"available providers={infer.onnx_available_providers}"
        dll_dirs = getattr(infer, "onnx_cuda_dll_dirs", None) or []
        dll_text = (
            f" | NVIDIA DLL dirs added={len(dll_dirs)}"
            if dll_dirs
            else ""
        )
        if infer.onnx_session_device_ids:
            emit_log(
                "ONNX backend ready on CUDA GPU(s): "
                f"{infer.onnx_session_device_ids} | sessions={len(infer.onnx_sessions)} | "
                f"requested batch size={infer.infer_batch_size} | {providers_text}{dll_text}"
            )
        else:
            cuda_error_text = ""
            cuda_errors = getattr(infer, "onnx_cuda_session_errors", None) or []
            if cuda_errors:
                first_error = str(cuda_errors[0]).splitlines()[0]
                cuda_error_text = f" | CUDA session error={first_error[:240]}"
            emit_log(
                "ONNX backend ready on CPU fallback | "
                f"sessions={len(infer.onnx_sessions)} | requested batch size={infer.infer_batch_size} | "
                f"{providers_text}{dll_text}{cuda_error_text}"
            )
    else:
        emit_log(
            f"PyTorch backend ready on device={infer.device} | requested batch size={infer.infer_batch_size}"
        )

    # Supported structures:
    #   {CHEMICAL_TYPE}_{CONCENTRATION} → sample_{SAMPLE_ID} → frame files
    #   {CHEMICAL_TYPE}_{CONCENTRATION} → SeriesXXX...      → frame files
    batch_combined_list = []
    total_samples = 0

    if load_to_viewer:
        img_list = []
    
    # Iterate over chemical_type_concentration directories
    chem_conc_dirs = sorted(
        d for d in os.listdir(root_dir) 
        if os.path.isdir(os.path.join(root_dir, d)) and not d.startswith('.')
    )
    
    # Count total samples for progress bar
    for chem_conc_dir in chem_conc_dirs:
        chem_conc_path = os.path.join(root_dir, chem_conc_dir)
        sample_dirs = _discover_sample_dirs(chem_conc_path, valid_exts=VALID_EXTS)
        total_samples += len(sample_dirs)
    
    # Main progress bar for all samples (only if not using GUI callback)
    if progress_callback is None:
        pbar_samples = tqdm(total=total_samples, desc="Processing samples", unit="sample", position=0, leave=True)
    else:
        pbar_samples = None  # Disable tqdm when using GUI progress
    
    for chem_conc_dir in chem_conc_dirs:
        chem_conc_path = os.path.join(root_dir, chem_conc_dir)
        
        # Parse chemical type and concentration from directory name
        # Format: {CHEMICAL_TYPE}_{CONCENTRATION} (e.g., "BaP_0.1", "Ter_CTRL")
        parts = chem_conc_dir.split('_', 1)
        if len(parts) == 2:
            chemical_type = parts[0]
            concentration = parts[1]
        else:
            # Fallback: assume it's just chemical type, concentration is unknown
            chemical_type = parts[0]
            concentration = "UNKNOWN"
        
        # Find all sample directories
        sample_dirs = _discover_sample_dirs(chem_conc_path, valid_exts=VALID_EXTS)
        condition_combined_list = []
        
        if pbar_samples is not None:
            pbar_samples.set_description(f"Processing {chem_conc_dir}")
        
        for sample_dir in sample_dirs:
            sample_path = os.path.join(chem_conc_path, sample_dir)
            sample_id = sample_dir
            sample_index = len(batch_combined_list) + 1
            
            # Create per-sample output directory
            sample_out = os.path.join(out_dir, chem_conc_dir, sample_dir)
            os.makedirs(sample_out, exist_ok=True)

            # Load single-frame images from sample directory
            # List and sort frame files
            frames = _sorted_frame_names(sample_path, VALID_EXTS)
            
            if not frames:
                emit_log(
                    f"[{sample_index}/{total_samples}] {chem_conc_dir}/{sample_dir}: no image files found, skipping"
                )
                if pbar_samples is not None:
                    tqdm.write(f"Warning: No image files found in {sample_path}")
                    pbar_samples.update(1)
                else:
                    print(f"Warning: No image files found in {sample_path}")
                if progress_callback:
                    progress_callback(len(batch_combined_list))
                continue

            # Create standardized video name for output: {DATE}_{CHEMICAL_TYPE}_{CONCENTRATION}_{SAMPLE_ID}
            current_date = datetime.now().strftime("%Y%m%d")
            date_str = _extract_date_yyyymmdd(sample_id)
            if date_str is None:
                for fn in frames[: min(10, len(frames))]:
                    date_str = _extract_date_yyyymmdd(fn)
                    if date_str is not None:
                        break
            if date_str is None:
                date_str = current_date
            video_name = f"{date_str}_{chemical_type}_{concentration}_{sample_id}"
            series_key = derive_series_key(frames, fallback=sample_id)

            # Resolve acquisition scale before heart detection/cropping. The
            # resulting 300x300 frames have an effective scale determined by
            # the physical field of view retained in the scaled crop.
            relative_times = None
            unit_str = "unknown"
            if meta_manual:
                meta_file = None
                original_um_per_px = _valid_um_per_px(resize_scale) or BASE_UM_PER_PX
            else:
                meta_file = _find_metadata_xml(
                    sample_path=sample_path,
                    sample_dir=sample_dir,
                    sample_id=sample_id,
                    chem_conc_path=chem_conc_path,
                    metadata_file=metadata_file,
                )
                if meta_file:
                    (
                        relative_times,
                        parsed_um_per_px,
                        _parsed_area_per_px2,
                        unit_str,
                    ) = parse_xml_times_and_scale(meta_file, num_images=len(frames))
                    original_um_per_px = (
                        _valid_um_per_px(parsed_um_per_px) or BASE_UM_PER_PX
                    )
                else:
                    original_um_per_px = BASE_UM_PER_PX

            crop_px, detect_px = _scaled_crop_sizes(original_um_per_px)
            effective_um_per_px = original_um_per_px * crop_px / BASE_CROP_PX
            analysis_meta_info = {
                "len_per_px": effective_um_per_px,
                "area_per_px2": effective_um_per_px**2,
                "unit_str": unit_str,
                "frame_interval": frame_interval,
                "relative_times": relative_times,
            }
            
            sample_t0 = perf_counter()
            stage_times = {}
            emit_log(
                f"[{sample_index}/{total_samples}] {chem_conc_dir}/{sample_dir}: "
                f"starting sample with {len(frames)} frame(s) | "
                f"source scale={original_um_per_px:.6g} um/px | "
                f"crop={crop_px}px | detect={detect_px}px | "
                f"normalized scale={effective_um_per_px:.6g} um/px"
            )

            # Stage progress: keeps progress visible during expensive post-segmentation analysis.
            stage_bar = None
            if progress_callback is None:
                stage_bar = tqdm(total=7, desc=f"  {sample_dir}", leave=False, unit="stage")

            # 1) Load + center-crop frames
            t0 = perf_counter()
            imgs = []
            emit_log(f"[{sample_index}/{total_samples}] {chem_conc_dir}/{sample_dir}: loading and center-cropping frames")
            if progress_callback is None:
                frame_bar = tqdm(frames, desc=f"    {sample_dir} load/crop", leave=False, unit="frame")
                for fn in frame_bar:
                    imgs.append(_read_frame_fast(os.path.join(sample_path, fn), as_gray=grayscale))
            else:
                imgs = [_read_frame_fast(os.path.join(sample_path, fn), as_gray=grayscale) for fn in frames]

            if imgs:
                est_center = _estimate_center_2stage(
                    imgs[0],
                    crop_size=detect_px,
                    coarse_thresh=0.10,
                    fine_thresh=0.10,
                )
                imgs = [
                    _crop_resize_center(
                        image,
                        est_center,
                        out_size=BASE_CROP_PX,
                        crop_size=crop_px,
                    )
                    for image in imgs
                ]
            stack = np.stack(imgs, axis=0)
            stack = dask.array.from_array(stack)
            stage_times["load_crop_s"] = perf_counter() - t0
            emit_log(
                f"[{sample_index}/{total_samples}] {chem_conc_dir}/{sample_dir}: "
                f"load/crop complete in {stage_times['load_crop_s']:.1f}s"
            )
            if stage_bar is not None:
                stage_bar.update(1)
                stage_bar.set_postfix(stage="load/crop", sec=f"{stage_times['load_crop_s']:.1f}")

            # 2) Segmentation
            t0 = perf_counter()
            emit_log(
                f"[{sample_index}/{total_samples}] {chem_conc_dir}/{sample_dir}: "
                f"starting segmentation on {int(stack.shape[0])} frame(s)"
            )
            masks, atrium_probabilities = infer.infer(
                stack,
                channel,
                return_atrium_probabilities=True,
            )
            stage_times["segment_s"] = perf_counter() - t0
            emit_log(
                f"[{sample_index}/{total_samples}] {chem_conc_dir}/{sample_dir}: "
                f"segmentation complete in {stage_times['segment_s']:.1f}s"
            )
            if stage_bar is not None:
                stage_bar.update(1)
                stage_bar.set_postfix(stage="segment", sec=f"{stage_times['segment_s']:.1f}")

            # 3) Save raw segmentation tiff
            t0 = perf_counter()
            emit_log(f"[{sample_index}/{total_samples}] {chem_conc_dir}/{sample_dir}: saving raw segmentation mask")
            mask_path = os.path.join(sample_out, f"{video_name}_segmentation.tif")
            tifffile.imwrite(mask_path, masks.astype(np.uint8))
            stage_times["save_mask_s"] = perf_counter() - t0
            emit_log(
                f"[{sample_index}/{total_samples}] {chem_conc_dir}/{sample_dir}: "
                f"raw mask saved in {stage_times['save_mask_s']:.1f}s"
            )
            if stage_bar is not None:
                stage_bar.update(1)
                stage_bar.set_postfix(stage="save-mask", sec=f"{stage_times['save_mask_s']:.1f}")

            _save_input_visualizations(
                sample_out,
                frames,
                imgs,
                masks,
                visualization_format,
            )
            gif_relative_times = _relative_times_for_frames(
                relative_times,
                len(frames),
                frame_interval,
            )
            _save_masked_gif(
                sample_out,
                video_name,
                imgs,
                masks,
                gif_relative_times,
                effective_um_per_px,
            )

            # 4) Segmentation statistics
            t0 = perf_counter()
            emit_log(f"[{sample_index}/{total_samples}] {chem_conc_dir}/{sample_dir}: computing segmentation statistics")
            seg_df, fs_overlay, cleaned_masks = compute_segmentation_statistics(
                stack,
                masks,
                video_name,
                sample_out,
                None,
                meta_info=analysis_meta_info,
                frame_filenames=frames,
                show_progress=(progress_callback is None),
                progress_desc=f"    {sample_dir} seg-stats",
                return_cleaned_masks=True,
                series_key=series_key,
                atrium_probabilities=atrium_probabilities,
                visualization_format=visualization_format,
            )
            stage_times["seg_stats_s"] = perf_counter() - t0
            emit_log(
                f"[{sample_index}/{total_samples}] {chem_conc_dir}/{sample_dir}: "
                f"segmentation statistics complete in {stage_times['seg_stats_s']:.1f}s"
            )
            if stage_bar is not None:
                stage_bar.update(1)
                stage_bar.set_postfix(stage="seg-stats", sec=f"{stage_times['seg_stats_s']:.1f}")

            # 5) Save cleaned segmentation tiff used for feature extraction
            t0 = perf_counter()
            emit_log(f"[{sample_index}/{total_samples}] {chem_conc_dir}/{sample_dir}: saving cleaned segmentation mask")
            cleaned_mask_path = os.path.join(sample_out, f"{video_name}_segmentation_cleaned.tif")
            tifffile.imwrite(cleaned_mask_path, cleaned_masks.astype(np.uint8, copy=False))
            stage_times["save_cleaned_mask_s"] = perf_counter() - t0
            emit_log(
                f"[{sample_index}/{total_samples}] {chem_conc_dir}/{sample_dir}: "
                f"cleaned mask saved in {stage_times['save_cleaned_mask_s']:.1f}s"
            )
            if stage_bar is not None:
                stage_bar.update(1)
                stage_bar.set_postfix(stage="save-cleaned", sec=f"{stage_times['save_cleaned_mask_s']:.1f}")

            # 6) Functional + synchrony analysis
            t0 = perf_counter()
            emit_log(f"[{sample_index}/{total_samples}] {chem_conc_dir}/{sample_dir}: computing functional and synchrony analysis")
            v_df, vFS_df, a_df, fig_v, fig_vfs, fig_a, fig_va = compute_functional_statistics(
                seg_df, video_name, sample_out, series_key=series_key
            )
            sync_df, fig_cav, fig_cc = compute_synchronize_analysis(
                seg_df, video_name, sample_out, series_key=series_key
            )
            stage_times["func_sync_s"] = perf_counter() - t0
            emit_log(
                f"[{sample_index}/{total_samples}] {chem_conc_dir}/{sample_dir}: "
                f"functional/synchrony analysis complete in {stage_times['func_sync_s']:.1f}s"
            )
            if stage_bar is not None:
                stage_bar.update(1)
                stage_bar.set_postfix(stage="func+sync", sec=f"{stage_times['func_sync_s']:.1f}")

            # 7) Combine + finalize
            t0 = perf_counter()
            emit_log(f"[{sample_index}/{total_samples}] {chem_conc_dir}/{sample_dir}: combining per-sample outputs")
            combined_df = combine_results(
                video_name, seg_df, v_df, vFS_df, a_df, sync_df, sample_out, series_key=series_key
            )
            batch_combined_list.append(combined_df)
            condition_combined_list.append(combined_df)

            if save_analysis_vis:
                analysis_vis_overlay = fs_overlay
            else:
                analysis_vis_overlay = None

            if load_to_viewer:
                if save_analysis_vis:
                    img_list.append((video_name, stack, masks, analysis_vis_overlay))
                else:
                    img_list.append((video_name, stack, masks, None))
            stage_times["finalize_s"] = perf_counter() - t0
            total_s = perf_counter() - sample_t0
            emit_log(
                f"[{sample_index}/{total_samples}] {chem_conc_dir}/{sample_dir}: "
                f"sample complete | load/crop={stage_times['load_crop_s']:.1f}s | "
                f"segment={stage_times['segment_s']:.1f}s | segStats={stage_times['seg_stats_s']:.1f}s | "
                f"func+sync={stage_times['func_sync_s']:.1f}s | total={total_s:.1f}s"
            )
            if stage_bar is not None:
                stage_bar.update(1)
                stage_bar.set_postfix(stage="finalize", sec=f"{stage_times['finalize_s']:.1f}")
                stage_bar.close()

                tqdm.write(
                    f"[timing] {chem_conc_dir}/{sample_dir} "
                    f"load/crop={stage_times['load_crop_s']:.1f}s "
                    f"segment={stage_times['segment_s']:.1f}s "
                    f"save={stage_times['save_mask_s']:.1f}s "
                    f"segStats={stage_times['seg_stats_s']:.1f}s "
                    f"func+sync={stage_times['func_sync_s']:.1f}s "
                    f"finalize={stage_times['finalize_s']:.1f}s "
                    f"total={total_s:.1f}s"
                )
            
            if progress_callback:
                progress_callback(len(batch_combined_list))
            else:
                pbar_samples.update(1)

        if condition_combined_list:
            cond_results_dir = os.path.join(out_dir, chem_conc_dir, "results")
            os.makedirs(cond_results_dir, exist_ok=True)
            cond_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            cond_df = pd.concat(condition_combined_list, ignore_index=True)
            export_perfish_df = perfish_export_dataframe(cond_df)
            emit_log(
                f"{chem_conc_dir}: writing condition workbook with {len(cond_df)} row(s) to results directory"
            )
            export_perfish_df.to_excel(
                os.path.join(cond_results_dir, f"PerFishMetrics_{chem_conc_dir}_{cond_timestamp}.xlsx"),
                index=False,
            )
    
    # Close main progress bar
    if pbar_samples is not None:
        pbar_samples.close()

    # Create one combined DataFrame for the entire batch
    if batch_combined_list:
        batch_combined_df = pd.concat(batch_combined_list, ignore_index=True)
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        # Save batch combined CSV
        batch_combined_path = os.path.join(out_dir, f"batch_combined_{timestamp}.csv")
        batch_combined_df.to_csv(batch_combined_path, index=False)
    else:
        batch_combined_df = pd.DataFrame()

    if load_to_viewer:
        return (batch_combined_df, img_list, None)
    else:
        return (batch_combined_df, None, None)
