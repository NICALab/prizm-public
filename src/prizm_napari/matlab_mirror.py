#!/usr/bin/env python3
"""
Utilities to rebuild PRIZM batch output into a MATLAB-like directory layout.
"""

from __future__ import annotations

import os
import re
import shutil
from datetime import datetime
from pathlib import Path
from time import perf_counter

import imageio.v2 as imageio
import numpy as np
import pandas as pd
import skimage.io as skio
import tifffile
from PIL import Image
from tqdm import tqdm

from prizm_napari.analysis import _matlab_preprocess_pdouble
from prizm_napari.batch_segmentation_core import _crop_resize_center, _estimate_center_2stage

VALID_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def extract_t_index(name: str) -> int | None:
    m = re.search(r"_t(\d+)", str(name))
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def sorted_frame_names(series_dir: Path) -> list[str]:
    files = [p.name for p in series_dir.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTS]

    def key_fn(n: str):
        t = extract_t_index(n)
        if t is None:
            return (1, n.lower())
        return (0, int(t), n.lower())

    return sorted(files, key=key_fn)


def ensure_condition_dirs(cond_out: Path) -> None:
    for d in ["masked", "cropped", "preprocessing", "FS", "results", "synchronize", "merged", "segmentation_masks"]:
        (cond_out / d).mkdir(parents=True, exist_ok=True)


def make_overlay(pre_u8: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if pre_u8.ndim == 2:
        rgb = np.stack([pre_u8, pre_u8, pre_u8], axis=-1).astype(np.uint8)
    else:
        rgb = pre_u8.astype(np.uint8).copy()
    if mask.ndim != 2:
        return rgb

    vent = mask == 1
    atr = mask == 2
    alpha = 0.70
    vent_col = np.array([0, 255, 255], dtype=np.float32)
    atr_col = np.array([255, 0, 255], dtype=np.float32)

    rgb_f = rgb.astype(np.float32)
    rgb_f[vent] = (1.0 - alpha) * rgb_f[vent] + alpha * vent_col
    rgb_f[atr] = (1.0 - alpha) * rgb_f[atr] + alpha * atr_col
    return np.clip(rgb_f, 0, 255).astype(np.uint8)


def save_strip_and_gif(paths: list[Path], out_jpg: Path, out_gif: Path) -> None:
    imgs = []
    for p in paths:
        try:
            imgs.append(Image.open(p).convert("RGB"))
        except Exception:
            continue
    if not imgs:
        return

    widths = [im.width for im in imgs]
    heights = [im.height for im in imgs]
    canvas = Image.new("RGB", (int(sum(widths)), int(max(heights))), color=(255, 255, 255))

    x = 0
    for im in imgs:
        canvas.paste(im, (x, 0))
        x += im.width

    out_jpg.parent.mkdir(parents=True, exist_ok=True)
    out_gif.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_jpg, quality=90)
    imgs[0].save(out_gif, save_all=True, append_images=imgs[1:], duration=62, loop=0)


def video_key_from_frame_name(frame_name: str) -> str:
    return re.sub(r"_t\d+.*$", "", frame_name)


def mirror_batch_output_to_matlab_layout(
    data_root: Path | str,
    output_root: Path | str,
    raw_batch_out: Path | str,
    make_merged_artifacts: bool = True,
) -> Path:
    """
    Convert core batch output into MATLAB-like condition-level folder layout.

    Returns the status CSV path.
    """
    data_root = Path(data_root)
    output_root = Path(output_root)
    raw_batch_out = Path(raw_batch_out)

    if not data_root.is_dir():
        raise FileNotFoundError(f"Data root not found: {data_root}")
    if not raw_batch_out.is_dir():
        raise FileNotFoundError(f"Raw batch output not found: {raw_batch_out}")

    output_root.mkdir(parents=True, exist_ok=True)

    batch_csvs = sorted(raw_batch_out.glob("batch_combined_*.csv"))
    latest_batch_csv = batch_csvs[-1] if batch_csvs else None

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_rows: list[dict] = []

    cond_dirs = sorted(
        p
        for p in raw_batch_out.iterdir()
        if p.is_dir() and not p.name.startswith(".") and p.name != "runlogs"
    )

    cond_pbar = tqdm(cond_dirs, desc="Mirror conditions", unit="condition")
    for cond_dir in cond_pbar:
        cond = cond_dir.name
        cond_out = output_root / cond
        ensure_condition_dirs(cond_out)
        perfish_rows = []

        sample_dirs = sorted([p for p in cond_dir.iterdir() if p.is_dir()])
        sample_pbar = tqdm(sample_dirs, desc=f"{cond}", leave=False, unit="series")
        for sample_dir in sample_pbar:
            t0 = perf_counter()
            series = sample_dir.name
            data_series_dir = data_root / cond / series
            if not data_series_dir.exists():
                run_rows.append(
                    {
                        "condition": cond,
                        "series": series,
                        "status": "missing_raw_series_dir",
                        "message": str(data_series_dir),
                    }
                )
                sample_pbar.set_postfix({"series": series, "sec": f"{perf_counter() - t0:.1f}"})
                continue

            seg_tif_candidates = sorted(sample_dir.glob("*_segmentation.tif"))
            if not seg_tif_candidates:
                run_rows.append(
                    {
                        "condition": cond,
                        "series": series,
                        "status": "missing_segmentation_tif",
                        "message": str(sample_dir),
                    }
                )
                sample_pbar.set_postfix({"series": series, "sec": f"{perf_counter() - t0:.1f}"})
                continue

            seg_tif = seg_tif_candidates[0]
            video_name = seg_tif.name.replace("_segmentation.tif", "")

            xlsx_path = sample_dir / f"{video_name}.xlsx"
            if xlsx_path.exists():
                shutil.copy2(xlsx_path, cond_out / xlsx_path.name)

            res_src = sample_dir / "results"
            if res_src.is_dir():
                for f in res_src.iterdir():
                    if f.is_file():
                        shutil.copy2(f, cond_out / "results" / f.name)

            sync_src = sample_dir / "synchronize"
            if sync_src.is_dir():
                for f in sync_src.iterdir():
                    if not f.is_file():
                        continue
                    if f.name == "SynchronizationResults.xlsx":
                        dst_name = f"{video_name}_SynchronyMetrics.xlsx"
                    else:
                        dst_name = f.name
                    shutil.copy2(f, cond_out / "synchronize" / dst_name)

            sample_combined_csv = res_src / f"{video_name}.csv"
            if sample_combined_csv.exists():
                try:
                    row_df = pd.read_csv(sample_combined_csv)
                    if len(row_df) > 0:
                        row = row_df.iloc[0].to_dict()
                        file_name = str(row.get("File Name", f"{video_name}.csv"))
                        row["FileKey"] = Path(file_name).stem
                        perfish_rows.append(row)
                except Exception:
                    pass

            masks = tifffile.imread(seg_tif)
            if masks.ndim == 2:
                masks = masks[np.newaxis, ...]

            frame_names = sorted_frame_names(data_series_dir)
            if not frame_names:
                run_rows.append(
                    {"condition": cond, "series": series, "status": "no_frames", "message": str(data_series_dir)}
                )
                sample_pbar.set_postfix({"series": series, "sec": f"{perf_counter() - t0:.1f}"})
                continue

            first = skio.imread(str(data_series_dir / frame_names[0]), as_gray=True)
            center_xy = _estimate_center_2stage(first, crop_size=400, coarse_thresh=0.10, fine_thresh=0.10)

            fs_src_dir = sample_dir / f"{video_name}_FS"
            fs_src_pngs = sorted(fs_src_dir.glob("*.png")) if fs_src_dir.is_dir() else []

            n = min(len(frame_names), masks.shape[0])
            seg_series_dir = cond_out / "segmentation_masks" / series
            seg_series_dir.mkdir(parents=True, exist_ok=True)

            for i in tqdm(range(n), desc=f"{series} rebuild", leave=False, unit="frame"):
                fn = frame_names[i]
                base = Path(fn).stem

                raw_img = skio.imread(str(data_series_dir / fn), as_gray=True)
                crop = _crop_resize_center(raw_img, center_xy, out_size=300)
                pre = _matlab_preprocess_pdouble(crop)
                pre_u8 = np.clip(pre * 255.0, 0, 255).astype(np.uint8)
                mask = masks[i].astype(np.uint8)

                imageio.imwrite(cond_out / "cropped" / f"cropped_{fn}", np.asarray(crop))
                imageio.imwrite(cond_out / "preprocessing" / f"preprocessing_{fn}", pre_u8)
                imageio.imwrite(cond_out / "masked" / f"labeled_{fn}", make_overlay(pre_u8, mask))

                vmask = ((mask == 1).astype(np.uint8) * 255)
                amask = ((mask == 2).astype(np.uint8) * 255)
                simple = (2 * (mask == 1).astype(np.uint8) + 4 * (mask == 2).astype(np.uint8))

                imageio.imwrite(seg_series_dir / f"{base}_VentricleMask.png", vmask)
                imageio.imwrite(seg_series_dir / f"{base}_AtriumMask.png", amask)
                imageio.imwrite(seg_series_dir / f"{base}_Simple Segmentation.png", simple)

                if i < len(fs_src_pngs):
                    try:
                        fs_img = imageio.imread(fs_src_pngs[i])
                        imageio.imwrite(cond_out / "FS" / f"FS_{fn}", fs_img)
                    except Exception:
                        pass

            run_rows.append(
                {
                    "condition": cond,
                    "series": series,
                    "status": "ok",
                    "message": video_name,
                    "n_frames": n,
                    "elapsed_sec": float(perf_counter() - t0),
                }
            )
            sample_pbar.set_postfix({"series": series, "sec": f"{perf_counter() - t0:.1f}"})

        if perfish_rows:
            per_df = pd.DataFrame(perfish_rows)
            cols = per_df.columns.tolist()
            if "FileKey" in cols:
                cols = ["FileKey"] + [c for c in cols if c != "FileKey"]
                per_df = per_df[cols]
            (cond_out / "results").mkdir(parents=True, exist_ok=True)
            per_df.to_excel(cond_out / "results" / f"PerFishMetrics_{cond}_{timestamp}.xlsx", index=False)

        if make_merged_artifacts:
            pre_dir = cond_out / "preprocessing"
            if pre_dir.is_dir():
                pre_files = sorted(
                    [p.name for p in pre_dir.iterdir() if p.is_file() and p.name.startswith("preprocessing_")]
                )
                groups: dict[str, list[str]] = {}
                for nm in pre_files:
                    orig = nm[len("preprocessing_") :]
                    vk = video_key_from_frame_name(orig)
                    groups.setdefault(vk, []).append(orig)

                for vk, frame_list in groups.items():
                    frame_list = sorted(
                        frame_list,
                        key=lambda n: (extract_t_index(n) is None, extract_t_index(n) or 0, n.lower()),
                    )
                    pre_paths = [
                        cond_out / "preprocessing" / f"preprocessing_{fn}"
                        for fn in frame_list
                        if (cond_out / "preprocessing" / f"preprocessing_{fn}").exists()
                    ]
                    msk_paths = [
                        cond_out / "masked" / f"labeled_{fn}"
                        for fn in frame_list
                        if (cond_out / "masked" / f"labeled_{fn}").exists()
                    ]
                    fs_paths = [
                        cond_out / "FS" / f"FS_{fn}"
                        for fn in frame_list
                        if (cond_out / "FS" / f"FS_{fn}").exists()
                    ]

                    if pre_paths:
                        save_strip_and_gif(
                            pre_paths,
                            cond_out / "merged" / f"{vk}_preprocessing_merged.jpg",
                            cond_out / "merged" / f"{vk}_preprocessing_merged.gif",
                        )
                    if msk_paths:
                        save_strip_and_gif(
                            msk_paths,
                            cond_out / "merged" / f"{vk}_masked_merged.jpg",
                            cond_out / "merged" / f"{vk}_masked_merged.gif",
                        )
                    if fs_paths:
                        save_strip_and_gif(
                            fs_paths,
                            cond_out / "merged" / f"{vk}_FS_merged.jpg",
                            cond_out / "merged" / f"{vk}_FS_merged.gif",
                        )

    if latest_batch_csv and latest_batch_csv.exists():
        shutil.copy2(latest_batch_csv, output_root / latest_batch_csv.name)

    status_csv = output_root / "python_matlab_mirror_run_status.csv"
    pd.DataFrame(run_rows).to_csv(status_csv, index=False)
    return status_csv


def remove_tree_if_exists(path: Path | str) -> None:
    p = Path(path)
    if p.exists():
        shutil.rmtree(p)
