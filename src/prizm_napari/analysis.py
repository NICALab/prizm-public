import os
from typing import Tuple, Optional, Sequence
import numpy as np
import pandas as pd
from skimage.measure import label, regionprops, regionprops_table
import re
import cv2
import torch
import xml.etree.ElementTree as ET
import math
from scipy.interpolate import CubicSpline, PchipInterpolator
from scipy import ndimage as ndi
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from scipy.signal import find_peaks, hilbert, peak_widths, correlate, correlation_lags, peak_prominences
from scipy.stats import pearsonr
from datetime import datetime
from PIL import Image
import imageio
from tqdm import tqdm
from pathlib import Path
from prizm_napari.utils import *

# ----------------------------
# Global visual/style settings
# ----------------------------
FIGSIZE = (12, 3.6)       # ~1200x360 px at dpi=100
DISPLAY_DPI = 100
LW_SIGNAL = 1.0           # signal lines
LW_ANNOT  = 1.0           # annotation (height/width/border) lines
GRID_LW   = 0.5
GRID_ALPHA= 0.5
PEAK_MS   = 5             # peak marker size
PEAK_MARK = 'v'           # filled downward triangle (to match MATLAB vibe)

# Colors: keep consistent across all figures
COLOR_V_SIGNAL = 'b'          # ventricle signal/peaks
COLOR_A_SIGNAL = 'r'          # atrium  signal/peaks
COLOR_HEIGHT   = 'tab:orange'
COLOR_WIDTH    = 'goldenrod'
COLOR_BORDER   = 'purple'
COLOR_OVERLAY_V= 'tab:blue'
COLOR_OVERLAY_A= 'tab:orange'
COLOR_CC       = 'k'
COLOR_CC_LAG   = 'r'

# fixed margins for a single-axes layout
_AX_RECT = dict(left=0.07, right=0.995, top=0.90, bottom=0.22)

MATLAB_FRAME_EXPORT_COLUMNS = [
    "FileName",
    "RelativeTime",
    "VArea_px",
    "VArea_real",
    "AArea_px",
    "AArea_real",
    "VCentroid_1",
    "VCentroid_2",
    "ACentroid_1",
    "ACentroid_2",
    "VCentroid_real_1",
    "VCentroid_real_2",
    "ACentroid_real_1",
    "ACentroid_real_2",
    "SVBA_Distance",
    "SVBA_Distance_Y",
    "MajorAxisLength",
    "MinorAxis_center",
    "MinorAxis_upper",
    "MinorAxis_lower",
    "VA_Distance_Center",
    "VA_Angle_Center_raw",
    "VA_Angle_Center",
    "VA_Distance_Bottom",
    "VA_Angle_Bottom_raw",
    "VA_Angle_Bottom",
    "VA_Distance_Top",
    "VA_Angle_Top_raw",
    "VA_Angle_Top",
    "LenPerPx",
    "AreaPerPx2",
    "UnitStr",
]

MATLAB_PERFISH_EXPORT_COLUMNS = [
    "FileKey",
    "V_HR_bpm",
    "A_HR_bpm",
    "Interval_SD_s",
    "Interval_CV",
    "SystolicDuration_mean",
    "DiastolicDuration_mean",
    "SystolicFraction",
    "EF_mean",
    "EF_SD",
    "FS_mean",
    "FS_SD",
    "V_ED_mean",
    "V_ES_mean",
    "V_ED_SD",
    "V_ES_SD",
    "SV_index_mean",
    "CO_index_mean",
    "Diastolic_AtoV_ratio",
    "MajorMinor_ratio_ED",
    "ContractilitySpeed",
    "RelaxationSpeed",
    "SVBA_Distance_mean",
    "SVBA_Distance_Y_mean",
    "A_ED_mean",
    "A_ES_mean",
    "A_ED_SD",
    "A_ES_SD",
    "A_ED_ES_Diff_mean",
    "VA_Dist_Center_mean",
    "VA_Dist_Bottom_mean",
    "VA_Dist_Top_mean",
    "VA_Ang_Center_raw_mean",
    "VA_Ang_Center_major_mean",
    "VA_Ang_Bottom_raw_mean",
    "VA_Ang_Bottom_major_mean",
    "VA_Ang_Top_raw_mean",
    "VA_Ang_Top_major_mean",
    "MaxCorrLag_s",
    "CrossCorrCoeff",
    "PhaseSynchronyIndex",
    "PLV",
    "PearsonCorr",
    "PeakTimeDiff_SD",
    "AV_Delay_Mean",
    "AV_Delay_SD",
]

def _save_svg(fig, path, dpi=300):
    # fix canvas size and margins for consistency
    fig.set_size_inches(FIGSIZE[0], FIGSIZE[1], forward=True)
    fig.subplots_adjust(**_AX_RECT)   # no tight_layout()
    fig.savefig(path, format='svg', dpi=dpi)   # fixed canvas every time


def derive_matlab_series_key(frame_filenames: Optional[Sequence[str]], fallback: Optional[str] = None) -> str:
    if frame_filenames:
        first_name = str(frame_filenames[0])
        base = os.path.splitext(os.path.basename(first_name))[0]
        base = re.sub(r"_t\d+.*$", "", base)
        if base:
            return base
    return str(fallback or "")


def matlab_style_segmentation_dataframe(seg_stats_df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=seg_stats_df.index)
    out["FileName"] = seg_stats_df.get("FileName", pd.Series(index=seg_stats_df.index, dtype=object))
    out["RelativeTime"] = seg_stats_df.get("RelativeTime", np.nan)
    out["VArea_px"] = seg_stats_df.get("VArea_px", seg_stats_df.get("VentricularCavitySize", np.nan))
    out["VArea_real"] = seg_stats_df.get("VArea_real", seg_stats_df.get("RealVentricularCavitySize", np.nan))
    out["AArea_px"] = seg_stats_df.get("AArea_px", seg_stats_df.get("AtriumCavitySize", np.nan))
    out["AArea_real"] = seg_stats_df.get("AArea_real", seg_stats_df.get("RealAtriumCavitySize", np.nan))
    out["VCentroid_1"] = seg_stats_df.get("VCentroid_X", np.nan)
    out["VCentroid_2"] = seg_stats_df.get("VCentroid_Y", np.nan)
    out["ACentroid_1"] = seg_stats_df.get("ACentroid_X", np.nan)
    out["ACentroid_2"] = seg_stats_df.get("ACentroid_Y", np.nan)
    out["VCentroid_real_1"] = seg_stats_df.get("VCentroid_real_X", np.nan)
    out["VCentroid_real_2"] = seg_stats_df.get("VCentroid_real_Y", np.nan)
    out["ACentroid_real_1"] = seg_stats_df.get("ACentroid_real_X", np.nan)
    out["ACentroid_real_2"] = seg_stats_df.get("ACentroid_real_Y", np.nan)
    out["SVBA_Distance"] = seg_stats_df.get("SVBA_Distance", seg_stats_df.get("VentricleAtriumDistance", np.nan))
    out["SVBA_Distance_Y"] = seg_stats_df.get("SVBA_Distance_Y", seg_stats_df.get("VentricleAtriumYDistance", np.nan))
    out["MajorAxisLength"] = seg_stats_df.get("MajorAxisLength", seg_stats_df.get("majorAxisLength", np.nan))
    out["MinorAxis_center"] = seg_stats_df.get("MinorAxis_center", seg_stats_df.get("minorAxis_center", np.nan))
    out["MinorAxis_upper"] = seg_stats_df.get("MinorAxis_upper", seg_stats_df.get("minorAxis_upper", np.nan))
    out["MinorAxis_lower"] = seg_stats_df.get("MinorAxis_lower", seg_stats_df.get("minorAxis_lower", np.nan))
    out["VA_Distance_Center"] = seg_stats_df.get("VA_Distance_Center", np.nan)
    out["VA_Angle_Center_raw"] = seg_stats_df.get("VA_Angle_Center_raw", seg_stats_df.get("Angle_Center", np.nan))
    out["VA_Angle_Center"] = seg_stats_df.get("VA_Angle_Center", seg_stats_df.get("Angle_Center", np.nan))
    out["VA_Distance_Bottom"] = seg_stats_df.get("VA_Distance_Bottom", np.nan)
    out["VA_Angle_Bottom_raw"] = seg_stats_df.get("VA_Angle_Bottom_raw", seg_stats_df.get("Angle_Bottom", np.nan))
    out["VA_Angle_Bottom"] = seg_stats_df.get("VA_Angle_Bottom", seg_stats_df.get("Angle_Bottom", np.nan))
    out["VA_Distance_Top"] = seg_stats_df.get("VA_Distance_Top", np.nan)
    out["VA_Angle_Top_raw"] = seg_stats_df.get("VA_Angle_Top_raw", seg_stats_df.get("Angle_Top", np.nan))
    out["VA_Angle_Top"] = seg_stats_df.get("VA_Angle_Top", seg_stats_df.get("Angle_Top", np.nan))
    out["LenPerPx"] = seg_stats_df.get("LenPerPx", np.nan)
    out["AreaPerPx2"] = seg_stats_df.get("AreaPerPx2", np.nan)
    out["UnitStr"] = seg_stats_df.get("UnitStr", "")
    return out.loc[:, MATLAB_FRAME_EXPORT_COLUMNS]


def matlab_style_perfish_dataframe(perfish_df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=perfish_df.index)
    if "FileKey" in perfish_df.columns:
        out["FileKey"] = perfish_df["FileKey"]
    elif "File Name" in perfish_df.columns:
        out["FileKey"] = perfish_df["File Name"].map(lambda x: os.path.splitext(os.path.basename(str(x)))[0])
    else:
        out["FileKey"] = ""
    for col in MATLAB_PERFISH_EXPORT_COLUMNS[1:]:
        out[col] = perfish_df[col] if col in perfish_df.columns else np.nan
    return out.loc[:, MATLAB_PERFISH_EXPORT_COLUMNS]

# Analysis pipeline
# stats_df, fig_v_axis, fig_a_axis, viz_data = compute_segmentation_statistics(
#     masks, f"{video_name}_ch{channel}", video_out, meta_file, meta_info=None
# )
# v_df, vFS_df, a_df, fig_v, fig_vfs, fig_a, fig_va, viz_data = compute_functional_statistics(
#     stats_df, f"{video_name}_ch{channel}", video_out, viz_data=viz_data
# )
# sync_df, fig_cav, fig_cc, viz_data = compute_synchronize_analysis(
#     stats_df, f"{video_name}_ch{channel}", video_out, viz_data=viz_data
# )
# combined_df, viz_data = combine_results(
#     f"{video_name}_ch{channel}", stats_df, v_df, vFS_df, a_df, sync_df, video_out, viz_data=viz_data
# )

def find_ventricle_major_axis_endpoints(mask):
    """Find the two endpoints of the ventricle's major axis using MATLAB approach"""
    m = (mask > 0).astype(np.uint8) * 255
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, None
    
    # Get the largest contour
    c = max(cnts, key=cv2.contourArea)
    if len(c) < 70:  # MATLAB uses 70 as minimum
        return None, None
    
    # Get centroid
    M = cv2.moments(c)
    if M["m00"] == 0:
        return None, None
    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]
    centroid = np.array([cx, cy])
    
    # Convert contour to boundary points (MATLAB bwboundaries equivalent)
    boundary_points = c.reshape(-1, 2)
    
    # Test different angles (MATLAB: 65:115)
    angle_range = range(65, 116)
    max_distance = -np.inf
    best_angle = np.nan
    best_pos_major = centroid.copy()
    best_neg_major = centroid.copy()
    
    for angle in angle_range:
        # Convert angle to radians and get direction vector
        angle_rad = np.radians(angle)
        direction = np.array([np.cos(angle_rad), np.sin(angle_rad)])
        
        # Start from centroid and walk in both directions
        pos_pt = centroid.copy()
        neg_pt = centroid.copy()
        
        # Walk in positive direction until we hit boundary
        while cv2.pointPolygonTest(c, (int(pos_pt[0]), int(pos_pt[1])), False) >= 0:
            pos_pt = pos_pt + direction
        
        # Walk in negative direction until we hit boundary
        while cv2.pointPolygonTest(c, (int(neg_pt[0]), int(neg_pt[1])), False) >= 0:
            neg_pt = neg_pt - direction
        
        # Calculate distance
        distance = np.linalg.norm(pos_pt - neg_pt)
        
        if distance > max_distance:
            max_distance = distance
            best_angle = angle
            best_pos_major = pos_pt
            best_neg_major = neg_pt
    
    if np.isnan(best_angle) or max_distance <= 0:
        return None, None
    
    return best_pos_major, best_neg_major

def find_boundary_points(mask):
    """Find top and bottom points of a mask (for atrium)"""
    m = (mask > 0).astype(np.uint8) * 255
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, None
    
    # Get the largest contour
    c = max(cnts, key=cv2.contourArea)
    if len(c) < 3:
        return None, None
    
    # Convert to numpy array and find top/bottom points
    contour_points = c.reshape(-1, 2)
    y_coords = contour_points[:, 1]
    
    # Find top (minimum y) and bottom (maximum y) points
    top_idx = np.argmin(y_coords)
    bottom_idx = np.argmax(y_coords)
    
    top_point = contour_points[top_idx]
    bottom_point = contour_points[bottom_idx]
    
    return top_point, bottom_point

def _extract_t_index(fname: Optional[str]) -> Optional[int]:
    """Extract frame index from a filename token like ..._t12...."""
    if not fname:
        return None
    m = re.search(r"_t(\d+)", str(fname))
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def parse_xml_times_and_scale(xml_file_path: str, num_images: int = 0):
    """
    MATLAB-parity parser for TimeStamp + length/area scale.

    Returns
    -------
    relative_times : np.ndarray
    len_per_px : float
    area_per_px2 : float
    unit_str : str
    """
    # Fallback defaults (kept consistent with MATLAB script)
    relative_times = np.array([0.062 * i for i in range(max(0, int(num_images)))], dtype=float)
    len_per_px = 0.9210
    unit_str = "unknown"
    area_per_px2 = float(len_per_px**2)

    if not xml_file_path or (not os.path.exists(xml_file_path)):
        return relative_times, float(len_per_px), float(area_per_px2), unit_str

    try:
        tree = ET.parse(xml_file_path)
        root = tree.getroot()

        # ---- TimeStamp ----
        times = []
        for timestamp in root.findall(".//TimeStamp"):
            relative_time_str = timestamp.get("RelativeTime")
            if not relative_time_str:
                continue
            m = re.search(r"(\d+\.?\d*)", str(relative_time_str))
            if m:
                try:
                    times.append(float(m.group(1)))
                except Exception:
                    pass
        if times:
            relative_times = np.asarray(times, dtype=float)

        # ---- DimensionDescription (Voxel + Unit) ----
        scale_x = np.nan
        scale_y = np.nan
        parsed_unit = ""
        for dimension in root.findall(".//DimensionDescription"):
            dim_id = str(dimension.get("DimID", ""))
            voxel_s = dimension.get("Voxel")
            try:
                voxel = float(voxel_s) if voxel_s is not None else np.nan
            except Exception:
                voxel = np.nan

            if not parsed_unit:
                u = dimension.get("Unit")
                if u:
                    parsed_unit = str(u)

            if dim_id == "X":
                scale_x = voxel
            elif dim_id == "Y":
                scale_y = voxel

        if np.isfinite(scale_x) and np.isfinite(scale_y) and scale_x > 0 and scale_y > 0:
            len_per_px = float(np.mean([scale_x, scale_y]))
            area_per_px2 = float(len_per_px**2)
            if parsed_unit:
                unit_str = parsed_unit
    except Exception:
        # Keep fallback defaults when XML parsing fails.
        pass

    relative_times = np.asarray(relative_times, dtype=float)
    relative_times = relative_times[np.isfinite(relative_times) & (relative_times >= 0)]
    if relative_times.size == 0:
        relative_times = np.array([0.062 * i for i in range(max(0, int(num_images)))], dtype=float)

    return relative_times, float(len_per_px), float(area_per_px2), unit_str


def read_xml_properties(xml_file_path, imsize=300, num_images=0):
    """
    Backward-compatible wrapper.
    Returns legacy pair: (resize_scale, relative_times)
    """
    relative_times, len_per_px, _area_per_px2, _unit_str = parse_xml_times_and_scale(
        xml_file_path, num_images=num_images
    )
    return float(len_per_px), relative_times

def compute_segmentation_stats(
    frame_img: np.ndarray,
    vent_mask: np.ndarray,
    atrium_mask: np.ndarray,
    len_per_px: float,
    area_per_px2: float,
    unit_str: str = "unknown",
    prev_major_angle: float | None = None,
    anchor_major_angle: float | None = None,
    axis_mode: str = "scan",  # "scan" (original) or "ellipse"
):
    """
    Replicates the MATLAB measurements for a single frame AND returns an FS overlay frame.

    Parameters
    ----------
    frame_img : np.ndarray
    vent_mask : np.ndarray (H,W) bool or binary
    atrium_mask : np.ndarray (H,W) bool or binary
    len_per_px : float
        Pixel-to-real length scale (physical length / pixel).
    area_per_px2 : float
        Pixel-to-real area scale (physical area / pixel^2).
    prev_major_angle : float | None
        Previous frame major-axis angle in the internal 0..180 image-angle basis.
    anchor_major_angle : float | None
        First accepted major-axis angle; later frames are constrained to remain within
        a wider anchor-centered tolerance, matching the updated MATLAB tracking logic.
    axis_mode : {"scan","ellipse"}
        "scan": original marching approach (default).
        "ellipse": estimate ventricle axes via cv2.fitEllipse and compute minor chords analytically.

    Returns
    -------
    stats: dict (fields exactly as requested)
    fs_overlay_frame: np.ndarray (H,W,3) BGR uint8 image with overlays drawn
    next_prev_major_angle: float | None
    next_anchor_major_angle: float | None
    """

    import cv2, math
    import numpy as np

    # ---- normalize frame to BGR uint8 for drawing ----
    if frame_img.ndim == 2:
        base = frame_img
        if base.dtype != np.uint8:
            # normalize to 0-255
            bmin, bmax = float(base.min()), float(base.max())
            if bmax > bmin:
                base = ((base - bmin) / (bmax - bmin) * 255.0).astype(np.uint8)
            else:
                base = np.zeros_like(base, dtype=np.uint8)
        fs = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    else:
        fs = frame_img.copy()
        if fs.dtype != np.uint8:
            fs = np.clip(fs, 0, 255).astype(np.uint8)

    def to_bool(m):
        return m.astype(bool) if m.dtype != np.bool_ else m

    vent_mask = to_bool(vent_mask)
    atrium_mask = to_bool(atrium_mask)
    H, W = vent_mask.shape

    def wrap360(x):
        return (x % 360.0 + 360.0) % 360.0

    # Colors (BGR like MATLAB-ish palette)
    BLUE  = (255,   0,   0)  # vent boundary
    GREEN = (  0, 255,   0)  # atrium boundary
    RED   = (  0,   0, 255)  # vent centroid & minor axes
    CYAN  = (255, 255,   0)  # atrium centroid & VA lines
    YELL  = (  0, 255, 255)  # major axis
    MAGENTA = (255,   0, 255)
    WHITE = (255, 255, 255)

    def largest_component(mask_bool):
        """Return largest 8-connected component by pixel area, MATLAB regionprops-like."""
        if not np.any(mask_bool):
            return None
        m = mask_bool.astype(np.uint8)
        n_lbl, lbls, stats_cc, cents = cv2.connectedComponentsWithStats(m, connectivity=8)
        if n_lbl <= 1:
            return None
        areas = stats_cc[1:, cv2.CC_STAT_AREA]
        if areas.size == 0:
            return None
        lab = int(np.argmax(areas) + 1)
        filled = (lbls == lab).astype(np.uint8)
        area = int(stats_cc[lab, cv2.CC_STAT_AREA])
        if area <= 0:
            return None
        cx = float(cents[lab, 0])
        cy = float(cents[lab, 1])
        x = int(stats_cc[lab, cv2.CC_STAT_LEFT])
        y = int(stats_cc[lab, cv2.CC_STAT_TOP])
        w = int(stats_cc[lab, cv2.CC_STAT_WIDTH])
        h = int(stats_cc[lab, cv2.CC_STAT_HEIGHT])
        cnts = cv2.findContours((filled * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
        if not cnts:
            return None
        cnt = max(cnts, key=lambda c: c.shape[0])
        cnt_xy = cnt.reshape(-1, 2)  # (x,y)
        return (filled > 0, cnt_xy, (cx, cy), (x, y, w, h), area)

    def longest_boundary(mask_bool):
        """Return the longest boundary on the full mask, MATLAB bwboundaries-like."""
        if not np.any(mask_bool):
            return None
        m = (mask_bool.astype(np.uint8) * 255)
        cnts = cv2.findContours(m, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0]
        if not cnts:
            return None
        cnt = max(cnts, key=lambda c: c.shape[0])
        pts = cnt.reshape(-1, 2).astype(float)
        if len(pts) <= 1:
            return pts

        # Normalize the trace to a MATLAB-like deterministic ordering:
        # start at the top-most/left-most boundary pixel and traverse clockwise.
        start_idx = int(np.lexsort((pts[:, 0], pts[:, 1]))[0])
        pts = np.roll(pts, -start_idx, axis=0)

        x = pts[:, 0]
        y = pts[:, 1]
        signed_area = 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))
        if signed_area < 0:
            pts = pts[::-1].copy()
            start_idx = int(np.lexsort((pts[:, 0], pts[:, 1]))[0])
            pts = np.roll(pts, -start_idx, axis=0)
        return pts

    def smooth_closed_boundary(points_xy, window: int = 9):
        if points_xy is None:
            return None
        pts = np.asarray(points_xy, dtype=float)
        if pts.ndim != 2 or pts.shape[0] == 0 or pts.shape[1] != 2:
            return pts
        if pts.shape[0] < 3 or window <= 1:
            return pts
        if window % 2 == 0:
            window += 1
        pad = window // 2
        kernel = np.ones(window, dtype=float) / float(window)
        xs = np.pad(pts[:, 0], (pad, pad), mode="wrap")
        ys = np.pad(pts[:, 1], (pad, pad), mode="wrap")
        xs = np.convolve(xs, kernel, mode="valid")
        ys = np.convolve(ys, kernel, mode="valid")
        return np.column_stack([xs, ys])

    def atrium_points(mask_bool):
        """
        MATLAB parity for atriumPoints(Amask):
        - centroid from largest-area connected component
        - top/bottom from the longest boundary on the full mask
        """
        cent = (float("nan"), float("nan"))
        bottom_pt = (float("nan"), float("nan"))
        top_pt = (float("nan"), float("nan"))

        comp = largest_component(mask_bool)
        if comp is not None:
            cent = (float(comp[2][0]), float(comp[2][1]))

        boundary = longest_boundary(mask_bool)
        if boundary is not None and len(boundary) > 0:
            by = boundary[:, 1]
            i_max = int(np.argmax(by))
            i_min = int(np.argmin(by))
            bottom_pt = (float(boundary[i_max, 0]), float(boundary[i_max, 1]))
            top_pt = (float(boundary[i_min, 0]), float(boundary[i_min, 1]))

        return cent, bottom_pt, top_pt, boundary

    # Defensive scale fallback
    if not np.isfinite(len_per_px) or len_per_px <= 0:
        len_per_px = 0.9210
    if not np.isfinite(area_per_px2) or area_per_px2 <= 0:
        area_per_px2 = float(len_per_px**2)

    # ---- cavity sizes over the whole masks ----
    v_size = int(np.count_nonzero(vent_mask))
    a_size = int(np.count_nonzero(atrium_mask))
    v_size_real = v_size * area_per_px2
    a_size_real = a_size * area_per_px2

    # ---- largest components & centroids ----
    V = largest_component(vent_mask)
    A = largest_component(atrium_mask)

    if V is None:
        v_centroid = (float("nan"), float("nan"))
        v_contour = None
        v_bbox = None
        v_mask_largest = None
        v_area_largest = 0
        v_contour_len = 0
    else:
        v_mask_largest, v_contour, v_centroid, v_bbox, v_area_largest = V
        v_contour_len = len(v_contour)

    if A is None:
        a_centroid = (float("nan"), float("nan"))
        a_contour = None
        a_top = (float("nan"), float("nan"))
        a_bottom = (float("nan"), float("nan"))
    else:
        a_mask_largest, _a_largest_contour, _a_centroid_largest, a_bbox, _ = A
        a_centroid, a_bottom, a_top, a_contour = atrium_points(atrium_mask)

    # ---- stats dict with defaults ----
    stats = {
        "VentricularCavitySize": float(v_size),
        "RealVentricularCavitySize": float(v_size_real),
        "AtriumCavitySize": float(a_size),
        "RealAtriumCavitySize": float(a_size_real),

        "VentricularCentroid": [float(v_centroid[0]), float(v_centroid[1])],
        "AtriumCentroid": [float(a_centroid[0]) if A else float("nan"),
                           float(a_centroid[1]) if A else float("nan")],
        "VentricularCentroid_real": [float(v_centroid[0] * len_per_px), float(v_centroid[1] * len_per_px)],
        "AtriumCentroid_real": [
            float(a_centroid[0] * len_per_px) if A else float("nan"),
            float(a_centroid[1] * len_per_px) if A else float("nan"),
        ],

        "majorAxisLength": float("nan"),
        "minorAxis_center": float("nan"),
        "minorAxis_upper": float("nan"),
        "minorAxis_lower": float("nan"),
        "majorAngle": float("nan"),

        "VA_Distance_Center": float("nan"),
        "VA_Angle_Center": 180.0,
        "Angle_Center": 180.0,

        "VA_Distance_Bottom": float("nan"),
        "VA_Angle_Bottom": 180.0,
        "Angle_Bottom": 180.0,

        "VA_Distance_Top": float("nan"),
        "VA_Angle_Top": 180.0,
        "Angle_Top": 180.0,

        "VentricleAtriumDistance": float("nan"),
        "VentricleAtriumYDistance": float("nan"),
        "LenPerPx": float(len_per_px),
        "AreaPerPx2": float(area_per_px2),
        "UnitStr": str(unit_str),
    }

    # ---- helpers for geometry ----
    def inside_mask(mask_bool, x, y):
        xi, yi = int(round(x)), int(round(y))
        if xi < 0 or yi < 0 or xi >= W or yi >= H:
            return False
        return mask_bool[yi, xi]

    def inside_boundary(contour_xy, x, y):
        if contour_xy is None or len(contour_xy) < 3:
            return False
        cnt = np.asarray(contour_xy, dtype=np.float32).reshape(-1, 1, 2)
        return cv2.pointPolygonTest(cnt, (float(x), float(y)), False) >= 0

    def march(center_xy, direction_xy, mask_bool):
        x, y = center_xy
        dx, dy = direction_xy
        last_in = (x, y)
        while inside_mask(mask_bool, x, y):
            last_in = (x, y)
            x += dx
            y += dy
        return (x, y), last_in   # (first_outside, last_inside)

    def march_boundary(center_xy, direction_xy, contour_xy):
        x, y = float(center_xy[0]), float(center_xy[1])
        dx, dy = float(direction_xy[0]), float(direction_xy[1])
        last_in = (x, y)
        while inside_boundary(contour_xy, x, y):
            last_in = (x, y)
            x += dx
            y += dy
        return (x, y), last_in

    def _clock_angle_deg(p_from, p_to):
        # 12 o'clock = 0, clockwise positive (MATLAB clock-angle convention)
        dx = float(p_to[0] - p_from[0])
        dy = float(p_to[1] - p_from[1])  # image y: down is +
        return wrap360(math.degrees(math.atan2(dx, -dy)))

    def ang_diff_180(a, b):
        a = np.asarray(a, dtype=float)
        return np.abs(np.mod(a - float(b) + 90.0, 180.0) - 90.0)

    def search_best_axis_angle(contour_xy, center_xy, angle_range):
        best_angle = float("nan")
        best_pos = (float(center_xy[0]), float(center_xy[1]))
        best_neg = (float(center_xy[0]), float(center_xy[1]))
        max_dist = -np.inf
        for ang in np.asarray(angle_range, dtype=float).reshape(-1):
            rad = math.radians(float(ang))
            dir_vec = (math.cos(rad), math.sin(rad))
            _, pos_in = march_boundary(center_xy, dir_vec, contour_xy)
            _, neg_in = march_boundary(center_xy, (-dir_vec[0], -dir_vec[1]), contour_xy)
            dist = math.hypot(pos_in[0] - neg_in[0], pos_in[1] - neg_in[1])
            if dist > max_dist:
                max_dist = dist
                best_angle = float(ang)
                best_pos = pos_in
                best_neg = neg_in
        return best_angle, best_pos, best_neg, max_dist

    def ventricle_top_bottom_on_major(best_pos_major, best_neg_major):
        if best_pos_major[1] < best_neg_major[1]:
            return best_pos_major, best_neg_major
        return best_neg_major, best_pos_major

    def va_metrics_from_vbottom(v_bottom, v_top, a_point):
        if not (np.all(np.isfinite(v_bottom)) and np.all(np.isfinite(v_top)) and np.all(np.isfinite(a_point))):
            return np.nan, np.nan, np.nan
        dist = float(np.linalg.norm(np.asarray(a_point, dtype=float) - np.asarray(v_bottom, dtype=float)) * len_per_px)
        maj_clock = _clock_angle_deg(v_bottom, v_top)
        raw_clock = _clock_angle_deg(v_bottom, a_point)
        major_rel = wrap360(raw_clock - maj_clock)
        return dist, major_rel, raw_clock

    def keep_seed_connected_component(mask_bool, seed_mask_bool):
        m = np.asarray(mask_bool, dtype=bool)
        seed = np.asarray(seed_mask_bool, dtype=bool)
        if not np.any(m) or not np.any(seed):
            return np.zeros_like(m, dtype=bool)
        n_lbl, lbls, _, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
        if n_lbl <= 1:
            return np.zeros_like(m, dtype=bool)
        seed_labels = np.unique(lbls[seed])
        seed_labels = seed_labels[seed_labels > 0]
        if seed_labels.size == 0:
            return np.zeros_like(m, dtype=bool)
        return np.isin(lbls, seed_labels)

    def binary_geodesic_distance(mask_bool, seed_mask_bool):
        from skimage.graph import MCP_Geometric

        m = np.asarray(mask_bool, dtype=bool)
        seed = np.asarray(seed_mask_bool, dtype=bool)
        out = np.full(m.shape, np.nan, dtype=float)
        if not np.any(m) or not np.any(seed):
            return out

        cost = np.where(m, 1.0, np.inf).astype(float)
        starts = [tuple(rc) for rc in np.column_stack(np.nonzero(seed))]
        if not starts:
            return out

        mcp = MCP_Geometric(cost)
        costs, _ = mcp.find_costs(starts=starts)
        costs = np.asarray(costs, dtype=float)
        costs[~np.isfinite(costs)] = np.nan
        return costs

    def heart_vtop_abottom_distance(frame_gray, vmask, amask):
        """
        MATLAB parity for the updated heartVtopAbottomDistance(P,...):
        - build whole-heart mask from thresholded preprocessing + dilated chambers
        - fill holes
        - geodesically split the whole-heart mask into ventricle/atrium regions
        - measure whole-ventricle top to whole-atrium bottom
        """
        g = np.asarray(frame_gray, dtype=float)
        if g.ndim == 3:
            g = cv2.cvtColor(g.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(float)
        if g.size == 0 or np.all(~np.isfinite(g)):
            return np.nan, np.nan, None, None, None, (np.nan, np.nan), (np.nan, np.nan)

        g = g.copy()
        g[~np.isfinite(g)] = 0.0
        g_min = float(np.min(g))
        g_max = float(np.max(g))
        if g_max > g_min:
            g = (g - g_min) / (g_max - g_min)
        else:
            g = np.zeros_like(g, dtype=float)

        if not np.any(vmask) or not np.any(amask):
            return np.nan, np.nan, None, None, None, (np.nan, np.nan), (np.nan, np.nan)

        bw = (g > 0.10).astype(np.uint8)
        n_lbl, lbls, stats_cc, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
        if n_lbl > 1:
            areas = stats_cc[1:, cv2.CC_STAT_AREA]
            if areas.size:
                keep = int(np.argmax(areas) + 1)
                bw = (lbls == keep).astype(np.uint8)

        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))  # disk radius ~= 4
        bw = np.logical_or(bw > 0, cv2.dilate(vmask.astype(np.uint8), ker) > 0)
        bw = np.logical_or(bw, cv2.dilate(amask.astype(np.uint8), ker) > 0)
        bw = ndi.binary_fill_holes(bw).astype(bool)

        best_boundary = longest_boundary(bw)
        if best_boundary is None or len(best_boundary) == 0:
            return np.nan, np.nan, None, None, None, (np.nan, np.nan), (np.nan, np.nan)

        seed_ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # disk radius ~= 2
        vseed = np.logical_and(bw, cv2.dilate(vmask.astype(np.uint8), seed_ker) > 0)
        aseed = np.logical_and(bw, cv2.dilate(amask.astype(np.uint8), seed_ker) > 0)
        if not np.any(vseed):
            vseed = np.logical_and(bw, vmask)
        if not np.any(aseed):
            aseed = np.logical_and(bw, amask)
        if not np.any(vseed) or not np.any(aseed):
            return np.nan, np.nan, best_boundary, None, None, (np.nan, np.nan), (np.nan, np.nan)

        dv = binary_geodesic_distance(bw, vseed)
        da = binary_geodesic_distance(bw, aseed)
        if np.all(np.isnan(dv)) or np.all(np.isnan(da)):
            return np.nan, np.nan, best_boundary, None, None, (np.nan, np.nan), (np.nan, np.nan)

        both_reach = np.isfinite(dv) & np.isfinite(da)
        only_v = np.isfinite(dv) & ~np.isfinite(da)
        only_a = ~np.isfinite(dv) & np.isfinite(da)

        vwhole = np.zeros_like(bw, dtype=bool)
        awhole = np.zeros_like(bw, dtype=bool)
        vwhole[both_reach] = dv[both_reach] <= da[both_reach]
        awhole[both_reach] = da[both_reach] < dv[both_reach]
        vwhole[only_v] = True
        awhole[only_a] = True
        vwhole = np.logical_and(bw, vwhole)
        awhole = np.logical_and(bw, awhole)

        vwhole = keep_seed_connected_component(vwhole, vseed)
        awhole = keep_seed_connected_component(awhole, aseed)
        if not np.any(vwhole) or not np.any(awhole):
            return np.nan, np.nan, best_boundary, None, None, (np.nan, np.nan), (np.nan, np.nan)

        vwhole_boundary = longest_boundary(vwhole)
        awhole_boundary = longest_boundary(awhole)
        if (
            vwhole_boundary is None
            or awhole_boundary is None
            or len(vwhole_boundary) == 0
            or len(awhole_boundary) == 0
        ):
            return np.nan, np.nan, best_boundary, vwhole_boundary, awhole_boundary, (np.nan, np.nan), (np.nan, np.nan)

        vtop = tuple(vwhole_boundary[int(np.argmin(vwhole_boundary[:, 1]))])
        abottom = tuple(awhole_boundary[int(np.argmax(awhole_boundary[:, 1]))])
        dist_xy = float(np.linalg.norm(np.asarray(vtop) - np.asarray(abottom)) * len_per_px)
        dist_y = float(abs(vtop[1] - abottom[1]) * len_per_px)
        return dist_xy, dist_y, best_boundary, vwhole_boundary, awhole_boundary, vtop, abottom

    (
        stats["VentricleAtriumDistance"],
        stats["VentricleAtriumYDistance"],
        heart_boundary,
        vwhole_boundary,
        awhole_boundary,
        vtop_whole,
        abottom_whole,
    ) = heart_vtop_abottom_distance(
        frame_img, vent_mask, atrium_mask
    )

    # ---- DRAW: updated MATLAB FS overlay order ----
    if heart_boundary is not None and len(heart_boundary) >= 3:
        smooth_heart = smooth_closed_boundary(heart_boundary, window=9)
        cv2.polylines(fs, [np.round(smooth_heart).astype(np.int32)], isClosed=True, color=WHITE, thickness=2)

    if v_size > 300:
        if v_contour is not None and len(v_contour) >= 3:
            cv2.polylines(fs, [v_contour.astype(np.int32)], isClosed=True, color=BLUE, thickness=2)
    if a_size > 100:
        if a_contour is not None and len(a_contour) >= 3:
            cv2.polylines(fs, [a_contour.astype(np.int32)], isClosed=True, color=GREEN, thickness=2)

    # ---- centroids (markers) ----
    if np.isfinite(stats["VentricularCentroid"][0]) and np.isfinite(stats["VentricularCentroid"][1]):
        cv2.circle(
            fs,
            (int(round(stats["VentricularCentroid"][0])), int(round(stats["VentricularCentroid"][1]))),
            radius=6, color=RED, thickness=1
        )
    if np.isfinite(stats["AtriumCentroid"][0]) and np.isfinite(stats["AtriumCentroid"][1]):
        cv2.circle(
            fs,
            (int(round(stats["AtriumCentroid"][0])), int(round(stats["AtriumCentroid"][1]))),
            radius=6, color=CYAN, thickness=1
        )

    # ---- MAJOR + MINOR AXES & VA lines (when vent area >=400 and boundary long enough) ----
    major_valid = False
    best_angle = float("nan")
    best_pos = None
    best_neg = None
    minor_segments = []  # for visualization

    if (V is not None) and (v_area_largest >= 400) and (v_contour_len >= 70):
        cx, cy = v_centroid

        if axis_mode.lower() == "ellipse":
            # ---- Fit an ellipse to the largest ventricle contour ----
            cnt_for_fit = v_contour.reshape(-1, 1, 2).astype(np.float32)
            if len(cnt_for_fit) >= 5:
                (ecx, ecy), (ax1, ax2), theta = cv2.fitEllipse(cnt_for_fit)
                # Determine major/minor and align angle to major axis direction
                if ax1 >= ax2:
                    major_len_px = float(ax1)
                    minor_len_px = float(ax2)
                    major_angle_deg = float(theta)
                else:
                    major_len_px = float(ax2)
                    minor_len_px = float(ax1)
                    major_angle_deg = float(theta + 90.0)

                # Semi-axes
                a = 0.5 * major_len_px
                b = 0.5 * minor_len_px

                # Unit vectors along major/minor in image coords
                rad = math.radians(major_angle_deg)
                u_major = np.array([math.cos(rad), math.sin(rad)], dtype=float)
                u_minor = np.array([math.cos(rad + math.pi/2.0), math.sin(rad + math.pi/2.0)], dtype=float)

                # Endpoints of major axis on ellipse
                E_center = np.array([ecx, ecy], dtype=float)
                P_pos = (E_center + a * u_major)
                P_neg = (E_center - a * u_major)

                # Draw major axis
                cv2.line(
                    fs,
                    (int(round(P_pos[0])), int(round(P_pos[1]))),
                    (int(round(P_neg[0])), int(round(P_neg[1]))),
                    color=YELL, thickness=2
                )

                # Minor chords at center and ±10% along MAJOR (by chord formula)
                offset = 0.1 * (2.0 * a)  # 10% of full major length
                u_offsets = [0.0, +offset, -offset]
                minor_lengths_real = []

                for u_off in u_offsets:
                    if abs(u_off) >= a - 1e-9:
                        # Degenerate chord (tangent or outside)
                        L_half = 0.0
                    else:
                        L_half = b * math.sqrt(max(0.0, 1.0 - (u_off / a) ** 2))
                    C = E_center + u_off * u_major    # chord center in image coords
                    pos_pt = C + L_half * u_minor
                    neg_pt = C - L_half * u_minor
                    minor_segments.append((pos_pt, neg_pt))
                    # Draw minor chord
                    cv2.line(
                        fs,
                        (int(round(pos_pt[0])), int(round(pos_pt[1]))),
                        (int(round(neg_pt[0])), int(round(neg_pt[1]))),
                        color=RED, thickness=2
                    )
                    minor_lengths_real.append((2.0 * L_half) * len_per_px)

                stats["majorAxisLength"] = major_len_px * len_per_px
                stats["minorAxis_center"] = float(minor_lengths_real[0])
                stats["minorAxis_upper"]  = float(minor_lengths_real[1])
                stats["minorAxis_lower"]  = float(minor_lengths_real[2])

                major_valid = True
                best_angle = major_angle_deg
                stats["majorAngle"] = float(best_angle)
                best_pos = P_pos
                best_neg = P_neg

                top_v, bot_v = ventricle_top_bottom_on_major(best_pos, best_neg)

                a_center = (
                    float(stats["AtriumCentroid"][0]),
                    float(stats["AtriumCentroid"][1]),
                )
                for tag, a_pt in (("Center", a_center), ("Bottom", a_bottom), ("Top", a_top)):
                    d_va, a_rel, a_raw = va_metrics_from_vbottom(bot_v, top_v, a_pt)
                    stats[f"VA_Distance_{tag}"] = float(d_va)
                    stats[f"VA_Angle_{tag}"] = float(a_rel)
                    stats[f"Angle_{tag}"] = float(a_raw)

        else:
            # ---- Updated MATLAB "scan" method with previous-angle + anchor tracking ----
            axis_center = np.array([cx, cy], dtype=float)
            if not inside_boundary(v_contour, axis_center[0], axis_center[1]):
                dist_in = cv2.distanceTransform(v_mask_largest.astype(np.uint8), cv2.DIST_L2, 5)
                if dist_in.size:
                    y0, x0 = np.unravel_index(int(np.argmax(dist_in)), dist_in.shape)
                    axis_center = np.array([float(x0), float(y0)], dtype=float)

            if prev_major_angle is None or (isinstance(prev_major_angle, float) and math.isnan(prev_major_angle)):
                coarse_range = np.arange(65.0, 178.0 + 1e-6, 3.0)
                coarse_angle, _, _, _ = search_best_axis_angle(v_contour, (axis_center[0], axis_center[1]), coarse_range)
                if not np.isfinite(coarse_angle):
                    return stats, fs, prev_major_angle, anchor_major_angle
                angle_range = np.mod(coarse_angle + np.arange(-3.0, 3.0 + 1e-6, 1.0), 180.0)
            else:
                angle_range = np.mod(prev_major_angle + np.arange(-3.0, 3.0 + 1e-6, 1.0), 180.0)

            if anchor_major_angle is not None and np.isfinite(anchor_major_angle):
                angle_range = angle_range[ang_diff_180(angle_range, anchor_major_angle) <= 15.0]

            if angle_range.size == 0:
                return stats, fs, prev_major_angle, anchor_major_angle

            best_angle, best_pos, best_neg, max_dist = search_best_axis_angle(
                v_contour,
                (axis_center[0], axis_center[1]),
                angle_range,
            )

            if np.isfinite(max_dist) and not math.isnan(best_angle) and max_dist > 0:
                major_valid = True
                best_angle = float(np.mod(best_angle, 180.0))
                if anchor_major_angle is None or (isinstance(anchor_major_angle, float) and math.isnan(anchor_major_angle)):
                    anchor_major_angle = best_angle
                prev_major_angle = best_angle
                stats["majorAxisLength"] = max_dist * len_per_px
                stats["majorAngle"] = float(best_angle)

                # Draw major axis (yellow)
                cv2.line(
                    fs,
                    (int(round(best_pos[0])), int(round(best_pos[1]))),
                    (int(round(best_neg[0])), int(round(best_neg[1]))),
                    color=YELL, thickness=2
                )

                # Minor axes at center, +10%, -10%
                v_major = np.array([best_pos[0] - best_neg[0], best_pos[1] - best_neg[1]], dtype=float)
                v_major /= (np.linalg.norm(v_major) + 1e-12)
                minor_angle = wrap360(best_angle + 90.0)
                radm = math.radians(minor_angle)
                v_minor = np.array([math.cos(radm), math.sin(radm)], dtype=float)

                offset = 0.1 * max_dist
                centers = [
                    np.array([axis_center[0], axis_center[1]], dtype=float),
                    np.array([axis_center[0], axis_center[1]], dtype=float) + offset * v_major,
                    np.array([axis_center[0], axis_center[1]], dtype=float) - offset * v_major,
                ]

                minor_lengths = []
                for C in centers:
                    pos_out, pos_in = march_boundary(C, v_minor, v_contour)
                    neg_out, neg_in = march_boundary(C, -v_minor, v_contour)
                    # Length from last-inside endpoints (parity with MATLAB)
                    length = math.hypot(pos_in[0] - neg_in[0], pos_in[1] - neg_in[1]) * len_per_px
                    minor_lengths.append(length)
                    minor_segments.append((pos_in, neg_in))
                    # Draw minor segment (red)
                    cv2.line(
                        fs,
                        (int(round(pos_in[0])), int(round(pos_in[1]))),
                        (int(round(neg_in[0])), int(round(neg_in[1]))),
                        color=RED, thickness=2
                    )

                stats["minorAxis_center"] = float(minor_lengths[0])
                stats["minorAxis_upper"]  = float(minor_lengths[1])
                stats["minorAxis_lower"]  = float(minor_lengths[2])

                top_v, bot_v = ventricle_top_bottom_on_major(best_pos, best_neg)

                a_center = (
                    float(stats["AtriumCentroid"][0]),
                    float(stats["AtriumCentroid"][1]),
                )
                for tag, a_pt in (("Center", a_center), ("Bottom", a_bottom), ("Top", a_top)):
                    d_va, a_rel, a_raw = va_metrics_from_vbottom(bot_v, top_v, a_pt)
                    stats[f"VA_Distance_{tag}"] = float(d_va)
                    stats[f"VA_Angle_{tag}"] = float(a_rel)
                    stats[f"Angle_{tag}"] = float(a_raw)

    if vwhole_boundary is not None and len(vwhole_boundary) >= 3:
        cv2.polylines(fs, [np.round(vwhole_boundary).astype(np.int32)], isClosed=True, color=CYAN, thickness=2)
    if awhole_boundary is not None and len(awhole_boundary) >= 3:
        cv2.polylines(fs, [np.round(awhole_boundary).astype(np.int32)], isClosed=True, color=MAGENTA, thickness=2)

    if (
        np.all(np.isfinite(vtop_whole))
        and np.all(np.isfinite(abottom_whole))
    ):
        cv2.line(
            fs,
            (int(round(vtop_whole[0])), int(round(vtop_whole[1]))),
            (int(round(abottom_whole[0])), int(round(abottom_whole[1]))),
            color=RED,
            thickness=3,
        )
        cv2.circle(fs, (int(round(vtop_whole[0])), int(round(vtop_whole[1]))), radius=7, color=RED, thickness=1)
        cv2.circle(fs, (int(round(abottom_whole[0])), int(round(abottom_whole[1]))), radius=7, color=YELL, thickness=1)

    return stats, fs, prev_major_angle, anchor_major_angle


def _im2gray_unit(frame: np.ndarray) -> np.ndarray:
    """MATLAB im2gray-like conversion to float image in [0,1]."""
    arr = np.asarray(frame)
    if arr.ndim == 3:
        if arr.shape[-1] >= 3:
            arr = (
                0.2989 * arr[..., 0]
                + 0.5870 * arr[..., 1]
                + 0.1140 * arr[..., 2]
            )
        else:
            arr = arr[..., 0]
    arr = np.asarray(arr, dtype=float)
    arr[~np.isfinite(arr)] = 0.0
    if arr.size == 0:
        return arr.astype(float)
    mx = float(np.max(arr))
    mn = float(np.min(arr))
    if mx > 1.0:
        arr = arr / 255.0
        mx = float(np.max(arr))
        mn = float(np.min(arr))
    if np.nanmax(arr) > 1.0 or np.nanmin(arr) < 0.0:
        if mx > mn:
            arr = (arr - mn) / (mx - mn)
        else:
            arr = np.zeros_like(arr, dtype=float)
    return np.clip(arr, 0.0, 1.0)


def _matlab_preprocess_pdouble(frame: np.ndarray) -> np.ndarray:
    """
    MATLAB parity preprocessing used for segmentation refinement:
    G=im2gray(I); J=stretchlim(G); J(2)=0.9*J(2); P=imadjust(G,J,[]); Pdouble=im2double(P)
    """
    g = _im2gray_unit(frame)
    if g.size == 0:
        return g.astype(float)
    j_lo = float(np.nanpercentile(g, 1.0))
    j_hi = float(np.nanpercentile(g, 99.0))
    if not np.isfinite(j_lo):
        j_lo = 0.0
    if not np.isfinite(j_hi):
        j_hi = 1.0
    j_hi = min(1.0, 0.90 * j_hi)
    if j_hi <= j_lo:
        return np.zeros_like(g, dtype=float)
    return np.clip((g - j_lo) / (j_hi - j_lo), 0.0, 1.0).astype(float)


def _to_gray_unit(frame: np.ndarray) -> np.ndarray:
    """Backward-compatible grayscale helper."""
    return _im2gray_unit(frame)


def _matlab_nanstd(x: np.ndarray) -> float:
    """MATLAB std(...,'omitnan') parity for vectors: empty->NaN, scalar->0, else sample std."""
    a = np.asarray(x, dtype=float).reshape(-1)
    a = a[np.isfinite(a)]
    n = int(a.size)
    if n == 0:
        return np.nan
    if n == 1:
        return 0.0
    return float(np.std(a, ddof=1))


def _remove_small_components(mask_bool: np.ndarray, min_area: int = 300) -> np.ndarray:
    """MATLAB bwareaopen(mask, min_area) equivalent with 8-connectivity."""
    m = np.asarray(mask_bool, dtype=bool)
    if not np.any(m):
        return m
    n_lbl, lbls, stats_cc, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
    if n_lbl <= 1:
        return m
    out = np.zeros_like(m, dtype=bool)
    for lab in range(1, n_lbl):
        if int(stats_cc[lab, cv2.CC_STAT_AREA]) >= int(min_area):
            out |= (lbls == lab)
    return out


def _largest_component_centroid(mask_bool: np.ndarray) -> Optional[tuple[float, float]]:
    """Centroid (x,y) of the largest connected component; None if empty."""
    m = np.asarray(mask_bool, dtype=bool)
    if not np.any(m):
        return None
    n_lbl, lbls, stats_cc, cents = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
    if n_lbl <= 1:
        return None
    areas = stats_cc[1:, cv2.CC_STAT_AREA]
    if areas.size == 0:
        return None
    lab = int(np.argmax(areas) + 1)
    cx, cy = cents[lab]
    if not (np.isfinite(cx) and np.isfinite(cy)):
        return None
    return float(cx), float(cy)


def _refine_mask_by_brightness_and_centroid(
    frame_gray_unit: np.ndarray,
    mask_bool: np.ndarray,
    prev_centroid_xy: Optional[tuple[float, float]],
    dark_th: float = 0.15,
    p_th: float = 0.70,
    dist_thresh: float = 30.0,
) -> tuple[np.ndarray, Optional[tuple[float, float]]]:
    """
    MATLAB parity for refineMasksByBrightnessAndCentroid:
    - Multi-component: keep only bright-enough components.
    - Single-component: if centroid jump is large AND component is dark, drop it.
    """
    m = np.asarray(mask_bool, dtype=bool)
    if not np.any(m):
        return m, prev_centroid_xy

    n_lbl, lbls, stats_cc, cents = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
    if n_lbl <= 1:
        return m, prev_centroid_xy

    comp_count = int(n_lbl - 1)
    if comp_count >= 2:
        out = np.zeros_like(m, dtype=bool)
        for lab in range(1, n_lbl):
            pix = (lbls == lab)
            if not np.any(pix):
                continue
            dark_ratio = float(np.mean(frame_gray_unit[pix] < float(dark_th)))
            if dark_ratio < float(p_th):
                out[pix] = True
        return out, prev_centroid_xy

    cx, cy = cents[1]
    if np.isfinite(cx) and np.isfinite(cy):
        cur = (float(cx), float(cy))
        if prev_centroid_xy is not None:
            dxy = np.hypot(cur[0] - prev_centroid_xy[0], cur[1] - prev_centroid_xy[1])
            if np.isfinite(dxy) and dxy > float(dist_thresh):
                pix = (lbls == 1)
                dark_ratio = float(np.mean(frame_gray_unit[pix] < float(dark_th))) if np.any(pix) else 1.0
                if dark_ratio >= float(p_th):
                    m = np.zeros_like(m, dtype=bool)
        # MATLAB updates prev centroid even when the component is dropped.
        return m, cur

    return m, prev_centroid_xy


def _cleanup_frame_labels(
    frame_i: np.ndarray,
    labels: np.ndarray,
    prev_v_centroid: Optional[tuple[float, float]],
    prev_a_centroid: Optional[tuple[float, float]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[tuple[float, float]], Optional[tuple[float, float]]]:
    """Apply MATLAB-style mask cleanup for one frame and return cleaned labels {0,1,2}."""
    frame_refine_unit = _matlab_preprocess_pdouble(frame_i)
    ventricle_mask = _remove_small_components(labels == 1, min_area=300)
    atrium_mask = _remove_small_components(labels == 2, min_area=300)
    ventricle_mask, prev_v_centroid = _refine_mask_by_brightness_and_centroid(
        frame_refine_unit, ventricle_mask, prev_v_centroid
    )
    atrium_mask, prev_a_centroid = _refine_mask_by_brightness_and_centroid(
        frame_refine_unit, atrium_mask, prev_a_centroid
    )
    cleaned_labels = ventricle_mask.astype(np.uint8) + 2 * atrium_mask.astype(np.uint8)
    return (
        frame_refine_unit,
        ventricle_mask,
        atrium_mask,
        cleaned_labels,
        prev_v_centroid,
        prev_a_centroid,
    )


def _circular_mean_deg(x: np.ndarray) -> float:
    """Mean of 0..360 degree angles with wrap handling (MATLAB circularMeanDeg parity)."""
    a = np.asarray(x, dtype=float).reshape(-1)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return np.nan
    ang = np.deg2rad(a)
    return float(np.mod(np.rad2deg(np.arctan2(np.mean(np.sin(ang)), np.mean(np.cos(ang)))), 360.0))


def compute_segmentation_statistics(
    stack,
    masks,
    experiment_id,
    analysis_dir,
    meta_file=None,
    meta_info=None,
    axis_mode="scan",
    frame_filenames: Optional[Sequence[str]] = None,
    show_progress: bool = False,
    progress_desc: Optional[str] = None,
    return_cleaned_masks: bool = False,
    apply_mask_cleanup: bool = True,
    frames_are_preprocessed: bool = False,
    matlab_series_key: Optional[str] = None,
):
    """
    stack : (T,H,W) uint8/float array (grayscale frames)
    masks : (T,H,W) int/uint8 with labels {0:bg, 1:ventricle, 2:atrium}
    """
    num_images = masks.shape[0]
    if meta_file is not None and meta_info is None:
        relative_times_vec, len_per_px, area_per_px2, unit_str = parse_xml_times_and_scale(
            meta_file, num_images=num_images
        )
    elif meta_file is None and meta_info is not None:
        len_per_px = float(meta_info.get("len_per_px", meta_info.get("resize_scale", 0.9210)))
        area_per_px2 = float(meta_info.get("area_per_px2", len_per_px**2))
        unit_str = str(meta_info.get("unit_str", "unknown"))
        if meta_info.get("relative_times") is not None:
            relative_times_vec = np.asarray(meta_info["relative_times"], dtype=float).reshape(-1)
        else:
            dt = float(meta_info.get("frame_interval", 0.062))
            relative_times_vec = np.array([dt * i for i in range(num_images)], dtype=float)
    else:
        relative_times_vec = np.array([0.062 * i for i in range(num_images)], dtype=float)
        len_per_px = 0.9210
        area_per_px2 = float(len_per_px**2)
        unit_str = "unknown"

    relative_times_vec = np.asarray(relative_times_vec, dtype=float).reshape(-1)
    if relative_times_vec.size == 0:
        relative_times_vec = np.array([0.062 * i for i in range(num_images)], dtype=float)

    fnames = None
    if frame_filenames is not None:
        fnames = [str(x) for x in frame_filenames]
        if len(fnames) != num_images:
            fnames = None

    seg_stats = []
    fs_frames_bgr = []
    cleaned_masks = [] if return_cleaned_masks else None

    # Track updated MATLAB major-axis state between frames in the same sample.
    prev_major_angle = None
    anchor_major_angle = None
    prev_v_centroid = None
    prev_a_centroid = None

    frame_iter = range(num_images)
    if show_progress:
        frame_iter = tqdm(
            frame_iter,
            total=num_images,
            desc=(progress_desc or f"{experiment_id} seg-stats"),
            leave=False,
            unit="frame",
            dynamic_ncols=True,
        )

    for i in frame_iter:
        frame_i = stack[i].compute() if hasattr(stack[i], "compute") else np.asarray(stack[i])
        labels = masks[i]
        if apply_mask_cleanup:
            if frames_are_preprocessed:
                frame_refine_unit = _im2gray_unit(frame_i)
                ventricle_mask = _remove_small_components(labels == 1, min_area=300)
                atrium_mask = _remove_small_components(labels == 2, min_area=300)
                ventricle_mask, prev_v_centroid = _refine_mask_by_brightness_and_centroid(
                    frame_refine_unit, ventricle_mask, prev_v_centroid
                )
                atrium_mask, prev_a_centroid = _refine_mask_by_brightness_and_centroid(
                    frame_refine_unit, atrium_mask, prev_a_centroid
                )
                cleaned_labels = ventricle_mask.astype(np.uint8) + 2 * atrium_mask.astype(np.uint8)
            else:
                (
                    frame_refine_unit,
                    ventricle_mask,
                    atrium_mask,
                    cleaned_labels,
                    prev_v_centroid,
                    prev_a_centroid,
                ) = _cleanup_frame_labels(frame_i, labels, prev_v_centroid, prev_a_centroid)
        else:
            frame_refine_unit = _im2gray_unit(frame_i) if frames_are_preprocessed else _matlab_preprocess_pdouble(frame_i)
            ventricle_mask = np.asarray(labels == 1, dtype=bool)
            atrium_mask = np.asarray(labels == 2, dtype=bool)
            cleaned_labels = ventricle_mask.astype(np.uint8) + 2 * atrium_mask.astype(np.uint8)

        stats_i, fs_overlay_i, prev_major_angle, anchor_major_angle = compute_segmentation_stats(
            frame_refine_unit,
            ventricle_mask,
            atrium_mask,
            len_per_px,
            area_per_px2,
            unit_str,
            prev_major_angle,
            anchor_major_angle,
            axis_mode,
        )

        frame_name = fnames[i] if fnames is not None else os.path.basename(experiment_id)
        t_idx = _extract_t_index(frame_name)
        if t_idx is not None and 0 <= t_idx < relative_times_vec.size:
            rel_t = float(relative_times_vec[t_idx])
        else:
            rel_t = float(relative_times_vec[min(i, relative_times_vec.size - 1)])

        seg_stats.append({
            "FileName": frame_name,
            "RelativeTime": rel_t,
            "VentricularCavitySize": stats_i["VentricularCavitySize"],
            "RealVentricularCavitySize": stats_i["RealVentricularCavitySize"],
            "AtriumCavitySize": stats_i["AtriumCavitySize"],
            "RealAtriumCavitySize": stats_i["RealAtriumCavitySize"],
            # MATLAB-friendly aliases
            "VArea_px": stats_i["VentricularCavitySize"],
            "VArea_real": stats_i["RealVentricularCavitySize"],
            "AArea_px": stats_i["AtriumCavitySize"],
            "AArea_real": stats_i["RealAtriumCavitySize"],
            # Kept in legacy order for compatibility with existing downstream files.
            "VentricularCentroid_X": stats_i["VentricularCentroid"][1],
            "VentricularCentroid_Y": stats_i["VentricularCentroid"][0],
            "AtriumCentroid_X": stats_i["AtriumCentroid"][1],
            "AtriumCentroid_Y": stats_i["AtriumCentroid"][0],
            "VCentroid": stats_i["VentricularCentroid"],
            "ACentroid": stats_i["AtriumCentroid"],
            "VCentroid_X": stats_i["VentricularCentroid"][0],
            "VCentroid_Y": stats_i["VentricularCentroid"][1],
            "ACentroid_X": stats_i["AtriumCentroid"][0],
            "ACentroid_Y": stats_i["AtriumCentroid"][1],
            "VCentroid_real": stats_i["VentricularCentroid_real"],
            "ACentroid_real": stats_i["AtriumCentroid_real"],
            "VCentroid_real_X": stats_i["VentricularCentroid_real"][0],
            "VCentroid_real_Y": stats_i["VentricularCentroid_real"][1],
            "ACentroid_real_X": stats_i["AtriumCentroid_real"][0],
            "ACentroid_real_Y": stats_i["AtriumCentroid_real"][1],
            "VentricularCentroid_Real_X": stats_i["VentricularCentroid_real"][0],
            "VentricularCentroid_Real_Y": stats_i["VentricularCentroid_real"][1],
            "AtriumCentroid_Real_X": stats_i["AtriumCentroid_real"][0],
            "AtriumCentroid_Real_Y": stats_i["AtriumCentroid_real"][1],
            "VentricleAtriumDistance": stats_i["VentricleAtriumDistance"],
            "VentricleAtriumYDistance": stats_i["VentricleAtriumYDistance"],
            "SVBA_Distance": stats_i["VentricleAtriumDistance"],
            "SVBA_Distance_Y": stats_i["VentricleAtriumYDistance"],
            "majorAxisLength": stats_i["majorAxisLength"],
            "minorAxis_center": stats_i["minorAxis_center"],
            "minorAxis_upper": stats_i["minorAxis_upper"],
            "minorAxis_lower": stats_i["minorAxis_lower"],
            "MajorAxisLength": stats_i["majorAxisLength"],
            "MinorAxis_center": stats_i["minorAxis_center"],
            "MinorAxis_upper": stats_i["minorAxis_upper"],
            "MinorAxis_lower": stats_i["minorAxis_lower"],
            "VA_Distance_Center": stats_i["VA_Distance_Center"],
            "VA_Angle_Center": stats_i["VA_Angle_Center"],
            "Angle_Center": stats_i["Angle_Center"],
            "VA_Angle_Center_raw": stats_i["Angle_Center"],
            "VA_Distance_Bottom": stats_i["VA_Distance_Bottom"],
            "VA_Angle_Bottom": stats_i["VA_Angle_Bottom"],
            "Angle_Bottom": stats_i["Angle_Bottom"],
            "VA_Angle_Bottom_raw": stats_i["Angle_Bottom"],
            "VA_Distance_Top": stats_i["VA_Distance_Top"],
            "VA_Angle_Top": stats_i["VA_Angle_Top"],
            "Angle_Top": stats_i["Angle_Top"],
            "VA_Angle_Top_raw": stats_i["Angle_Top"],
            "LenPerPx": stats_i["LenPerPx"],
            "AreaPerPx2": stats_i["AreaPerPx2"],
            "UnitStr": stats_i["UnitStr"],
        })

        fs_frames_bgr.append(fs_overlay_i)
        if cleaned_masks is not None:
            cleaned_masks.append(cleaned_labels)

    # ---- Write Excel like MATLAB (one file per group/experiment) ----
    seg_stats_df = pd.DataFrame(seg_stats)
    os.makedirs(analysis_dir, exist_ok=True)
    excel_file = os.path.join(analysis_dir, f"{experiment_id}.xlsx")
    seg_stats_df.to_excel(excel_file, index=False)
    matlab_key = str(matlab_series_key or derive_matlab_series_key(fnames, fallback=experiment_id))
    matlab_seg_df = matlab_style_segmentation_dataframe(seg_stats_df)
    matlab_excel_file = os.path.join(analysis_dir, f"{matlab_key}.xlsx")
    if os.path.abspath(matlab_excel_file) != os.path.abspath(excel_file):
        matlab_seg_df.to_excel(matlab_excel_file, index=False)

    # ---- Save FS overlay as GIF ----
    # Duration per-frame based on RelativeTime diffs (fallback to 0.062s)
    rt = seg_stats_df["RelativeTime"].to_numpy(dtype=float)
    if len(rt) >= 2:
        diffs = np.diff(rt, prepend=rt[0])
        # Replace first zero with median (or default 0.062)
        if diffs[0] <= 0:
            diffs[0] = (np.median(diffs[1:]) if np.any(diffs[1:] > 0) else 0.062)
        durations = [max(1e-3, float(d)) for d in diffs]  # seconds per frame
    else:
        durations = [0.062] * len(rt)

    # Convert BGR -> RGB for imageio
    fs_frames_rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in fs_frames_bgr]
    gif_path = os.path.join(analysis_dir, f"{experiment_id}_FS.gif")
    imageio.mimsave(gif_path, fs_frames_rgb, duration=durations)
    # Also save a directory of PNGs
    png_dir = os.path.join(analysis_dir, f"{experiment_id}_FS")
    os.makedirs(png_dir, exist_ok=True)
    png_iter = enumerate(fs_frames_rgb)
    if show_progress:
        png_iter = tqdm(
            png_iter,
            total=len(fs_frames_rgb),
            desc=(progress_desc or f"{experiment_id} FS-export"),
            leave=False,
            unit="frame",
            dynamic_ncols=True,
        )

    for i, f in png_iter:
        png_path = os.path.join(png_dir, f"{experiment_id}_FS_{i:04d}.png")
        imageio.imwrite(png_path, f)
    # Return as np array of shape (T, H, W, 3)
    fs_frames_rgb_np = np.stack(fs_frames_rgb, axis=0)

    if cleaned_masks is not None:
        cleaned_masks_np = np.stack(cleaned_masks, axis=0).astype(np.uint8, copy=False)
        return seg_stats_df, fs_frames_rgb_np, cleaned_masks_np

    return seg_stats_df, fs_frames_rgb_np


def fix_relative_time_stitch(
    time: np.ndarray,
    gap_factor: float = 5.0,
    expected_duration_sec: float | None = None,
) -> np.ndarray:
    """
    Remove large jumps in relative time by stitching each detected gap to the
    median frame interval (MATLAB fixRelativeTimeStitch parity).
    """
    t_fix = np.asarray(time, dtype=float).reshape(-1).copy()
    if t_fix.size < 3:
        return t_fix

    dt = np.diff(t_fix)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if dt.size == 0:
        return t_fix

    dt_med = float(np.median(dt))
    if not np.isfinite(dt_med) or dt_med <= 0:
        return t_fix

    gap_thr = float(gap_factor) * dt_med
    dt_all = np.diff(t_fix)
    jump_idx = np.where(np.isfinite(dt_all) & (dt_all > gap_thr))[0]

    for j in jump_idx:
        if j + 1 >= t_fix.size:
            continue
        if not (np.isfinite(t_fix[j]) and np.isfinite(t_fix[j + 1])):
            continue
        desired_next = t_fix[j] + dt_med
        shift = t_fix[j + 1] - desired_next
        t_fix[j + 1 :] -= shift

    if np.isfinite(t_fix[0]):
        t_fix -= t_fix[0]

    if expected_duration_sec is not None and np.isfinite(expected_duration_sec):
        cur_dur = t_fix[-1] - t_fix[0]
        if np.isfinite(cur_dur) and cur_dur > 0:
            t_fix = t_fix * (float(expected_duration_sec) / float(cur_dur))

    return t_fix


def _prctile_matlab_default(x: np.ndarray, q: float) -> float:
    """
    MATLAB prctile default parity.
    MATLAB default "midpoint"/legacy "exact" aligns with Hazen quantiles.
    """
    arr = np.asarray(x, dtype=float).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan

    # Prefer NumPy's built-in Hazen implementation when available.
    try:
        return float(np.percentile(arr, q, method="hazen"))
    except TypeError:
        # NumPy fallback for versions without "method": manual Hazen quantile.
        arr = np.sort(arr)
        n = arr.size
        p = float(q) / 100.0
        p = min(max(p, 0.0), 1.0)
        h = n * p + 0.5
        if h <= 1.0:
            return float(arr[0])
        if h >= n:
            return float(arr[-1])
        lo = int(np.floor(h))
        hi = int(np.ceil(h))
        g = h - lo
        x_lo = arr[lo - 1]
        x_hi = arr[hi - 1]
        return float((1.0 - g) * x_lo + g * x_hi)


def _unique_stable(time: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """MATLAB unique(x,'stable') parity: keep first occurrence, preserve order."""
    arr = np.asarray(time)
    _, idx = np.unique(arr, return_index=True)
    idx = np.sort(idx)
    return arr[idx], idx


def _matlab_colon_grid(start: float, stop: float, step: float) -> np.ndarray:
    """Return MATLAB-style `start:step:stop` values without overshooting stop."""
    if not (np.isfinite(start) and np.isfinite(stop) and np.isfinite(step)) or step <= 0:
        return np.array([], dtype=float)
    if stop < start:
        return np.array([], dtype=float)
    n_steps = int(np.floor(((stop - start) / step) + 1e-12))
    return start + step * np.arange(n_steps + 1, dtype=float)


def _find_peaks_matlab(
    y: np.ndarray,
    distance: int | None = None,
    prominence: float | None = None,
) -> tuple[np.ndarray, dict]:
    """
    MATLAB findpeaks parity for flat peaks:
      - MATLAB chooses the lowest index of a plateau
      - SciPy defaults to midpoint for plateaus
    """
    peaks, props = find_peaks(y, distance=distance, prominence=prominence, plateau_size=1)
    if peaks.size == 0:
        return peaks, props

    left_edges = props.get("left_edges")
    if left_edges is not None and len(left_edges) == len(peaks):
        peaks = np.asarray(left_edges, dtype=int)
        # Recompute prominence for adjusted indices so downstream thresholds match.
        proms, left_bases, right_bases = peak_prominences(y, peaks)
        props["prominences"] = proms
        props["left_bases"] = left_bases
        props["right_bases"] = right_bases

    return peaks, props


def _auto_find_peaks(sig: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    MATLAB-style robust peak picking (autoFindPeaks in PRIZM_20260127.m).
    Returns (peak_values, peak_times, peak_indices).
    """
    y = np.asarray(sig, dtype=float).reshape(-1)
    tt = np.asarray(t, dtype=float).reshape(-1)
    if y.size == 0 or tt.size != y.size:
        return np.array([]), np.array([]), np.array([], dtype=int)
    if np.all(~np.isfinite(y)):
        return np.array([]), np.array([]), np.array([], dtype=int)

    y_f = y[np.isfinite(y)]
    if y_f.size == 0 or np.allclose(y_f, 0):
        return np.array([]), np.array([]), np.array([], dtype=int)

    p95 = _prctile_matlab_default(y_f, 95.0)
    p5 = _prctile_matlab_default(y_f, 5.0)
    sig_range = float(p95 - p5)
    if not np.isfinite(sig_range) or sig_range <= 0:
        sig_range = float(np.nanmax(y_f) - np.nanmin(y_f))
    if not np.isfinite(sig_range) or sig_range <= 0:
        return np.array([]), np.array([]), np.array([], dtype=int)

    dy = np.diff(y_f)
    if dy.size:
        noise = 1.4826 * np.nanmedian(np.abs(dy - np.nanmedian(dy))) / np.sqrt(2.0)
    else:
        noise = np.nan
    if not np.isfinite(noise) or noise <= 0:
        noise = 1.4826 * np.nanmedian(np.abs(y_f - np.nanmedian(y_f)))
    if not np.isfinite(noise):
        noise = 0.0

    dt = np.diff(tt)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    dt_med = float(np.median(dt)) if dt.size else 0.01
    min_dist_samples = max(1, int(round(0.2 / max(dt_med, 1e-9))))

    prom1 = max(0.15 * sig_range, 3.0 * noise)
    idx1, props1 = _find_peaks_matlab(y, distance=min_dist_samples, prominence=prom1)
    p1 = y[idx1]
    t1 = tt[idx1]

    if p1.size < 3:
        return p1, t1, idx1

    prom_vals = props1.get("prominences", np.array([], dtype=float))
    prom_floor = 0.5 * float(np.nanmedian(prom_vals)) if prom_vals.size else 0.0
    border_guess = _prctile_matlab_default(y_f, 20.0)
    prom_dyn = (float(np.nanmean(p1)) - border_guess) * 0.20
    prom2 = max(0.06 * sig_range, prom_dyn, prom_floor, 5.0 * noise)

    idx2, _ = _find_peaks_matlab(y, distance=min_dist_samples, prominence=prom2)
    if idx2.size == 0:
        return p1, t1, idx1
    return y[idx2], tt[idx2], idx2


def _troughs_between_peaks(
    sig: np.ndarray, t: np.ndarray, peak_times: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return minima values/times between consecutive peaks (including tail)."""
    y = np.asarray(sig, dtype=float).reshape(-1)
    tt = np.asarray(t, dtype=float).reshape(-1)
    locs = np.asarray(peak_times, dtype=float).reshape(-1)
    if locs.size == 0:
        return np.array([]), np.array([])

    border_vals = np.full(locs.shape, np.nan, dtype=float)
    border_times = np.full(locs.shape, np.nan, dtype=float)

    for i in range(locs.size):
        if i < locs.size - 1:
            idx = np.where((tt >= locs[i]) & (tt <= locs[i + 1]))[0]
        else:
            idx = np.where(tt >= locs[i])[0]
        if idx.size == 0:
            continue
        seg = y[idx]
        if seg.size == 0 or np.all(~np.isfinite(seg)):
            continue
        k = int(np.nanargmin(seg))
        border_vals[i] = float(seg[k])
        border_times[i] = float(tt[idx[k]])
    return border_vals, border_times


def _calc_fraction_from_peak_and_border(
    peaks: np.ndarray, borders: np.ndarray
) -> tuple[np.ndarray, float, float]:
    n = min(len(peaks), len(borders))
    if n == 0:
        return np.array([]), np.nan, np.nan
    p = np.asarray(peaks[:n], dtype=float)
    b = np.asarray(borders[:n], dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        vec = (p - b) / p * 100.0
    return vec, float(np.nanmean(vec)), _matlab_nanstd(vec)


def _hr_and_interval(locs: np.ndarray) -> tuple[float, float, float]:
    locs = np.asarray(locs, dtype=float)
    locs = locs[np.isfinite(locs)]
    if locs.size < 2:
        return np.nan, np.nan, np.nan
    intervals = np.diff(locs)
    mean_int = float(np.nanmean(intervals))
    int_sd = _matlab_nanstd(intervals)
    int_cv = float(int_sd / mean_int) if np.isfinite(mean_int) and mean_int > 0 else np.nan
    duration = locs[-1] - locs[0]
    hr = float((locs.size - 1) * 60.0 / duration) if duration > 0 else np.nan
    return hr, int_sd, int_cv


def _contract_relax_speed(
    peaks: np.ndarray,
    peak_times: np.ndarray,
    borders: np.ndarray,
    border_times: np.ndarray,
) -> tuple[float, float]:
    """Average contraction/relaxation slope in MATLAB style."""
    n = min(len(peaks), len(peak_times), len(borders), len(border_times))
    if n < 2:
        return np.nan, np.nan

    p = np.asarray(peaks[:n], dtype=float)
    pt = np.asarray(peak_times[:n], dtype=float)
    b = np.asarray(borders[:n], dtype=float)
    bt = np.asarray(border_times[:n], dtype=float)

    cs = np.full(n, np.nan, dtype=float)
    rs = np.full(n, np.nan, dtype=float)
    for i in range(n):
        dtc = bt[i] - pt[i]
        if np.isfinite(dtc) and dtc > 0:
            cs[i] = (p[i] - b[i]) / dtc
        if i < n - 1:
            dtr = pt[i + 1] - bt[i]
            if np.isfinite(dtr) and dtr > 0:
                rs[i] = (p[i + 1] - b[i]) / dtr
    return float(np.nanmean(cs)), float(np.nanmean(rs))


def _major_minor_ratio_at_ed(
    time: np.ndarray, major: np.ndarray, minor: np.ndarray, vent_peak_times: np.ndarray
) -> float:
    if len(vent_peak_times) == 0:
        return np.nan
    t = np.asarray(time, dtype=float)
    maj = np.asarray(major, dtype=float)
    mi = np.asarray(minor, dtype=float)
    vals = []
    for p in np.asarray(vent_peak_times, dtype=float):
        if not np.isfinite(p):
            continue
        idx = int(np.nanargmin(np.abs(t - p)))
        if idx < 0 or idx >= len(t):
            continue
        if np.isfinite(maj[idx]) and np.isfinite(mi[idx]) and mi[idx] > 0:
            vals.append(float(maj[idx] / mi[idx]))
    if not vals:
        return np.nan
    return float(np.nanmean(vals))


def _pair_nearest_peaks_monotonic(
    t_v: np.ndarray, t_a: np.ndarray, max_win: float
) -> np.ndarray:
    """
    Monotonic one-to-one nearest peak pairing with a max time window.
    Mirrors MATLAB pairNearestPeaksMonotonic.
    """
    tv = np.asarray(t_v, dtype=float).reshape(-1)
    ta = np.asarray(t_a, dtype=float).reshape(-1)
    if tv.size == 0 or ta.size == 0 or not np.isfinite(max_win) or max_win <= 0:
        return np.array([])

    diffs = np.full(tv.shape, np.nan, dtype=float)
    a_start = 0
    for i, tv_i in enumerate(tv):
        if a_start >= ta.size:
            break
        lo, hi = tv_i - max_win, tv_i + max_win
        cand = np.where((ta[a_start:] >= lo) & (ta[a_start:] <= hi))[0]
        if cand.size == 0:
            continue
        cand = cand + a_start
        d = np.abs(ta[cand] - tv_i)
        k = int(np.argmin(d))
        diffs[i] = float(d[k])
        a_start = int(cand[k] + 1)
    return diffs[np.isfinite(diffs)]


# ===========================================================
# Functional analysis (findpeaks-like visuals with extents)
# ===========================================================
def compute_functional_statistics(
    seg_df: pd.DataFrame,
    video_name: str,
    video_out: str,
    dt_interp: float = 0.01,
    dpi: int = 300,
    marker_offset_frac: float = 0.03,
    interp_method: str = "pchip",
    apply_time_stitch: bool = True,
    gap_factor: float = 5.0,
    expected_duration_sec: float | None = None,
    matlab_file_key: Optional[str] = None,
):
    """
    MATLAB-aligned functional analysis with additional 2026 metrics:
    interval CV, systolic/diastolic durations, SV/CO index, major/minor ED
    ratio, contractility/relaxation speeds, and diastolic A/V ratio.
    """
    os.makedirs(video_out, exist_ok=True)
    results_dir = os.path.join(video_out, "results")
    os.makedirs(results_dir, exist_ok=True)

    def _pad_to_len(vec, L):
        v = np.asarray(vec).reshape(-1)
        if len(v) >= L:
            return v[:L]
        out = np.full((L,), np.nan, dtype=float)
        out[: len(v)] = v
        return out

    def _interp(t_src, y_src, t_dst, method="pchip"):
        t_src = np.asarray(t_src, dtype=float)
        y_src = np.asarray(y_src, dtype=float)
        good = np.isfinite(t_src) & np.isfinite(y_src)
        if np.count_nonzero(good) < 2:
            return np.full_like(t_dst, np.nan, dtype=float)
        t_use = t_src[good]
        y_use = y_src[good]
        if np.nanmax(t_use) <= np.nanmin(t_use):
            return np.full_like(t_dst, np.nan, dtype=float)
        if str(method).lower() == "spline":
            fn = CubicSpline(t_use, y_use)
        else:
            fn = PchipInterpolator(t_use, y_use)
        return fn(t_dst)

    # --------------------------
    # Pull and clean series
    # --------------------------
    seg_df = seg_df.sort_values("RelativeTime").reset_index(drop=True)
    time_raw = pd.to_numeric(seg_df.get("RelativeTime"), errors="coerce").to_numpy(dtype=float)

    if "RealVentricularCavitySize" in seg_df.columns:
        vent_raw = pd.to_numeric(seg_df["RealVentricularCavitySize"], errors="coerce").to_numpy(dtype=float)
    else:
        vent_raw = pd.to_numeric(seg_df.get("VentricularCavitySize"), errors="coerce").to_numpy(dtype=float)
    if "RealAtriumCavitySize" in seg_df.columns:
        atr_raw = pd.to_numeric(seg_df["RealAtriumCavitySize"], errors="coerce").to_numpy(dtype=float)
    else:
        atr_raw = pd.to_numeric(seg_df.get("AtriumCavitySize"), errors="coerce").to_numpy(dtype=float)

    mc = pd.to_numeric(seg_df.get("minorAxis_center"), errors="coerce").to_numpy(dtype=float)
    mu = pd.to_numeric(seg_df.get("minorAxis_upper"), errors="coerce").to_numpy(dtype=float)
    ml = pd.to_numeric(seg_df.get("minorAxis_lower"), errors="coerce").to_numpy(dtype=float)
    maj_raw = pd.to_numeric(seg_df.get("majorAxisLength"), errors="coerce").to_numpy(dtype=float)

    minor_triplet = np.stack([mc, mu, ml], axis=1)
    minor_raw = np.full((len(seg_df),), np.nan, dtype=float)
    valid_minor = np.all(np.isfinite(minor_triplet), axis=1)
    minor_raw[valid_minor] = np.min(minor_triplet[valid_minor], axis=1)

    if apply_time_stitch:
        time_fix = fix_relative_time_stitch(
            time_raw, gap_factor=gap_factor, expected_duration_sec=expected_duration_sec
        )
    else:
        time_fix = np.asarray(time_raw, dtype=float)

    good = np.isfinite(time_fix) & np.isfinite(vent_raw) & np.isfinite(atr_raw)
    time = time_fix[good]
    vent = vent_raw[good]
    atr = atr_raw[good]
    minor = minor_raw[good]
    major = maj_raw[good]

    if time.size < 4:
        empty = pd.DataFrame(columns=["locs", "interval", "peaks", "Border", "vEF", "vFS"])
        fig_v, fig_vfs, fig_a, fig_va = Figure(), Figure(), Figure(), Figure()
        return empty, empty.copy(), pd.DataFrame(columns=["locs"]), fig_v, fig_vfs, fig_a, fig_va

    time, uniq_idx = _unique_stable(time)
    vent = vent[uniq_idx]
    atr = atr[uniq_idx]
    minor = minor[uniq_idx]
    major = major[uniq_idx]

    t0 = float(np.nanmin(time))
    t1 = float(np.nanmax(time))
    if not np.isfinite(t0) or not np.isfinite(t1) or t1 <= t0:
        empty = pd.DataFrame(columns=["locs", "interval", "peaks", "Border", "vEF", "vFS"])
        fig_v, fig_vfs, fig_a, fig_va = Figure(), Figure(), Figure(), Figure()
        return empty, empty.copy(), pd.DataFrame(columns=["locs"]), fig_v, fig_vfs, fig_a, fig_va

    t = _matlab_colon_grid(t0, t1, dt_interp)
    y_vent = _interp(time, vent, t, method=interp_method)
    y_atr = _interp(time, atr, t, method=interp_method)
    # MATLAB parity: minor-axis interpolation is only allowed when at least
    # 4 finite samples exist (runFunctionalAnalysis: nnz(gM) >= 4).
    g_minor = np.isfinite(minor) & np.isfinite(time)
    if np.count_nonzero(g_minor) >= 4:
        y_minor = _interp(time[g_minor], minor[g_minor], t, method=interp_method)
    else:
        y_minor = np.full_like(t, np.nan, dtype=float)
    dt = (t[1] - t[0]) if len(t) > 1 else dt_interp

    # --------------------------
    # Peak/trough extraction
    # --------------------------
    pks_v, locs_v, idx_v = _auto_find_peaks(y_vent, t)
    border_v, border_tv = _troughs_between_peaks(y_vent, t, locs_v)
    EF_vec, EF_mean, EF_sd = _calc_fraction_from_peak_and_border(pks_v, border_v)
    V_HR, intSD, intCV = _hr_and_interval(locs_v)

    pks_a, locs_a, idx_a = _auto_find_peaks(y_atr, t)
    border_a, border_at = _troughs_between_peaks(y_atr, t, locs_a)
    A_HR, _, _ = _hr_and_interval(locs_a)

    pks_m, locs_m, idx_m = _auto_find_peaks(y_minor, t)
    border_m, border_mt = _troughs_between_peaks(y_minor, t, locs_m)
    FS_vec, FS_mean, FS_sd = _calc_fraction_from_peak_and_border(pks_m, border_m)

    # --------------------------
    # MATLAB 2026 added metrics
    # --------------------------
    systolic_mean = np.nan
    diastolic_mean = np.nan
    systolic_fraction = np.nan
    if len(locs_v) and len(border_tv):
        nbd = min(len(locs_v), len(border_tv))
        sdur = np.asarray(border_tv[:nbd] - locs_v[:nbd], dtype=float)
        sdur[~np.isfinite(sdur) | (sdur <= 0)] = np.nan
        systolic_mean = float(np.nanmean(sdur))
        if nbd >= 2:
            ddur = np.asarray(locs_v[1:nbd] - border_tv[: nbd - 1], dtype=float)
            ddur[~np.isfinite(ddur) | (ddur <= 0)] = np.nan
            diastolic_mean = float(np.nanmean(ddur))
            intervals = np.asarray(np.diff(locs_v[:nbd]), dtype=float)
            intervals[~np.isfinite(intervals) | (intervals <= 0)] = np.nan
            frac = sdur[: nbd - 1] / intervals
            frac[~np.isfinite(frac) | (frac < 0) | (frac > 1.2)] = np.nan
            systolic_fraction = float(np.nanmean(frac))

    n_sv = min(len(pks_v), len(border_v))
    sv_index = (np.asarray(pks_v[:n_sv]) - np.asarray(border_v[:n_sv])) if n_sv else np.array([])
    sv_index_mean = float(np.nanmean(sv_index)) if sv_index.size else np.nan
    co_index_mean = (
        float(sv_index_mean * (V_HR / 60.0))
        if np.isfinite(sv_index_mean) and np.isfinite(V_HR)
        else np.nan
    )

    c_speed, r_speed = _contract_relax_speed(pks_v, locs_v, border_v, border_tv)
    diastolic_ratio = (
        float(np.nanmean(pks_a) / np.nanmean(pks_v))
        if len(pks_a) and len(pks_v) and np.isfinite(np.nanmean(pks_v)) and np.nanmean(pks_v) != 0
        else np.nan
    )
    mm_ratio = _major_minor_ratio_at_ed(time, major, minor, locs_v)

    if len(locs_v) >= 3:
        peak_intervals = np.diff(locs_v)
        interval_col = np.concatenate(([_matlab_nanstd(peak_intervals)], peak_intervals))
    else:
        interval_col = np.array([], dtype=float)

    # --------------------------
    # DataFrames + save Excel
    # --------------------------
    L = max(
        len(locs_v),
        len(interval_col),
        len(pks_v),
        len(border_v),
        len(EF_vec),
        len(FS_vec),
        1,
    )

    summary_row = np.full((L,), np.nan, dtype=float)
    summary_row[0] = 1.0

    v_df = pd.DataFrame(
        {
            "locs": _pad_to_len(locs_v, L),
            "interval": _pad_to_len(interval_col, L),
            "peaks": _pad_to_len(pks_v, L),
            "Border": _pad_to_len(border_v, L),
            "vEF": _pad_to_len(EF_vec, L),
            "vFS": _pad_to_len(FS_vec, L),
            "SV_index": _pad_to_len(sv_index, L),
            "Interval_CV": np.where(np.isfinite(summary_row), intCV, np.nan),
            "SystolicDuration_mean": np.where(np.isfinite(summary_row), systolic_mean, np.nan),
            "DiastolicDuration_mean": np.where(np.isfinite(summary_row), diastolic_mean, np.nan),
            "SystolicFraction": np.where(np.isfinite(summary_row), systolic_fraction, np.nan),
            "CO_index_mean": np.where(np.isfinite(summary_row), co_index_mean, np.nan),
            "Diastolic_AtoV_ratio": np.where(np.isfinite(summary_row), diastolic_ratio, np.nan),
            "MajorMinor_ratio_ED": np.where(np.isfinite(summary_row), mm_ratio, np.nan),
            "ContractilitySpeed": np.where(np.isfinite(summary_row), c_speed, np.nan),
            "RelaxationSpeed": np.where(np.isfinite(summary_row), r_speed, np.nan),
        }
    )

    vFS_df = pd.DataFrame({"locs": _pad_to_len(locs_m, len(FS_vec)), "vFS": FS_vec})
    a_df = pd.DataFrame(
        {
            "locs": _pad_to_len(locs_a, max(len(locs_a), len(border_a))),
            "peaks": _pad_to_len(pks_a, max(len(locs_a), len(border_a))),
            "Border": _pad_to_len(border_a, max(len(locs_a), len(border_a))),
        }
    )

    v_path_xlsx = os.path.join(results_dir, f"{video_name}_result_ventricle.xlsx")
    a_path_xlsx = os.path.join(results_dir, f"{video_name}_result_atrium.xlsx")
    v_df.to_excel(v_path_xlsx, index=False)
    a_df.to_excel(a_path_xlsx, index=False)
    matlab_key = str(matlab_file_key or "")
    if matlab_key and matlab_key != video_name:
        v_df.to_excel(os.path.join(results_dir, f"{matlab_key}_result_ventricle.xlsx"), index=False)
        a_df.to_excel(os.path.join(results_dir, f"{matlab_key}_result_atrium.xlsx"), index=False)

    # --------------------------
    # Plotting
    # --------------------------
    def _idx_to_time(idx_float):
        return t[0] + idx_float * dt

    def _y_for_limits(y, idx, marker_y):
        if len(idx) and marker_y is not None:
            y_ext_max = np.nanmax([np.nanmax(y), np.nanmax(marker_y)])
            y_ext_min = np.nanmin(y) if np.isfinite(y).any() else 0.0
            return y_ext_min, y_ext_max
        return (
            np.nanmin(y) if np.isfinite(y).any() else 0.0,
            np.nanmax(y) if np.isfinite(y).any() else 0.0,
        )

    def _set_axes_limits(ax, ymin, ymax, minor=False):
        ax.set_xlim(0.0, 30.05)
        if not np.isfinite(ymax) or ymax == 0.0:
            ax.set_ylim(0, 50 if minor else 5000)
        else:
            ax.set_ylim(ymin / 1.2, ymax * 1.5)
        ax.grid(True, which="both", linewidth=GRID_LW, alpha=GRID_ALPHA)

    def _peak_width_prom(y, idx):
        if len(idx):
            widths, width_heights, left_ips, right_ips = peak_widths(y, idx, rel_height=0.5)
            proms, left_bases, right_bases = peak_prominences(y, idx)
            return widths, width_heights, left_ips, right_ips, proms, left_bases, right_bases
        return (
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([]),
        )

    def _make_peak_figure(
        t_sig,
        y_sig,
        idx,
        left_ips,
        right_ips,
        width_heights,
        proms,
        left_bases,
        right_bases,
        title,
        ylabel,
        minor=False,
    ):
        fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DISPLAY_DPI)
        if len(idx):
            max_peak_val = float(np.nanmax(y_sig[idx]))
        else:
            max_peak_val = float(np.nanmax(y_sig)) if np.isfinite(y_sig).any() else 0.0
        marker_off = marker_offset_frac * max_peak_val if np.isfinite(max_peak_val) else 0.0
        marker_y = (y_sig[idx] + marker_off) if len(idx) else None

        sig_line, = ax.plot(t_sig, y_sig, lw=LW_SIGNAL, color=COLOR_V_SIGNAL, label="signal")
        if len(idx):
            ax.plot(
                t_sig[idx],
                marker_y,
                marker=PEAK_MARK,
                linestyle="None",
                ms=PEAK_MS,
                mfc=COLOR_V_SIGNAL,
                mec=COLOR_V_SIGNAL,
                label="peak",
            )
        if len(left_ips):
            for li, ri, hi in zip(left_ips, right_ips, width_heights):
                ax.hlines(hi, _idx_to_time(li), _idx_to_time(ri), colors=COLOR_WIDTH, linestyles="-", lw=LW_ANNOT)
        if len(idx):
            height_labeled = False
            border_labeled = False
            for k, pk in enumerate(idx):
                if k < len(proms):
                    lb, rb = int(left_bases[k]), int(right_bases[k])
                    base_y = max(y_sig[lb], y_sig[rb])
                else:
                    base_y = np.nan
                xpk = t_sig[pk]
                if np.isfinite(base_y):
                    ax.vlines(
                        xpk,
                        0.0,
                        base_y,
                        colors=COLOR_BORDER,
                        lw=LW_ANNOT,
                        label=None if border_labeled else "border",
                    )
                    border_labeled = True
                ax.vlines(
                    xpk,
                    0.0,
                    y_sig[pk],
                    colors=COLOR_HEIGHT,
                    lw=LW_ANNOT,
                    label=None if height_labeled else "height",
                )
                height_labeled = True

        ymin, ymax = _y_for_limits(y_sig, idx, marker_y)
        _set_axes_limits(ax, ymin, ymax, minor=minor)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)

        handles, labels = [sig_line], ["signal"]
        if len(idx):
            handles.append(
                ax.plot([], [], marker=PEAK_MARK, linestyle="None", ms=PEAK_MS, mfc=COLOR_V_SIGNAL, mec=COLOR_V_SIGNAL)[0]
            )
            labels.append("peak")
            handles.append(ax.plot([], [], color=COLOR_HEIGHT, lw=LW_ANNOT)[0])
            labels.append("height")
            handles.append(ax.plot([], [], color=COLOR_WIDTH, lw=LW_ANNOT)[0])
            labels.append("width (half-height)")
            handles.append(ax.plot([], [], color=COLOR_BORDER, lw=LW_ANNOT)[0])
            labels.append("border")
        ax.legend(handles, labels, loc="upper right", frameon=True)
        return fig

    (
        _,
        width_heights_v,
        left_ips_v,
        right_ips_v,
        proms_v,
        left_bases_v,
        right_bases_v,
    ) = _peak_width_prom(y_vent, idx_v)
    (
        _,
        width_heights_m,
        left_ips_m,
        right_ips_m,
        proms_m,
        left_bases_m,
        right_bases_m,
    ) = _peak_width_prom(y_minor, idx_m)
    (
        _,
        width_heights_a,
        left_ips_a,
        right_ips_a,
        proms_a,
        left_bases_a,
        right_bases_a,
    ) = _peak_width_prom(y_atr, idx_a)

    title_v = video_name.replace("_", "-") + " - Ventricular Cavity Peaks"
    title_vfs = video_name.replace("_", "-") + " - Minor Axis Peaks"
    title_a = video_name.replace("_", "-") + " - Atrium Cavity Peaks"
    fig_v = _make_peak_figure(
        t,
        y_vent,
        idx_v,
        left_ips_v,
        right_ips_v,
        width_heights_v,
        proms_v,
        left_bases_v,
        right_bases_v,
        title_v,
        "Ventricular Cavity Size",
        minor=False,
    )
    fig_vfs = _make_peak_figure(
        t,
        y_minor,
        idx_m,
        left_ips_m,
        right_ips_m,
        width_heights_m,
        proms_m,
        left_bases_m,
        right_bases_m,
        title_vfs,
        "Ventricular Minor Axis Length",
        minor=True,
    )
    fig_a = _make_peak_figure(
        t,
        y_atr,
        idx_a,
        left_ips_a,
        right_ips_a,
        width_heights_a,
        proms_a,
        left_bases_a,
        right_bases_a,
        title_a,
        "Atrium Cavity Size",
        minor=False,
    )

    _save_svg(fig_v, os.path.join(results_dir, f"{video_name}_find_selected_peaks_ventricle.svg"), dpi=dpi)
    _save_svg(fig_vfs, os.path.join(results_dir, f"{video_name}_minoraxis_peaks.svg"), dpi=dpi)
    _save_svg(fig_a, os.path.join(results_dir, f"{video_name}_find_selected_peaks_atrium.svg"), dpi=dpi)
    plt.close(fig_v)
    plt.close(fig_vfs)
    plt.close(fig_a)

    fig_va, ax_va = plt.subplots(figsize=FIGSIZE, dpi=DISPLAY_DPI)

    def _norm(y):
        y = np.asarray(y, dtype=float)
        if not np.isfinite(y).any():
            return np.zeros_like(y)
        lo, hi = np.nanmin(y), np.nanmax(y)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi == lo:
            return np.zeros_like(y)
        return (y - lo) / (hi - lo)

    ax_va.plot(t, _norm(y_vent), lw=LW_SIGNAL, color=COLOR_OVERLAY_V, label="Ventricle (norm.)")
    ax_va.plot(t, _norm(y_atr), lw=LW_SIGNAL, color=COLOR_OVERLAY_A, label="Atrium (norm.)")
    ax_va.set_xlabel("Time (s)")
    ax_va.set_ylabel("Normalized amplitude")
    ax_va.set_title(video_name.replace("_", "-") + " - Ventricle vs Atrium (normalized)")
    ax_va.set_xlim(0.0, 30.05)
    ax_va.margins(x=0.01, y=0.0)
    ax_va.set_ylim(-0.05, 1.05)
    ax_va.grid(True, which="both", linewidth=GRID_LW, alpha=GRID_ALPHA)
    ax_va.legend(loc="upper right", frameon=True, borderaxespad=0.6)
    _save_svg(fig_va, os.path.join(results_dir, f"{video_name}_ventricle_atrium_overlay.svg"), dpi=dpi)
    plt.close(fig_va)

    v_df.attrs["summary_metrics"] = {
        "V_HR_bpm": V_HR,
        "A_HR_bpm": A_HR,
        "Interval_SD_s": intSD,
        "Interval_CV": intCV,
        "SystolicDuration_mean": systolic_mean,
        "DiastolicDuration_mean": diastolic_mean,
        "SystolicFraction": systolic_fraction,
        "EF_mean": EF_mean,
        "EF_SD": EF_sd,
        "FS_mean": FS_mean,
        "FS_SD": FS_sd,
        "V_ED_mean": float(np.nanmean(pks_v)) if len(pks_v) else np.nan,
        "V_ES_mean": float(np.nanmean(border_v)) if len(border_v) else np.nan,
        "V_ED_SD": _matlab_nanstd(pks_v),
        "V_ES_SD": _matlab_nanstd(border_v),
        "SV_index_mean": sv_index_mean,
        "CO_index_mean": co_index_mean,
        "Diastolic_AtoV_ratio": diastolic_ratio,
        "MajorMinor_ratio_ED": mm_ratio,
        "ContractilitySpeed": c_speed,
        "RelaxationSpeed": r_speed,
        "A_ED_mean": float(np.nanmean(pks_a)) if len(pks_a) else np.nan,
        "A_ES_mean": float(np.nanmean(border_a)) if len(border_a) else np.nan,
        "A_ED_SD": _matlab_nanstd(pks_a),
        "A_ES_SD": _matlab_nanstd(border_a),
        "A_ED_ES_Diff_mean": float(np.nanmean(np.asarray(pks_a[: min(len(pks_a), len(border_a))]) - np.asarray(border_a[: min(len(pks_a), len(border_a))])))
        if len(pks_a) and len(border_a)
        else np.nan,
    }

    return v_df, vFS_df, a_df, fig_v, fig_vfs, fig_a, fig_va


# # ===========================================================
# # Synchronization analysis (xcorr + Hilbert + peak timing)
# # ===========================================================

def _zscore_sample(x: np.ndarray) -> np.ndarray:
    """Z-score with sample std (ddof=1), matching MATLAB std default."""
    mu = np.mean(x)
    sd = np.std(x, ddof=1)
    if not np.isfinite(sd) or sd == 0:
        return np.zeros_like(x, dtype=float)
    return (x - mu) / sd


def _xcorr_coeff(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Cross-correlation with MATLAB 'coeff' normalization.
    Inputs must be same-length 1D arrays.
    Returns (corr, lags) where lags run from -(N-1)..(N-1).
    """
    # Full correlation (same definition/sign convention as MATLAB's xcorr)
    raw = correlate(x, y, mode="full", method="auto")
    lags = correlation_lags(x.size, y.size, mode="full")
    # MATLAB 'coeff' divides by sqrt(sum(x.^2) * sum(y.^2))
    denom = np.sqrt(np.sum(x * x) * np.sum(y * y))
    if not np.isfinite(denom) or denom <= 0:
        corr = np.full(raw.shape, np.nan, dtype=float)
    else:
        corr = raw / denom
    return corr, lags


# ----------------------- main function -----------------------
def compute_synchronize_analysis(
    seg_df: pd.DataFrame,
    video_name: str,
    video_out: str,
    num_points: int = 1000,
    dpi: int = 300,
    interp_method: str = "pchip",
    apply_time_stitch: bool = True,
    gap_factor: float = 5.0,
    expected_duration_sec: float | None = None,
    matlab_file_key: Optional[str] = None,
):
    """
    MATLAB-aligned ventricle/atrium synchrony analysis with PLV and AV-delay SD.
    """
    os.makedirs(video_out, exist_ok=True)
    sync_dir = os.path.join(video_out, "synchronize")
    os.makedirs(sync_dir, exist_ok=True)

    def _interp_sync(t_src, y_src, t_dst, method="pchip"):
        t_src = np.asarray(t_src, dtype=float)
        y_src = np.asarray(y_src, dtype=float)
        good = np.isfinite(t_src) & np.isfinite(y_src)
        if np.count_nonzero(good) < 2:
            return np.full_like(t_dst, np.nan, dtype=float)
        tx = t_src[good]
        yx = y_src[good]
        if np.nanmax(tx) <= np.nanmin(tx):
            return np.full_like(t_dst, np.nan, dtype=float)
        if str(method).lower() == "spline":
            fn = CubicSpline(tx, yx)
        else:
            fn = PchipInterpolator(tx, yx)
        return fn(t_dst)

    time_raw = pd.to_numeric(seg_df.get("RelativeTime"), errors="coerce").to_numpy(dtype=float)
    if "RealVentricularCavitySize" in seg_df.columns:
        vent_raw = pd.to_numeric(seg_df["RealVentricularCavitySize"], errors="coerce").to_numpy(dtype=float)
    else:
        vent_raw = pd.to_numeric(seg_df.get("VentricularCavitySize"), errors="coerce").to_numpy(dtype=float)
    if "RealAtriumCavitySize" in seg_df.columns:
        atr_raw = pd.to_numeric(seg_df["RealAtriumCavitySize"], errors="coerce").to_numpy(dtype=float)
    else:
        atr_raw = pd.to_numeric(seg_df.get("AtriumCavitySize"), errors="coerce").to_numpy(dtype=float)

    if apply_time_stitch:
        time_fix = fix_relative_time_stitch(
            time_raw, gap_factor=gap_factor, expected_duration_sec=expected_duration_sec
        )
    else:
        time_fix = np.asarray(time_raw, dtype=float)

    good = np.isfinite(time_fix) & np.isfinite(vent_raw) & np.isfinite(atr_raw)
    time = time_fix[good]
    vent = vent_raw[good]
    atr = atr_raw[good]

    if time.size < 4:
        sync_df = pd.DataFrame(
            [
                {
                    "FileName": f"{video_name}.xlsx",
                    "MaxCorrelationLag": np.nan,
                    "MaxCorrLag_s": np.nan,
                    "Cross_Correlation_Based_Synchrony_Index": np.nan,
                    "CrossCorrCoeff": np.nan,
                    "Phase_Synchrony_Index": np.nan,
                    "PhaseSynchronyIndex": np.nan,
                    "PLV": np.nan,
                    "PearsonCorrelation": np.nan,
                    "PearsonCorr": np.nan,
                    "Standard_deviation_of_peak_times": np.nan,
                    "PeakTimeDiff_SD": np.nan,
                    "AV_Delay_Mean": np.nan,
                    "AV_Delay_SD": np.nan,
                }
            ]
        )
        out_xlsx = os.path.join(sync_dir, "SynchronizationResults.xlsx")
        sync_df.to_excel(out_xlsx, index=False)
        return sync_df, Figure(), Figure()

    time, uniq_idx = _unique_stable(time)
    vent = vent[uniq_idx]
    atr = atr[uniq_idx]

    t_min = float(np.nanmin(time))
    t_max = float(np.nanmax(time))
    if not np.isfinite(t_min) or not np.isfinite(t_max) or t_max <= t_min:
        sync_df = pd.DataFrame(
            [{"FileName": f"{video_name}.xlsx", "MaxCorrelationLag": np.nan}]
        )
        out_xlsx = os.path.join(sync_dir, "SynchronizationResults.xlsx")
        sync_df.to_excel(out_xlsx, index=False)
        return sync_df, Figure(), Figure()

    time_fine = np.linspace(t_min, t_max, int(max(10, num_points)))
    vent_spline = _interp_sync(time, vent, time_fine, method=interp_method)
    atr_spline = _interp_sync(time, atr, time_fine, method=interp_method)

    vent_norm = _zscore_sample(vent_spline)
    atr_norm = _zscore_sample(atr_spline)

    correlation, lags = _xcorr_coeff(vent_norm, atr_norm)
    max_idx = int(np.argmax(correlation))
    max_corr_lag = int(lags[max_idx])
    max_corr_value = float(correlation[max_idx])
    dt = (time_fine[1] - time_fine[0]) if len(time_fine) > 1 else np.nan
    lag_sec = float(max_corr_lag * dt) if np.isfinite(dt) else np.nan

    if (
        np.isfinite(np.nanstd(vent_spline))
        and np.isfinite(np.nanstd(atr_spline))
        and np.nanstd(vent_spline) > 0
        and np.nanstd(atr_spline) > 0
    ):
        pearson_corr = float(pearsonr(vent_spline, atr_spline)[0])
    else:
        pearson_corr = np.nan

    # MATLAB parity: runSynchronyAnalysis uses plain findpeaks(vN, tf) and
    # findpeaks(aN, tf) with default settings (no autoFindPeaks here).
    v_peak_idx, _ = _find_peaks_matlab(np.asarray(vent_norm, dtype=float))
    a_peak_idx, _ = _find_peaks_matlab(np.asarray(atr_norm, dtype=float))
    vent_peak_times = (
        np.asarray(time_fine[v_peak_idx], dtype=float).reshape(-1)
        if v_peak_idx.size
        else np.array([], dtype=float)
    )
    atr_peak_times = (
        np.asarray(time_fine[a_peak_idx], dtype=float).reshape(-1)
        if a_peak_idx.size
        else np.array([], dtype=float)
    )
    v_rr = np.diff(vent_peak_times)
    a_rr = np.diff(atr_peak_times)
    period = float(np.nanmedian(v_rr)) if v_rr.size else np.nan
    if not np.isfinite(period) or period <= 0:
        period = float(np.nanmedian(a_rr)) if a_rr.size else np.nan
    if np.isfinite(period) and period > 0:
        peak_time_diffs = _pair_nearest_peaks_monotonic(
            vent_peak_times, atr_peak_times, 0.45 * period
        )
        std_peak_diff = _matlab_nanstd(peak_time_diffs)
    else:
        peak_time_diffs = np.array([])
        std_peak_diff = np.nan

    vent_hilbert = hilbert(vent_norm)
    atr_hilbert = hilbert(atr_norm)
    phase_difference = np.angle(vent_hilbert * np.conj(atr_hilbert))
    synchrony_index_phase = float(1.0 - np.nanmean(np.abs(phase_difference) / np.pi))
    plv = float(np.abs(np.nanmean(np.exp(1j * phase_difference))))

    av_delays = []
    if atr_peak_times.size and vent_peak_times.size:
        for t_a in atr_peak_times:
            idx = np.searchsorted(vent_peak_times, t_a, side="right")
            if idx < vent_peak_times.size:
                av_delays.append(float(vent_peak_times[idx] - t_a))
    av_delay_mean = float(np.nanmean(av_delays)) if len(av_delays) else np.nan
    av_delay_sd = _matlab_nanstd(np.asarray(av_delays, dtype=float))

    replace_title = video_name.replace("_", "-")
    fig_cav, ax = plt.subplots(figsize=FIGSIZE, dpi=DISPLAY_DPI)
    ax.plot(time, vent, "o", color=COLOR_V_SIGNAL, ms=3.5, mfc="none", label="Original Ventricular Size")
    ax.plot(time, atr, "o", color=COLOR_A_SIGNAL, ms=3.5, mfc="none", label="Original Atrium Size")
    ax.plot(time_fine, vent_spline, "-", color=COLOR_V_SIGNAL, lw=LW_SIGNAL, label="Spline Ventricular Size")
    ax.plot(time_fine, atr_spline, "-", color=COLOR_A_SIGNAL, lw=LW_SIGNAL, label="Spline Atrium Size")
    ax.set_xlabel("Relative Time (s)")
    ax.set_ylabel("Cavity Size")
    ax.set_title(f"Ventricle and Atrium Cavity Sizes - {replace_title}")
    ax.set_xlim(0.0, 30.05)
    ax.grid(True, which="both", linewidth=GRID_LW, alpha=GRID_ALPHA)
    ax.legend(loc="best", frameon=True)
    _save_svg(fig_cav, os.path.join(sync_dir, f"TimeSeries_{video_name}.svg"), dpi=dpi)

    fig_cc, ax2 = plt.subplots(figsize=FIGSIZE, dpi=DISPLAY_DPI)
    ax2.plot(lags, correlation, "-", color=COLOR_CC, lw=LW_SIGNAL, label="xcorr (coeff)")
    ax2.axvline(max_corr_lag, color=COLOR_CC_LAG, linestyle="--", label=f"Max Correlation Lag: {max_corr_lag}")
    ax2.set_xlabel("Lag")
    ax2.set_ylabel("Cross-correlation")
    ax2.set_title(f"Cross-correlation - {replace_title}")
    ax2.grid(True, which="both", linewidth=GRID_LW, alpha=GRID_ALPHA)
    ax2.legend(loc="best", frameon=True)
    _save_svg(fig_cc, os.path.join(sync_dir, f"CrossCorrelation_{video_name}.svg"), dpi=dpi)

    sync_df = pd.DataFrame(
        [
            {
                "FileName": f"{video_name}.xlsx",
                "MaxCorrelationLag": max_corr_lag,
                "MaxCorrLag_s": lag_sec,
                "Cross_Correlation_Based_Synchrony_Index": max_corr_value,
                "CrossCorrCoeff": max_corr_value,
                "Phase_Synchrony_Index": synchrony_index_phase,
                "PhaseSynchronyIndex": synchrony_index_phase,
                "PLV": plv,
                "PearsonCorrelation": pearson_corr,
                "PearsonCorr": pearson_corr,
                "Standard_deviation_of_peak_times": std_peak_diff,
                "PeakTimeDiff_SD": std_peak_diff,
                "AV_Delay_Mean": av_delay_mean,
                "AV_Delay_SD": av_delay_sd,
            }
        ]
    )

    out_xlsx = os.path.join(sync_dir, "SynchronizationResults.xlsx")
    sync_df.to_excel(out_xlsx, index=False)

    time_series_xlsx = os.path.join(sync_dir, f"TimeSeries_{video_name}.xlsx")
    with pd.ExcelWriter(time_series_xlsx, engine="openpyxl") as writer:
        pd.DataFrame(
            {
                "Time_s": time_fine,
                "V_raw": vent_spline,
                "A_raw": atr_spline,
                "V_z": vent_norm,
                "A_z": atr_norm,
            }
        ).to_excel(writer, index=False, sheet_name="TimeSeries")
        Lp = min(len(vent_peak_times), len(atr_peak_times), len(peak_time_diffs))
        pd.DataFrame(
            {
                "V_PeakTime_s": vent_peak_times[:Lp],
                "A_PeakTime_s": atr_peak_times[:Lp],
                "AbsDiff_s": peak_time_diffs[:Lp],
            }
        ).to_excel(writer, index=False, sheet_name="Peaks")
        pd.DataFrame({"AV_Delay_s": av_delays}).to_excel(writer, index=False, sheet_name="AVdelay")
        pd.DataFrame(
            {
                "PhaseDiff_rad": phase_difference,
                "PLV_component_real": np.real(np.exp(1j * phase_difference)),
                "PLV_component_imag": np.imag(np.exp(1j * phase_difference)),
            }
        ).to_excel(writer, index=False, sheet_name="Phase")
        sync_df.to_excel(writer, index=False, sheet_name="Summary")

    cc_xlsx = os.path.join(sync_dir, f"CrossCorrelation_{video_name}.xlsx")
    with pd.ExcelWriter(cc_xlsx, engine="openpyxl") as writer:
        pd.DataFrame(
            {
                "Lag_samples": lags,
                "Lag_s": lags * dt if np.isfinite(dt) else np.full_like(lags, np.nan, dtype=float),
                "Correlation": correlation,
            }
        ).to_excel(writer, index=False, sheet_name="CrossCorrelation")
        pd.DataFrame(
            [
                {
                    "MaxLag_samples": max_corr_lag,
                    "MaxLag_s": lag_sec,
                    "MaxCorrelation": max_corr_value,
                }
            ]
        ).to_excel(writer, index=False, sheet_name="Summary")

    matlab_key = str(matlab_file_key or "")
    if matlab_key and matlab_key != video_name:
        _save_svg(fig_cav, os.path.join(sync_dir, f"TimeSeries_{matlab_key}.svg"), dpi=dpi)
        _save_svg(fig_cc, os.path.join(sync_dir, f"CrossCorrelation_{matlab_key}.svg"), dpi=dpi)
        with pd.ExcelWriter(os.path.join(sync_dir, f"TimeSeries_{matlab_key}.xlsx"), engine="openpyxl") as writer:
            pd.DataFrame(
                {
                    "Time_s": time_fine,
                    "V_raw": vent_spline,
                    "A_raw": atr_spline,
                    "V_z": vent_norm,
                    "A_z": atr_norm,
                }
            ).to_excel(writer, index=False, sheet_name="TimeSeries")
            Lp = min(len(vent_peak_times), len(atr_peak_times), len(peak_time_diffs))
            pd.DataFrame(
                {
                    "V_PeakTime_s": vent_peak_times[:Lp],
                    "A_PeakTime_s": atr_peak_times[:Lp],
                    "AbsDiff_s": peak_time_diffs[:Lp],
                }
            ).to_excel(writer, index=False, sheet_name="Peaks")
            pd.DataFrame({"AV_Delay_s": av_delays}).to_excel(writer, index=False, sheet_name="AVdelay")
            pd.DataFrame(
                {
                    "PhaseDiff_rad": phase_difference,
                    "PLV_component_real": np.real(np.exp(1j * phase_difference)),
                    "PLV_component_imag": np.imag(np.exp(1j * phase_difference)),
                }
            ).to_excel(writer, index=False, sheet_name="Phase")
            sync_df.to_excel(writer, index=False, sheet_name="Summary")
        with pd.ExcelWriter(os.path.join(sync_dir, f"CrossCorrelation_{matlab_key}.xlsx"), engine="openpyxl") as writer:
            pd.DataFrame(
                {
                    "Lag_samples": lags,
                    "Lag_s": lags * dt if np.isfinite(dt) else np.full_like(lags, np.nan, dtype=float),
                    "Correlation": correlation,
                }
            ).to_excel(writer, index=False, sheet_name="CrossCorrelation")
            pd.DataFrame(
                [
                    {
                        "MaxLag_samples": max_corr_lag,
                        "MaxLag_s": lag_sec,
                        "MaxCorrelation": max_corr_value,
                    }
                ]
            ).to_excel(writer, index=False, sheet_name="Summary")

    return sync_df, fig_cav, fig_cc

def combine_results(video_name: str,
                    seg_df: pd.DataFrame,
                    v_df: pd.DataFrame,
                    vFS_df: pd.DataFrame,   # not strictly needed, kept for API parity
                    a_df: pd.DataFrame,
                    sync_df: pd.DataFrame,
                    video_out: str,
                    matlab_file_key: Optional[str] = None) -> pd.DataFrame:
    """
    Build one combined row (one video) with the exact same columns as the MATLAB
    '기능 분석 모든 결과 하나의 파일로 정리' script and save it under <video_out>/results
    as 'combined_results_<timestamp>.xlsx' (Sheet1).

    Returns
    -------
    combined_df : pd.DataFrame  # one-row table with the MATLAB column order.
    """
    # ----------------------------
    # Paths (mirror previous modules)
    # ----------------------------
    os.makedirs(video_out, exist_ok=True)
    results_dir = os.path.join(video_out, "results")
    os.makedirs(results_dir, exist_ok=True)

    # Helpers
    def nanmean(x):
        x = np.asarray(x, dtype=float)
        return float(np.nanmean(x)) if x.size and np.any(np.isfinite(x)) else np.nan

    def nanstd(x, ddof=1):
        x = np.asarray(x, dtype=float)
        return _matlab_nanstd(x)

    def valid_series(df, col):
        if col not in df.columns:
            return np.array([], dtype=float)
        return pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)

    # ----------------------------
    # 1) Heart-rate style helpers from locs arrays (ventricle & atrium)
    # ----------------------------
    def heart_rate_from_locs(locs: np.ndarray) -> float:
        locs = np.asarray(locs, dtype=float)
        locs = locs[np.isfinite(locs)]
        if locs.size >= 2:
            return float((locs.size - 1) * 60.0 / (locs[-1] - locs[0])) if locs[-1] > locs[0] else np.nan
        return np.nan

    # ----------------------------
    # 2) Pull ventricle functional metrics from v_df (like MATLAB result_ventricle.xlsx)
    #    Columns expected (per our compute_functional_statistics):
    #      'locs','interval','peaks','Border','vEF','vFS'
    # ----------------------------
    v_locs   = valid_series(v_df, "locs")
    v_interv = valid_series(v_df, "interval")   # MATLAB recomputes std on whole column (incl. the first SD row)
    v_peaks  = valid_series(v_df, "peaks")      # ED candidates
    v_border = valid_series(v_df, "Border")     # ES candidates
    v_vEF    = valid_series(v_df, "vEF")
    v_vFS    = valid_series(v_df, "vFS")
    func_summary = dict(v_df.attrs.get("summary_metrics", {})) if hasattr(v_df, "attrs") else {}

    heartrate_v = heart_rate_from_locs(v_locs)
    sd_interval = nanstd(v_interv, ddof=1)  # MATLAB: std(...,'omitnan') → ddof=1

    ED_mean = nanmean(v_peaks)
    ES_mean = nanmean(v_border)
    ED_sd   = nanstd(v_peaks, ddof=1)
    ES_sd   = nanstd(v_border, ddof=1)
    Difv    = (ED_mean - ES_mean) if (np.isfinite(ED_mean) and np.isfinite(ES_mean)) else np.nan

    vEF_mean = nanmean(v_vEF)
    vEF_sd   = nanstd(v_vEF, ddof=1)
    vFS_mean = nanmean(v_vFS)
    vFS_sd   = nanstd(v_vFS, ddof=1)

    # Prefer MATLAB-aligned values from functional summary when available.
    heartrate_v = float(func_summary.get("V_HR_bpm", heartrate_v))
    heartrate_a = float(func_summary.get("A_HR_bpm", np.nan))
    sd_interval = float(func_summary.get("Interval_SD_s", sd_interval))
    interval_cv = float(func_summary.get("Interval_CV", np.nan))
    systolic_duration_mean = float(func_summary.get("SystolicDuration_mean", np.nan))
    diastolic_duration_mean = float(func_summary.get("DiastolicDuration_mean", np.nan))
    systolic_fraction = float(func_summary.get("SystolicFraction", np.nan))
    sv_index_mean = float(func_summary.get("SV_index_mean", np.nan))
    co_index_mean = float(func_summary.get("CO_index_mean", np.nan))
    diastolic_a_to_v_ratio = float(func_summary.get("Diastolic_AtoV_ratio", np.nan))
    major_minor_ratio_ed = float(func_summary.get("MajorMinor_ratio_ED", np.nan))
    contractility_speed = float(func_summary.get("ContractilitySpeed", np.nan))
    relaxation_speed = float(func_summary.get("RelaxationSpeed", np.nan))
    A_ED_mean = float(func_summary.get("A_ED_mean", np.nan))
    A_ES_mean = float(func_summary.get("A_ES_mean", np.nan))
    A_ED_sd = float(func_summary.get("A_ED_SD", np.nan))
    A_ES_sd = float(func_summary.get("A_ES_SD", np.nan))
    A_ED_ES_diff_mean = float(func_summary.get("A_ED_ES_Diff_mean", np.nan))

    # ----------------------------
    # 3) Atrium heart-rate from a_df (like MATLAB result_atrium.xlsx col1)
    #    a_df only has 'locs'
    # ----------------------------
    a_locs = valid_series(a_df, "locs")
    if not np.isfinite(heartrate_a):
        heartrate_a = heart_rate_from_locs(a_locs)

    # ----------------------------
    # 4) Geometry means from seg_df (these replace MATLAB's angleData column indices)
    #    MATLAB used:
    #      col11 → SV-BA (X,Y), col12 → SV-BA (Y), but in our seg_df they are:
    #        'VentricleAtriumDistance', 'VentricleAtriumYDistance'
    #    VA Center/Bottom/Top: use named columns directly (means, omit NaNs).
    # ----------------------------
    def mean_col(name):  # safe mean
        return nanmean(valid_series(seg_df, name)) if name in seg_df.columns else np.nan

    SVBA_XY = mean_col("VentricleAtriumDistance")
    SVBA_Y  = mean_col("VentricleAtriumYDistance")

    VA_Dist_C = mean_col("VA_Distance_Center")
    VA_Angle_C = _circular_mean_deg(valid_series(seg_df, "VA_Angle_Center"))

    VA_Dist_B = mean_col("VA_Distance_Bottom")
    VA_Angle_B = _circular_mean_deg(valid_series(seg_df, "VA_Angle_Bottom"))

    VA_Dist_T = mean_col("VA_Distance_Top")
    VA_Angle_T = _circular_mean_deg(valid_series(seg_df, "VA_Angle_Top"))
    VA_Angle_C_raw = _circular_mean_deg(
        valid_series(seg_df, "VA_Angle_Center_raw")
        if "VA_Angle_Center_raw" in seg_df.columns
        else valid_series(seg_df, "Angle_Center")
    )
    VA_Angle_B_raw = _circular_mean_deg(
        valid_series(seg_df, "VA_Angle_Bottom_raw")
        if "VA_Angle_Bottom_raw" in seg_df.columns
        else valid_series(seg_df, "Angle_Bottom")
    )
    VA_Angle_T_raw = _circular_mean_deg(
        valid_series(seg_df, "VA_Angle_Top_raw")
        if "VA_Angle_Top_raw" in seg_df.columns
        else valid_series(seg_df, "Angle_Top")
    )

    # ----------------------------
    # 5) Synchronization values (one-row DataFrame from compute_synchronize_analysis)
    #    Expected columns:
    #      'MaxCorrelationLag','Cross_Correlation_Based_Synchrony_Index',
    #      'Phase_Synchrony_Index','PearsonCorrelation',
    #      'Standard_deviation_of_peak_times','AV_Delay_Mean'
    # ----------------------------
    if sync_df is None or len(sync_df) == 0:
        sync_vals = {
            "MaxCorrelationLag": np.nan,
            "MaxCorrLag_s": np.nan,
            "Cross_Correlation_Based_Synchrony_Index": np.nan,
            "CrossCorrCoeff": np.nan,
            "Phase_Synchrony_Index": np.nan,
            "PLV": np.nan,
            "PearsonCorrelation": np.nan,
            "PearsonCorr": np.nan,
            "Standard_deviation_of_peak_times": np.nan,
            "PeakTimeDiff_SD": np.nan,
            "AV_Delay_Mean": np.nan,
            "AV_Delay_SD": np.nan,
        }
    else:
        row0 = sync_df.iloc[0]
        sync_vals = {
            "MaxCorrelationLag": float(row0.get("MaxCorrelationLag", np.nan)),
            "MaxCorrLag_s": float(row0.get("MaxCorrLag_s", np.nan)),
            "Cross_Correlation_Based_Synchrony_Index": float(row0.get("Cross_Correlation_Based_Synchrony_Index", np.nan)),
            "CrossCorrCoeff": float(row0.get("CrossCorrCoeff", row0.get("Cross_Correlation_Based_Synchrony_Index", np.nan))),
            "Phase_Synchrony_Index": float(row0.get("Phase_Synchrony_Index", np.nan)),
            "PLV": float(row0.get("PLV", np.nan)),
            "PearsonCorrelation": float(row0.get("PearsonCorrelation", np.nan)),
            "PearsonCorr": float(row0.get("PearsonCorr", row0.get("PearsonCorrelation", np.nan))),
            "Standard_deviation_of_peak_times": float(row0.get("Standard_deviation_of_peak_times", np.nan)),
            "PeakTimeDiff_SD": float(row0.get("PeakTimeDiff_SD", row0.get("Standard_deviation_of_peak_times", np.nan))),
            "AV_Delay_Mean": float(row0.get("AV_Delay_Mean", np.nan)),
            "AV_Delay_SD": float(row0.get("AV_Delay_SD", np.nan)),
        }

    # ----------------------------
    # 6) Build the row (exact MATLAB header order & labels)
    # ----------------------------
    # Format filename for chemical analysis plugin compatibility: {DATE}_{CHEMICAL_TYPE}_{CONCENTRATION}_{SESSION_ID}.csv
    # If video_name already follows this format, use it; otherwise create it
    if '_' in video_name and len(video_name.split('_')) >= 4:
        # Already in correct format
        vent_result_filename = f"{video_name}.csv"
    else:
        # Legacy format, keep as is but change extension
        vent_result_filename = f"{video_name}_result_ventricle.csv"

    columns = [
        'File Name',
        'SD of interval',
        'Ejaction Fraction (EF)', 'SD of EF',
        'Fractional shortening (FS)', 'SD of FS',
        'heartrate_ventricle', 'heartrate_atrium',
        'End-diastolic', 'End-systolic', 'Differ-Ventricle',
        'SD of End-diastolic', 'SD of End-systolic',
        'SV-BA distance (X,Y)', 'SV-BA distance (Y)',
        'V-A Angle (Center)', 'V-A distance (Center)',
        'V-A Angle (Bottom)', 'V-A distance (Bottom)',
        'V-A Angle (Top)', 'V-A distance (Top)',
        'Max_Correlation_Lag', 'Cross_Correlation_Synchrony_Index',
        'Phase_Synchrony_Index', 'Pearson_Correlation',
        'Std_Deviation_of_PeakTimes', 'AV_Delay_Mean'
    ]

    row_vals = [
        vent_result_filename,
        sd_interval,
        vEF_mean, vEF_sd,
        vFS_mean, vFS_sd,
        heartrate_v, heartrate_a,
        ED_mean, ES_mean, Difv,
        ED_sd, ES_sd,
        SVBA_XY, SVBA_Y,
        VA_Angle_C, VA_Dist_C,
        VA_Angle_B, VA_Dist_B,
        VA_Angle_T, VA_Dist_T,
        sync_vals["MaxCorrelationLag"],
        sync_vals["Cross_Correlation_Based_Synchrony_Index"],
        sync_vals["Phase_Synchrony_Index"],
        sync_vals["PearsonCorrelation"],
        sync_vals["Standard_deviation_of_peak_times"],
        sync_vals["AV_Delay_Mean"],
    ]

    combined_df = pd.DataFrame([row_vals], columns=columns)
    # FileKey is the canonical sample identifier used for cross-pipeline joins.
    # Default to the original sample/video key instead of the exported ventricle
    # filename stem so MATLAB-style names like `..._Series016` do not become
    # `..._Series016_result_ventricle`.
    file_key = str(matlab_file_key or video_name)
    combined_df.insert(0, "FileKey", file_key)

    # MATLAB 2026 metrics (added while preserving existing schema).
    combined_df["V_HR_bpm"] = heartrate_v
    combined_df["A_HR_bpm"] = heartrate_a
    combined_df["Interval_SD_s"] = sd_interval
    combined_df["Interval_CV"] = interval_cv
    combined_df["SystolicDuration_mean"] = systolic_duration_mean
    combined_df["DiastolicDuration_mean"] = diastolic_duration_mean
    combined_df["SystolicFraction"] = systolic_fraction
    combined_df["EF_mean"] = vEF_mean
    combined_df["EF_SD"] = vEF_sd
    combined_df["FS_mean"] = vFS_mean
    combined_df["FS_SD"] = vFS_sd
    combined_df["V_ED_mean"] = ED_mean
    combined_df["V_ES_mean"] = ES_mean
    combined_df["V_ED_SD"] = ED_sd
    combined_df["V_ES_SD"] = ES_sd
    combined_df["SV_index_mean"] = sv_index_mean
    combined_df["CO_index_mean"] = co_index_mean
    combined_df["Diastolic_AtoV_ratio"] = diastolic_a_to_v_ratio
    combined_df["MajorMinor_ratio_ED"] = major_minor_ratio_ed
    combined_df["ContractilitySpeed"] = contractility_speed
    combined_df["RelaxationSpeed"] = relaxation_speed
    combined_df["SVBA_Distance_mean"] = SVBA_XY
    combined_df["SVBA_Distance_Y_mean"] = SVBA_Y
    combined_df["A_ED_mean"] = A_ED_mean
    combined_df["A_ES_mean"] = A_ES_mean
    combined_df["A_ED_SD"] = A_ED_sd
    combined_df["A_ES_SD"] = A_ES_sd
    combined_df["A_ED_ES_Diff_mean"] = A_ED_ES_diff_mean
    combined_df["VA_Dist_Center_mean"] = VA_Dist_C
    combined_df["VA_Dist_Bottom_mean"] = VA_Dist_B
    combined_df["VA_Dist_Top_mean"] = VA_Dist_T
    combined_df["VA_Ang_Center_raw_mean"] = VA_Angle_C_raw
    combined_df["VA_Ang_Center_major_mean"] = VA_Angle_C
    combined_df["VA_Ang_Bottom_raw_mean"] = VA_Angle_B_raw
    combined_df["VA_Ang_Bottom_major_mean"] = VA_Angle_B
    combined_df["VA_Ang_Top_raw_mean"] = VA_Angle_T_raw
    combined_df["VA_Ang_Top_major_mean"] = VA_Angle_T
    combined_df["MaxCorrLag_s"] = sync_vals["MaxCorrLag_s"]
    combined_df["CrossCorrCoeff"] = sync_vals["CrossCorrCoeff"]
    combined_df["PhaseSynchronyIndex"] = sync_vals["Phase_Synchrony_Index"]
    combined_df["PLV"] = sync_vals["PLV"]
    combined_df["PearsonCorr"] = sync_vals["PearsonCorr"]
    combined_df["PeakTimeDiff_SD"] = sync_vals["PeakTimeDiff_SD"]
    combined_df["AV_Delay_SD"] = sync_vals["AV_Delay_SD"]

    preferred_order = ["FileKey"] + [c for c in MATLAB_PERFISH_EXPORT_COLUMNS if c != "FileKey"] + [
        c for c in combined_df.columns if c not in (["FileKey"] + [c for c in MATLAB_PERFISH_EXPORT_COLUMNS if c != "FileKey"])
    ]
    combined_df = combined_df.loc[:, preferred_order]

    # ----------------------------
    # 7) Save as CSV file compatible with chemical analysis plugin
    # ----------------------------
    # Save CSV with standardized filename format
    out_csv = os.path.join(results_dir, vent_result_filename)
    combined_df.to_csv(out_csv, index=False)
    
    # Also save as Excel for backward compatibility
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    out_xlsx = os.path.join(results_dir, f"combined_results_{timestamp}.xlsx")
    # Sheet name & header as in MATLAB (first row is header)
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        combined_df.to_excel(writer, index=False, sheet_name="Sheet1")

    return combined_df

# Main function for testing
if __name__ == "__main__":
    from tqdm import tqdm
    import tifffile
    import skimage.io as skio
    import dask.array
    from prizm_napari.infer import PRIZMInference

    data_path = r"C:\Users\jyyka\workspace\250806-prizm-napari-for-snu\data"
    # axis_mode = "ellipse"
    axis_mode = "scan"
    grayscale = True
    if grayscale:
        channel = 0
    else:
        channel = 1
    infer = PRIZMInference(
        r"C:\Users\jyyka\workspace\250806-prizm-napari-for-snu\pretrained_models\250725_same_params_fixed_aug_0_bs8_lr1e-3_lrschstep_opt_adam_encd3_decch256_encstr8_ldf0.3_ldp5_atr3x6x9_aug1_gpu2_slot0_run10_epoch_89.pth",        
        # r"C:\Users\jyyka\workspace\250806-prizm-napari-for-snu\pretrained_models\model_epoch_4.pth",
        num_classes=3,
        backbone="resnet50",
        encoder_depth=3,
        decoder_channels=256,
        encoder_output_stride=8,
        decoder_atrous_rates=(3, 6, 9),
    )

    batch_combined_list = []
    for video_name in tqdm(os.listdir(data_path), desc="Processing videos"):
        # sub_path = r"C:\Users\jyyka\workspace\250806-prizm-napari-for-snu\data_single\20250521_BaP_Series025"
        # meta_dir = r"C:\Users\jyyka\workspace\250806-prizm-napari-for-snu\data_single\20250521_BaP_Series025\metadata"
        # video_name = "20250521_BaP_Series025"
        sub_path = fr"{data_path}\{video_name}"
        print(sub_path)

        # Create per-video output directory
        video_out = fr"C:\Users\jyyka\workspace\250806-prizm-napari-for-snu\test_outputs\{video_name}"
        os.makedirs(video_out, exist_ok=True)

        # Load single-frame images
        # define which extensions you consider “images”
        VALID_EXTS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}

        # list and sort only files whose lower‐cased extension is in VALID_EXTS
        frames = sorted(
            fn for fn in os.listdir(sub_path)
            if os.path.splitext(fn)[1].lower() in VALID_EXTS
        )
        imgs = [skio.imread(os.path.join(sub_path, fn), as_gray=grayscale) for fn in frames]
        stack = np.stack(imgs, axis=0)
        stack = dask.array.from_array(stack)

        # Run segmentation
        masks = infer.infer(stack, channel)

        # Save mask TIFF
        mask_path = os.path.join(video_out, f"{video_name}_segmentation.tif")
        tifffile.imwrite(mask_path, masks.astype(np.uint8))

        # # Save mask TIFF
        # mask_path = fr"C:\Users\jyyka\workspace\250806-prizm-napari-for-snu\{video_name}_ch0_segmentation.tif"
        # masks = tifffile.imread(mask_path)

        # Center crop stack and mask to 300x300
        stack = stack[:, stack.shape[1]//2-150:stack.shape[1]//2+150, stack.shape[2]//2-150:stack.shape[2]//2+150]
        masks = masks[:, masks.shape[1]//2-150:masks.shape[1]//2+150, masks.shape[2]//2-150:masks.shape[2]//2+150]

        # Metadata lookup
        meta_file = os.path.join(sub_path, "metadata", f"{video_name}_Properties.xml")

        seg_df, fs_overlay = compute_segmentation_statistics(
            stack, masks, f"{video_name}", video_out, meta_file, meta_info=None, axis_mode=axis_mode
        )

        v_df, vFS_df, a_df, fig_v, fig_vfs, fig_a, fig_va = compute_functional_statistics(
            seg_df, f"{video_name}", video_out
        )

        sync_df, fig_cav, fig_cc = compute_synchronize_analysis(
            seg_df, f"{video_name}", video_out
        )

        combined_df = combine_results(
            f"{video_name}", seg_df, v_df, vFS_df, a_df, sync_df, video_out
        )
    
        batch_combined_list.append(combined_df)

    # Create one combined DataFrame for the entire batch
    if batch_combined_list:
        batch_combined_df = pd.concat(batch_combined_list, ignore_index=True)
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        # Save batch combined CSV
        batch_combined_path = fr"C:\Users\jyyka\workspace\250806-prizm-napari-for-snu\test_outputs\batch_combined_{timestamp}.csv"
        batch_combined_df.to_csv(batch_combined_path, index=False)
    else:
        batch_combined_df = pd.DataFrame()
