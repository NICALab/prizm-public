from __future__ import annotations

from pathlib import Path
import re
from typing import Iterable, List


def _iter_excel_candidates(root: Path, recursive: bool = True) -> Iterable[Path]:
    patterns = ("*.xlsx", "*.xls")
    for pattern in patterns:
        if recursive:
            yield from root.rglob(pattern)
        else:
            yield from root.glob(pattern)


def discover_perfish_workbooks(folder: str | Path, recursive: bool = True) -> List[Path]:
    """
    Discover PRIZM condition-level workbooks under a folder.

    This prefers `PerFishMetrics_*.xlsx` recursively so callers can point
    directly at `segmentation_outputs/` without flattening the results first.
    If no PerFishMetrics workbooks are found, it falls back to broader Excel
    discovery while still excluding obvious downstream result files.
    """
    root = Path(folder)
    if not root.is_dir():
        return []

    all_files = []
    for fp in _iter_excel_candidates(root, recursive=recursive):
        name = fp.name.lower()
        if name.startswith("~$") or name.startswith("."):
            continue
        all_files.append(fp)

    excluded_tokens = (
        "prediction",
        "master_unknown",
        "train_2stage",
        "fig_files",
        "stats_significance",
        "batch_combined",
    )
    all_files = sorted(set(all_files))
    perfish = [
        fp for fp in all_files
        if fp.name.lower().startswith("perfishmetrics_")
        and not any(token in fp.name.lower() for token in excluded_tokens)
    ]
    if perfish:
        return sorted(perfish)

    filtered = [
        fp for fp in all_files
        if not any(token in fp.name.lower() for token in excluded_tokens)
    ]
    return sorted(filtered)


def infer_perfish_group_name(file_path_or_name: str | Path) -> str:
    stem = Path(file_path_or_name).stem
    group = re.sub(r"^PerFishMetrics_", "", stem, flags=re.I)
    group = re.sub(r"_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$", "", group)
    group = re.sub(r"^_+|_+$", "", group).strip()
    return group or stem


def is_control_like_workbook(file_path_or_name: str | Path) -> bool:
    stem = Path(file_path_or_name).stem.lower()
    return any(token in stem for token in ("control", "vehicle", "ctrl", "veh", "dmso"))
