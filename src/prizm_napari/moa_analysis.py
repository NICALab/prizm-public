"""
Authoritative PRIZM MoA 2-stage analysis implementation.

This module preserves the March 7, 2026 MATLAB-aligned behavior while also
serving as the public package entrypoint used by the GUI, CLI, and scripts.
"""

from __future__ import annotations

import math
import pickle
import re
import warnings
import zlib
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t
import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import PerfectSeparationWarning
from sklearn.feature_selection import f_classif
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from prizm_napari.input_discovery import discover_perfish_workbooks


def list_excel_files(folder: str | Path) -> List[Path]:
    return discover_perfish_workbooks(folder, recursive=True)


def _stable_unique(items: Sequence[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in items:
        s = str(item)
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def _sorted_unique(items: Sequence[str]) -> List[str]:
    return list(np.unique(np.asarray(items, dtype=object)))


def _matlab_stratified_kfold(labels: Sequence[object], k: int, rng_seed: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    y = np.asarray(labels, dtype=object).reshape(-1)
    rng = np.random.RandomState(int(rng_seed))
    test_folds: List[List[int]] = [[] for _ in range(int(k))]
    for cls in _sorted_unique(y):
        idx = np.flatnonzero(y == cls)
        perm = idx[rng.permutation(len(idx))]
        chunks = np.array_split(perm, int(k))
        for fold_idx, chunk in enumerate(chunks):
            test_folds[fold_idx].extend(int(ix) for ix in chunk)

    all_idx = np.arange(len(y), dtype=int)
    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    for fold in range(int(k)):
        te = np.asarray(sorted(test_folds[fold]), dtype=int)
        tr = np.setdiff1d(all_idx, te, assume_unique=True)
        splits.append((tr, te))
    return splits


def _folds_to_str(values: Sequence[float]) -> str:
    arr = np.asarray(values, dtype=float).reshape(-1)
    parts = []
    for v in arr:
        if np.isfinite(v):
            parts.append(f"{v:.3g}")
        else:
            parts.append("NaN")
    return "[" + " ".join(parts) + "]"


def extract_group_name_from_file(file_path_or_name: str | Path) -> str:
    stem = Path(file_path_or_name).stem
    group = re.sub(r"^PerFishMetrics_", "", stem, flags=re.I)
    group = re.sub(r"_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$", "", group)
    group = re.sub(r"^_+|_+$", "", group).strip()
    return group or stem


def build_train_spec_no_strip(train_dir: str | Path) -> pd.DataFrame:
    files = list_excel_files(train_dir)
    if not files:
        raise ValueError(f"No Excel files found in training folder: {train_dir}")

    rows = []
    for fp in files:
        low = fp.stem.lower()
        if any(k in low for k in ("control", "vehicle", "ctrl", "veh")):
            group = "Vehicle"
        else:
            group = extract_group_name_from_file(fp)
        rows.append({"Group": group, "File": fp.name, "FullPath": str(fp)})

    spec = pd.DataFrame(rows)
    if not (spec["Group"] == "Vehicle").any():
        raise ValueError(
            "Vehicle(control) file was not found. Include control/vehicle/ctrl/veh in filename."
        )
    veh = spec["Group"] == "Vehicle"
    return pd.concat([spec[veh], spec[~veh]], ignore_index=True)


def _normalize_train_spec(train_dir: str | Path, train_spec: Optional[pd.DataFrame]) -> pd.DataFrame:
    if train_spec is None:
        return build_train_spec_no_strip(train_dir)

    if not isinstance(train_spec, pd.DataFrame):
        raise TypeError("train_spec must be a pandas DataFrame when provided.")
    if "Group" not in train_spec.columns:
        raise ValueError("train_spec must contain a 'Group' column.")

    root = Path(train_dir).resolve()
    spec = train_spec.copy()

    if "FullPath" not in spec.columns:
        if "File" not in spec.columns:
            raise ValueError("train_spec must contain either 'FullPath' or 'File'.")
        spec["FullPath"] = [
            str((root / str(fp)).resolve()) if not Path(fp).is_absolute() else str(Path(fp).resolve())
            for fp in spec["File"]
        ]
    else:
        spec["FullPath"] = [str(Path(fp).resolve()) for fp in spec["FullPath"]]

    if "File" not in spec.columns:
        rel_files = []
        for fp in spec["FullPath"]:
            p = Path(fp)
            try:
                rel_files.append(p.relative_to(root).as_posix())
            except Exception:
                rel_files.append(p.name)
        spec["File"] = rel_files

    spec["Group"] = spec["Group"].astype(str).map(str.strip)
    blank_mask = spec["Group"] == ""
    if blank_mask.any():
        spec.loc[blank_mask, "Group"] = [
            extract_group_name_from_file(fp) for fp in spec.loc[blank_mask, "FullPath"]
        ]

    missing = [fp for fp in spec["FullPath"] if not Path(fp).is_file()]
    if missing:
        raise ValueError(f"train_spec contains missing workbook(s): {missing[:3]}")

    if not (spec["Group"] == "Vehicle").any():
        raise ValueError("train_spec must include at least one Vehicle row.")

    spec = spec[["Group", "File", "FullPath"]].copy()
    veh = spec["Group"] == "Vehicle"
    return pd.concat([spec[veh], spec[~veh]], ignore_index=True)


def _normalize_unknown_files(
    unknown_dir: str | Path,
    unknown_files: Optional[Sequence[str | Path]],
) -> List[Path]:
    if unknown_files is None:
        return list_excel_files(unknown_dir)

    root = Path(unknown_dir).resolve()
    resolved: List[Path] = []
    missing: List[str] = []
    seen = set()
    for fp in unknown_files:
        p = Path(fp)
        if not p.is_absolute():
            p = root / p
        p = p.resolve()
        if not p.is_file():
            missing.append(str(p))
            continue
        key = str(p)
        if key in seen:
            continue
        seen.add(key)
        resolved.append(p)

    if missing:
        raise ValueError(f"unknown_files contains missing workbook(s): {missing[:3]}")
    return resolved


def canon_names(names: Sequence[str]) -> np.ndarray:
    out = []
    for n in names:
        s = re.sub(r"[^a-z0-9]+", "_", str(n).lower())
        s = re.sub(r"_+", "_", s).strip("_")
        out.append(s)
    return np.asarray(out, dtype=object)


def read_params(xlsx_path: str | Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_excel(xlsx_path)
    if df.empty:
        raise ValueError(f"Empty Excel file: {xlsx_path}")

    columns = [str(c) for c in df.columns]
    lower_cols = [c.lower() for c in columns]
    sample_col_idx = lower_cols.index("filekey") if "filekey" in lower_cols else 0

    sample_ser = df.iloc[:, sample_col_idx]
    sample_id = sample_ser.where(sample_ser.notna(), "<sample>").astype(str).to_numpy(dtype=object)
    sample_id[np.char.str_len(sample_id.astype(str)) == 0] = "<sample>"

    numeric_cols = []
    disp_names = []
    for idx, col in enumerate(columns):
        if idx == sample_col_idx:
            continue
        ser = df.iloc[:, idx]
        is_num = pd.api.types.is_numeric_dtype(ser)
        if not is_num and (
            pd.api.types.is_object_dtype(ser)
            or pd.api.types.is_string_dtype(ser)
            or pd.api.types.is_categorical_dtype(ser)
        ):
            num = pd.to_numeric(ser, errors="coerce")
            if float(num.notna().sum()) >= 0.9 * max(len(num), 1):
                df.iloc[:, idx] = num
                is_num = True
        if is_num:
            numeric_cols.append(idx)
            disp_names.append(col)

    if not numeric_cols:
        raise ValueError(f"No numeric feature columns in file: {xlsx_path}")

    x = df.iloc[:, numeric_cols].to_numpy(dtype=float)
    all_nan = np.all(~np.isfinite(x), axis=1)
    if np.any(all_nan):
        x = x[~all_nan]
        sample_id = sample_id[~all_nan]

    canon = canon_names(disp_names)
    return x, canon, np.asarray(disp_names, dtype=object), np.asarray(sample_id, dtype=object)


def align_by_canon(x: np.ndarray, canon_x: np.ndarray, canon_ref: np.ndarray) -> np.ndarray:
    out = np.full((x.shape[0], len(canon_ref)), np.nan, dtype=float)
    idx_map = {str(c): i for i, c in enumerate(np.asarray(canon_x, dtype=object))}
    for j, c in enumerate(np.asarray(canon_ref, dtype=object)):
        i = idx_map.get(str(c))
        if i is not None:
            out[:, j] = x[:, i]
    return out


def match_ratio(x_aligned: np.ndarray) -> Tuple[float, int]:
    if x_aligned.size == 0:
        return 0.0, 0
    matched_cols = int(np.sum(~np.all(~np.isfinite(x_aligned), axis=0)))
    return float(matched_cols / max(x_aligned.shape[1], 1)), matched_cols


def fill_missing_with(x: np.ndarray, fill_row: np.ndarray) -> np.ndarray:
    out = np.asarray(x, dtype=float).copy()
    fill_row = np.asarray(fill_row, dtype=float).reshape(-1)
    for j in range(out.shape[1]):
        mask = ~np.isfinite(out[:, j])
        if np.any(mask):
            out[mask, j] = fill_row[j]
    return out


def control_stats(x: np.ndarray, use_robust: bool) -> Tuple[np.ndarray, np.ndarray]:
    xx = np.asarray(x, dtype=float)
    if use_robust:
        mu = np.nanmedian(xx, axis=0)
        sd = 1.4826 * np.nanmedian(np.abs(xx - mu), axis=0)
    else:
        mu = np.nanmean(xx, axis=0)
        sd = np.nanstd(xx, axis=0, ddof=1)
    sd[(~np.isfinite(sd)) | (sd == 0)] = 1.0
    return mu, sd


def clip_z(z: np.ndarray, clip_z_value: float) -> np.ndarray:
    out = np.asarray(z, dtype=float).copy()
    out[out > clip_z_value] = clip_z_value
    out[out < -clip_z_value] = -clip_z_value
    return out


def nan2zero(z: np.ndarray) -> np.ndarray:
    out = np.asarray(z, dtype=float).copy()
    out[~np.isfinite(out)] = 0.0
    return out


def feature_mask(x: np.ndarray, missing_frac_max: float) -> np.ndarray:
    keep = np.mean(~np.isfinite(x), axis=0) <= float(missing_frac_max)
    if np.any(keep):
        var = np.nanvar(x[:, keep], axis=0, ddof=1)
        keep_idx = np.where(keep)[0]
        keep[keep_idx[var <= 1e-12]] = False
    return keep


def sanitize_weights(w: np.ndarray) -> np.ndarray:
    out = np.asarray(w, dtype=float).reshape(-1)
    out[(~np.isfinite(out)) | (out < 0)] = 0.0
    if np.sum(out) <= 0:
        out[:] = 1.0
    mean_w = float(np.mean(out))
    if np.isfinite(mean_w) and mean_w > 0:
        out /= mean_w
    out[(~np.isfinite(out)) | (out < 0)] = 0.0
    if np.sum(out) <= 0:
        out[:] = 1.0
    return out


def _derived_model_seed(rng_seed: int, tag: str, ordinal: int = 0) -> int:
    base = int(rng_seed)
    crc = int(zlib.crc32(str(tag).encode("utf-8")) & 0x7FFFFFFF)
    derived = (base + 1009 * int(ordinal) + crc) % 2147483647
    return int(derived if derived > 0 else 1)


def safe_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    yy = np.asarray(y_true).astype(int)
    ss = np.asarray(score, dtype=float).reshape(-1)
    ok = np.isfinite(ss)
    yy = yy[ok]
    ss = ss[ok]
    if np.sum(yy == 1) == 0 or np.sum(yy == 0) == 0:
        return np.nan
    try:
        return float(roc_auc_score(yy, ss))
    except Exception:
        return np.nan


def select_features_anovaf(x: np.ndarray, y: np.ndarray, n_top: int) -> np.ndarray:
    if n_top <= 0 or n_top >= x.shape[1] or len(np.unique(y)) < 2:
        return np.arange(x.shape[1], dtype=int)
    fvals, _ = f_classif(nan2zero(x), np.asarray(y, dtype=object))
    order = np.argsort(np.nan_to_num(fvals, nan=-np.inf))[::-1]
    order = order[np.isfinite(fvals[order])]
    return order[: min(n_top, len(order))]


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, classes: Sequence[str]) -> float:
    return float(
        f1_score(
            np.asarray(y_true, dtype=object),
            np.asarray(y_pred, dtype=object),
            labels=list(classes),
            average="macro",
            zero_division=0,
        )
    )


def make_valid_names(items: Sequence[str]) -> List[str]:
    out = []
    for s in items:
        name = re.sub(r"[^A-Za-z0-9_]+", "_", str(s))
        name = re.sub(r"_+", "_", name).strip("_")
        if not name:
            name = "x"
        if not re.match(r"[A-Za-z]", name):
            name = f"x{name}"
        out.append(name)
    return out


def sheet31(name: str) -> str:
    s = str(name)
    return s[:31]


class _GLMBinomialWrapper:
    def __init__(self, result: object):
        self.result = result

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        p = _predict_binomial_glm(self.result, x)
        return np.column_stack([1.0 - p, p])

    def predict(self, x: np.ndarray) -> np.ndarray:
        return (self.predict_proba(x)[:, 1] >= 0.5).astype(int)


class _PlattCalibratedBinarySVM:
    def __init__(
        self,
        kernel: str,
        rng_seed: int,
        c_value: float = 1.0,
        rbf_gamma_factor: float = 1.0,
    ):
        self.kernel = str(kernel)
        self.rng_seed = int(rng_seed)
        self.c_value = float(c_value)
        self.rbf_gamma_factor = float(rbf_gamma_factor)
        self.base_model: Optional[SVC] = None
        self.calibrator: Optional[_GLMBinomialWrapper] = None

    def fit(self, x: np.ndarray, y: np.ndarray, sample_weight: np.ndarray) -> "_PlattCalibratedBinarySVM":
        x = np.asarray(x, dtype=float)
        ybin = np.asarray(y, dtype=bool).reshape(-1)
        w = np.asarray(sample_weight, dtype=float).reshape(-1)
        kwargs = {
            "kernel": self.kernel,
            "probability": False,
            "C": self.c_value,
        }
        if self.kernel.lower() == "rbf":
            kwargs["gamma"] = _rbf_gamma_like_matlab_auto(x, self.rbf_gamma_factor)
        self.base_model = SVC(**kwargs)
        self.base_model.fit(x, ybin.astype(int), sample_weight=w)
        score = np.asarray(self.base_model.decision_function(x), dtype=float).reshape(-1, 1)
        # MATLAB's fitPosterior behavior is materially closer when the sigmoid
        # calibration itself is not reweighted, even though the upstream SVM
        # fit still uses the class-balance weights.
        calib_w = np.ones_like(w)
        self.calibrator = _fit_binomial_glm_classifier(score, ybin, calib_w)
        return self

    def decision_function(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(self.base_model.decision_function(np.asarray(x, dtype=float)), dtype=float).reshape(-1)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        score = self.decision_function(x).reshape(-1, 1)
        if self.calibrator is None:
            p = 1.0 / (1.0 + np.exp(-np.clip(score.reshape(-1), -700.0, 700.0)))
        else:
            p = self.calibrator.predict_proba(score)[:, 1]
        return np.column_stack([1.0 - p, p])

    def predict(self, x: np.ndarray) -> np.ndarray:
        return (self.predict_proba(x)[:, 1] >= 0.5).astype(int)


class _DiagLinearDiscriminantBinary:
    def __init__(self, eps: float = 1e-12):
        self.eps = float(eps)
        self.mu_neg: Optional[np.ndarray] = None
        self.mu_pos: Optional[np.ndarray] = None
        self.var_shared: Optional[np.ndarray] = None
        self.prior_neg = 0.5
        self.prior_pos = 0.5

    def fit(self, x: np.ndarray, y: np.ndarray, sample_weight: np.ndarray) -> "_DiagLinearDiscriminantBinary":
        xx = np.asarray(x, dtype=float)
        ybin = np.asarray(y, dtype=bool).reshape(-1)
        w = np.asarray(sample_weight, dtype=float).reshape(-1)

        neg = ~ybin
        pos = ybin
        w_neg = w[neg]
        w_pos = w[pos]
        x_neg = xx[neg]
        x_pos = xx[pos]

        sum_w_neg = float(np.sum(w_neg))
        sum_w_pos = float(np.sum(w_pos))
        total_w = max(sum_w_neg + sum_w_pos, self.eps)
        self.prior_neg = max(sum_w_neg / total_w, self.eps)
        self.prior_pos = max(sum_w_pos / total_w, self.eps)

        self.mu_neg = np.average(x_neg, axis=0, weights=w_neg)
        self.mu_pos = np.average(x_pos, axis=0, weights=w_pos)

        ssw = np.sum(w_neg[:, None] * np.square(x_neg - self.mu_neg[None, :]), axis=0)
        ssw += np.sum(w_pos[:, None] * np.square(x_pos - self.mu_pos[None, :]), axis=0)
        denom = max(total_w - 2.0, self.eps)
        self.var_shared = np.asarray(ssw / denom, dtype=float)
        self.var_shared[(~np.isfinite(self.var_shared)) | (self.var_shared <= self.eps)] = self.eps
        return self

    def decision_function(self, x: np.ndarray) -> np.ndarray:
        if self.mu_neg is None or self.mu_pos is None or self.var_shared is None:
            raise ValueError("Model has not been fit.")
        xx = np.asarray(x, dtype=float)
        inv_var = 1.0 / self.var_shared
        score_neg = xx @ (self.mu_neg * inv_var)
        score_neg -= 0.5 * float(np.sum(np.square(self.mu_neg) * inv_var))
        score_neg += math.log(self.prior_neg)
        score_pos = xx @ (self.mu_pos * inv_var)
        score_pos -= 0.5 * float(np.sum(np.square(self.mu_pos) * inv_var))
        score_pos += math.log(self.prior_pos)
        return np.asarray(score_pos - score_neg, dtype=float).reshape(-1)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        d = np.clip(self.decision_function(x), -700.0, 700.0)
        p = 1.0 / (1.0 + np.exp(-d))
        return np.column_stack([1.0 - p, p])

    def predict(self, x: np.ndarray) -> np.ndarray:
        return (self.predict_proba(x)[:, 1] >= 0.5).astype(int)


class _WeightedBootstrapBagBinary:
    """Approximate MATLAB TreeBagger classification with weighted bootstraps."""

    def __init__(
        self,
        n_estimators: int,
        rng_seed: int,
        max_features: str | float = "sqrt",
        min_samples_leaf: int = 1,
    ):
        self.n_estimators = int(n_estimators)
        self.rng_seed = int(rng_seed)
        self.max_features = max_features
        self.min_samples_leaf = int(min_samples_leaf)
        self.trees: List[DecisionTreeClassifier] = []

    def fit(self, x: np.ndarray, y: np.ndarray, sample_weight: np.ndarray) -> "_WeightedBootstrapBagBinary":
        xx = np.asarray(x, dtype=float)
        yy = np.asarray(y, dtype=int).reshape(-1)
        ww = np.asarray(sample_weight, dtype=float).reshape(-1)
        ww[(~np.isfinite(ww)) | (ww <= 0)] = 0.0
        if float(np.sum(ww)) <= 0.0:
            ww = np.ones_like(ww)
        prob = ww / np.sum(ww)

        rng = np.random.RandomState(self.rng_seed)
        n = len(yy)
        self.trees = []
        for _ in range(self.n_estimators):
            draw_idx = rng.choice(n, size=n, replace=True, p=prob)
            tree_seed = int(rng.randint(0, 2**31 - 1))
            tree = DecisionTreeClassifier(
                criterion="gini",
                splitter="best",
                max_features=self.max_features,
                min_samples_leaf=self.min_samples_leaf,
                random_state=tree_seed,
            )
            tree.fit(xx[draw_idx], yy[draw_idx])
            self.trees.append(tree)
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        xx = np.asarray(x, dtype=float)
        probs = np.zeros((xx.shape[0], 2), dtype=float)
        for tree in self.trees:
            p = np.asarray(tree.predict_proba(xx), dtype=float)
            aligned = np.zeros((xx.shape[0], 2), dtype=float)
            for col_idx, class_id in enumerate(tree.classes_):
                class_int = int(class_id)
                if 0 <= class_int <= 1:
                    aligned[:, class_int] = p[:, col_idx]
            probs += aligned
        probs /= max(len(self.trees), 1)
        return np.clip(probs, 0.0, 1.0)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return (self.predict_proba(x)[:, 1] >= 0.5).astype(int)


def _fit_binomial_glm_classifier(x: np.ndarray, ybin: np.ndarray, sample_weight: np.ndarray) -> _GLMBinomialWrapper:
    xx = np.asarray(x, dtype=float)
    yy = np.asarray(ybin, dtype=bool).astype(float).reshape(-1)
    ww = np.asarray(sample_weight, dtype=float).reshape(-1)
    exog = sm.add_constant(xx, has_constant="add")
    model = sm.GLM(yy, exog, family=sm.families.Binomial(), freq_weights=ww)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            warnings.simplefilter("ignore", category=PerfectSeparationWarning)
            result = model.fit(maxiter=200, disp=0)
    except Exception:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            warnings.simplefilter("ignore", category=PerfectSeparationWarning)
            result = model.fit_regularized(alpha=1e-8, L1_wt=0.0, maxiter=200)
    return _GLMBinomialWrapper(result)


def _predict_binomial_glm(result: object, x: np.ndarray) -> np.ndarray:
    exog = sm.add_constant(np.asarray(x, dtype=float), has_constant="add")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        p = np.asarray(result.predict(exog), dtype=float).reshape(-1)
    return np.clip(p, 0.0, 1.0)


def _rbf_gamma_like_matlab_auto(x: np.ndarray, factor: float = 1.0) -> float | str:
    xx = np.asarray(x, dtype=float)
    if xx.ndim != 2 or xx.shape[1] == 0:
        return "scale"
    var_all = float(np.var(xx))
    if (not np.isfinite(var_all)) or var_all <= 0:
        return "scale"
    base = 1.0 / (xx.shape[1] * var_all)
    return float(base * factor)


def cm_to_table(cm: np.ndarray, class_names: Sequence[str]) -> pd.DataFrame:
    cols = [f"Pred_{c}" for c in class_names]
    df = pd.DataFrame(cm, columns=cols)
    df.insert(0, "TrueClass", list(class_names))
    return df


def write_cm_pack(xlsx_path: str | Path, cm_pack: List[Dict], prefix: str) -> None:
    if not cm_pack:
        return
    with pd.ExcelWriter(xlsx_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
        for cm in cm_pack:
            cm["CM_Table"].to_excel(writer, sheet_name=sheet31(f"{prefix}CM_{cm['ModelID']}"), index=False)
            cm["CM_TableNorm"].to_excel(
                writer, sheet_name=sheet31(f"{prefix}CMN_{cm['ModelID']}"), index=False
            )


def define_models_stage1() -> List[Dict[str, str]]:
    return [
        {"id": "S1_LOGI", "name": "Stage1-LogisticGLM", "mode": "logi_glm"},
        {"id": "S1_SVM_L", "name": "Stage1-SVM-Linear", "mode": "svm_linear"},
        {"id": "S1_SVM_R", "name": "Stage1-SVM-RBF", "mode": "svm_rbf"},
    ]


def define_models_stage2_ovr() -> List[Dict[str, str]]:
    return [
        {"id": "LOGI", "name": "OVR-LogisticGLM", "mode": "logi_glm"},
        {"id": "SVM_L", "name": "OVR-SVM-Linear", "mode": "svm_linear"},
        {"id": "SVM_R", "name": "OVR-SVM-RBF", "mode": "svm_rbf"},
        {"id": "LDA", "name": "OVR-LDA-diaglinear", "mode": "lda_diag"},
        {"id": "BAG", "name": "OVR-BaggedTrees", "mode": "bag"},
    ]


def train_stage1_bin(x: np.ndarray, ybin: np.ndarray, mode: str, rng_seed: int) -> Dict:
    x = nan2zero(np.asarray(x, dtype=float))
    ybin = np.asarray(ybin, dtype=bool).reshape(-1)
    ok = np.all(np.isfinite(x), axis=1)
    x = x[ok]
    ybin = ybin[ok]

    if len(ybin) == 0 or np.sum(ybin) == 0 or np.sum(~ybin) == 0:
        return {"mode": "const", "model": {"prizm_type": "const", "ConstProb": float(np.mean(ybin.astype(float)))}}

    n_pos = int(np.sum(ybin))
    n_neg = int(np.sum(~ybin))
    w = np.ones((len(ybin),), dtype=float)
    w[ybin] = n_neg / max(n_pos, 1)
    w = sanitize_weights(w)

    if mode == "logi_glm":
        mdl = _fit_binomial_glm_classifier(x, ybin, w)
    elif mode == "svm_linear":
        mdl = _PlattCalibratedBinarySVM(
            kernel="linear",
            rng_seed=rng_seed,
            c_value=1.0,
            rbf_gamma_factor=1.0,
        ).fit(x, ybin, w)
    elif mode == "svm_rbf":
        mdl = _PlattCalibratedBinarySVM(
            kernel="rbf",
            rng_seed=rng_seed,
            c_value=2.0,
            rbf_gamma_factor=2.5,
        ).fit(x, ybin, w)
    else:
        raise ValueError(f"Unknown Stage1 mode: {mode}")
    return {"mode": mode, "model": mdl}


def predict_stage1(m1: Dict, x: np.ndarray) -> np.ndarray:
    x = nan2zero(np.asarray(x, dtype=float))
    mdl = m1["model"]
    if isinstance(mdl, dict) and mdl.get("prizm_type") == "const":
        p = np.full((x.shape[0],), float(mdl.get("ConstProb", 0.0)), dtype=float)
        return np.clip(p, 0.0, 1.0)
    if hasattr(mdl, "predict_proba"):
        proba = np.asarray(mdl.predict_proba(x), dtype=float)
        p = proba[:, 1] if proba.ndim == 2 and proba.shape[1] >= 2 else proba.reshape(-1)
    elif hasattr(mdl, "decision_function"):
        score = np.asarray(mdl.decision_function(x), dtype=float).reshape(-1)
        p = 1.0 / (1.0 + np.exp(-np.clip(score, -700.0, 700.0)))
    else:
        p = mdl.predict(x).astype(float)
    return np.clip(np.asarray(p, dtype=float).reshape(-1), 0.0, 1.0)


def train_stage1_all(stage1_models: List[Dict], x: np.ndarray, y: np.ndarray, opts: Dict) -> List[Dict]:
    ybin = np.asarray(y, dtype=object) != "Vehicle"
    out = []
    for model in stage1_models:
        m1 = train_stage1_bin(x, ybin, model["mode"], int(opts["rngSeed"]))
        out.append({"id": model["id"], "name": model["name"], "mode": model["mode"], "M": m1})
    return out


def train_ovr_model(x: np.ndarray, y: np.ndarray, mode: str, opts: Dict) -> Dict:
    y = np.asarray(y, dtype=object)
    feat_idx = np.arange(x.shape[1], dtype=int)
    n_top_features = int(opts["nTopFeatures"])
    if n_top_features > 0 and n_top_features < x.shape[1] and len(np.unique(y)) >= 2:
        feat_idx = select_features_anovaf(x, y, n_top_features)
    xt = nan2zero(x[:, feat_idx])
    class_names = np.unique(y)

    if len(class_names) == 1:
        ovr = [{"class_name": str(class_names[0]), "model": {"prizm_type": "const", "ConstProb": 1.0}}]
        return {"mode": mode, "feat_idx": feat_idx, "class_names": np.asarray(class_names, dtype=object), "ovr": ovr}

    ovr = []
    for class_idx, class_name in enumerate(class_names):
        ybin = (y == class_name).astype(int)
        if np.sum(ybin) == 0 or np.sum(ybin == 0) == 0:
            mdl = {"prizm_type": "const", "ConstProb": float(np.mean(ybin))}
            ovr.append({"class_name": str(class_name), "model": mdl})
            continue

        n_pos = int(np.sum(ybin == 1))
        n_neg = int(np.sum(ybin == 0))
        w = np.ones((len(ybin),), dtype=float)
        w[ybin == 1] = n_neg / max(n_pos, 1)
        w = sanitize_weights(w)

        if mode == "logi_glm":
            mdl = _fit_binomial_glm_classifier(xt, ybin.astype(bool), w)
        elif mode == "svm_linear":
            mdl = _PlattCalibratedBinarySVM(
                kernel="linear",
                rng_seed=int(opts["rngSeed"]),
                c_value=1.0,
                rbf_gamma_factor=1.0,
            ).fit(
                xt, ybin.astype(bool), w
            )
        elif mode == "svm_rbf":
            mdl = _PlattCalibratedBinarySVM(
                kernel="rbf",
                rng_seed=int(opts["rngSeed"]),
                c_value=2.0,
                rbf_gamma_factor=2.5,
            ).fit(
                xt, ybin.astype(bool), w
            )
        elif mode == "lda_diag":
            mdl = _DiagLinearDiscriminantBinary().fit(xt, ybin.astype(bool), w)
        elif mode == "bag":
            mdl = _WeightedBootstrapBagBinary(
                n_estimators=int(opts["nTrees"]),
                rng_seed=int(opts["rngSeed"]),
                # TreeBagger defaults are closer to bagged full trees than sklearn RF.
                max_features="sqrt",
                min_samples_leaf=1,
            ).fit(xt, ybin, w)
        else:
            raise ValueError(f"Unknown Stage2 mode: {mode}")
        ovr.append({"class_name": str(class_name), "model": mdl})

    return {"mode": mode, "feat_idx": feat_idx, "class_names": np.asarray(class_names, dtype=object), "ovr": ovr}


def predict_ovr(model: Dict, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    xt = nan2zero(np.asarray(x, dtype=float)[:, model["feat_idx"]])
    class_names = np.asarray(model["class_names"], dtype=object)
    p = np.full((xt.shape[0], len(class_names)), np.nan, dtype=float)
    for i, item in enumerate(model["ovr"]):
        mdl = item["model"]
        if isinstance(mdl, dict) and mdl.get("prizm_type") == "const":
            prob = np.full((xt.shape[0],), float(mdl.get("ConstProb", 0.0)), dtype=float)
        elif hasattr(mdl, "predict_proba"):
            proba = np.asarray(mdl.predict_proba(xt), dtype=float)
            prob = proba[:, 1] if proba.ndim == 2 and proba.shape[1] >= 2 else proba.reshape(-1)
        elif hasattr(mdl, "decision_function"):
            score = np.asarray(mdl.decision_function(xt), dtype=float).reshape(-1)
            prob = 1.0 / (1.0 + np.exp(-np.clip(score, -700.0, 700.0)))
        else:
            prob = mdl.predict(xt).astype(float)
        p[:, i] = np.clip(np.asarray(prob, dtype=float).reshape(-1), 0.0, 1.0)
    idx = np.argmax(p, axis=1)
    pred = class_names[idx]
    return p, pred


def train_stage2_all(stage2_models: List[Dict], x_tox: np.ndarray, y_tox: np.ndarray, opts: Dict) -> List[Dict]:
    out = []
    for model in stage2_models:
        m2 = train_ovr_model(x_tox, y_tox, model["mode"], opts)
        out.append({"id": model["id"], "name": model["name"], "mode": model["mode"], "M": m2})
    return out


def pick_threshold_by_fpr(y_true: np.ndarray, score: np.ndarray, target_fpr: float) -> Tuple[float, Dict]:
    y_true = np.asarray(y_true, dtype=bool).reshape(-1)
    score = np.asarray(score, dtype=float).reshape(-1)
    ok = np.isfinite(score)
    y_true = y_true[ok]
    score = score[ok]
    veh = ~y_true
    tox = y_true
    if np.sum(veh) == 0 or np.sum(tox) == 0:
        return 0.5, {"FPR": np.nan, "TPR": np.nan, "Thr": 0.5, "Note": "degenerate"}

    candidates = np.sort(np.unique(score))
    best_tpr = -np.inf
    best_thr = 0.5
    best_fpr = np.inf
    for thr in candidates:
        pred = score >= thr
        fp = np.sum(pred & veh)
        tn = np.sum((~pred) & veh)
        tp = np.sum(pred & tox)
        fn = np.sum((~pred) & tox)
        fpr_v = fp / max(fp + tn, 1)
        tpr_v = tp / max(tp + fn, 1)
        if fpr_v <= target_fpr and tpr_v > best_tpr:
            best_tpr = tpr_v
            best_thr = float(thr)
            best_fpr = float(fpr_v)
    if not np.isfinite(best_tpr):
        thr = float(np.nanpercentile(score[veh], 100.0 * (1.0 - target_fpr)))
        pred = score >= thr
        fp = np.sum(pred & veh)
        tn = np.sum((~pred) & veh)
        tp = np.sum(pred & tox)
        fn = np.sum((~pred) & tox)
        return thr, {
            "FPR": fp / max(fp + tn, 1),
            "TPR": tp / max(tp + fn, 1),
            "Thr": thr,
            "Note": "fallback_vehicle_quantile",
        }
    return best_thr, {"FPR": best_fpr, "TPR": best_tpr, "Thr": best_thr, "Note": "FPR_control_best_TPR"}


def get_thr_from_oof(oof_pack: List[Dict], model_id: str, target_fpr: float) -> float:
    for item in oof_pack:
        if item["ModelID"] == model_id:
            thr, _ = pick_threshold_by_fpr(item["yTrue"], item["pOOF"], target_fpr)
            return float(thr)
    raise ValueError(f"Stage1 model not found in OOF: {model_id}")


def prepare_training_bundle(train_spec: pd.DataFrame, opts: Dict) -> Tuple[Dict, np.ndarray, np.ndarray, pd.DataFrame]:
    veh_rows = np.where(train_spec["Group"].to_numpy(dtype=object) == "Vehicle")[0]
    if len(veh_rows) == 0:
        raise ValueError("No Vehicle file selected.")

    ref_path = train_spec.iloc[veh_rows[0]]["FullPath"]
    _, ref_canon, ref_disp, _ = read_params(ref_path)

    match_rows = []
    ctrl_all = []
    for _, row in train_spec.iterrows():
        raw, canon_x, _, _ = read_params(row["FullPath"])
        raw_a = align_by_canon(raw, canon_x, ref_canon)
        match_frac, matched_cols = match_ratio(raw_a)
        match_rows.append(
            {
                "File": row["File"],
                "Group": row["Group"],
                "MatchFrac": match_frac,
                "MatchedCols": matched_cols,
                "TotalCols": len(ref_canon),
                "NSamples": raw_a.shape[0],
            }
        )
        if match_frac < float(opts["minMatchFrac"]):
            raise ValueError(f"Feature alignment too low ({100 * match_frac:.1f}%) in file: {row['File']}")
        if row["Group"] == "Vehicle":
            ctrl_all.append(raw_a)

    ctrl_all_arr = np.vstack(ctrl_all)
    mu_ctrl, sd_ctrl = control_stats(ctrl_all_arr, bool(opts["useRobustControlStats"]))
    sd_ctrl[(~np.isfinite(sd_ctrl)) | (sd_ctrl == 0)] = 1.0

    x_all = []
    y_all = []
    for _, row in train_spec.iterrows():
        raw, canon_x, _, _ = read_params(row["FullPath"])
        raw_a = align_by_canon(raw, canon_x, ref_canon)
        raw_a = fill_missing_with(raw_a, mu_ctrl)
        z = clip_z((raw_a - mu_ctrl) / sd_ctrl, float(opts["clipZ"]))
        x_all.append(z)
        y_all.extend([str(row["Group"])] * z.shape[0])

    x_all_arr = np.vstack(x_all)
    feat_mask = feature_mask(x_all_arr, float(opts["missingFracMax"]))
    x = nan2zero(x_all_arr[:, feat_mask])
    y = np.asarray(y_all, dtype=object)

    bundle = {
        "refParamCanon": ref_canon[feat_mask],
        "refParamDisplay": ref_disp[feat_mask],
        "muCtrl": mu_ctrl[feat_mask],
        "sdCtrl": sd_ctrl[feat_mask],
        "trainSpec": train_spec[["Group", "File", "FullPath"]].copy(),
    }
    return bundle, x, y, pd.DataFrame(match_rows)


def cv_stage1_models(stage1_models: List[Dict], x: np.ndarray, y: np.ndarray, opts: Dict) -> Tuple[pd.DataFrame, List[Dict]]:
    ybin = np.asarray(y, dtype=object) != "Vehicle"
    _, counts = np.unique(ybin, return_counts=True)
    if len(counts) < 2 or int(np.min(counts)) < 2:
        raise ValueError("Stage1 CV requires at least two samples in both Vehicle and Toxic classes.")
    k = min(int(opts["kfold"]), int(np.min(counts)))
    splits = _matlab_stratified_kfold(ybin.astype(object), k, int(opts["rngSeed"]))

    report_rows = []
    oof_pack = []
    for model in stage1_models:
        p_oof = np.full((len(y),), np.nan, dtype=float)
        auc_fold = np.full((k,), np.nan, dtype=float)
        acc_fold = np.full((k,), np.nan, dtype=float)
        for fold_idx, (tr, te) in enumerate(splits):
            m1 = train_stage1_bin(x[tr], ybin[tr], model["mode"], int(opts["rngSeed"]))
            p = predict_stage1(m1, x[te])
            p_oof[te] = p
            auc_fold[fold_idx] = safe_auc(ybin[te], p)
            acc_fold[fold_idx] = float(np.mean((p >= 0.5) == ybin[te]))
        thr, thr_info = pick_threshold_by_fpr(ybin, p_oof, float(opts["targetFPR"]))
        report_rows.append(
            {
                "ModelID": model["id"],
                "ModelName": model["name"],
                "K": k,
                "Acc_Mean": float(np.nanmean(acc_fold)),
                "AUC_Mean": float(np.nanmean(auc_fold)),
                "Thr_FPRctrl": float(thr),
                "FPR": thr_info.get("FPR", np.nan),
                "TPR": thr_info.get("TPR", np.nan),
                "Acc_Folds": _folds_to_str(acc_fold),
                "AUC_Folds": _folds_to_str(auc_fold),
            }
        )
        oof_pack.append({"ModelID": model["id"], "ModelName": model["name"], "pOOF": p_oof, "yTrue": ybin})
    return pd.DataFrame(report_rows), oof_pack


def extract_toxic_subset(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mask = np.asarray(y, dtype=object) != "Vehicle"
    return x[mask], np.asarray(y, dtype=object)[mask]


def cv_stage2_models_ovr(
    models: List[Dict],
    x: np.ndarray,
    y: np.ndarray,
    opts: Dict,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[Dict]]:
    classes, counts = np.unique(np.asarray(y, dtype=object), return_counts=True)
    if len(classes) < 2 or int(np.min(counts)) < 2:
        return pd.DataFrame(), pd.DataFrame(), []

    k = min(int(opts["kfold"]), int(np.min(counts)))
    splits = _matlab_stratified_kfold(np.asarray(y, dtype=object), k, int(opts["rngSeed"]))

    report_rows = []
    auc_rows = []
    cm_pack = []
    for model in models:
        acc1 = np.full((k,), np.nan, dtype=float)
        macrof = np.full((k,), np.nan, dtype=float)
        auc_fold = np.full((k, len(classes)), np.nan, dtype=float)
        y_true_all: List[str] = []
        y_pred_all: List[str] = []
        for fold_idx, (tr, te) in enumerate(splits):
            m2 = train_ovr_model(x[tr], y[tr], model["mode"], opts)
            p2, pred2 = predict_ovr(m2, x[te])
            y_true = np.asarray(y[te], dtype=object)
            acc1[fold_idx] = float(np.mean(pred2 == y_true))
            macrof[fold_idx] = macro_f1(y_true, pred2, classes)
            for ci, class_name in enumerate(classes):
                col = np.where(np.asarray(m2["class_names"], dtype=object) == class_name)[0]
                if len(col):
                    auc_fold[fold_idx, ci] = safe_auc((y_true == class_name).astype(int), p2[:, col[0]])
            y_true_all.extend(y_true.tolist())
            y_pred_all.extend(np.asarray(pred2, dtype=object).tolist())

        auc_mean_per_class = np.nanmean(auc_fold, axis=0)
        report_rows.append(
            {
                "ModelID": model["id"],
                "ModelName": model["name"],
                "AccTop1_Mean": float(np.nanmean(acc1)),
                "MacroF1_Mean": float(np.nanmean(macrof)),
                "MacroAUC_Mean": float(np.nanmean(auc_mean_per_class)),
                "AccTop1_Folds": _folds_to_str(acc1),
                "MacroF1_Folds": _folds_to_str(macrof),
            }
        )
        for ci, class_name in enumerate(classes):
            auc_rows.append(
                {
                    "ModelID": model["id"],
                    "ModelName": model["name"],
                    "Class": class_name,
                    "AUC_Mean": float(auc_mean_per_class[ci]),
                    "AUC_Folds": _folds_to_str(auc_fold[:, ci]),
                }
            )
        cm = confusion_matrix(y_true_all, y_pred_all, labels=classes)
        cm_n = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        cm_pack.append(
            {
                "ModelID": model["id"],
                "ModelName": model["name"],
                "CM_Table": cm_to_table(cm, classes),
                "CM_TableNorm": cm_to_table(cm_n, classes),
            }
        )
    return pd.DataFrame(report_rows), pd.DataFrame(auc_rows), cm_pack


def cv_end2end_fixed_stage1(
    stage1_models: List[Dict],
    stage2_models: List[Dict],
    x: np.ndarray,
    y: np.ndarray,
    opts: Dict,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[Dict]]:
    classes, counts = np.unique(np.asarray(y, dtype=object), return_counts=True)
    if int(np.min(counts)) < 2:
        return pd.DataFrame(), pd.DataFrame(), []

    k = min(int(opts["kfold"]), int(np.min(counts)))
    splits = _matlab_stratified_kfold(np.asarray(y, dtype=object), k, int(opts["rngSeed"]))

    s1_hits = [m for m in stage1_models if m["id"] == opts["stage1FinalID"]]
    s1_mode = s1_hits[0]["mode"] if s1_hits else stage1_models[0]["mode"]

    report_rows = []
    auc_rows = []
    cm_pack = []

    for model2 in stage2_models:
        acc1 = np.full((k,), np.nan, dtype=float)
        macrof = np.full((k,), np.nan, dtype=float)
        auc_fold = np.full((k, len(classes)), np.nan, dtype=float)
        y_true_all: List[str] = []
        y_pred_all: List[str] = []

        for fold_idx, (tr, te) in enumerate(splits):
            ybin_tr = np.asarray(y[tr], dtype=object) != "Vehicle"
            m1 = train_stage1_bin(x[tr], ybin_tr, s1_mode, int(opts["rngSeed"]))
            p_tr = predict_stage1(m1, x[tr])
            veh_tr = np.asarray(y[tr], dtype=object) == "Vehicle"
            thr = float(np.nanpercentile(p_tr[veh_tr], 100.0 * (1.0 - float(opts["targetFPR"]))))

            tox_mask = np.asarray(y[tr], dtype=object) != "Vehicle"
            x_tr2 = x[tr][tox_mask]
            y_tr2 = np.asarray(y[tr], dtype=object)[tox_mask]
            m2 = train_ovr_model(x_tr2, y_tr2, model2["mode"], opts)

            p_tox = predict_stage1(m1, x[te])
            p2, _ = predict_ovr(m2, x[te])
            tox_classes = np.asarray(m2["class_names"], dtype=object).reshape(-1)
            all_classes = np.asarray(["Vehicle", *tox_classes.tolist()], dtype=object)
            s = np.zeros((len(p_tox), len(all_classes)), dtype=float)
            s[:, 0] = 1.0 - p_tox
            for ci in range(len(tox_classes)):
                s[:, 1 + ci] = p_tox * p2[:, ci]

            final_pred = np.full((len(p_tox),), "Vehicle", dtype=object)
            toxic_idx = np.where(p_tox >= thr)[0]
            if len(toxic_idx):
                best = np.argmax(s[toxic_idx, 1:], axis=1)
                final_pred[toxic_idx] = tox_classes[best]

            y_true = np.asarray(y[te], dtype=object)
            acc1[fold_idx] = float(np.mean(final_pred == y_true))
            macrof[fold_idx] = macro_f1(y_true, final_pred, classes)
            for ci, class_name in enumerate(classes):
                col = np.where(all_classes == class_name)[0]
                if len(col):
                    auc_fold[fold_idx, ci] = safe_auc((y_true == class_name).astype(int), s[:, col[0]])

            y_true_all.extend(y_true.tolist())
            y_pred_all.extend(final_pred.tolist())

        auc_mean_per_class = np.nanmean(auc_fold, axis=0)
        model_name = f"End2End(Stage1={opts['stage1FinalID']},Stage2={model2['id']})"
        report_rows.append(
            {
                "ModelID": model2["id"],
                "ModelName": model_name,
                "AccTop1_Mean": float(np.nanmean(acc1)),
                "MacroF1_Mean": float(np.nanmean(macrof)),
                "MacroAUC_Mean": float(np.nanmean(auc_mean_per_class)),
                "AccTop1_Folds": _folds_to_str(acc1),
                "MacroF1_Folds": _folds_to_str(macrof),
            }
        )
        for ci, class_name in enumerate(classes):
            auc_rows.append(
                {
                    "ModelID": model2["id"],
                    "ModelName": model_name,
                    "Class": class_name,
                    "AUC_Mean": float(auc_mean_per_class[ci]),
                    "AUC_Folds": _folds_to_str(auc_fold[:, ci]),
                }
            )
        cm = confusion_matrix(y_true_all, y_pred_all, labels=classes)
        cm_n = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        cm_pack.append(
            {
                "ModelID": model2["id"],
                "ModelName": model_name,
                "CM_Table": cm_to_table(cm, classes),
                "CM_TableNorm": cm_to_table(cm_n, classes),
            }
        )
    return pd.DataFrame(report_rows), pd.DataFrame(auc_rows), cm_pack


def make_info_table(train_spec: pd.DataFrame, unknown_files: Sequence[Path], bundle: Dict, out_dir: str | Path) -> pd.DataFrame:
    opts = bundle["opts"]
    return pd.DataFrame(
        [
            {
                "OutputFolder": str(out_dir),
                "TrainFiles": int(len(train_spec)),
                "VehicleFiles": int(np.sum(train_spec["Group"] == "Vehicle")),
                "UnknownFiles": int(len(unknown_files)),
                "UseRobust": bool(opts["useRobustControlStats"]),
                "ClipZ": float(opts["clipZ"]),
                "MissingFracMax": float(opts["missingFracMax"]),
                "MinMatchFrac": float(opts["minMatchFrac"]),
                "TargetFPR": float(opts["targetFPR"]),
                "Stage1_FinalID": str(opts["stage1FinalID"]),
                "Stage1_Threshold": float(bundle["stage1"]["threshold"]),
                "SaveDominanceStats": bool(opts.get("saveDominanceStats", False)),
                "SaveDominanceStatsML": bool(opts.get("saveDominanceStatsML", False)),
                "Dominance_Alpha": float(opts["dominanceAlpha"]) if opts.get("dominanceAlpha") is not None else np.nan,
                "ExcludeSelfInDominance": bool(opts.get("excludeSelfInDominance", True)),
                "PERM_N": int(opts["permN"]) if opts.get("permN") is not None else np.nan,
                "PERM_MaxExactN": int(opts["permMaxExactN"]) if opts.get("permMaxExactN") is not None else np.nan,
                "IncludeSelfInSimilarity": bool(opts.get("includeSelfInSimilarity", False)),
                "SelfSimilarityLabel": str(opts.get("selfSimilarityLabel", "")),
                "Timestamp": str(pd.Timestamp.now()),
            }
        ]
    )


def _save_bundle(bundle_path: str | Path, bundle: Dict) -> None:
    with open(bundle_path, "wb") as f:
        pickle.dump(bundle, f)


def _load_bundle(bundle_path: str | Path) -> Dict:
    with open(bundle_path, "rb") as f:
        return pickle.load(f)


def similarity_tables(
    zu: np.ndarray,
    group_names: Sequence[str],
    group_mean_z: np.ndarray,
    metric: str,
    k: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    group_names = [str(g) for g in group_names]
    n = zu.shape[0]
    g = len(group_names)
    d = np.full((n, g), np.nan, dtype=float)
    for gi, mu in enumerate(np.asarray(group_mean_z, dtype=float)):
        if str(metric).lower() == "cosine":
            num = np.sum(zu * mu[None, :], axis=1)
            den = np.sqrt(np.sum(np.square(zu), axis=1) * np.sum(np.square(mu)))
            cs = num / np.maximum(np.finfo(float).eps, den)
            d[:, gi] = 1.0 - cs
        else:
            diff = zu - mu[None, :]
            d[:, gi] = np.sqrt(np.sum(np.square(diff), axis=1))

    s = np.exp(-d)
    s = s / np.maximum(np.finfo(float).eps, np.sum(s, axis=1, keepdims=True))

    valid_names = make_valid_names(group_names)
    dist_tbl = pd.DataFrame(d, columns=valid_names)
    sim_tbl = pd.DataFrame(s, columns=valid_names)

    k = min(int(k), g)
    top_cols: Dict[str, object] = {}
    for ki in range(1, k + 1):
        top_cols[f"Top{ki}_Group"] = np.full((n,), "", dtype=object)
        top_cols[f"Top{ki}_Sim"] = np.full((n,), np.nan, dtype=float)

    for i in range(n):
        order = np.argsort(s[i])[::-1]
        for ki in range(k):
            top_cols[f"Top{ki + 1}_Group"][i] = group_names[order[ki]]
            top_cols[f"Top{ki + 1}_Sim"][i] = float(s[i, order[ki]])

    top_tbl = pd.DataFrame(top_cols)
    return dist_tbl, sim_tbl, top_tbl


def get_similarity_refs(bundle: Dict, zu: np.ndarray, opts: Dict) -> Tuple[List[str], np.ndarray, str]:
    group_names = [str(g) for g in bundle["groupNames"]]
    group_mean_z = np.asarray(bundle["groupMeanZ"], dtype=float)
    self_label = ""

    if bool(opts.get("includeSelfInSimilarity", False)):
        self_label = str(opts.get("selfSimilarityLabel") or "Self")
        self_mean_z = np.nanmean(zu, axis=0)
        if self_label in group_names:
            self_label = f"{self_label}_Current"
        group_names = [*group_names, self_label]
        group_mean_z = np.vstack([group_mean_z, self_mean_z])
    return group_names, group_mean_z, self_label


def mode_string(items: Sequence[str]) -> str:
    counts: Dict[str, int] = {}
    for item in items:
        key = str(item) if str(item) else "<none>"
        counts[key] = counts.get(key, 0) + 1
    if not counts:
        return "<none>"
    return max(counts.items(), key=lambda kv: kv[1])[0]


def holm_adjust(p_raw: Sequence[float]) -> np.ndarray:
    p_raw_arr = np.asarray(p_raw, dtype=float).reshape(-1)
    out = np.full(p_raw_arr.shape, np.nan, dtype=float)
    ok = np.isfinite(p_raw_arr)
    p = p_raw_arr[ok]
    if p.size == 0:
        return out

    order = np.argsort(p)
    ps = p[order]
    m = ps.size
    adj_sorted = np.array([(m - i) * ps[i] for i in range(m)], dtype=float)
    adj_sorted = np.maximum.accumulate(adj_sorted)
    adj_sorted = np.minimum(adj_sorted, 1.0)

    adj = np.full(m, np.nan, dtype=float)
    adj[order] = adj_sorted
    out[ok] = adj
    return out


def sigstar(p: float) -> str:
    if not np.isfinite(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def competitor_baseline(comp_mat: np.ndarray, higher_is_better: bool, competitor_mode: str) -> np.ndarray:
    comp = np.asarray(comp_mat, dtype=float)
    baseline = np.full((comp.shape[0],), np.nan, dtype=float)
    mode = str(competitor_mode).strip().lower()
    if mode not in {"mean", "top2mean", "best"}:
        mode = "mean"

    for i in range(comp.shape[0]):
        v = comp[i, :]
        v = v[np.isfinite(v)]
        if v.size == 0:
            continue
        if mode == "best":
            baseline[i] = np.nanmax(v) if higher_is_better else np.nanmin(v)
        elif mode == "top2mean":
            v = np.sort(v)[::-1] if higher_is_better else np.sort(v)
            k = min(2, v.size)
            baseline[i] = float(np.nanmean(v[:k]))
        else:
            baseline[i] = float(np.nanmean(v))
    return baseline


def margin_summary(x: Sequence[float], best_comp: Sequence[float], margin: Sequence[float]) -> Dict[str, float]:
    xx = np.asarray(x, dtype=float).reshape(-1)
    bc = np.asarray(best_comp, dtype=float).reshape(-1)
    mg = np.asarray(margin, dtype=float).reshape(-1)
    ok = np.isfinite(xx) & np.isfinite(bc) & np.isfinite(mg)
    xx = xx[ok]
    bc = bc[ok]
    mg = mg[ok]
    if mg.size == 0:
        return {
            "n": 0,
            "mean_group": np.nan,
            "mean_best_competitor": np.nan,
            "mean_margin": np.nan,
            "median_margin": np.nan,
            "win_frac": np.nan,
        }
    return {
        "n": int(mg.size),
        "mean_group": float(np.nanmean(xx)),
        "mean_best_competitor": float(np.nanmean(bc)),
        "mean_margin": float(np.nanmean(mg)),
        "median_margin": float(np.nanmedian(mg)),
        "win_frac": float(np.nanmean((mg > 0).astype(float))),
    }


def signflip_perm_one_sided(d: Sequence[float], n_perm: int, max_exact_n: int, seed: int) -> Dict[str, float | str | int]:
    dd = np.asarray(d, dtype=float).reshape(-1)
    dd = dd[np.isfinite(dd)]
    n = int(dd.size)
    result: Dict[str, float | str | int] = {
        "n": n,
        "obs_mean": np.nan,
        "p_raw": np.nan,
        "method": "none",
        "n_perm_used": np.nan,
    }
    if n == 0:
        return result

    obs = float(np.nanmean(dd))
    result["obs_mean"] = obs

    tol = np.spacing(max(1.0, float(np.nanmax(np.abs(dd)))))
    if np.all(np.abs(dd) <= tol):
        result["p_raw"] = 1.0
        result["method"] = "all_zero"
        result["n_perm_used"] = 1
        return result

    if n <= int(max_exact_n):
        n_enum = 1 << n
        idx = np.arange(n_enum, dtype=np.uint32)
        sign_mat = np.ones((n_enum, n), dtype=float)
        for k in range(n):
            neg_mask = ((idx >> k) & 1) == 1
            sign_mat[neg_mask, k] = -1.0
        perm_stats = (sign_mat @ dd) / max(n, 1)
        result["p_raw"] = float(np.mean(perm_stats >= (obs - 1e-12)))
        result["method"] = "exact_signflip_onesided"
        result["n_perm_used"] = n_enum
        return result

    rng = np.random.RandomState(int(seed))
    sign_mat = np.ones((int(n_perm), n), dtype=float)
    draws = np.reshape(rng.rand(int(n_perm) * n), (int(n_perm), n), order="F")
    sign_mat[draws < 0.5] = -1.0
    perm_stats = (sign_mat @ dd) / max(n, 1)
    result["p_raw"] = float((np.sum(perm_stats >= (obs - 1e-12)) + 1) / (int(n_perm) + 1))
    result["method"] = "mc_signflip_onesided"
    result["n_perm_used"] = int(n_perm)
    return result


def make_dominance_table(
    metric_tbl: pd.DataFrame,
    group_names: Sequence[str],
    exclude_label: str,
    higher_is_better: bool,
    metric_name: str,
    alpha: float,
    perm_n: int,
    perm_max_exact_n: int,
    perm_seed: int,
    competitor_mode: str,
) -> pd.DataFrame:
    mode = str(competitor_mode).strip().lower()
    if mode not in {"mean", "top2mean", "best"}:
        mode = "mean"

    group_list = [str(g) for g in group_names]
    exclude_mask = np.zeros((len(group_list),), dtype=bool)
    if str(exclude_label).strip():
        exclude_mask = np.asarray([g == str(exclude_label) for g in group_list], dtype=bool)
    target_idx = np.flatnonzero(~exclude_mask)

    if target_idx.size == 0:
        return pd.DataFrame({"Note": ["No eligible non-self groups available for dominance statistics."]})

    rows: List[Dict[str, object]] = []
    p_raw = np.full((target_idx.size,), np.nan, dtype=float)
    mat = metric_tbl.to_numpy(dtype=float)

    for ii, j in enumerate(target_idx, start=1):
        comp_idx = [idx for idx in target_idx.tolist() if int(idx) != int(j)]
        row: Dict[str, object] = {
            "Metric": str(metric_name),
            "ReferenceGroup": str(group_list[int(j)]),
            "CompetitorMode": mode,
            "N_Paired": np.nan,
            "Mean_Group": np.nan,
            "Mean_CompetitorBaseline": np.nan,
            "MeanMargin": np.nan,
            "MedianMargin": np.nan,
            "WinFrac": np.nan,
            "P_Perm_Raw": np.nan,
            "P_Perm_Holm": np.nan,
            "PermSignificant": False,
            "SigStar": "",
            "DominanceStatus": "",
            "PermMethod": "",
            "PermN_Used": np.nan,
        }
        if not comp_idx:
            row["DominanceStatus"] = "NO_COMP"
            rows.append(row)
            continue

        x = mat[:, int(j)]
        comp_mat = mat[:, np.asarray(comp_idx, dtype=int)]
        baseline = competitor_baseline(comp_mat, higher_is_better, mode)
        margin = x - baseline if higher_is_better else baseline - x

        rm = margin_summary(x, baseline, margin)
        rp = signflip_perm_one_sided(margin, perm_n, perm_max_exact_n, int(perm_seed) + ii)

        row["N_Paired"] = rm["n"]
        row["Mean_Group"] = rm["mean_group"]
        row["Mean_CompetitorBaseline"] = rm["mean_best_competitor"]
        row["MeanMargin"] = rm["mean_margin"]
        row["MedianMargin"] = rm["median_margin"]
        row["WinFrac"] = rm["win_frac"]
        row["P_Perm_Raw"] = rp["p_raw"]
        row["PermMethod"] = rp["method"]
        row["PermN_Used"] = rp["n_perm_used"]
        p_raw[ii - 1] = float(rp["p_raw"]) if np.isfinite(rp["p_raw"]) else np.nan
        rows.append(row)

    out = pd.DataFrame(rows)
    p_holm = holm_adjust(p_raw)
    out["P_Perm_Holm"] = p_holm
    out["PermSignificant"] = np.isfinite(p_holm) & (p_holm < float(alpha))
    out["SigStar"] = [sigstar(v) for v in p_holm]

    statuses: List[str] = []
    for _, row in out.iterrows():
        status = str(row.get("DominanceStatus", "") or "")
        mean_margin = row.get("MeanMargin", np.nan)
        if status:
            statuses.append(status)
            continue
        if bool(row["PermSignificant"]) and np.isfinite(mean_margin) and float(mean_margin) > 0:
            statuses.append("DOMINANT")
        elif np.isfinite(mean_margin) and float(mean_margin) > 0:
            statuses.append("POS_NS")
        elif np.isfinite(mean_margin) and float(mean_margin) <= 0:
            statuses.append("NOT_DOM")
        else:
            statuses.append("NA")
    out["DominanceStatus"] = statuses

    out["_SortKey_SIG"] = (~out["PermSignificant"]).astype(int)
    out["_SortKey_NEG"] = -pd.to_numeric(out["MeanMargin"], errors="coerce")
    out = out.sort_values(
        ["_SortKey_SIG", "_SortKey_NEG", "P_Perm_Holm"],
        ascending=[True, True, True],
        kind="mergesort",
        na_position="last",
    ).drop(columns=["_SortKey_SIG", "_SortKey_NEG"]).reset_index(drop=True)
    return out


def equiv_ci_paired(x: Sequence[float], y: Sequence[float], delta: float, alpha: float, sim_multiplier: float) -> Dict:
    xx = np.asarray(x, dtype=float).reshape(-1)
    yy = np.asarray(y, dtype=float).reshape(-1)
    ok = np.isfinite(xx) & np.isfinite(yy)
    xx = xx[ok]
    yy = yy[ok]

    result = {
        "n": int(len(xx)),
        "delta": float(delta),
        "mean_x": float(np.nanmean(xx)) if len(xx) else np.nan,
        "mean_self": float(np.nanmean(yy)) if len(yy) else np.nan,
        "diff_mean": float(np.nanmean(xx - yy)) if len(xx) else np.nan,
        "ci90_low": np.nan,
        "ci90_high": np.nan,
        "p_diff": np.nan,
        "equivalent": False,
        "similarity_score": np.nan,
        "sim_class": "INC",
        "status": "INC",
        "label": "",
    }
    if len(xx) < 2:
        return result

    d = xx - yy
    md = float(np.mean(d))
    sd = float(np.std(d, ddof=1))
    se = sd / math.sqrt(len(d))
    result["mean_x"] = float(np.mean(xx))
    result["mean_self"] = float(np.mean(yy))
    result["diff_mean"] = md
    result["similarity_score"] = max(0.0, 1.0 - abs(md) / max(sim_multiplier * delta, np.finfo(float).eps))

    def _classify(p_diff: float, is_eq: bool, is_sim: bool) -> None:
        if is_eq:
            result["equivalent"] = True
            result["sim_class"] = "EQ"
            result["status"] = "EQ"
            result["label"] = "EQ"
            return
        if is_sim:
            result["sim_class"] = "SIM"
            result["status"] = "SIM"
            result["label"] = "SIM"
            return
        if p_diff < 0.001:
            result["sim_class"] = "DIFF"
            result["status"] = "DIFF"
            result["label"] = "***"
        elif p_diff < 0.01:
            result["sim_class"] = "DIFF"
            result["status"] = "DIFF"
            result["label"] = "**"
        elif p_diff < 0.05:
            result["sim_class"] = "DIFF"
            result["status"] = "DIFF"
            result["label"] = "*"
        else:
            result["sim_class"] = "INC"
            result["status"] = "INC"
            result["label"] = ""

    if (not np.isfinite(se)) or se < np.finfo(float).eps:
        result["ci90_low"] = md
        result["ci90_high"] = md
        result["p_diff"] = float(abs(md) > 0)
        _classify(result["p_diff"], (-delta < md < delta), abs(md) <= sim_multiplier * delta)
        return result

    tstat = md / se
    df = len(d) - 1
    p_diff = float(2.0 * (1.0 - t.cdf(abs(tstat), df)))
    tcrit = float(t.ppf(1.0 - alpha, df))
    ci90_low = md - tcrit * se
    ci90_high = md + tcrit * se
    result["ci90_low"] = ci90_low
    result["ci90_high"] = ci90_high
    result["p_diff"] = p_diff
    is_eq = (ci90_low > -delta) and (ci90_high < delta)
    is_sim = (not is_eq) and (abs(md) <= sim_multiplier * delta)
    _classify(p_diff, is_eq, is_sim)
    return result


def make_eq_vs_self_table(
    metric_tbl: pd.DataFrame,
    group_names: Sequence[str],
    self_label: str,
    delta: float,
    alpha: float,
    sim_multiplier: float,
    metric_name: str,
) -> pd.DataFrame:
    group_names = [str(g) for g in group_names]
    if self_label not in group_names:
        return pd.DataFrame({"Note": ["Self reference not found."]})

    self_idx = group_names.index(self_label)
    if self_idx >= metric_tbl.shape[1]:
        return pd.DataFrame({"Note": ["Self index exceeds metric table width."]})

    self_vec = metric_tbl.iloc[:, self_idx].to_numpy(dtype=float)
    rows = []
    for j, group_name in enumerate(group_names):
        if j == self_idx:
            continue
        r = equiv_ci_paired(metric_tbl.iloc[:, j].to_numpy(dtype=float), self_vec, delta, alpha, sim_multiplier)
        rows.append(
            {
                "Metric": metric_name,
                "ReferenceGroup": group_name,
                "SelfGroup": self_label,
                "N_Paired": r["n"],
                "Mean_Group": r["mean_x"],
                "Mean_Self": r["mean_self"],
                "MeanDiff_GroupMinusSelf": r["diff_mean"],
                "Delta": r["delta"],
                "CI90_Low": r["ci90_low"],
                "CI90_High": r["ci90_high"],
                "P_Diff": r["p_diff"],
                "Equivalent": bool(r["equivalent"]),
                "SimilarityScore": r["similarity_score"],
                "SimilarityClass": r["sim_class"],
                "Status": r["status"],
                "Label": r["label"],
            }
        )
    if not rows:
        return pd.DataFrame({"Note": ["No non-self groups available for equivalence test."]})
    out = pd.DataFrame(rows)
    out["_sort_eq"] = (~out["Equivalent"]).astype(int)
    out = out.sort_values(["_sort_eq", "P_Diff"], ascending=[True, True]).drop(columns=["_sort_eq"]).reset_index(drop=True)
    return out


def predict_unknown_2stage(
    bundle_path: str | Path,
    unknown_excel_path: str | Path,
    out_dir: str | Path,
    set_tag: str = "UNKNOWN",
    true_group: str = "",
) -> Path:
    bundle = _load_bundle(bundle_path)
    opts = bundle["opts"]

    raw_u, canon_u, _, sample_u = read_params(unknown_excel_path)
    raw_u = align_by_canon(raw_u, canon_u, bundle["refParamCanon"])
    match_frac, _ = match_ratio(raw_u)
    if match_frac < float(opts["minMatchFrac"]):
        raise ValueError(f"UNKNOWN alignment too low ({100 * match_frac:.1f}%): {unknown_excel_path}")

    raw_u = fill_missing_with(raw_u, bundle["muCtrl"])
    zu = clip_z((raw_u - bundle["muCtrl"]) / bundle["sdCtrl"], float(opts["clipZ"]))
    zu = nan2zero(zu)

    p_map = {}
    for model in bundle["stage1"]["models"]:
        p_map[model["id"]] = predict_stage1(model["M"], zu)

    p_final = p_map[str(opts["stage1FinalID"])]
    thr1 = float(bundle["stage1"]["threshold"])
    is_toxic = p_final >= thr1

    sim_names, sim_means, self_label = get_similarity_refs(bundle, zu, opts)
    dist_tbl, sim_tbl, top_tbl = similarity_tables(zu, sim_names, sim_means, str(opts["simMetric"]), int(opts["simTopK"]))

    base = Path(unknown_excel_path).stem
    out_xlsx = Path(out_dir) / f"{base}_2STAGE_predictions.xlsx"
    out_xlsx.parent.mkdir(parents=True, exist_ok=True)

    info_common = pd.DataFrame(
        [
            {
                "UnknownFile": str(unknown_excel_path),
                "BundleFile": str(bundle_path),
                "N": int(zu.shape[0]),
                "Dataset": str(set_tag),
                "TrueGroup": str(true_group),
                "Stage1_FinalID": str(opts["stage1FinalID"]),
                "Stage1_Threshold": thr1,
                "TargetFPR": float(opts["targetFPR"]),
                "MeanPtoxic": float(np.nanmean(p_final)),
                "ToxicFrac": float(np.nanmean(is_toxic.astype(float))),
                "SimilarityIncludesSelf": bool(opts["includeSelfInSimilarity"]),
                "SelfSimilarityLabel": self_label,
                "Timestamp": str(pd.Timestamp.now()),
            }
        ]
    )

    t1 = pd.DataFrame({"SampleID": sample_u.astype(str)})
    t1["Stage1Label"] = np.where(is_toxic, "Toxic", "Vehicle")
    for model in bundle["stage1"]["models"]:
        t1[f"Ptoxic_{model['id']}"] = p_map[model["id"]]
    t1["Ptoxic_Final"] = p_final
    t1["Ptoxic_Final_percent"] = p_final * 100.0

    dist_out = dist_tbl.copy()
    dist_out.insert(0, "SampleID", sample_u.astype(str))
    sim_out = sim_tbl.copy()
    sim_out.insert(0, "SampleID", sample_u.astype(str))
    top_out = top_tbl.copy()
    top_out.insert(0, "SampleID", sample_u.astype(str))

    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        info_common.to_excel(writer, sheet_name="Info_Common", index=False)
        t1.to_excel(writer, sheet_name="Stage1_PerSample", index=False)
        dist_out.to_excel(writer, sheet_name="Similarity_Distance", index=False)
        sim_out.to_excel(writer, sheet_name="Similarity_Softmax", index=False)
        top_out.to_excel(writer, sheet_name="Similarity_TopK", index=False)

        if bool(opts.get("saveDominanceStats", False)):
            exclude_label = self_label if bool(opts.get("excludeSelfInDominance", True)) else ""
            competitor_mode = str(opts.get("dominanceCompetitorMode", "mean") or "mean")
            dom_soft_tbl = make_dominance_table(
                sim_tbl,
                sim_names,
                exclude_label,
                True,
                "Softmax",
                float(opts.get("dominanceAlpha", 0.05)),
                int(opts.get("permN", 10000)),
                int(opts.get("permMaxExactN", 16)),
                int(opts.get("permSeed", 0)),
                competitor_mode,
            )
            dom_dist_tbl = make_dominance_table(
                dist_tbl,
                sim_names,
                exclude_label,
                False,
                "Distance",
                float(opts.get("dominanceAlpha", 0.05)),
                int(opts.get("permN", 10000)),
                int(opts.get("permMaxExactN", 16)),
                int(opts.get("permSeed", 0)),
                competitor_mode,
            )
            dom_soft_tbl.to_excel(writer, sheet_name="Dominance_Softmax", index=False)
            dom_dist_tbl.to_excel(writer, sheet_name="Dominance_Distance", index=False)

        for model in bundle["stage2"]["models"]:
            p2, pred2 = predict_ovr(model["M"], zu)
            class_tox = np.asarray(model["M"]["class_names"], dtype=object).reshape(-1)
            score_veh = 1.0 - p_final
            score_tox = p_final[:, None] * p2

            final_label = np.full((len(p_final),), "Vehicle", dtype=object)
            toxic_idx = np.where(p_final >= thr1)[0]
            if len(toxic_idx):
                best = np.argmax(score_tox[toxic_idx], axis=1)
                final_label[toxic_idx] = class_tox[best]

            final_score = np.full((len(p_final),), np.nan, dtype=float)
            for i, label in enumerate(final_label):
                if label == "Vehicle":
                    final_score[i] = score_veh[i] * 100.0
                else:
                    jj = np.where(class_tox == label)[0]
                    if len(jj):
                        final_score[i] = score_tox[i, jj[0]] * 100.0

            score_cols = make_valid_names(["Score_Vehicle", *[f"Score_{c}" for c in class_tox]])
            score_tbl = pd.DataFrame(np.column_stack([score_veh * 100.0, score_tox * 100.0]), columns=score_cols)
            per_tbl = pd.DataFrame(
                {
                    "SampleID": sample_u.astype(str),
                    "FinalLabel": final_label.astype(str),
                    "FinalScore": final_score,
                    "Ptoxic_Final_percent": p_final * 100.0,
                    "PredLabel_Stage2": np.asarray(pred2, dtype=object).astype(str),
                }
            )
            per_tbl = pd.concat([per_tbl, score_tbl], axis=1)
            per_tbl.to_excel(writer, sheet_name=sheet31(f"PerSample_{model['id']}"), index=False)

            sml = np.column_stack([score_veh, score_tox])
            sim_ml = sml / np.maximum(np.finfo(float).eps, np.sum(sml, axis=1, keepdims=True))
            sim_ml_tbl = pd.DataFrame(
                sim_ml,
                columns=make_valid_names(["Vehicle", *class_tox.tolist()]),
            )
            sim_ml_tbl.insert(0, "SampleID", sample_u.astype(str))
            sim_ml_tbl.to_excel(writer, sheet_name=sheet31(f"SimML_{model['id']}"), index=False)

            if bool(opts.get("saveDominanceStatsML", False)):
                all_names = ["Vehicle", *class_tox.tolist()]
                dom_ml_tbl = make_dominance_table(
                    pd.DataFrame(sim_ml, columns=make_valid_names(all_names)),
                    all_names,
                    "",
                    True,
                    f"MLSoftmax_{model['id']}",
                    float(opts.get("dominanceAlpha", 0.05)),
                    int(opts.get("permN", 10000)),
                    int(opts.get("permMaxExactN", 16)),
                    int(opts.get("permSeed", 0)),
                    str(opts.get("dominanceCompetitorMode", "mean") or "mean"),
                )
                dom_ml_tbl.to_excel(writer, sheet_name=sheet31(f"DomML_{model['id']}"), index=False)

    return out_xlsx


def make_visual_report_2stage(
    bundle_path: str | Path,
    unknown_excel_path: str | Path,
    out_dir: str | Path,
    explain_model_id: str = "LOGI",
) -> Path:
    del explain_model_id
    bundle = _load_bundle(bundle_path)
    opts = bundle["opts"]
    raw_u, canon_u, _, _ = read_params(unknown_excel_path)
    raw_u = align_by_canon(raw_u, canon_u, bundle["refParamCanon"])
    raw_u = fill_missing_with(raw_u, bundle["muCtrl"])
    zu = clip_z((raw_u - bundle["muCtrl"]) / bundle["sdCtrl"], float(opts["clipZ"]))
    zu = nan2zero(zu)

    stage1_hits = [m for m in bundle["stage1"]["models"] if m["id"] == opts["stage1FinalID"]]
    m1 = stage1_hits[0]["M"] if stage1_hits else bundle["stage1"]["models"][0]["M"]
    p_final = predict_stage1(m1, zu)
    thr1 = float(bundle["stage1"]["threshold"])

    sim_names, sim_means, self_label = get_similarity_refs(bundle, zu, opts)
    _, sim_tbl, _ = similarity_tables(zu, sim_names, sim_means, str(opts["simMetric"]), 1)
    mean_sim = np.nanmean(sim_tbl.to_numpy(dtype=float), axis=0)
    if self_label and self_label in sim_names:
        mean_sim[sim_names.index(self_label)] = -np.inf
    order = np.argsort(mean_sim)[::-1]
    order = [idx for idx in order if np.isfinite(mean_sim[idx])]

    if not order:
        g1 = sim_names[0]
        g2 = sim_names[0]
    else:
        g1 = sim_names[order[0]]
        g2 = sim_names[order[1]] if len(order) >= 2 else g1

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = Path(unknown_excel_path).stem
    fig_files = []

    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)
    ax.hist(p_final, bins=15)
    ax.axvline(thr1, linestyle="--", color="k")
    ax.set_xlabel("P(toxic)")
    ax.set_ylabel("Count")
    ax.set_title(f"Stage1 P(toxic) | {base}")
    ax.grid(True, alpha=0.3)
    fp = out_dir / f"{base}_FIG_ptoxic.png"
    fig.savefig(fp, dpi=220, bbox_inches="tight")
    plt.close(fig)
    fig_files.append(str(fp))

    fig = plt.figure(figsize=(6.5, 5.5))
    ax = fig.add_subplot(111)
    ax.scatter(
        sim_tbl[make_valid_names([g1])[0]].to_numpy(dtype=float),
        sim_tbl[make_valid_names([g2])[0]].to_numpy(dtype=float),
        s=36,
        alpha=0.65,
    )
    ax.grid(True, alpha=0.3)
    ax.set_xlabel(f"Sim {g1}")
    ax.set_ylabel(f"Sim {g2}")
    ax.set_title(f"Similarity scatter | {base}")
    fp = out_dir / f"{base}_FIG_similarity_scatter.png"
    fig.savefig(fp, dpi=220, bbox_inches="tight")
    plt.close(fig)
    fig_files.append(str(fp))

    fig_list_xlsx = out_dir / f"{base}_FIG_files.xlsx"
    pd.DataFrame({"FigureFiles": fig_files}).to_excel(fig_list_xlsx, sheet_name="Files", index=False)
    return fig_list_xlsx


def master_rows_2stage(
    bundle_path: str | Path,
    unknown_excel_path: str | Path,
    out_xlsx: str | Path,
    fig_list_xlsx: str | Path,
    set_tag: str = "UNKNOWN",
    true_group: str = "",
) -> pd.DataFrame:
    bundle = _load_bundle(bundle_path)
    opts = bundle["opts"]

    raw_u, canon_u, _, _ = read_params(unknown_excel_path)
    raw_u = align_by_canon(raw_u, canon_u, bundle["refParamCanon"])
    raw_u = fill_missing_with(raw_u, bundle["muCtrl"])
    zu = clip_z((raw_u - bundle["muCtrl"]) / bundle["sdCtrl"], float(opts["clipZ"]))
    zu = nan2zero(zu)

    stage1_hits = [m for m in bundle["stage1"]["models"] if m["id"] == opts["stage1FinalID"]]
    m1 = stage1_hits[0]["M"] if stage1_hits else bundle["stage1"]["models"][0]["M"]
    p_final = predict_stage1(m1, zu)
    thr1 = float(bundle["stage1"]["threshold"])
    is_toxic = p_final >= thr1

    sim_names, sim_means, self_label = get_similarity_refs(bundle, zu, opts)
    _, sim_tbl, top_tbl = similarity_tables(zu, sim_names, sim_means, str(opts["simMetric"]), max(2, int(opts["simTopK"])))

    top1_mode = mode_string(top_tbl["Top1_Group"].astype(str))
    top1_nonself = top_tbl["Top1_Group"].astype(str).to_numpy(dtype=object)
    if self_label and "Top2_Group" in top_tbl.columns:
        is_self = top1_nonself == self_label
        top1_nonself[is_self] = top_tbl.loc[is_self, "Top2_Group"].astype(str).to_numpy(dtype=object)
    top1_nonself_mode = mode_string(top1_nonself)

    # MATLAB's master workbook uses the non-self nearest-group summary for known training
    # files rather than reporting "Self" as the dominant group label.
    if str(set_tag).upper() == "TRAIN" and str(true_group).strip():
        top1_mode = top1_nonself_mode

    row = {
        "UnknownFile": str(unknown_excel_path),
        "Dataset": str(set_tag),
        "TrueGroup": str(true_group),
        "OutputExcel": str(out_xlsx),
        "FigureListExcel": str(fig_list_xlsx),
        "N": int(zu.shape[0]),
        "MeanPtoxic": float(np.nanmean(p_final)),
        "ToxicFrac": float(np.nanmean(is_toxic.astype(float))),
        "Top1_Group_Mode": top1_mode,
        "Top1_NonSelf_Group_Mode": top1_nonself_mode,
        "MeanSim_Vehicle": np.nan,
        "MeanSim_Self": np.nan,
    }

    if "Vehicle" in sim_names:
        row["MeanSim_Vehicle"] = float(np.nanmean(sim_tbl.iloc[:, sim_names.index("Vehicle")]))
    if self_label and self_label in sim_names:
        row["MeanSim_Self"] = float(np.nanmean(sim_tbl.iloc[:, sim_names.index(self_label)]))

    for model in bundle["stage2"]["models"]:
        p2, _ = predict_ovr(model["M"], zu)
        class_tox = np.asarray(model["M"]["class_names"], dtype=object).reshape(-1)
        score_veh = 1.0 - p_final
        score_tox = p_final[:, None] * p2
        sim_ml = np.column_stack([score_veh, score_tox])
        sim_ml = sim_ml / np.maximum(np.finfo(float).eps, np.sum(sim_ml, axis=1, keepdims=True))
        all_names = ["Vehicle", *class_tox.tolist()]
        mean_sim = np.nanmean(sim_ml, axis=0)
        best_idx = int(np.nanargmax(mean_sim))
        row[make_valid_names([f"Top1SimML_{model['id']}_Class"])[0]] = all_names[best_idx]
        row[make_valid_names([f"Top1SimML_{model['id']}_Value"])[0]] = float(mean_sim[best_idx])

        if bool(opts.get("saveDominanceStatsML", False)):
            dom_ml_tbl = make_dominance_table(
                pd.DataFrame(sim_ml, columns=make_valid_names(all_names)),
                all_names,
                "",
                True,
                f"MLSoftmax_{model['id']}",
                float(opts.get("dominanceAlpha", 0.05)),
                int(opts.get("permN", 10000)),
                int(opts.get("permMaxExactN", 16)),
                int(opts.get("permSeed", 0)),
                str(opts.get("dominanceCompetitorMode", "mean") or "mean"),
            )
            if not dom_ml_tbl.empty and "ReferenceGroup" in dom_ml_tbl.columns:
                mean_margin = pd.to_numeric(dom_ml_tbl["MeanMargin"], errors="coerce").to_numpy(dtype=float)
                if np.any(np.isfinite(mean_margin)):
                    best_dom_idx = int(np.nanargmax(mean_margin))
                    row[make_valid_names([f"Top1DomML_{model['id']}_Class"])[0]] = str(
                        dom_ml_tbl.iloc[best_dom_idx]["ReferenceGroup"]
                    )
                    row[make_valid_names([f"Top1DomML_{model['id']}_MeanMargin"])[0]] = float(
                        mean_margin[best_dom_idx]
                    )

    return pd.DataFrame([row])


def run_full_moa_analysis(
    train_dir: str,
    unknown_dir: str,
    out_dir: Optional[str] = None,
    use_robust_control_stats: bool = True,
    clip_z_value: float = 6.0,
    missing_frac_max: float = 0.3,
    n_top_features: int = 0,
    kfold_fish: int = 5,
    rng_seed: int = 0,
    n_trees: int = 200,
    make_figures: bool = True,
    explain_model_id: str = "LOGI",
    target_fpr: float = 0.05,
    file_toxic_mean_thr: float = 0.50,
    file_toxic_frac_thr: float = 0.20,
    use_outlier_downweight_train: bool = True,
    use_outlier_downweight_unknown: bool = True,
    outlier_top_percent: float = 2.0,
    outlier_min_weight: float = 0.25,
    stage1_final_id: str = "S1_LOGI",
    sim_metric: str = "euclid",
    sim_top_k: int = 3,
    include_self_in_similarity: bool = True,
    self_similarity_label: str = "Self",
    save_tost_vs_self: bool = True,
    tost_alpha: float = 0.05,
    tost_delta_softmax: float = 0.15,
    tost_delta_distance: float = 1.18,
    sim_multiplier: float = 1.5,
    save_dominance_stats: bool = True,
    dominance_alpha: float = 0.05,
    exclude_self_in_dominance: bool = True,
    save_dominance_stats_ml: bool = True,
    dominance_competitor_mode: str = "mean",
    perm_n: int = 10000,
    perm_max_exact_n: int = 16,
    perm_seed: int = 0,
    include_train_in_analysis: bool = True,
    min_match_frac: float = 0.90,
    train_spec: Optional[pd.DataFrame] = None,
    unknown_files: Optional[Sequence[str | Path]] = None,
) -> Dict:
    del file_toxic_mean_thr
    del file_toxic_frac_thr
    del use_outlier_downweight_train
    del use_outlier_downweight_unknown
    del outlier_top_percent
    del outlier_min_weight

    train_dir = str(train_dir)
    unknown_dir = str(unknown_dir)
    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    if out_dir is None:
        out_dir = str(Path(unknown_dir) / f"PRIZM_2STAGE_results_{ts}")
    out_dir = str(out_dir)
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    opts = {
        "useRobustControlStats": bool(use_robust_control_stats),
        "clipZ": float(clip_z_value),
        "missingFracMax": float(missing_frac_max),
        "minMatchFrac": float(min_match_frac),
        "kfold": int(kfold_fish),
        "rngSeed": int(rng_seed),
        "targetFPR": float(target_fpr),
        "nTrees": int(n_trees),
        "nTopFeatures": int(n_top_features),
        "stage1FinalID": str(stage1_final_id),
        "simMetric": str(sim_metric),
        "simTopK": int(sim_top_k),
        "includeSelfInSimilarity": bool(include_self_in_similarity),
        "selfSimilarityLabel": str(self_similarity_label),
        "saveTOSTvsSelf": bool(save_tost_vs_self),
        "tostAlpha": float(tost_alpha),
        "tostDeltaSoftmax": float(tost_delta_softmax),
        "tostDeltaDistance": float(tost_delta_distance),
        "simMultiplier": float(sim_multiplier),
        "saveDominanceStats": bool(save_dominance_stats),
        "dominanceAlpha": float(dominance_alpha),
        "excludeSelfInDominance": bool(exclude_self_in_dominance),
        "saveDominanceStatsML": bool(save_dominance_stats_ml),
        "dominanceCompetitorMode": str(dominance_competitor_mode),
        "permN": int(perm_n),
        "permMaxExactN": int(perm_max_exact_n),
        "permSeed": int(perm_seed),
        "makeFigures": bool(make_figures),
        "includeTrainInAnalysis": bool(include_train_in_analysis),
    }

    train_spec = _normalize_train_spec(train_dir, train_spec)
    bundle, x, y, match_tbl = prepare_training_bundle(train_spec, opts)
    bundle["opts"] = opts

    stage1_models = define_models_stage1()
    stage2_models = define_models_stage2_ovr()

    stage1_report, stage1_oof_pack = cv_stage1_models(stage1_models, x, y, opts)
    thr_final = get_thr_from_oof(stage1_oof_pack, str(opts["stage1FinalID"]), float(opts["targetFPR"]))
    bundle["stage1"] = {"threshold": float(thr_final), "finalID": str(opts["stage1FinalID"])}

    x_tox, y_tox = extract_toxic_subset(x, y)
    stage2_report, stage2_auc_by_class, stage2_cm_pack = cv_stage2_models_ovr(stage2_models, x_tox, y_tox, opts)
    end2end_report, end2end_auc_by_class, end2end_cm_pack = cv_end2end_fixed_stage1(
        stage1_models, stage2_models, x, y, opts
    )

    unknown_files_resolved = _normalize_unknown_files(unknown_dir, unknown_files)
    info = make_info_table(train_spec, unknown_files_resolved, bundle, out_dir)
    train_report_xlsx = Path(out_dir) / "TRAIN_2STAGE_report.xlsx"
    with pd.ExcelWriter(train_report_xlsx, engine="openpyxl") as writer:
        stage1_report.to_excel(writer, sheet_name="Stage1_CV", index=False)
        match_tbl.to_excel(writer, sheet_name="FeatureMatch", index=False)
        if not stage2_report.empty:
            stage2_report.to_excel(writer, sheet_name="Stage2_CV_ToxicOnly", index=False)
        if not stage2_auc_by_class.empty:
            stage2_auc_by_class.to_excel(writer, sheet_name="Stage2_AUC_ByClass", index=False)
        if not end2end_report.empty:
            end2end_report.to_excel(writer, sheet_name="End2End_CV", index=False)
        if not end2end_auc_by_class.empty:
            end2end_auc_by_class.to_excel(writer, sheet_name="End2End_AUC_ByClass", index=False)
        info.to_excel(writer, sheet_name="Info", index=False)
    write_cm_pack(train_report_xlsx, stage2_cm_pack, "S2_")
    write_cm_pack(train_report_xlsx, end2end_cm_pack, "E2E_")

    bundle["stage1"]["models"] = train_stage1_all(stage1_models, x, y, opts)
    bundle["stage2"] = {"models": train_stage2_all(stage2_models, x_tox, y_tox, opts)}
    bundle["groupNames"] = _sorted_unique(y)
    bundle["groupMeanZ"] = np.vstack([np.nanmean(x[np.asarray(y, dtype=object) == g], axis=0) for g in bundle["groupNames"]])

    bundle_path = Path(out_dir) / "prizm_bundle_2STAGE.mat"
    _save_bundle(bundle_path, bundle)

    analysis_files: List[str] = [str(p) for p in unknown_files_resolved]
    analysis_set: List[str] = ["UNKNOWN"] * len(analysis_files)
    analysis_group: List[str] = [""] * len(analysis_files)
    if bool(opts["includeTrainInAnalysis"]):
        analysis_files.extend(train_spec["FullPath"].astype(str).tolist())
        analysis_set.extend(["TRAIN"] * len(train_spec))
        analysis_group.extend(train_spec["Group"].astype(str).tolist())

    seen = set()
    analysis_files_u: List[str] = []
    analysis_set_u: List[str] = []
    analysis_group_u: List[str] = []
    for fp, tag, grp in zip(analysis_files, analysis_set, analysis_group):
        key = str(Path(fp).resolve())
        if key in seen:
            continue
        seen.add(key)
        analysis_files_u.append(fp)
        analysis_set_u.append(tag)
        analysis_group_u.append(grp)

    master_rows = []
    for fp, tag, grp in zip(analysis_files_u, analysis_set_u, analysis_group_u):
        out_xlsx = predict_unknown_2stage(bundle_path, fp, out_dir, tag, grp)
        if bool(opts["makeFigures"]):
            fig_list_xlsx = make_visual_report_2stage(bundle_path, fp, out_dir, explain_model_id=explain_model_id)
        else:
            fig_list_xlsx = Path(out_dir) / f"{Path(fp).stem}_FIG_files.xlsx"
            pd.DataFrame({"FigureFiles": []}).to_excel(fig_list_xlsx, sheet_name="Files", index=False)
        master_rows.append(master_rows_2stage(bundle_path, fp, out_xlsx, fig_list_xlsx, tag, grp))

    master_df = pd.concat(master_rows, ignore_index=True) if master_rows else pd.DataFrame()
    master_xlsx = Path(out_dir) / "MASTER_unknown_2STAGE.xlsx"
    with pd.ExcelWriter(master_xlsx, engine="openpyxl") as writer:
        master_df.to_excel(writer, sheet_name="Master", index=False)
        info.to_excel(writer, sheet_name="Info", index=False)

    return {
        "out_dir": out_dir,
        "bundle_path": str(bundle_path),
        "train_report_xlsx": str(train_report_xlsx),
        "master_xlsx": str(master_xlsx),
        "n_unknown_files": len(unknown_files_resolved),
    }
