"""
PRIZM mini-panel + heatmap + LDA/PCA/t-SNE analysis.

Python port of:
- PRIZM_minipanel_heatmap_clustering_20260129.mlx
- PRIZM_make_mini_bar_panel_20260326.m
"""

from __future__ import annotations

import math
import re
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator
from scipy import linalg
from scipy.stats import chi2, f, mannwhitneyu, spearmanr, t
from sklearn.covariance import MinCovDet
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from prizm_napari.input_discovery import discover_perfish_workbooks
from prizm_napari.plot_colors import distinct_categorical_colors, neutral_categorical_colors


# Use the font family established for PRIZM figure exports.
plt.rcParams["font.family"] = "Nimbus Sans"


# PRIZM functional metric partition used by the major/minor reports.
# Keep this explicit: substring-based numeric-column discovery incorrectly pulled
# uncertainty/count outputs into the functional panels (and even treated the
# ``id`` inside ``Valid`` as an identifier exclusion).
MAJOR_METRICS: Tuple[str, ...] = (
    "V_HR_bpm",
    "A_HR_bpm",
    "Interval_CV",
    "SystolicDuration_mean",
    "DiastolicDuration_mean",
    "EF_mean",
    "FS_mean",
    "V_ED_mean",
    "V_ES_mean",
    "SV_index_mean",
    "CO_index_mean",
    "ContractilitySpeed",
    "RelaxationSpeed",
    "A_ED_mean",
    "A_ES_mean",
    "Diastolic_AtoV_ratio",
    "MajorMinor_ratio_ED",
    "SVBA_Distance_mean",
    "VA_Dist_Center_mean",
    "VA_Ang_Center_major_mean",
    "AV_Delay_Mean",
    "PLV",
)

MINOR_METRICS: Tuple[str, ...] = (
    "Interval_SD_s",
    "SystolicFraction",
    "EF_SD",
    "FS_SD",
    "V_ED_SD",
    "V_ES_SD",
    "A_ED_SD",
    "A_ES_SD",
    "A_ED_ES_Diff_mean",
    "SVBA_Distance_Y_mean",
    "VA_Dist_Bottom_mean",
    "VA_Dist_Top_mean",
    "VA_Ang_Center_raw_mean",
    "VA_Ang_Bottom_raw_mean",
    "VA_Ang_Bottom_major_mean",
    "VA_Ang_Top_raw_mean",
    "VA_Ang_Top_major_mean",
    "MaxCorrLag_s",
    "CrossCorrCoeff",
    "PhaseSynchronyIndex",
    "PearsonCorr",
    "PeakTimeDiff_SD",
    "AV_Delay_SD",
)

FUNCTIONAL_METRICS: Tuple[str, ...] = MAJOR_METRICS + MINOR_METRICS

# Compact 5x5 tiles use a deliberately sparse number of y intervals.
# Matplotlib's default locator uses many more ticks at the same physical tile
# height. These interval ceilings preserve the established density while still
# allowing numeric tick values to adapt to a new dataset.
_PANEL_Y_NBINS = dict(
    zip(
        FUNCTIONAL_METRICS,
        (
            3, 3, 4, 4, 3, 5, 4, 6, 4, 3, 4, 3, 3, 4, 4, 4, 4, 3, 3, 3, 3, 4,
            3, 5, 4, 5, 4, 3, 5, 3, 4, 5, 3, 4, 4, 3, 5, 4, 3, 2, 3, 4, 3, 3, 3,
        ),
    )
)


def infer_group_from_filename(path: str | Path) -> str:
    stem = Path(path).stem
    prefix = "PerFishMetrics_"
    if stem.startswith(prefix):
        rest = stem[len(prefix) :]
        m = re.search(r"_(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})$", rest)
        if m:
            return rest[: m.start()].strip()
        return rest.strip()
    return stem.strip()


def parse_drug_dose(group_name: str) -> Tuple[str, float]:
    s = str(group_name).strip()
    m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
    if m is None:
        return re.sub(r"[_\-\s]+$", "", s), np.nan
    dose = float(m.group(0))
    prefix = re.sub(r"[_\-\s]+$", "", s[: m.start()].strip())
    return (prefix if prefix else s), dose


def find_ctrl_idx(group_names: Sequence[str], control_group_name: str = "CTRL") -> int:
    groups = np.asarray(group_names, dtype=object)
    if control_group_name:
        hit = np.where(np.char.lower(groups.astype(str)) == str(control_group_name).lower())[0]
        if len(hit):
            return int(hit[0])
    low = np.char.lower(groups.astype(str))
    for c in ("ctrl", "control", "vehicle", "dmso", "veh"):
        idx = np.where(np.char.find(low, c) >= 0)[0]
        if len(idx):
            return int(idx[0])
    return 0


def fdr_bh(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, dtype=float).reshape(-1)
    q = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    pp = p[ok]
    if len(pp) == 0:
        return q
    order = np.argsort(pp)
    ps = pp[order]
    m = len(ps)
    qs = ps * m / (np.arange(m) + 1.0)
    for i in range(m - 2, -1, -1):
        qs[i] = min(qs[i], qs[i + 1])
    qs = np.minimum(qs, 1.0)
    tmp = np.full_like(pp, np.nan)
    tmp[order] = qs
    q[ok] = tmp
    return q


def sig_star(p: float, alpha: float = 0.05) -> str:
    if not np.isfinite(p):
        return ""
    if p < 1e-4:
        return "****"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < alpha:
        return "*"
    return ""


def welch_ttest2_details(
    a: Sequence[float],
    b: Sequence[float],
    alpha: float = 0.05,
) -> Tuple[float, float, float, float, float, float, float]:
    xa = np.asarray(a, dtype=float).reshape(-1)
    xb = np.asarray(b, dtype=float).reshape(-1)
    xa = xa[np.isfinite(xa)]
    xb = xb[np.isfinite(xb)]
    if len(xa) < 2 or len(xb) < 2:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    n1 = len(xa)
    n2 = len(xb)
    m1 = float(np.mean(xa))
    m2 = float(np.mean(xb))
    v1 = float(np.var(xa, ddof=1))
    v2 = float(np.var(xb, ddof=1))
    se = math.sqrt(v1 / n1 + v2 / n2)
    mean_diff = m1 - m2
    tstat = mean_diff / (se + np.finfo(float).eps)
    denom = (v1 * v1) / (n1 * n1 * max(n1 - 1, 1)) + (v2 * v2) / (n2 * n2 * max(n2 - 1, 1))
    dfw = ((v1 / n1 + v2 / n2) ** 2) / max(denom, np.finfo(float).eps)
    if not np.isfinite(dfw) or dfw <= 0:
        return np.nan, mean_diff, np.nan, np.nan, tstat, np.nan, se
    pval = float(2.0 * (1.0 - t.cdf(abs(tstat), dfw)))
    tcrit = float(t.ppf(1.0 - alpha / 2.0, dfw))
    ci_lo = mean_diff - tcrit * se
    ci_hi = mean_diff + tcrit * se
    return pval, mean_diff, ci_lo, ci_hi, tstat, dfw, se


def welch_anova_1way(xcell: Sequence[Sequence[float]]) -> Tuple[float, float, float, float, int, int]:
    means = []
    variances = []
    ns = []
    n_total = 0

    for x in xcell:
        xx = np.asarray(x, dtype=float).reshape(-1)
        xx = xx[np.isfinite(xx)]
        ni = len(xx)
        if ni < 2:
            continue
        means.append(float(np.mean(xx)))
        variances.append(max(float(np.var(xx, ddof=1)), np.finfo(float).eps))
        ns.append(ni)
        n_total += ni

    k_used = len(ns)
    if k_used < 2:
        return np.nan, np.nan, np.nan, np.nan, k_used, n_total

    means = np.asarray(means, dtype=float)
    variances = np.asarray(variances, dtype=float)
    ns = np.asarray(ns, dtype=float)
    weights = ns / variances
    w_sum = np.sum(weights)
    xw = float(np.sum(weights * means) / w_sum)
    df1 = float(k_used - 1)
    a_term = float(np.sum(weights * np.square(means - xw)) / df1)
    tmp = (1.0 / (ns - 1.0)) * np.square(1.0 - (weights / w_sum))
    b_term = 1.0 + (2.0 * (k_used - 2.0) / (k_used * k_used - 1.0)) * float(np.sum(tmp))
    fw = a_term / b_term
    df2 = float((k_used * k_used - 1.0) / (3.0 * np.sum(tmp)))
    pval = float(1.0 - f.cdf(fw, df1, df2))
    return pval, fw, df1, df2, k_used, int(n_total)


def stats_welchanova_vs_ref(
    tables: List[pd.DataFrame],
    params: Sequence[str],
    group_names: Sequence[str],
    ref_idx: int,
    alpha_sig: float = 0.05,
    pair_method: str = "welch",
    use_fdr: bool = False,
    save_all_pairs_excel: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    params = np.asarray(params, dtype=object)
    groups = np.asarray(group_names, dtype=object)
    n_p, n_g = len(params), len(groups)
    p_raw = np.full((n_p, n_g), np.nan, dtype=float)
    p_adj = np.full((n_p, n_g), np.nan, dtype=float)
    stars = np.full((n_p, n_g), "", dtype=object)
    rows = []
    anova_rows = []
    pairwise_rows = []

    for pi, p in enumerate(params):
        xcell = []
        for gi in range(n_g):
            xg = pd.to_numeric(tables[gi][p], errors="coerce").to_numpy(dtype=float)
            xg = xg[np.isfinite(xg)]
            xcell.append(xg)

        p_a, fw, df1, df2, k_used, n_total = welch_anova_1way(xcell)
        anova_rows.append(
            {
                "Parameter": p,
                "WelchANOVA_p": p_a,
                "WelchANOVA_F": fw,
                "df1": df1,
                "df2": df2,
                "GroupsUsed": k_used,
                "Ntotal": n_total,
            }
        )

        xref = xcell[ref_idx]
        pvec = np.full((n_g,), np.nan, dtype=float)
        mean_diff = np.full((n_g,), np.nan, dtype=float)
        ci_lo = np.full((n_g,), np.nan, dtype=float)
        ci_hi = np.full((n_g,), np.nan, dtype=float)
        se_diff = np.full((n_g,), np.nan, dtype=float)
        t_stat = np.full((n_g,), np.nan, dtype=float)
        df_w = np.full((n_g,), np.nan, dtype=float)

        for gi in range(n_g):
            if gi == ref_idx:
                continue
            xg = xcell[gi]
            if len(xref) >= 2 and len(xg) >= 2:
                if str(pair_method).lower() == "ranksum":
                    pp = float(mannwhitneyu(xref, xg, alternative="two-sided").pvalue)
                    md = float(np.mean(xref) - np.mean(xg))
                    clo = chi = tt = dfi = se = np.nan
                else:
                    pp, md, clo, chi, tt, dfi, se = welch_ttest2_details(xref, xg, alpha_sig)
                pvec[gi] = pp
                mean_diff[gi] = md
                ci_lo[gi] = clo
                ci_hi[gi] = chi
                se_diff[gi] = se
                t_stat[gi] = tt
                df_w[gi] = dfi

        idx_test = np.array([j for j in range(n_g) if j != ref_idx], dtype=int)
        if use_fdr and len(idx_test):
            padj_vec = np.array(pvec, copy=True)
            padj_vec[idx_test] = fdr_bh(pvec[idx_test])
        else:
            padj_vec = pvec

        for gi, g in enumerate(groups):
            xg = xcell[gi]
            if gi != ref_idx:
                p_raw[pi, gi] = pvec[gi]
                p_adj[pi, gi] = padj_vec[gi]
                stars[pi, gi] = sig_star(padj_vec[gi], alpha_sig)
            rows.append(
                {
                    "Parameter": p,
                    "Group": g,
                    "N": int(len(xg)),
                    "Mean": float(np.mean(xg)) if len(xg) else np.nan,
                    "SD": float(np.std(xg, ddof=1)) if len(xg) > 1 else np.nan,
                    "SEM": float(np.std(xg, ddof=1) / math.sqrt(len(xg))) if len(xg) > 1 else np.nan,
                    "ReferenceGroup": groups[ref_idx],
                    "WelchANOVA_p": p_a,
                    "WelchANOVA_F": fw,
                    "df1": df1,
                    "df2": df2,
                    "P_raw_vs_REF": pvec[gi] if gi != ref_idx else np.nan,
                    "P_adj_vs_REF": padj_vec[gi] if gi != ref_idx else np.nan,
                    "Star": stars[pi, gi] if gi != ref_idx else "",
                    "MeanDiff_REFminusGroup": mean_diff[gi] if gi != ref_idx else np.nan,
                    "CI95_Lo": ci_lo[gi] if gi != ref_idx else np.nan,
                    "CI95_Hi": ci_hi[gi] if gi != ref_idx else np.nan,
                    "SE_diff": se_diff[gi] if gi != ref_idx else np.nan,
                    "t": t_stat[gi] if gi != ref_idx else np.nan,
                    "DF": df_w[gi] if gi != ref_idx else np.nan,
                    "PairwiseMethod": str(pair_method),
                    "Alpha": float(alpha_sig),
                    "UseFDR": bool(use_fdr),
                }
            )

        if save_all_pairs_excel:
            p_pairs = []
            pair_meta = []
            pair_i = []
            pair_j = []
            for i in range(n_g - 1):
                for j in range(i + 1, n_g):
                    xi = xcell[i]
                    xj = xcell[j]
                    if len(xi) >= 2 and len(xj) >= 2:
                        pp, md, clo, chi, tt, dfi, se = welch_ttest2_details(xi, xj, alpha_sig)
                    else:
                        pp = md = clo = chi = tt = dfi = se = np.nan
                    p_pairs.append(pp)
                    pair_meta.append((md, clo, chi, se, tt, dfi))
                    pair_i.append(i)
                    pair_j.append(j)
            q_pairs = fdr_bh(np.asarray(p_pairs, dtype=float)) if use_fdr else np.asarray(p_pairs, dtype=float)
            for k, pp in enumerate(p_pairs):
                md, clo, chi, se, tt, dfi = pair_meta[k]
                pairwise_rows.append(
                    {
                        "Parameter": p,
                        "Group1": groups[pair_i[k]],
                        "Group2": groups[pair_j[k]],
                        "P_raw": pp,
                        "P_adj": q_pairs[k],
                        "Star": sig_star(q_pairs[k], alpha_sig),
                        "MeanDiff_G1minusG2": md,
                        "CI95_Lo": clo,
                        "CI95_Hi": chi,
                        "SE_diff": se,
                        "t": tt,
                        "DF": dfi,
                        "PairwiseMethod": "welch",
                        "Alpha": float(alpha_sig),
                        "UseFDR": bool(use_fdr),
                    }
                )

    return (
        p_raw,
        p_adj,
        stars,
        pd.DataFrame(rows),
        pd.DataFrame(anova_rows),
        pd.DataFrame(pairwise_rows),
    )


def build_group_meta_and_colors(
    classes: Sequence[str],
    control_keywords: Sequence[str],
    t_min: float = 0.20,
    t_max: float = 0.95,
) -> Tuple[pd.DataFrame, np.ndarray]:
    classes = np.asarray(classes, dtype=object)
    low = np.char.lower(classes.astype(str))
    is_control = np.zeros((len(classes),), dtype=bool)
    for k in control_keywords:
        is_control |= np.char.find(low, str(k).lower()) >= 0

    drugs = np.full((len(classes),), "", dtype=object)
    doses = np.full((len(classes),), np.nan, dtype=float)
    for i, c in enumerate(classes):
        if is_control[i]:
            drugs[i] = "Control"
            doses[i] = -np.inf
        else:
            d, dose = parse_drug_dose(str(c))
            drugs[i] = d
            doses[i] = dose

    colors = np.zeros((len(classes), 3), dtype=float)
    control_idx = np.where(is_control)[0]
    non_control_idx = np.where(~is_control)[0]
    if len(control_idx):
        colors[control_idx, :] = neutral_categorical_colors(len(control_idx))
    if len(non_control_idx):
        colors[non_control_idx, :] = distinct_categorical_colors(len(non_control_idx))

    tord = pd.DataFrame(
        {"groupStr": classes, "drug": drugs, "dose": doses, "isControl": is_control}
    )
    return tord, colors


def fisher_lda_2d(x: np.ndarray, y: np.ndarray, classes: Optional[Sequence[str]] = None) -> Tuple[np.ndarray, np.ndarray]:
    if classes is None:
        classes = pd.unique(np.asarray(y, dtype=object))
    else:
        classes = np.asarray(classes, dtype=object)
    n, p = x.shape
    mu = np.nanmean(x, axis=0)
    sw = np.zeros((p, p), dtype=float)
    sb = np.zeros((p, p), dtype=float)
    for c in classes:
        xi = x[y == c]
        if len(xi) == 0:
            continue
        mi = np.nanmean(xi, axis=0)
        ni = len(xi)
        ci = np.cov(xi, rowvar=False, bias=True)
        sw += ni * ci
        d = (mi - mu).reshape(-1, 1)
        sb += ni * (d @ d.T)
    sw += 1e-6 * np.eye(p)
    vals, vecs = linalg.eig(sb, sw)
    order = np.argsort(np.real(vals))[::-1]
    w = np.real(vecs[:, order[:2]])
    z = x @ w
    return z, w


def orient_lda_axes_by_top_corr(xz: np.ndarray, z: np.ndarray, w: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    corr = np.full((xz.shape[1], 2), np.nan, dtype=float)
    for j in range(2):
        r = np.array(
            [
                np.corrcoef(xz[:, k], z[:, j])[0, 1]
                if np.std(xz[:, k]) > 0 and np.std(z[:, j]) > 0
                else np.nan
                for k in range(xz.shape[1])
            ]
        )
        idx = int(np.nanargmax(np.abs(r)))
        if np.isfinite(r[idx]) and r[idx] < 0:
            z[:, j] *= -1
            w[:, j] *= -1
            r *= -1
        corr[:, j] = r
    return z, w, corr


def cov_ellipse_points(
    mu: np.ndarray, s: np.ndarray, conf: float = 0.68, ellipse_scale: float = 0.85, n_points: int = 240
) -> np.ndarray:
    if np.any(~np.isfinite(s)) or np.linalg.cond(s) > 1e12:
        s = s + 1e-6 * np.eye(2)
    k2 = chi2.ppf(conf, 2) * (ellipse_scale**2)
    vals, vecs = np.linalg.eigh(s)
    vals = np.maximum(vals, 1e-12)
    t = np.linspace(0, 2 * np.pi, n_points)
    a = np.sqrt(k2) * np.sqrt(vals[0])
    b = np.sqrt(k2) * np.sqrt(vals[1])
    p = (vecs @ np.vstack([a * np.cos(t), b * np.sin(t)])) + mu.reshape(2, 1)
    return p.T


def plot2d_filled_ellipse(
    z: np.ndarray,
    y: np.ndarray,
    classes: Sequence[str],
    colors: np.ndarray,
    title: str,
    xlabel: str,
    ylabel: str,
    conf: float = 0.68,
    face_alpha: float = 0.08,
    ellipse_scale: float = 0.85,
    point_size: int = 24,
    use_robust_cov: bool = True,
) -> plt.Figure:
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    classes = np.asarray(classes, dtype=object)
    for i, c in enumerate(classes):
        idx = y == c
        ax.scatter(
            z[idx, 0],
            z[idx, 1],
            s=point_size,
            c=[colors[i]],
            edgecolors="black",
            linewidths=0.55,
            alpha=0.95,
            label=str(c),
        )
    for i, c in enumerate(classes):
        idx = y == c
        zi = z[idx, :2]
        if len(zi) < 3:
            continue
        # Robust covariance requires enough observations relative to the number
        # of dimensions. Fall back deterministically instead of aborting the
        # MiniPanel run for small groups.
        if use_robust_cov and len(zi) >= 2 * zi.shape[1]:
            try:
                mcd = MinCovDet(random_state=1).fit(zi)
                mu = mcd.location_
                s = mcd.covariance_
            except Exception:
                mu = np.nanmean(zi, axis=0)
                s = np.cov(zi, rowvar=False, bias=True)
        else:
            mu = np.nanmean(zi, axis=0)
            s = np.cov(zi, rowvar=False, bias=True)
        e = cov_ellipse_points(mu, s, conf=conf, ellipse_scale=ellipse_scale)
        ax.fill(e[:, 0], e[:, 1], color=colors[i], alpha=face_alpha)
        ax.plot(e[:, 0], e[:, 1], color=colors[i], lw=1.2)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


def group_centroids_ld(z: np.ndarray, y: np.ndarray, classes: Sequence[str], tord: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for i, c in enumerate(classes):
        idx = y == c
        rows.append(
            {
                "Group": c,
                "Drug": tord.iloc[i]["drug"],
                "Dose": tord.iloc[i]["dose"],
                "IsControl": bool(tord.iloc[i]["isControl"]),
                "N": int(np.sum(idx)),
                "Mean_LD1": float(np.nanmean(z[idx, 0])) if np.any(idx) else np.nan,
                "Mean_LD2": float(np.nanmean(z[idx, 1])) if np.any(idx) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def write_auto_interpretation(
    txt_path: str | Path,
    tcent: pd.DataFrame,
    tcorr_ld1: pd.DataFrame,
    tcorr_ld2: pd.DataFrame,
    tord: pd.DataFrame,
) -> None:
    lines = []
    lines.append("=== Fisher LDA (LD1/LD2) Auto Interpretation ===")
    lines.append(f"Generated: {pd.Timestamp.now()}")
    lines.append("")
    top_n = min(8, len(tcorr_ld1))
    lines.append("[Top LD1 features by |corr|]")
    for i in range(top_n):
        lines.append(f"  - {tcorr_ld1.iloc[i]['Feature']} (corr={tcorr_ld1.iloc[i]['corr_LD1']:.3f})")
    lines.append("")
    top_n = min(8, len(tcorr_ld2))
    lines.append("[Top LD2 features by |corr|]")
    for i in range(top_n):
        lines.append(f"  - {tcorr_ld2.iloc[i]['Feature']} (corr={tcorr_ld2.iloc[i]['corr_LD2']:.3f})")
    lines.append("")
    lines.append("[Group centroids]")
    for _, r in tcent.iterrows():
        lines.append(
            f"  - {r['Group']:<20} (N={int(r['N'])}) LD1={r['Mean_LD1']:.3f}, LD2={r['Mean_LD2']:.3f}"
        )
    lines.append("")
    k = len(tcent)
    d = np.full((k, k), np.nan, dtype=float)
    for i in range(k):
        for j in range(i + 1, k):
            d[i, j] = np.hypot(
                tcent.iloc[i]["Mean_LD1"] - tcent.iloc[j]["Mean_LD1"],
                tcent.iloc[i]["Mean_LD2"] - tcent.iloc[j]["Mean_LD2"],
            )
    flat = np.dstack(np.unravel_index(np.argsort(d.ravel())[::-1], d.shape))[0]
    lines.append("[Top centroid separations]")
    cnt = 0
    for i, j in flat:
        if i >= j or not np.isfinite(d[i, j]) or d[i, j] <= 0:
            continue
        cnt += 1
        lines.append(
            f"  {cnt}) {tcent.iloc[i]['Group']} vs {tcent.iloc[j]['Group']} (dist={d[i, j]:.3f})"
        )
        if cnt >= 5:
            break
    lines.append("")
    lines.append("[Dose trends within drugs]")
    drug_list = pd.unique(tord.loc[~tord["isControl"], "drug"])
    for dn in drug_list:
        idx = (
            (tcent["Drug"] == dn)
            & (~tcent["IsControl"])
            & np.isfinite(tcent["Dose"].to_numpy(dtype=float))
        )
        td = tcent[idx]
        if len(td) < 3:
            continue
        rho1, p1 = spearmanr(td["Dose"], td["Mean_LD1"], nan_policy="omit")
        rho2, p2 = spearmanr(td["Dose"], td["Mean_LD2"], nan_policy="omit")
        lines.append(
            f"  - {dn}: dose vs LD1 rho={rho1:.3f} (p={p1:.3g}), dose vs LD2 rho={rho2:.3f} (p={p2:.3g})"
        )
    lines.append("")
    lines.append("[Notes]")
    lines.append("  - LD axes are linear combinations of PRIZM features and can change by dataset composition.")
    lines.append("  - Axis signs were oriented for interpretation stability.")
    lines.append("  - Validate separation with cross-validation-based classification performance.")
    Path(txt_path).write_text("\n".join(lines), encoding="utf-8")


def recommend_barpanel_size(n_cols: int, n_rows: int, n_groups: int) -> Tuple[float, float]:
    """Return the established PRIZM mini-bar-panel size in inches."""
    n_cols = max(int(n_cols), 1)
    n_rows = max(int(n_rows), 1)
    n_groups = max(int(n_groups), 1)
    group_extra = max(n_groups - 5, 0)
    if n_cols == 5:
        tile_w_in = 1.82 + 0.02 * group_extra
        tile_h_in = 1.58
        fig_w_in = n_cols * tile_w_in + 0.28
        fig_h_in = max(14.5, n_rows * tile_h_in + 0.50)
    else:
        tile_w_in = 1.90 + 0.02 * group_extra
        tile_h_in = 1.45
        fig_w_in = n_cols * tile_w_in + 0.35
        fig_h_in = max(9.0, n_rows * tile_h_in + 0.50)
    return fig_w_in, fig_h_in


def save_triplet(
    fig: plt.Figure,
    png_path: Path,
    pdf_path: Path,
    tif_path: Path,
    dpi: int,
    target_raster_size: Optional[Tuple[int, int]] = None,
) -> None:
    fig.savefig(pdf_path, facecolor="white", bbox_inches="tight", pad_inches=0.03)
    with tempfile.NamedTemporaryFile(
        suffix=".png",
        dir=str(png_path.parent),
        delete=False,
    ) as tmp:
        tmp_png = Path(tmp.name)
    try:
        fig.savefig(tmp_png, dpi=dpi, facecolor="white", bbox_inches="tight", pad_inches=0.03)
        try:
            from PIL import Image

            # Reuse the exact RGB pixel array for TIFF so PNG/TIFF dimensions
            # and color channels cannot diverge.
            with Image.open(tmp_png) as im:
                rgb = im.convert("RGB")
                if target_raster_size is not None and rgb.size != tuple(target_raster_size):
                    rgb = rgb.resize(tuple(target_raster_size), Image.Resampling.LANCZOS)
                rgb.save(png_path, dpi=(dpi, dpi))
                rgb.save(tif_path, dpi=(dpi, dpi))
        except Exception:
            tmp_png.replace(png_path)
            fig.savefig(tif_path, dpi=dpi, facecolor="white")
    finally:
        if tmp_png.exists():
            tmp_png.unlink()


def _metric_summary(
    tables: Sequence[pd.DataFrame],
    params: Sequence[str],
    error_type: str,
) -> Tuple[np.ndarray, np.ndarray]:
    m = np.full((len(params), len(tables)), np.nan, dtype=float)
    e = np.full_like(m, np.nan)
    for pi, metric in enumerate(params):
        for gi, table in enumerate(tables):
            x = pd.to_numeric(table[metric], errors="coerce").to_numpy(dtype=float)
            x = x[np.isfinite(x)]
            if len(x) == 0:
                continue
            m[pi, gi] = float(np.mean(x))
            if str(error_type).lower() == "sem":
                e[pi, gi] = float(np.std(x, ddof=1) / np.sqrt(len(x))) if len(x) > 1 else np.nan
            elif str(error_type).lower() == "none":
                e[pi, gi] = 0.0
            else:
                e[pi, gi] = float(np.std(x, ddof=1)) if len(x) > 1 else np.nan
    return m, e


def _render_metric_partition(
    *,
    partition_name: str,
    tables: Sequence[pd.DataFrame],
    params: Sequence[str],
    group_names: np.ndarray,
    ref_idx: int,
    panel_dir: Path,
    error_type: str,
    n_cols: int,
    same_y_lim: bool,
    main_title: str,
    dpi: int,
    make_heatmap: bool,
    heatmap_title_prefix: str,
    include_ctrl_in_heatmap: bool,
    alpha_sig: float,
    use_fdr: bool,
    stat_method: str,
    save_all_pairs_excel: bool,
    show_sig_on_bar: bool,
    show_sig_on_heatmap: bool,
) -> Dict[str, object]:
    params = list(params)
    n_p = len(params)
    n_g = len(group_names)
    m, e = _metric_summary(tables, params, error_type)
    p_raw, p_adj, stars, stats_long, anova_summary, pairwise_all = stats_welchanova_vs_ref(
        list(tables),
        params,
        group_names,
        ref_idx,
        alpha_sig,
        stat_method,
        use_fdr,
        save_all_pairs_excel,
    )

    show_xlabels_only_bottom_row = True
    x_tick_rotation = 28
    axes_font_size = 7.8
    title_font_size = 8.8
    sig_font_size = 10
    bar_width = 0.78
    bar_line_width = 0.75
    err_line_width = 0.80
    panel_pad_ratio = 0.18
    sig_offset_ratio = 0.05
    same_y_extra_top_ratio = 0.20
    same_y_extra_bottom_ratio = 0.10

    n_cols = max(int(n_cols), 1)
    n_rows = int(math.ceil(n_p / n_cols))
    fig_w_in, fig_h_in = recommend_barpanel_size(n_cols, n_rows, n_g)
    fig_bar, axes2d = plt.subplots(
        n_rows,
        n_cols,
        figsize=(fig_w_in, fig_h_in),
        facecolor="white",
        squeeze=False,
    )
    axes = axes2d.reshape(-1)
    if main_title:
        fig_bar.suptitle(main_title, fontsize=10, fontweight="bold")

    global_y = None
    if same_y_lim:
        epos = np.where(np.isfinite(e), e, 0.0)
        y0, y1 = np.nanmin(m - epos), np.nanmax(m + epos)
        if np.isfinite(y0) and np.isfinite(y1) and y1 != y0:
            yr = y1 - y0
            global_y = (
                y0 - same_y_extra_bottom_ratio * yr,
                y1 + same_y_extra_top_ratio * yr,
            )

    for pi, metric in enumerate(params):
        ax = axes[pi]
        y = m[pi, :]
        err = np.where(np.isfinite(e[pi, :]), e[pi, :], 0.0)
        x = np.arange(1, n_g + 1)
        ax.bar(
            x,
            y,
            width=bar_width,
            color=(0.0, 0.4470, 0.7410),
            edgecolor="black",
            linewidth=bar_line_width,
        )
        if str(error_type).lower() != "none":
            ax.errorbar(
                x,
                y,
                yerr=err,
                fmt="none",
                ecolor="k",
                elinewidth=err_line_width,
                capsize=6,
                capthick=err_line_width,
            )

        # Start from the exact data extent plus the zero baseline. Matplotlib's
        # default 5% margin would be padded a second time below and produce
        # visibly different y ranges in nearly every panel.
        finite_low = (y - err)[np.isfinite(y - err)]
        finite_high = (y + err)[np.isfinite(y + err)]
        if finite_low.size and finite_high.size:
            raw_low = min(0.0, float(np.min(finite_low)))
            raw_high = max(0.0, float(np.max(finite_high)))
            if raw_high <= raw_low:
                raw_high = raw_low + max(abs(raw_low), 1.0)
            ax.set_ylim(raw_low, raw_high)
        if show_sig_on_bar:
            yl = ax.get_ylim()
            yr = yl[1] - yl[0] + np.finfo(float).eps
            off = sig_offset_ratio * yr
            y_min_need, y_max_need = yl
            for gi in range(n_g):
                if gi == ref_idx or not stars[pi, gi] or not np.isfinite(y[gi]):
                    continue
                if y[gi] >= 0:
                    y_star = y[gi] + err[gi] + off
                    va = "bottom"
                    y_max_need = max(y_max_need, y_star + off)
                else:
                    y_star = y[gi] - err[gi] - off
                    va = "top"
                    y_min_need = min(y_min_need, y_star - off)
                ax.text(
                    x[gi],
                    y_star,
                    stars[pi, gi],
                    ha="center",
                    va=va,
                    fontsize=sig_font_size,
                    fontweight="bold",
                )
            ax.set_ylim([y_min_need, y_max_need])
        ax.set_title(str(metric), fontsize=title_font_size, fontweight="bold")
        ax.set_xlim(0.4, n_g + 0.6)
        ax.set_xticks(x)
        is_bottom_row = (pi // n_cols) == (n_rows - 1)
        if show_xlabels_only_bottom_row and not is_bottom_row:
            ax.set_xticklabels([])
        else:
            ax.set_xticklabels(group_names, rotation=x_tick_rotation, ha="right")
        ax.tick_params(axis="both", labelsize=axes_font_size, width=0.75, length=3)
        if metric in {"ContractilitySpeed", "RelaxationSpeed"}:
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
        for spine in ax.spines.values():
            spine.set_linewidth(0.75)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, linewidth=0.6, alpha=0.35)
        ax.set_axisbelow(True)
        if global_y is not None:
            ax.set_ylim(global_y)
        else:
            yl = ax.get_ylim()
            pad = panel_pad_ratio * (yl[1] - yl[0] + np.finfo(float).eps)
            ax.set_ylim([yl[0] - pad, yl[1] + pad])
        ax.yaxis.set_major_locator(
            MaxNLocator(
                nbins=_PANEL_Y_NBINS.get(str(metric), 4),
                steps=[1.0, 2.0, 2.5, 5.0, 10.0],
            )
        )

    for i in range(n_p, len(axes)):
        axes[i].axis("off")
    suffix = str(partition_name).strip().lower()
    if n_cols == 5 and n_rows == 5 and suffix in {"major", "minor"}:
        # Established compact positions for the 22-major and 23-minor
        # partitions. Tight-layout is a different layout engine and visibly
        # changes tile widths and gaps.
        x0 = 0.065
        x_step = 0.18777514826757665
        tile_width = 0.12889940692969326
        if suffix == "major":
            y_top = 0.8103928540401532
            y_step = 0.19042611103861714
            tile_height = 0.14540695395049463
        else:
            y_top = 0.8097376819771368
            y_step = 0.19026231802286308
            tile_height = 0.14606212601351098
        for index, ax in enumerate(axes):
            row, col = divmod(index, n_cols)
            ax.set_position(
                [
                    x0 + col * x_step,
                    y_top - row * y_step,
                    tile_width,
                    tile_height,
                ]
            )
    else:
        fig_bar.tight_layout(pad=0.70, w_pad=0.45, h_pad=0.65, rect=(0.0, 0.0, 0.97, 1.0))

    bar_png = panel_dir / f"mini_bar_panel_{suffix}.png"
    bar_target = None
    if n_cols == 5 and suffix in {"major", "minor"}:
        width_crop_in = 0.63143 if suffix == "major" else 0.74000
        if suffix == "major":
            # The standard major panel is eight pixels narrower at 350 dpi when
            # it expands from four to five groups.
            width_crop_in += (8.0 / 350.0) * max(n_g - 4, 0)
        bar_target = (
            int(round((fig_w_in - width_crop_in) * dpi)),
            int(round((fig_h_in - 0.60857) * dpi)),
        )
    save_triplet(
        fig_bar,
        bar_png,
        panel_dir / f"mini_bar_panel_{suffix}.pdf",
        panel_dir / f"mini_bar_panel_{suffix}.tif",
        dpi,
        target_raster_size=bar_target,
    )
    plt.close(fig_bar)

    heat_png: Optional[Path] = None
    if make_heatmap:
        base = m[:, ref_idx : ref_idx + 1]
        denom = np.maximum(np.abs(base), np.finfo(float).eps)
        rel = (m - base) / denom
        hplot = np.sign(rel) * np.log2(1 + np.abs(rel))
        plot_cols = list(range(n_g)) if include_ctrl_in_heatmap else [i for i in range(n_g) if i != ref_idx]
        heat_width_in = max(4.0, 0.50 * n_g + 0.6)
        heat_height_in = max(9.5, 0.23 * n_p + 2.0)
        fig_heat = plt.figure(
            figsize=(heat_width_in, heat_height_in),
            facecolor="white",
        )
        standard_heatmap = suffix in {"major", "minor"} and n_g == 5
        if standard_heatmap:
            # Crop the canvas around its decorations. These positions reproduce
            # the five-column matrix and colorbar geometry in the established
            # PNG/TIFF outputs.
            ax = fig_heat.add_axes(
                [0.408, 0.11825, 0.379, 0.8153]
            )
        else:
            ax = fig_heat.add_subplot(111)
        cmap = LinearSegmentedColormap.from_list(
            "tripoint",
            [(0.20, 0.40, 0.85), (1.0, 1.0, 1.0), (0.85, 0.20, 0.20)],
            N=256,
        )
        im = ax.imshow(hplot[:, plot_cols], aspect="auto", cmap=cmap, interpolation="nearest")
        v = hplot[np.isfinite(hplot)]
        lim = float(np.percentile(np.abs(v), 92)) if len(v) else 1.0
        if not np.isfinite(lim) or lim <= 0:
            lim = float(np.max(np.abs(v))) if len(v) else 1.0
        if not np.isfinite(lim) or lim <= 0:
            lim = 1.0
        im.set_clim(-lim, lim)
        if standard_heatmap:
            cax = fig_heat.add_axes(
                [0.814, 0.1186, 0.0437, 0.8154]
            )
        else:
            cax = fig_heat.add_axes([0.88, 0.11, 0.04, 0.77])
        cb = fig_heat.colorbar(im, cax=cax)
        cb.set_label("Signed log2(1+|Δ/|REF||)", fontsize=9)
        cb.ax.tick_params(labelsize=8)
        raw_cb_step = (2.0 * lim) / 10.0
        cb_magnitude = 10.0 ** np.floor(np.log10(raw_cb_step))
        cb_normalized = raw_cb_step / cb_magnitude
        cb_candidate = min((1.0, 2.0, 5.0, 10.0), key=lambda value: abs(value - cb_normalized))
        cb_step = cb_candidate * cb_magnitude
        cb_ticks = np.arange(
            np.ceil((-lim - cb_step * 1e-9) / cb_step) * cb_step,
            np.floor((lim + cb_step * 1e-9) / cb_step) * cb_step + cb_step * 0.5,
            cb_step,
        )
        cb_ticks[np.abs(cb_ticks) < cb_step * 1e-10] = 0.0
        cb.set_ticks(cb_ticks)
        ax.set_xlabel(f"Group (vs {group_names[ref_idx]})", fontsize=9)
        ax.set_ylabel("Parameter", fontsize=9)
        ax.set_title(
            f"{heatmap_title_prefix} (Blue=low, Red=high) vs {group_names[ref_idx]}",
            fontsize=10,
            fontweight="bold",
        )
        ax.set_xticks(np.arange(len(plot_cols)))
        ax.set_xticklabels(group_names[plot_cols], rotation=35, ha="right")
        ax.set_yticks(np.arange(n_p))
        ax.set_yticklabels(params)
        ax.tick_params(axis="both", labelsize=8)
        ax.tick_params(axis="both", which="both", top=True, right=True, direction="in")
        if show_sig_on_heatmap:
            for pi in range(n_p):
                for k, gi in enumerate(plot_cols):
                    if gi != ref_idx and stars[pi, gi]:
                        ax.text(
                            k,
                            pi,
                            stars[pi, gi],
                            ha="center",
                            va="center",
                            fontsize=6,
                            fontweight="bold",
                            color="black",
                        )
        if not standard_heatmap:
            fig_heat.tight_layout(rect=(0.0, 0.0, 0.86, 1.0))
        heat_png = panel_dir / f"heatmap_{suffix}.png"
        heat_target = None
        if suffix in {"major", "minor"}:
            width_crop_in = 0.22571 if suffix == "major" else 0.22857
            heat_target = (
                int(round((heat_width_in - width_crop_in) * dpi)),
                int(round((heat_height_in - 0.86857) * dpi)),
            )
        save_triplet(
            fig_heat,
            heat_png,
            panel_dir / f"heatmap_{suffix}.pdf",
            panel_dir / f"heatmap_{suffix}.tif",
            dpi,
            target_raster_size=heat_target,
        )
        plt.close(fig_heat)

    stats_xlsx = panel_dir / f"stats_significance_{suffix}.xlsx"
    if n_cols == 5:
        tile_w_in = 1.82 + 0.02 * max(n_g - 5, 0)
        tile_h_in = 1.58
    else:
        tile_w_in = 1.90 + 0.02 * max(n_g - 5, 0)
        tile_h_in = 1.45
    with pd.ExcelWriter(stats_xlsx, engine="openpyxl") as writer:
        stats_long.to_excel(writer, sheet_name="Long", index=False)
        anova_summary.to_excel(writer, sheet_name="WelchANOVA", index=False)
        pd.DataFrame(m, index=params, columns=group_names).to_excel(writer, sheet_name="Mean", index_label="Row")
        pd.DataFrame(e, index=params, columns=group_names).to_excel(writer, sheet_name="Error", index_label="Row")
        pd.DataFrame(p_raw, index=params, columns=group_names).to_excel(writer, sheet_name="P_raw_vs_REF", index_label="Row")
        pd.DataFrame(p_adj, index=params, columns=group_names).to_excel(writer, sheet_name="P_adj_vs_REF", index_label="Row")
        pd.DataFrame(stars, index=params, columns=group_names).to_excel(writer, sheet_name="Stars_vs_REF", index_label="Row")
        pd.DataFrame(
            [
                {
                    "PairwiseMethod": stat_method,
                    "Alpha": alpha_sig,
                    "UseFDR": use_fdr,
                    "ReferenceGroupName": str(group_names[ref_idx]),
                    "ReferenceIndex": ref_idx + 1,
                    "IncludeREF_in_Heatmap": include_ctrl_in_heatmap,
                    "SaveAllPairsExcel": save_all_pairs_excel,
                    "TileW_in": tile_w_in,
                    "TileH_in": tile_h_in,
                    "ShowXLabelsOnlyBottomRow": True,
                }
            ]
        ).to_excel(writer, sheet_name="Info", index=False)
        if save_all_pairs_excel and not pairwise_all.empty:
            pairwise_all.to_excel(writer, sheet_name="AllPairs_WelchT", index=False)

    return {
        "name": partition_name.title(),
        "params": params,
        "n_params": n_p,
        "bar_png": str(bar_png),
        "heatmap_png": str(heat_png) if heat_png is not None else None,
        "stats_xlsx": str(stats_xlsx),
    }


def run_minipanel_analysis(
    data_folder: str,
    output_dir: Optional[str] = None,
    selected_files: Optional[Sequence[str | Path]] = None,
    ordered_files: Optional[Sequence[str]] = None,
    include_ctrl_in_heatmap: bool = True,
    sheet_name: int | str = 0,
    error_type: str = "sd",
    n_cols: int = 5,
    same_y_lim: bool = False,
    main_title: str = "",
    dpi: int = 350,
    param_order: Optional[Sequence[str]] = None,
    exclude_cols_exact: Optional[Sequence[str]] = None,
    exclude_cols_contains: Optional[Sequence[str]] = None,
    make_heatmap: bool = True,
    heatmap_title_prefix: str = "Mean heatmap",
    control_group_name: str = "CTRL",
    alpha_sig: float = 0.05,
    reference_group_name: Optional[str] = None,
    use_fdr: bool = False,
    stat_method: str = "welch",
    save_all_pairs_excel: bool = False,
    show_sig_on_bar: bool = True,
    show_sig_on_heatmap: bool = True,
    do_lda: bool = True,
    do_pca: bool = True,
    do_tsne: bool = True,
    t_min: float = 0.20,
    t_max: float = 0.95,
    ellipse_conf: float = 0.68,
    ellipse_scale: float = 0.85,
    ellipse_alpha: float = 0.08,
    use_robust_cov: bool = True,
    point_size: int = 24,
) -> Dict:
    data_folder = str(data_folder)
    p_data = Path(data_folder)
    if not p_data.is_dir():
        raise ValueError(f"Data folder not found: {data_folder}")

    if selected_files:
        files = []
        for fp in selected_files:
            p = Path(fp)
            if not p.is_absolute():
                p = p_data / p
            files.append(p.resolve())
        files = [fp for fp in files if fp.is_file()]
        if not files:
            raise ValueError("selected_files did not resolve to any readable .xlsx files.")
    else:
        files = discover_perfish_workbooks(p_data, recursive=True)
        if not files:
            raise ValueError(f"No PerFishMetrics .xlsx files found in: {data_folder}")

    if ordered_files and not selected_files:
        name_map = {f.name: f for f in files}
        files = [name_map[n] for n in ordered_files if n in name_map]
        if not files:
            raise ValueError("ordered_files did not match any .xlsx files.")

    group_names = np.asarray([infer_group_from_filename(f.name) for f in files], dtype=object)
    n_g = len(files)

    if exclude_cols_exact is None:
        exclude_cols_exact = ("FileKey", "ID", "FishID")
    if exclude_cols_contains is None:
        # Functional selection is governed by the explicit major/minor contract.
        # A generic ``id`` substring rule is unsafe (for example, it matches
        # ``Valid`` in an uncertainty-count column).
        exclude_cols_contains = ()

    tables = []
    num_vars_by_group = []
    for fp in files:
        t = pd.read_excel(fp, sheet_name=sheet_name)
        vnames = list(t.columns)
        keep_numeric = []
        for c in vnames:
            ser = t[c]
            is_num = pd.api.types.is_numeric_dtype(ser)
            if not is_num and (
                pd.api.types.is_object_dtype(ser) or pd.api.types.is_string_dtype(ser)
            ):
                num = pd.to_numeric(ser, errors="coerce")
                valid_ratio = float(num.notna().sum()) / max(len(num), 1)
                if valid_ratio >= 0.60 and num.notna().sum() >= 3:
                    t[c] = num
                    is_num = True
            if is_num:
                keep_numeric.append(c)
        keep_numeric = [c for c in keep_numeric if c not in exclude_cols_exact]
        keep_numeric2 = []
        for c in keep_numeric:
            low = c.lower()
            if any(tok.lower() in low for tok in exclude_cols_contains):
                continue
            keep_numeric2.append(c)
        tables.append(t)
        num_vars_by_group.append(keep_numeric2)

    common_vars = list(num_vars_by_group[0])
    for nv in num_vars_by_group[1:]:
        common_vars = [c for c in common_vars if c in nv]
    if not common_vars:
        raise ValueError("No common numeric parameter columns across selected groups.")

    expected_metrics = list(FUNCTIONAL_METRICS)
    if param_order:
        params = [str(metric) for metric in param_order]
        missing = [metric for metric in params if metric not in common_vars]
        if missing:
            raise ValueError(
                "Selected MiniPanel metrics are missing from one or more workbooks: "
                + ", ".join(missing)
            )
        partitions = {"custom": params}
    else:
        missing = [metric for metric in expected_metrics if metric not in common_vars]
        if missing:
            raise ValueError(
                "The PRIZM major/minor MiniPanel contract requires all 45 functional metrics. "
                "Missing from one or more selected workbooks: "
                + ", ".join(missing)
            )
        params = expected_metrics
        partitions = {"major": list(MAJOR_METRICS), "minor": list(MINOR_METRICS)}
    n_p = len(params)

    ctrl_idx = find_ctrl_idx(group_names, control_group_name)
    if reference_group_name:
        ref_hits = np.where(np.char.lower(group_names.astype(str)) == str(reference_group_name).lower())[0]
        ref_idx = int(ref_hits[0]) if len(ref_hits) else ctrl_idx
    else:
        ref_idx = ctrl_idx

    if output_dir is None:
        ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        out_dir = p_data / f"output_{ts}"
    else:
        out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    panel_dir = out_dir / "panel_heatmap"
    panel_dir.mkdir(exist_ok=True)

    partition_results: Dict[str, Dict[str, object]] = {}
    for partition_name, partition_metrics in partitions.items():
        partition_results[partition_name] = _render_metric_partition(
            partition_name=partition_name,
            tables=tables,
            params=partition_metrics,
            group_names=group_names,
            ref_idx=ref_idx,
            panel_dir=panel_dir,
            error_type=error_type,
            n_cols=n_cols,
            same_y_lim=same_y_lim,
            main_title=main_title,
            dpi=dpi,
            make_heatmap=make_heatmap,
            heatmap_title_prefix=heatmap_title_prefix,
            include_ctrl_in_heatmap=include_ctrl_in_heatmap,
            alpha_sig=alpha_sig,
            use_fdr=use_fdr,
            stat_method=stat_method,
            save_all_pairs_excel=save_all_pairs_excel,
            show_sig_on_bar=show_sig_on_bar,
            show_sig_on_heatmap=show_sig_on_heatmap,
        )

    selected_partition = {
        metric: partition_name.title()
        for partition_name, partition_metrics in partitions.items()
        for metric in partition_metrics
    }
    selection_rows = [
        {
            "Metric": metric,
            "Partition": selected_partition.get(metric, ""),
            "Selected": metric in selected_partition,
            "Reason": "PRIZM major/minor mapping" if metric in selected_partition else "Not a mapped functional metric",
        }
        for metric in common_vars
    ]
    selection_xlsx = panel_dir / "metric_selection.xlsx"
    with pd.ExcelWriter(selection_xlsx, engine="openpyxl") as writer:
        pd.DataFrame(selection_rows).to_excel(writer, sheet_name="MetricSelection", index=False)
        pd.DataFrame(
            {
                "Metric": params,
                "Partition": [selected_partition.get(metric, "Custom") for metric in params],
            }
        ).to_excel(writer, sheet_name="SelectedOrder", index=False)

    stats_paths = [str(result["stats_xlsx"]) for result in partition_results.values()]
    stats_xlsx = stats_paths[0] if len(stats_paths) == 1 else "; ".join(stats_paths)

    # --- LDA/PCA/t-SNE ---
    fig_dir = out_dir / "FIGURES_300dpi"
    report_dir = out_dir / "LDA_REPORT"
    if do_lda or do_pca or do_tsne:
        fig_dir.mkdir(exist_ok=True)
        report_dir.mkdir(exist_ok=True)

        xraw = []
        ystr = []
        for gi in range(n_g):
            ti = tables[gi]
            xi = np.full((len(ti), n_p), np.nan, dtype=float)
            for pi, v in enumerate(params):
                xi[:, pi] = pd.to_numeric(ti[v], errors="coerce").to_numpy(dtype=float)
            xraw.append(xi)
            ystr.extend([group_names[gi]] * len(ti))
        xraw = np.vstack(xraw)
        ystr = np.asarray(ystr, dtype=object)
        med = np.nanmedian(xraw, axis=0)
        x = np.array(xraw, copy=True)
        for j in range(x.shape[1]):
            mcol = np.isfinite(x[:, j])
            fill = med[j] if np.isfinite(med[j]) else 0.0
            x[~mcol, j] = fill
        mu = np.nanmean(x, axis=0)
        sd = np.nanstd(x, axis=0, ddof=1)
        embed_keep = np.isfinite(sd) & (sd > 1e-12)
        if np.sum(embed_keep) < 2:
            raise ValueError("LDA/PCA/t-SNE require at least two non-constant functional metrics.")
        pd.DataFrame(
            {
                "Metric": params,
                "Included": embed_keep,
                "Reason": np.where(embed_keep, "Non-constant functional metric", "Constant or invalid"),
            }
        ).to_csv(report_dir / "Embedding_FeatureSelection.csv", index=False)
        x = x[:, embed_keep]
        mu = mu[embed_keep]
        sd = sd[embed_keep]
        xz = (x - mu) / sd

        classes = np.asarray(group_names, dtype=object)
        control_keywords = np.unique(
            np.asarray([control_group_name.lower(), "control", "ctrl", "vehicle", "dmso", "veh"])
        )
        tord, colors = build_group_meta_and_colors(classes, control_keywords, t_min, t_max)
        feat_names = np.asarray(params, dtype=object)[embed_keep]

        if do_lda:
            zlda, w = fisher_lda_2d(xz, ystr, classes=classes)
            zlda, w, struct_corr = orient_lda_axes_by_top_corr(xz, zlda, w)
            h = plot2d_filled_ellipse(
                zlda,
                ystr,
                classes,
                colors,
                "Fisher LDA (2D) - Group Separation",
                "LD1",
                "LD2",
                conf=ellipse_conf,
                face_alpha=ellipse_alpha,
                ellipse_scale=ellipse_scale,
                point_size=point_size,
                use_robust_cov=use_robust_cov,
            )
            save_triplet(
                h,
                fig_dir / "FisherLDA_filledEllipse.png",
                fig_dir / "FisherLDA_filledEllipse.pdf",
                fig_dir / "FisherLDA_filledEllipse.tif",
                300,
            )
            plt.close(h)

            tload = pd.DataFrame(
                {
                    "Feature": feat_names,
                    "LD1_weight": w[:, 0],
                    "absLD1": np.abs(w[:, 0]),
                    "LD2_weight": w[:, 1],
                    "absLD2": np.abs(w[:, 1]),
                }
            )
            tcorr = pd.DataFrame(
                {
                    "Feature": feat_names,
                    "corr_LD1": struct_corr[:, 0],
                    "absCorrLD1": np.abs(struct_corr[:, 0]),
                    "corr_LD2": struct_corr[:, 1],
                    "absCorrLD2": np.abs(struct_corr[:, 1]),
                }
            )
            tload.to_csv(report_dir / "LDA_Loadings_all.csv", index=False)
            tload.sort_values("absLD1", ascending=False).to_csv(
                report_dir / "LDA_Loadings_sorted_by_LD1.csv", index=False
            )
            tload.sort_values("absLD2", ascending=False).to_csv(
                report_dir / "LDA_Loadings_sorted_by_LD2.csv", index=False
            )
            tcorr.to_csv(report_dir / "LDA_StructureCorr_all.csv", index=False)
            tcorr_ld1 = tcorr.sort_values("absCorrLD1", ascending=False)
            tcorr_ld2 = tcorr.sort_values("absCorrLD2", ascending=False)
            tcorr_ld1.to_csv(report_dir / "LDA_StructureCorr_sorted_by_LD1.csv", index=False)
            tcorr_ld2.to_csv(report_dir / "LDA_StructureCorr_sorted_by_LD2.csv", index=False)
            tcent = group_centroids_ld(zlda, ystr, classes, tord)
            tcent.to_csv(report_dir / "LDA_GroupCentroids.csv", index=False)
            write_auto_interpretation(
                report_dir / "LDA_AutoInterpretation.txt", tcent, tcorr_ld1, tcorr_ld2, tord
            )

        if do_pca:
            try:
                score = PCA(n_components=2, random_state=0).fit_transform(xz)
                h = plot2d_filled_ellipse(
                    score[:, :2],
                    ystr,
                    classes,
                    colors,
                    "PCA (PC1 vs PC2)",
                    "PC1",
                    "PC2",
                    conf=ellipse_conf,
                    face_alpha=ellipse_alpha,
                    ellipse_scale=ellipse_scale,
                    point_size=point_size,
                    use_robust_cov=use_robust_cov,
                )
                save_triplet(
                    h,
                    fig_dir / "PCA_filledEllipse.png",
                    fig_dir / "PCA_filledEllipse.pdf",
                    fig_dir / "PCA_filledEllipse.tif",
                    300,
                )
                plt.close(h)
            except Exception:
                pass

        if do_tsne:
            try:
                n = xz.shape[0]
                perp = max(5, min(35, int(math.floor((n - 1) / 3))))
                ytsne = TSNE(
                    n_components=2,
                    perplexity=perp,
                    init="pca",
                    method="exact",
                    n_jobs=1,
                    random_state=1,
                ).fit_transform(xz)
                h = plot2d_filled_ellipse(
                    ytsne[:, :2],
                    ystr,
                    classes,
                    colors,
                    f"t-SNE (perplexity={perp})",
                    "tSNE1",
                    "tSNE2",
                    conf=ellipse_conf,
                    face_alpha=ellipse_alpha,
                    ellipse_scale=ellipse_scale,
                    point_size=point_size,
                    use_robust_cov=use_robust_cov,
                )
                save_triplet(
                    h,
                    fig_dir / f"tSNE_perp{perp}_filledEllipse.png",
                    fig_dir / f"tSNE_perp{perp}_filledEllipse.pdf",
                    fig_dir / f"tSNE_perp{perp}_filledEllipse.tif",
                    300,
                )
                plt.close(h)
            except Exception:
                pass

    return {
        "output_dir": str(out_dir),
        "panel_dir": str(panel_dir),
        "stats_xlsx": str(stats_xlsx),
        "partition_results": partition_results,
        "metric_selection_xlsx": str(selection_xlsx),
        "groups": list(map(str, group_names)),
        "n_groups": int(n_g),
        "n_params": int(n_p),
        "ctrl_index_1based": int(ctrl_idx + 1),
        "reference_index_1based": int(ref_idx + 1),
    }
