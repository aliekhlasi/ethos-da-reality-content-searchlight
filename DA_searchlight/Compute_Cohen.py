#!/usr/bin/env python3
# ------------------------------------------------------------------------------
# Part of: "Is This the Real Life? Is This Just Fantasy? A Domain Adaptation
#          Framework for Identifying Higher-Order Reality Representations"
# Edited by: Ali Ekhlasi, Postdoctoral Researcher,
#            Department of Cognitive Sciences, University of California, Irvine
#            https://github.com/aliekhlasi
# License: GNU General Public License v3.0 (see LICENSE)
# ------------------------------------------------------------------------------
from __future__ import annotations

import argparse
import ast
import re
from pathlib import Path

import numpy as np
import pandas as pd
from nilearn import masking
from nilearn.image import new_img_like

from da_paths import DATA_ROOT, REPO_ROOT, RESULTS_ROOT
from dataManipulation.whole_brain import load_data

SCORE_KEYS = ("score", "accuracy")
COV_KEYS = ("cov_distance", "cov_dist")
METRIC_TO_SUMMARY_COLUMNS = {
    SCORE_KEYS: ("score_mean", "score"),
    COV_KEYS: ("cov_distance_mean", "cov_distance", "cov_dist"),
}


def normalize_numpy_scalar_repr(s: str) -> str:
    """Convert NumPy scalar reprs to literals ast.literal_eval can parse."""
    number = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
    s = re.sub(rf"np\.(?:float|int)\d*\(({number})\)", r"\1", s)
    s = re.sub(r"np\.nan", "None", s)
    return s


def cell_to_dict(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    if not isinstance(x, str):
        return None
    s = x.strip()
    if not s:
        return None
    try:
        # Replace all 'nan' with None, which is a valid Python literal
        # This preserves the data structure - we'll use np.nan where needed in extract_metric
        s = normalize_numpy_scalar_repr(s)
        s = s.replace(': nan,', ': None,').replace(': nan}', ': None}').replace(': nan', ': None')
        s = s.replace(', nan,', ', None,').replace(', nan]', ', None]').replace(', nan}', ', None}')
        return ast.literal_eval(s)
    except Exception:
        return None


def extract_metric(cell, keys):
    d = cell_to_dict(cell)
    if not isinstance(d, dict):
        return np.nan
    for key in keys:
        if key in d:
            try:
                return float(d[key])
            except Exception:
                return np.nan
    return np.nan


def extract_voxel_indices(index_str):
    """Parse voxel_indices from a dict string or list string."""
    d = cell_to_dict(index_str)
    if isinstance(d, dict) and 'voxel_indices' in d:
        return tuple(sorted(d['voxel_indices']))
    if isinstance(d, list):
        return tuple(sorted(d))
    return None


def summary_csv_for_raw(csv_path: Path, method_label: str | None = None) -> Path | None:
    """Return the summary CSV corresponding to a raw searchlight CSV if it exists."""
    match = re.match(r"map_(\d+)iter_(\d+)mm_(.+)\.csv$", csv_path.name)
    if not match:
        return None
    niter, radius, label = match.groups()
    label = method_label or label
    if label.upper() in {"BASELINE", "NAIVE"}:
        candidate = csv_path.with_name(f"map_summary_{niter}iter_{radius}mm_{label}.csv")
    else:
        candidate = csv_path.parent / f"map_summary_{niter}iter_{radius}mm_{label}.csv"
    return candidate if candidate.exists() else None


def metric_columns_for_summary(keys) -> tuple[str, ...]:
    for key_tuple, columns in METRIC_TO_SUMMARY_COLUMNS.items():
        if tuple(keys) == tuple(key_tuple):
            return columns
    return tuple(keys)


def load_summary_metric_matrix(summary_csv: Path, keys) -> pd.DataFrame:
    """Load one-column metric matrix indexed by voxel_indices from a summary CSV."""
    df_raw = pd.read_csv(summary_csv, dtype=str)
    if "voxel_indices" not in df_raw.columns:
        raise ValueError(f"Summary CSV lacks voxel_indices column: {summary_csv}")

    metric_col = None
    for col in metric_columns_for_summary(tuple(keys)):
        if col in df_raw.columns:
            metric_col = col
            break
    if metric_col is None:
        raise ValueError(f"Summary CSV lacks metric columns {metric_columns_for_summary(tuple(keys))}: {summary_csv}")

    metric_name = metric_col
    df = pd.DataFrame({metric_name: pd.to_numeric(df_raw[metric_col], errors="coerce")})
    df.index = df_raw["voxel_indices"].astype(str)
    return df


def align_dfs_by_voxel_indices(base_df: pd.DataFrame, da_df: pd.DataFrame):
    """
    Align baseline and DA dataframes by voxel_indices instead of row index.
    
    Returns:
        aligned_base: subset of base_df with voxels in da_df
        aligned_da: subset of da_df with voxels in base_df (same order as aligned_base)
    """
    # Extract voxel_indices from both dataframes' index
    base_voxels = {}
    for idx, idx_str in enumerate(base_df.index):
        voxels = extract_voxel_indices(idx_str)
        if voxels:
            base_voxels[voxels] = idx
    
    da_voxels = {}
    for idx, idx_str in enumerate(da_df.index):
        voxels = extract_voxel_indices(idx_str)
        if voxels:
            da_voxels[voxels] = idx
    
    # Find common voxel sets
    common_voxels = set(base_voxels.keys()) & set(da_voxels.keys())
    
    if not common_voxels:
        raise ValueError(
            f"No common voxel patterns found between baseline ({len(base_voxels)} patterns) "
            f"and DA ({len(da_voxels)} patterns)"
        )
    
    # Extract rows in the same order
    base_indices = [base_voxels[v] for v in sorted(common_voxels)]
    da_indices = [da_voxels[v] for v in sorted(common_voxels)]
    
    aligned_base = base_df.iloc[base_indices].copy()
    aligned_da = da_df.iloc[da_indices].copy()
    
    print(f"  Aligned by voxel_indices: {len(common_voxels)} common searchlight spheres")
    
    return aligned_base, aligned_da


def rowwise_voxel_indices_match(base_index, da_index) -> bool:
    """Check if voxel_indices match row-by-row (order preserved)."""
    if len(base_index) != len(da_index):
        return False
    for b_idx, d_idx in zip(base_index, da_index):
        b_vox = extract_voxel_indices(b_idx)
        d_vox = extract_voxel_indices(d_idx)
        if b_vox is None or d_vox is None:
            return False
        if b_vox != d_vox:
            return False
    return True


def load_metric_matrix(csv_path: Path, keys, method_label: str | None = None) -> pd.DataFrame:
    df_raw = pd.read_csv(csv_path, dtype=str, index_col=0)
    # Handle pandas version differences (applymap -> map)
    if hasattr(df_raw, 'map'):
        df = df_raw.map(lambda x: extract_metric(x, keys))
    else:
        df = df_raw.applymap(lambda x: extract_metric(x, keys))
    raw_metrics = df.apply(pd.to_numeric, errors="coerce")
    usable_columns = raw_metrics.columns[raw_metrics.notna().any(axis=0)]
    if len(usable_columns):
        print(f"Using raw iteration CSV for metrics: {csv_path}")
        return raw_metrics.loc[:, usable_columns]

    summary_csv = summary_csv_for_raw(csv_path, method_label=method_label)
    if summary_csv is not None:
        print(f"Raw iteration metrics unavailable; using summary CSV fallback: {summary_csv}")
        return load_summary_metric_matrix(summary_csv, tuple(keys))

    return raw_metrics


def load_and_align_metrics_by_voxel(baseline_csv: str, da_csv: str) -> tuple:
    """
    Load baseline and DA CSVs and align them by voxel_indices.
    
    Returns:
        (aligned_base_df, aligned_da_df)
    """
    base_df = load_metric_matrix(Path(baseline_csv), SCORE_KEYS)
    da_df = load_metric_matrix(Path(da_csv), SCORE_KEYS)
    
    aligned_base, aligned_da = align_dfs_by_voxel_indices(base_df, da_df)
    return aligned_base, aligned_da


def cohens_dz_from_diff(diff_1d: np.ndarray) -> float:
    diff_1d = np.asarray(diff_1d, dtype=float)
    diff_1d = diff_1d[np.isfinite(diff_1d)]
    if diff_1d.size < 2:
        return np.nan
    sd = diff_1d.std(ddof=1)
    if sd == 0 or not np.isfinite(sd):
        return np.nan
    return diff_1d.mean() / sd


def load_mask(target_domain: str, subject: str, data_root: Path):
    _, _, tgt_masker = load_data(target_domain, subject, data_root=data_root)
    mask_img = getattr(tgt_masker, "mask_img_", None) or tgt_masker.mask_img
    try:
        process_mask, _ = masking.load_mask_img(mask_img)
    except Exception:
        process_mask, _ = masking._load_mask_img(mask_img)
    return process_mask, mask_img


def write_map(vec, process_mask, mask_img, out_path: Path) -> None:
    arr = np.zeros(process_mask.shape, dtype=float)
    arr[process_mask] = np.asarray(vec, dtype=float)
    img = new_img_like(mask_img, arr, copy_header=False)
    img.to_filename(str(out_path))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute Cohen's dz from searchlight CSVs.")
    parser.add_argument("--repo-root", default=str(REPO_ROOT), help="Repo root path.")
    parser.add_argument(
        "--results-root",
        default="",
        help="Override results root (searchlight/Cross_Domain_Content_Decoding).",
    )
    parser.add_argument(
        "--data-root",
        default="",
        help="Override DATA_ROOT (used to load the mask).",
    )
    parser.add_argument("--subject", default="3", help="Subject id (folder name).")
    parser.add_argument("--target-domain", default="imagery", help="Target domain used in DA run.")
    parser.add_argument("--mode", default="P2I", help="Mode folder, e.g., P2I.")
    parser.add_argument("--da-method", default="SA", help="DA method folder name.")
    parser.add_argument("--baseline-label", default="BASELINE", help="Baseline CSV label.")
    parser.add_argument("--niter", type=int, default=10, help="Number of iterations.")
    parser.add_argument("--radius", type=int, default=12, help="Searchlight radius (mm).")
    parser.add_argument("--baseline-csv", default="", help="Override baseline CSV path.")
    parser.add_argument("--sa-csv", default="", help="Override DA/SA CSV path.")
    parser.add_argument("--out-dir", default="", help="Output directory.")
    parser.add_argument("--tag", default="", help="Output tag name.")
    parser.add_argument("--skip-mask-check", action="store_true", help="Skip mask/row count check.")
    parser.add_argument("--no-nifti", action="store_true", help="Skip writing NIfTI outputs.")
    return parser.parse_args()


def compute_and_save_dz(
    base_df: pd.DataFrame,
    da_df: pd.DataFrame,
    out_dir: Path,
    base_name: str,
    metric_label: str,
    process_mask=None,
    mask_img=None,
    write_nifti: bool = True,
    n_mask_vox: int | None = None,
) -> None:
    common_cols = [c for c in base_df.columns if c in da_df.columns]
    if not common_cols:
        raise ValueError(f"No common iteration columns for {metric_label}.")

    base_df = base_df[common_cols]
    da_df = da_df[common_cols]

    print(f"{metric_label} BASE shape:", base_df.shape)
    print(f"{metric_label} DA shape:", da_df.shape)
    print(f"{metric_label} iterations:", len(common_cols), common_cols[:5], "..." if len(common_cols) > 5 else "")

    # Align by voxel_indices only if row order doesn't already match.
    # This avoids collapsing duplicate voxel_indices when the CSVs are already aligned by row.
    original_index = None
    if base_df.index.equals(da_df.index):
        print("  Index match: skipping voxel_indices alignment.")
        original_index = base_df.index
        base_df = base_df.reset_index(drop=True)
        da_df = da_df.reset_index(drop=True)
    elif rowwise_voxel_indices_match(base_df.index, da_df.index):
        print("  Row-wise voxel_indices match: skipping alignment.")
        original_index = base_df.index
        base_df = base_df.reset_index(drop=True)
        da_df = da_df.reset_index(drop=True)
    else:
        base_df, da_df = align_dfs_by_voxel_indices(base_df, da_df)
        original_index = base_df.index
        base_df = base_df.reset_index(drop=True)
        da_df = da_df.reset_index(drop=True)
    print(f"{metric_label} after alignment - BASE shape: {base_df.shape}, DA shape: {da_df.shape}")

    diff = da_df - base_df

    dz_iter = []
    n_used = []
    mean_diff_iter = []

    for col in common_cols:
        v = diff[col].to_numpy(dtype=float)
        m = np.isfinite(v)
        n_used.append(int(m.sum()))
        mean_diff_iter.append(float(np.nanmean(v)) if m.any() else np.nan)
        dz_iter.append(cohens_dz_from_diff(v))

    dz_iter_df = pd.DataFrame({
        "iteration": common_cols,
        "n_voxels_used": n_used,
        "mean_diff_DA_minus_BASE": mean_diff_iter,
        "cohen_dz": dz_iter,
        "metric": metric_label,
    })

    mean_diff_vox = diff.mean(axis=1)
    std_diff_vox = diff.std(axis=1, ddof=1).replace(0, np.nan)
    dz_vox = mean_diff_vox / std_diff_vox

    dz_vox_df = pd.DataFrame({
        "dz_per_voxel": dz_vox,
        "mean_diff_DA_minus_BASE": mean_diff_vox,
        "std_diff_DA_minus_BASE": std_diff_vox,
        "metric": metric_label,
    })

    p_iter = out_dir / f"{base_name}_per_iter.csv"
    p_vox = out_dir / f"{base_name}_per_voxel.csv"

    dz_iter_df.to_csv(p_iter, index=False)
    dz_vox_df.to_csv(p_vox, index=True, index_label="row")

    print("✅ Saved CSVs:")
    print(" -", p_iter)
    print(" -", p_vox)

    if write_nifti:
        if process_mask is None or mask_img is None:
            raise ValueError("Mask not loaded; cannot write NIfTI.")
        mask_voxels = n_mask_vox or int(process_mask.sum())
        if dz_vox.shape[0] == mask_voxels:
            p_dz = out_dir / f"{base_name}.nii.gz"
            write_map(dz_vox.to_numpy(dtype=float), process_mask, mask_img, p_dz)
            print(" -", p_dz)
        elif original_index is not None and len(original_index) == dz_vox.shape[0]:
            full_vec = np.full(mask_voxels, np.nan, dtype=float)
            for row_pos, idx_value in enumerate(original_index):
                try:
                    mask_row = int(idx_value)
                except (TypeError, ValueError):
                    break
                if 0 <= mask_row < mask_voxels:
                    full_vec[mask_row] = float(dz_vox.iloc[row_pos])
            else:
                p_dz = out_dir / f"{base_name}.nii.gz"
                write_map(full_vec, process_mask, mask_img, p_dz)
                print(" -", p_dz)
        else:
            print(f"  Note: {dz_vox.shape[0]} aligned rows, {mask_voxels} mask voxels. Skipping NIfTI output.")

    dz_valid = dz_iter_df["cohen_dz"].to_numpy(dtype=float)
    dz_valid = dz_valid[np.isfinite(dz_valid)]
    if dz_valid.size:
        sd = dz_valid.std(ddof=1) if dz_valid.size > 1 else np.nan
        print(f"{metric_label} dz mean={dz_valid.mean():.4f}, sd={sd:.4f}")
    else:
        print(f"No valid dz values for {metric_label} (check NaNs / parsing / zero-variance diffs).")


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root)
    if args.results_root:
        results_root = Path(args.results_root)
    else:
        results_root = RESULTS_ROOT / "searchlight" / "Cross_Domain_Content_Decoding"
        fallback_root = repo_root / "results" / "searchlight" / "Cross_Domain_Content_Decoding"
        if not results_root.exists() and fallback_root.exists():
            results_root = fallback_root

    baseline_csv = Path(args.baseline_csv) if args.baseline_csv else (
        results_root / args.mode / str(args.subject) / f"map_{args.niter}iter_{args.radius}mm_{args.baseline_label}.csv"
    )
    sa_csv = Path(args.sa_csv) if args.sa_csv else (
        results_root / args.mode / str(args.subject) / args.da_method / f"map_{args.niter}iter_{args.radius}mm_{args.da_method}.csv"
    )

    out_dir = Path(args.out_dir) if args.out_dir else (
        results_root / args.mode / str(args.subject) / args.da_method
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    suffix = args.tag or f"{args.niter}iter_{args.radius}mm_{args.da_method}{args.baseline_label}"
    base_name = f"map_cohen_dz_{suffix}"

    if not baseline_csv.exists():
        raise FileNotFoundError(baseline_csv)
    if not sa_csv.exists():
        raise FileNotFoundError(sa_csv)

    base_score = load_metric_matrix(baseline_csv, SCORE_KEYS, method_label=args.baseline_label)
    da_score = load_metric_matrix(sa_csv, SCORE_KEYS, method_label=args.da_method)

    process_mask = None
    mask_img = None
    if not args.skip_mask_check or not args.no_nifti:
        data_root = Path(args.data_root) if args.data_root else DATA_ROOT
        process_mask, mask_img = load_mask(args.target_domain, args.subject, data_root=data_root)

    if not args.skip_mask_check:
        n_mask_vox = int(process_mask.sum())
        print("Mask voxels:", n_mask_vox)
        if base_score.shape[0] != n_mask_vox or da_score.shape[0] != n_mask_vox:
            raise ValueError(
                f"Row count mismatch:\\n"
                f"  baseline rows={base_score.shape[0]}, DA rows={da_score.shape[0]}\\n"
                f"  mask voxels={n_mask_vox}\\n\\n"
                "This means you used a different mask than the one used to generate these CSVs, "
                "or your CSV rows are not 1-to-1 with mask voxels."
            )

    compute_and_save_dz(
        base_score,
        da_score,
        out_dir,
        base_name,
        "score",
        process_mask=process_mask,
        mask_img=mask_img,
        write_nifti=not args.no_nifti,
        n_mask_vox=int(process_mask.sum()) if process_mask is not None else None,
    )

    base_cov = load_metric_matrix(baseline_csv, COV_KEYS, method_label=args.baseline_label)
    da_cov = load_metric_matrix(sa_csv, COV_KEYS, method_label=args.da_method)
    cov_base_name = f"map_cohen_dz_cov_distance_{suffix}"
    compute_and_save_dz(
        base_cov,
        da_cov,
        out_dir,
        cov_base_name,
        "cov_distance",
        process_mask=process_mask,
        mask_img=mask_img,
        write_nifti=not args.no_nifti,
        n_mask_vox=int(process_mask.sum()) if process_mask is not None else None,
    )


if __name__ == "__main__":
    main()
