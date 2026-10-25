#!/usr/bin/env python3
# -*- coding: utf-8 -*-#
# ------------------------------------------------------------------------------
# Part of: "Is This the Real Life? Is This Just Fantasy? A Domain Adaptation
#          Framework for Identifying Higher-Order Reality Representations"
# Edited by: Ali Ekhlasi, Postdoctoral Researcher,
#            Department of Cognitive Sciences, University of California, Irvine
#            https://github.com/aliekhlasi
# License: GNU General Public License v3.0 (see LICENSE)
# ------------------------------------------------------------------------------
"""
WITHIN-CONTENT DOMAIN decoding script (PvI within one content).

Goal:
- Decode DOMAIN (Perception vs Imagery) using ONLY trials from ONE content (e.g., content1).
- No DA. No cross-content baseline.
- Use searchlight_cv_DA ONLY to generate leakage-safe splits using group IDs.

Usage:
  python within_content_domain_decoding.py <subject_index> <n_iterations> [content_value]

Examples:
  python within_content_domain_decoding.py 0 10
  python within_content_domain_decoding.py 0 10 face
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import re
import sys
import glob
import warnings
from pathlib import Path

# Use centralized path configuration (handles sys.path bootstrapping)
from da_paths import DATA_ROOT, RESULTS_ROOT, FIGURES_ROOT

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"

import numpy as np
import pandas as pd
from tqdm import tqdm

from joblib import Parallel, cpu_count, delayed
from nilearn.decoding.searchlight import GroupIterator, _group_iter_search_light
from sklearn.exceptions import ConvergenceWarning

from nilearn import masking
from nilearn.image import new_img_like
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score

# metrics
from adapt.metrics import (
    cov_distance,
    # normalized_frechet_distance,
    normalized_linear_discrepancy,
)

# your project imports
from algorithms.searchlight_DA_save import get_sphere_data, searchlight_cv_DA
from dataManipulation.whole_brain import load_data


# -----------------------------
# Args
# -----------------------------
if len(sys.argv) not in (3, 4):
    print("Usage: python within_content_domain_decoding.py <subject_index> <n_iterations> [content_value]")
    sys.exit(1)

subject_idx = int(sys.argv[1])
NITER = int(sys.argv[2])
content_arg = sys.argv[3] if len(sys.argv) == 4 else None

source_domain = "perception"
target_domain = "imagery"
radius = 12

subjects = sorted(
    [S.split("/")[-1] for S in glob.glob(str(DATA_ROOT / source_domain / "*"))],
    key=lambda x: int(x)
)
subject = subjects[subject_idx]
s = re.sub(r"[0-9]+_", "", subject).capitalize()

print(f"--- Running WITHIN-CONTENT DOMAIN searchlight (PvI within one content) ---")
print(f"Subject: {subject} | NITER: {NITER} | radius: {radius}mm")


# -----------------------------
# Output dirs
# -----------------------------
base_out_dir = RESULTS_ROOT / "searchlight" / "Cross_Content_Domain_Decoding" / "Ci2Ci" / subject
base_fig_dir = FIGURES_ROOT / "searchlight" / "Cross_Content_Domain_Decoding" / "Ci2Ci" / subject
os.makedirs(base_out_dir, exist_ok=True)
os.makedirs(base_fig_dir, exist_ok=True)


# -----------------------------
# Estimator factory
# -----------------------------
def make_estimator():
    # stable + fast for many fits; increase max_iter if needed
    return LogisticRegression(solver="liblinear", max_iter=1000)


# -----------------------------
# Searchlight (no DA) function
# -----------------------------
def search_light_ALI(
    X,
    y,
    estimator_factory,
    A,
    X_test,
    y_test,
    groups=None,
    scoring=None,
    n_jobs=-1,
    verbose=0,
    i_iter=0,
):
    # jobs
    n_jobs_limit = cpu_count() - 2 if n_jobs == -1 else n_jobs
    n_jobs_actual = max(1, min(n_jobs_limit, os.cpu_count() or 1))

    # IMPORTANT: use n_jobs_actual for GroupIterator too
    group_iter = GroupIterator(A.shape[0], n_jobs_actual)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        scores = Parallel(n_jobs=n_jobs_actual, verbose=verbose)(
            delayed(_group_iter_search_light)(
                A.rows[list_i],
                estimator_factory,
                X,
                y,
                groups,
                scoring,
                X_test,
                y_test,
                None,   # DA
                None,   # X_tgt
                None,   # y_tgt
                thread_id + 1,
                A.shape[0],
                verbose,
                i_iter
            )
            for thread_id, list_i in enumerate(group_iter)
        )
    return np.concatenate(scores)


def _group_iter_search_light(
    list_rows,
    estimator_factory,
    X,
    y,
    groups,
    scoring,
    X_test,
    y_test,
    DA,
    X_tgt,
    y_tgt,
    thread_id,
    total,
    verbose=0,
    i_iter=0
):
    par_scores = np.zeros(len(list_rows))
    #sphere_results = []

    # -----------------------------
    # DEBUG LIMITER (your old check)
    # -----------------------------
    # Set to float("inf") for full runs
    MAX_SPHERES_TO_RUN = float("inf")
    #MAX_SPHERES_TO_RUN = 50

    out = np.empty(len(list_rows), dtype=object)

    for i, row in enumerate(list_rows):
        if i >= MAX_SPHERES_TO_RUN:
            if thread_id == 1:
                print(f"\nStopping early: processed {MAX_SPHERES_TO_RUN} spheres per worker (debug mode).")
            break

        # fit/predict
        fitted_estimator = estimator_factory().fit(X[:, row], y)
        y_pred = fitted_estimator.predict(X_test[:, row])

        # metrics: TRAIN vs TEST (within same content pooled PvI)
        cov = cov_distance(X[:, row], X_test[:, row])
        lin = normalized_linear_discrepancy(X[:, row], X_test[:, row])
        # nf = normalized_frechet_distance(X[:, row], X_test[:, row])
        # nf = np.nan  # normalized_frechet_distance disabled

        score = scoring(y_test, y_pred)

        out[i] = str({
            "score": float(score),
            "cov_distance": float(cov),
            "linear_discrepancy": float(lin),
            # "normalized_frechet_distance": float(nf),
            "da_attributes": {},
            "voxel_indices": row,
        })

    #return pd.Series([str(r) for r in sphere_results])
    return out


# -----------------------------
# Parsing helpers
# -----------------------------
def parse_and_extract_metric(data_string, metric_keys):
    if isinstance(data_string, str):
        for key in metric_keys:
            try:
                pattern = fr'["\']?{key}["\']?\s*:\s*([0-9\.\-eE]+)'
                match = re.search(pattern, data_string)
                if match:
                    return float(match.group(1))
            except Exception:
                continue
    return np.nan

def parse_and_extract_score(data_string):
    return parse_and_extract_metric(data_string, ["score", "accuracy"])

def parse_and_extract_cov_distance(data_string):
    return parse_and_extract_metric(data_string, ["cov_distance", "cov_dist"])

def parse_and_extract_lin_disc(data_string):
    return parse_and_extract_metric(data_string, ["linear_discrepancy", "lin_disc"])

# def parse_and_extract_norm_frechet(data_string):
    # return parse_and_extract_metric(
        # data_string,
        # [
            # "normalized_frechet_distance",
            # "norm_frechet_distance",
            # "normalized_frechet",
        # ],
    # )
# 
def extract_metadata(data_string, key):
    if isinstance(data_string, str) and data_string.startswith("{"):
        processed = data_string.replace("\n", " ").replace("'", '"')
        if key == "voxel_indices":
            m = re.search(r'"voxel_indices":\s*(\[.*?\])', processed)
            return m.group(1) if m else "N/A"
    return "N/A_Error"

def load_and_parse_searchlight(pfile, NITER, metric_parser):
    if Path(pfile).is_file():
        try:
            df = pd.read_csv(pfile, dtype={col: str for col in range(NITER)})
            for col in df.columns:
                df[col] = df[col].apply(metric_parser)
            return df
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()

def load_raw_data_for_metadata(pfile):
    if Path(pfile).is_file():
        return pd.read_csv(pfile, usecols=[0], dtype={0: str})
    return pd.DataFrame()

def is_numeric_df(df):
    return (not df.empty) and df.iloc[:, 0].notna().any()


# -----------------------------
# MAIN
# -----------------------------
# Load data
bold_data, events, masker = load_data(source_domain, subject)
tgt_data, tgt_events, tgt_masker = load_data(target_domain, subject)

# Build groups (run+trial) for BOTH domains
g = (events.run + "trial" + events.trial_idx.astype(str)).reset_index(drop=True)
g_tgt = (tgt_events.run + "trial" + tgt_events.trial_idx.astype(str)).reset_index(drop=True)

# Content labels in each domain
content_P = events.target_category.values
content_I = tgt_events.target_category.values

# Decide which content to use
all_contents = np.unique(np.concatenate([content_P, content_I]))
if len(all_contents) < 1:
    raise ValueError("No content labels found.")

if content_arg is None:
    # default: first content (sorted by string repr) to keep deterministic behavior
    all_contents_sorted = sorted([str(c) for c in all_contents])
    content_value_str = all_contents_sorted[0]
else:
    content_value_str = str(content_arg)

# Map requested content string back to actual dtype by string match
content_value_candidates = [c for c in all_contents if str(c) == content_value_str]
if len(content_value_candidates) == 0:
    raise ValueError(f"Requested content '{content_value_str}' not found. Available: {[str(c) for c in all_contents]}")

content_value = content_value_candidates[0]
print(f"Within-content PvI decoding using content = {content_value}")

# Extract indices for that content in BOTH domains
# idx_P = np.where(content_P == content_value)[0]
# idx_I = np.where(content_I == content_value)[0]

idx_P_all = np.where(content_P == content_value)[0]
idx_I_all = np.where(content_I == content_value)[0]

if len(idx_P_all) == 0 or len(idx_I_all) == 0:
    raise ValueError(f"Need trials for content={content_value} in BOTH domains. Found P={len(idx_P_all)}, I={len(idx_I_all)}")

# Sphere data
X, A, _, _ = get_sphere_data(masker, bold_data, radius)
X_tgt, _, _, _ = get_sphere_data(tgt_masker, tgt_data, radius)

# Build pooled dataset: content fixed, label = domain (P vs I)
# X_c = np.vstack([X[idx_P], X_tgt[idx_I]])
# y_dom = np.concatenate([np.zeros(len(idx_P), dtype=int), np.ones(len(idx_I), dtype=int)])
# g_c = pd.concat([g.iloc[idx_P], g_tgt.iloc[idx_I]], ignore_index=True)

# ------------------------------------------------------------
# Pool dataset (content fixed): X_pool = [P_trials; I_trials]
# Label = DOMAIN (0=P, 1=I)
# Group = run+trial pooled similarly
# ------------------------------------------------------------
X_pool = np.vstack([X[idx_P_all], X_tgt[idx_I_all]])
y_pool = np.concatenate([
    np.zeros(len(idx_P_all), dtype=int),
    np.ones(len(idx_I_all), dtype=int)
])
g_pool = pd.concat([g.iloc[idx_P_all], g_tgt.iloc[idx_I_all]], ignore_index=True)

# Output files for this content
content_tag = re.sub(r"\W+", "_", str(content_value))
out_dir = os.path.join(base_out_dir, f"content_{content_tag}")
os.makedirs(out_dir, exist_ok=True)

pwithin = os.path.join(out_dir, f"map_{NITER}iter_{radius}mm_BASELINE.csv")

# Compute (or skip)
searchlight_raw = pd.DataFrame()

if Path(pwithin).is_file():
    print(f"Found existing results: {pwithin}. Skipping computation.")
else:
    # Use searchlight_cv_DA to get splits (same arrays passed twice)
    Source_train_is, Target_train_is = searchlight_cv_DA(
        y_pool,   # y_src
        y_pool,   # y_tgt
        g_pool,   # g_src
        g_pool,   # g_tgt
        NITER
    )

    # process mask for NIfTI mapping later
    try:
        process_mask, process_mask_affine = masking.load_mask_img(tgt_masker.mask_img)
    except Exception:
        process_mask, process_mask_affine = masking._load_mask_img(tgt_masker.mask_img)

    # run iterations
    for i in tqdm(range(NITER), desc=f"Within-content PvI ({content_value})"):
        idx_train_pool = Source_train_is[i]

        test_mask = np.ones(len(y_pool), dtype=bool)
        test_mask[Target_train_is[i]] = False

        X_train = X_pool[idx_train_pool]
        y_train = y_pool[idx_train_pool]

        # X_test = X_pool[test_mask]
        # y_test = y_pool[test_mask]

        X_test = np.delete(X_pool, idx_train_pool, axis=0)
        y_test = np.delete(y_pool, idx_train_pool, axis=0)

        searchlight_raw[i] = search_light_ALI(
            X_train, y_train,
            make_estimator,  # factory
            A,
            X_test, y_test,
            scoring=balanced_accuracy_score,
            verbose=0,
            i_iter=i
        )

    if not searchlight_raw.empty:
        searchlight_raw.to_csv(pwithin, index=False)
        print(f"✅ Saved raw within-content results to: {pwithin}")


# -----------------------------
# Load + summarize + write NIfTIs
# -----------------------------
searchlight_score = load_and_parse_searchlight(pwithin, NITER, parse_and_extract_score)
searchlight_cov   = load_and_parse_searchlight(pwithin, NITER, parse_and_extract_cov_distance)
searchlight_lin   = load_and_parse_searchlight(pwithin, NITER, parse_and_extract_lin_disc)
# searchlight_nf    = load_and_parse_searchlight(pwithin, NITER, parse_and_extract_norm_frechet)

if not is_numeric_df(searchlight_score):
    print("No numeric results found; nothing to summarize.")
    sys.exit(0)

score_mean = searchlight_score.mean(axis=1)
score_std  = searchlight_score.std(axis=1)

cov_mean = searchlight_cov.mean(axis=1)
cov_std  = searchlight_cov.std(axis=1)

lin_mean = searchlight_lin.mean(axis=1)
lin_std  = searchlight_lin.std(axis=1)

# normalized frechet distance disabled; keep NaN columns
# nf_mean = pd.Series(np.nan, index=score_mean.index)
# nf_std  = pd.Series(np.nan, index=score_mean.index)

raw_df = load_raw_data_for_metadata(pwithin)
summary_df = pd.DataFrame({
    "score_mean": score_mean,
    "score_std": score_std,
    "cov_distance_mean": cov_mean,
    "cov_distance_std": cov_std,
    "linear_discrepancy_mean": lin_mean,
    "linear_discrepancy_std": lin_std,
    "task": "PvI within-content",
    "content": str(content_value),
    "voxel_indices": raw_df.iloc[:, 0].apply(lambda x: extract_metadata(x, "voxel_indices")) if not raw_df.empty else "N/A"
})
psummary = os.path.join(out_dir, f"map_summary_{NITER}iter_{radius}mm_BASELINE.csv")
summary_df.to_csv(psummary, index=True, index_label="row")
print(f"✅ Summary CSV saved to: {psummary}")

# Ensure process_mask exists
if "process_mask" not in locals():
    try:
        process_mask, process_mask_affine = masking.load_mask_img(tgt_masker.mask_img)
    except Exception:
        process_mask, process_mask_affine = masking._load_mask_img(tgt_masker.mask_img)

# Write NIfTIs
def write_map(vec, filename):
    arr = np.zeros(process_mask.shape, dtype=float)
    arr[process_mask] = vec.values
    img = new_img_like(tgt_masker.mask_img, arr, copy_header=False)
    img.to_filename(os.path.join(out_dir, filename))

write_map(score_mean, f"map_acc_{NITER}iter_{radius}mm_BASELINE.nii.gz")
write_map(score_std,  f"map_std_{NITER}iter_{radius}mm_BASELINE.nii.gz")
write_map(cov_mean,   f"map_cov_{NITER}iter_{radius}mm_BASELINE.nii.gz")
write_map(cov_std,    f"map_cov_std_{NITER}iter_{radius}mm_BASELINE.nii.gz")
write_map(lin_mean,   f"map_lin_disc_{NITER}iter_{radius}mm_BASELINE.nii.gz")
write_map(lin_std,    f"map_lin_disc_std_{NITER}iter_{radius}mm_BASELINE.nii.gz")

print(f"--- Finished WITHIN-CONTENT DOMAIN (PvI) for Subject: {subject} | content: {content_value} ---")
