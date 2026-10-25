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
v10 CONSISTENT SCRIPT (I→I baseline, no DA, no naive)
- Same directory structure as P→I v10
- Uses search_light_ALI and stores per-sphere dict strings in CSV
- Keeps MAX_SPHERES_TO_RUN=50 option for quick checking

Usage:
  python searchlight_PP_v10_BASELINE.py <subject_index> <n_iterations>

Example:
  python searchlight_PP_v10_BASELINE.py 0 10
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import re
import sys
import glob
import warnings
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from joblib import Parallel, cpu_count, delayed
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score

from nilearn import masking
from nilearn.image import new_img_like

from adapt.metrics import (
    cov_distance,
    # normalized_frechet_distance,
    normalized_linear_discrepancy,
)

# Use centralized path configuration (handles sys.path bootstrapping)
from da_paths import DATA_ROOT, RESULTS_ROOT, FIGURES_ROOT

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"

from algorithms.searchlight_DA_save import get_sphere_data, searchlight_cv_DA
from dataManipulation.whole_brain import load_data


# -----------------------------
# PARAMETERS / CLI
# -----------------------------
if len(sys.argv) != 3:
    print("Usage: python searchlight_Cross_Domain_Content_Decoding_IMAGERY.py <subject_index> <n_iterations>")
    sys.exit(1)

source_domain = "imagery"
target_domain = "imagery"
radius = 12

subjects = sorted(
    [S.split("/")[-1] for S in glob.glob(str(DATA_ROOT / source_domain / "*"))],
    key=lambda x: int(x),
)

subject_idx = int(sys.argv[1])
NITER = int(sys.argv[2])
subject = subjects[subject_idx]
s = re.sub(r"[0-9]+_", "", subject).capitalize()

print(f"--- Running I→I baseline (v10) | Subject: {subject} | NITER={NITER} ---")


# -----------------------------
# v10 OUTPUT DIRS (MATCH P→I)
# -----------------------------
base_out_dir = RESULTS_ROOT / "searchlight" / "Cross_Domain_Content_Decoding" / "I2I" / subject
base_fig_dir = FIGURES_ROOT / "searchlight" / "Cross_Domain_Content_Decoding" / "I2I" / subject
os.makedirs(base_out_dir, exist_ok=True)
os.makedirs(base_fig_dir, exist_ok=True)

# Keep naming parallel to your P→I scheme but tag as IMAGERY
pbase = os.path.join(base_out_dir, f"map_{NITER}iter_{radius}mm_BASELINE.csv")


# ----------------------------------------------------------------------------------
# SEARCHLIGHT (ALI style) — returns per-voxel dict strings
# ----------------------------------------------------------------------------------
def search_light_ALI(
    X,
    y,
    estimator,
    A,
    X_test,
    y_test,
    groups=None,
    scoring=None,
    n_jobs=2,
    verbose=0,
    i_iter=0,
):
    # keep robust but simple
    n_jobs_limit = cpu_count() - 2 if n_jobs == -1 else n_jobs
    n_jobs_actual = max(1, min(n_jobs_limit, os.cpu_count() or 1))

    from nilearn.decoding.searchlight import GroupIterator
    group_iter = GroupIterator(A.shape[0], n_jobs_actual)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        scores = Parallel(n_jobs=n_jobs_actual, verbose=verbose)(
            delayed(_group_iter_search_light)(
                A.rows[list_i],
                estimator,
                X,
                y,
                groups,
                scoring,
                X_test,
                y_test,
                thread_id + 1,
                A.shape[0],
                verbose,
                i_iter=i_iter,
            )
            for thread_id, list_i in enumerate(group_iter)
        )

    return np.concatenate(scores)


def _group_iter_search_light(
    list_rows,
    estimator,
    X,
    y,
    groups,
    scoring,
    X_test,
    y_test,
    thread_id,
    total,
    verbose=0,
    i_iter=0,
):
    """
    IMPORTANT:
    - Must return an array of length == len(list_rows),
      otherwise np.concatenate() will shrink the output and break CSV shape.
    - We support MAX_SPHERES_TO_RUN=50 by filling the remainder with NaNs.
    """
    t0 = time.time()

    # QUICK CHECK OPTION (you requested to keep this)
    #MAX_SPHERES_TO_RUN = 50  # set to float('inf') for full run
    MAX_SPHERES_TO_RUN = float('inf')

    out = np.empty(len(list_rows), dtype=object)

    for i, row in enumerate(list_rows):
        if i >= MAX_SPHERES_TO_RUN:
            # Fill remaining to keep shape consistent
            # for j in range(i, len(list_rows)):
            #     out[j] = str({
            #         "score": np.nan,
            #         "cov_distance": np.nan,
            #         "linear_discrepancy": np.nan,
            #         "da_attributes": {},
            #         "voxel_indices": list_rows[j],
            #     })
            print(f"\nStopping: Processed requested {MAX_SPHERES_TO_RUN} spheres.")
            break

        # baseline I→I: train on X, test on X_test
        fitted = estimator().fit(X[:, row], y)
        y_pred = fitted.predict(X_test[:, row])

        score = scoring(y_test, y_pred)

        # baseline metrics here are TRAIN-vs-TEST within imagery (not cross-domain)
        cov = cov_distance(X[:, row], X_test[:, row])
        lin = normalized_linear_discrepancy(X[:, row], X_test[:, row])
        # nf = normalized_frechet_distance(X[:, row], X_test[:, row])
        # nf = np.nan  # normalized_frechet_distance disabled

        out[i] = str({
            "score": float(score),
            "cov_distance": float(cov),
            "linear_discrepancy": float(lin),
            # "normalized_frechet_distance": float(nf),
            "da_attributes": {},
            "voxel_indices": row,
        })

    return out


# ----------------------------------------------------------------------------------
# PARSERS (same style as P→I)
# ----------------------------------------------------------------------------------
def parse_and_extract_metric(data_string, metric_keys):
    if isinstance(data_string, str):
        for key in metric_keys:
            try:
                pattern = fr'["\']?{key}["\']?\s*:\s*([0-9\.\-]+)'
                m = re.search(pattern, data_string)
                if m:
                    return float(m.group(1))
            except Exception:
                continue
    return np.nan

def parse_score(x): return parse_and_extract_metric(x, ["score", "accuracy"])
def parse_cov(x):   return parse_and_extract_metric(x, ["cov_distance", "cov_dist"])
def parse_lin(x):   return parse_and_extract_metric(x, ["linear_discrepancy", "lin_disc"])
# def parse_nf(x):    return parse_and_extract_metric(
    # x,
    # [
        # "normalized_frechet_distance",
        # "norm_frechet_distance",
        # "normalized_frechet",
    # ],
# )

def load_and_parse_searchlight(pfile, niter, metric_parser):
    if Path(pfile).is_file():
        try:
            df = pd.read_csv(pfile, dtype={col: str for col in range(niter)})
            for col in df.columns:
                df[col] = df[col].apply(metric_parser)
            return df
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()

def is_numeric_df(df):
    return (df is not None) and (not df.empty) and df.iloc[:, 0].notna().any()


# ----------------------------------------------------------------------------------
# MAIN COMPUTE
# ----------------------------------------------------------------------------------
if Path(pbase).is_file():
    print(f"Results file found: {pbase} — Nothing to do.")
else:
    # Load data
    bold_data, events, masker = load_data(source_domain, subject)
    tgt_data, tgt_events, tgt_masker = load_data(target_domain, subject)

    # Groups (same as DA scripts)
    g = events.run + "trial" + events.trial_idx.astype(str)
    g_tgt = tgt_events.run + "trial" + tgt_events.trial_idx.astype(str)

    # Sphere data
    X, A, _, _ = get_sphere_data(masker, bold_data, radius)
    X_tgt, A_tgt, _, _ = get_sphere_data(tgt_masker, tgt_data, radius)

    # Splits (same helper as your P→I script)
    Source_train_is, Target_train_is = searchlight_cv_DA(
        events.target_category.values,
        tgt_events.target_category.values,
        g,
        g_tgt,
        NITER
    )

    searchlight = pd.DataFrame()

    for i in tqdm(range(NITER), desc="Processing IMAGERY"):
        # define test set in the same style as P→I:
        Target_test_is = np.ones(tgt_events.target_category.values.size, dtype=bool)
        Target_test_is[Target_train_is[i]] = False

        xtrain = X[Source_train_is[i]]
        ytrain = events.target_category.values[Source_train_is[i]]

        # xtest = X_tgt[Target_test_is]
        # ytest = tgt_events.target_category.values[Target_test_is]

        xtest = np.delete(X, Source_train_is[i], axis=0)
        ytest = np.delete(events.target_category.values,Source_train_is[i],axis=0)

        searchlight[i] = search_light_ALI(
            xtrain, ytrain,
            LogisticRegression,
            A,
            xtest, ytest,
            scoring=balanced_accuracy_score,
            verbose=0,
            i_iter=i
        )

    searchlight.to_csv(pbase, index=False)
    print(f"✅ Raw CSV saved: {pbase}")


# ----------------------------------------------------------------------------------
# LOAD + SUMMARY + NIFTI (same idea as P→I)
# ----------------------------------------------------------------------------------
searchlight = load_and_parse_searchlight(pbase, NITER, parse_score)
searchlight_cov = load_and_parse_searchlight(pbase, NITER, parse_cov)
searchlight_lin = load_and_parse_searchlight(pbase, NITER, parse_lin)
# searchlight_nf = load_and_parse_searchlight(pbase, NITER, parse_nf)

if is_numeric_df(searchlight):
    print("--- Generating summary + NIfTIs: IMAGERY ---")

    mean_acc = searchlight.mean(axis=1)
    std_acc = searchlight.std(axis=1)

    mean_cov = searchlight_cov.mean(axis=1) if is_numeric_df(searchlight_cov) else pd.Series(np.nan, index=mean_acc.index)
    std_cov = searchlight_cov.std(axis=1)   if is_numeric_df(searchlight_cov) else pd.Series(np.nan, index=mean_acc.index)

    mean_lin = searchlight_lin.mean(axis=1) if is_numeric_df(searchlight_lin) else pd.Series(np.nan, index=mean_acc.index)
    std_lin = searchlight_lin.std(axis=1)   if is_numeric_df(searchlight_lin) else pd.Series(np.nan, index=mean_acc.index)

    # normalized frechet distance disabled; keep NaN columns
    # mean_nf = pd.Series(np.nan, index=mean_acc.index)
    # std_nf = pd.Series(np.nan, index=mean_acc.index)

    # Summary CSV (same naming style as your v10)
    psummary = os.path.join(base_out_dir, f"map_summary_{NITER}iter_{radius}mm_BASELINE.csv")
    pd.DataFrame({
        "score_mean": mean_acc,
        "score_std": std_acc,
        "cov_distance_mean": mean_cov,
        "cov_distance_std": std_cov,
        "linear_discrepancy_mean": mean_lin,
        "linear_discrepancy_std": std_lin,
        "method": "IMAGERY"
    }).to_csv(psummary, index=True, index_label="row")
    print(f"✅ Summary CSV saved to: {psummary}")

    # Load mask for writing images
    _, _, masker = load_data(source_domain, subject)
    try:
        process_mask, _ = masking.load_mask_img(masker.mask_img)
    except Exception:
        process_mask, _ = masking._load_mask_img(masker.mask_img)

    def series_to_nii(series, fname):
        vol = np.zeros(process_mask.shape, dtype=float)
        vol[process_mask] = series.values
        img = new_img_like(masker.mask_img, vol, copy_header=False)
        img.to_filename(os.path.join(base_out_dir, fname))

    series_to_nii(mean_acc, f"map_acc_{NITER}iter_{radius}mm_BASELINE.nii.gz")
    series_to_nii(std_acc,  f"map_std_{NITER}iter_{radius}mm_BASELINE.nii.gz")
    series_to_nii(mean_cov, f"map_cov_{NITER}iter_{radius}mm_BASELINE.nii.gz")
    series_to_nii(std_cov,  f"map_cov_std_{NITER}iter_{radius}mm_BASELINE.nii.gz")
    series_to_nii(mean_lin, f"map_lin_disc_{NITER}iter_{radius}mm_BASELINE.nii.gz")
    series_to_nii(std_lin,  f"map_lin_disc_std_{NITER}iter_{radius}mm_BASELINE.nii.gz")

print(f"--- Finished I→I IMAGERY for Subject: {subject} ---")
