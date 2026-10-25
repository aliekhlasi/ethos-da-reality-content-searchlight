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
Cross-CONTENT DOMAIN decoding script.

- Decode domain (Perception vs Imagery) as labels.
- Use CONTENT (e.g., content1 vs content2) as the domain for DA.
- Train on content1 (P+I), test on content2 (P+I).
- Compute Baseline (no DA) and DA maps, plus covariance / discrepancy metrics,
  using the shared searchlight and summary code.

Adapted from the cross-domain (P2I / I2P) script, with the training/testing
design changed to cross-content decoding.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import re
import sys
import glob
import warnings
from pathlib import Path
import ast
import json
import joblib

from joblib import Parallel, cpu_count, delayed
from nilearn.decoding.searchlight import search_light, GroupIterator, _group_iter_search_light
from sklearn.exceptions import ConvergenceWarning
import time

# Use centralized path configuration (handles sys.path bootstrapping)
from da_paths import DATA_ROOT, RESULTS_ROOT, FIGURES_ROOT

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"

import pandas as pd
from tqdm import tqdm
import numpy as np
from nilearn import plotting
from nilearn import masking
from nilearn.image import new_img_like
from matplotlib import pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import LabelBinarizer
from adapt.utils import check_arrays

# Import DA methods
from adapt.parameter_based import RegularTransferLR, RegularTransferLC
from adapt.feature_based import PRED, FA, CORAL, SA, fMMD, DANN, TCA
from adapt.instance_based import (
    LDM, KLIEP, KMM, ULSIF, RULSIF,
    NearestNeighborsWeighting, IWC, IWN,
    BalancedWeighting, TrAdaBoost
)

# Metrics
from adapt.metrics import (
    neg_j_score, 
    cov_distance,
    # normalized_frechet_distance,
    normalized_linear_discrepancy,
    # frechet_distance,
    domain_classifier
)

# We keep these imports but will NOT use searchlight_cv_DA anymore in the main loop
from algorithms.searchlight_DA_save import get_sphere_data, searchlight_cv_DA, search_light
from dataManipulation.whole_brain import load_data

# --- DA method maps ---
DA_METHODS_MAP = {
    "PRED": PRED, "FA": FA, "CORAL": CORAL, "SA": SA, "fMMD": fMMD, "DANN": DANN, "TCA": TCA,
    "LDM": LDM, "KLIEP": KLIEP, "KMM": KMM, "ULSIF": ULSIF, "RULSIF": RULSIF,
    "NearestNeighborsWeighting": NearestNeighborsWeighting, "IWC": IWC, "IWN": IWN,
    "BalancedWeighting": BalancedWeighting, "TrAdaBoost": TrAdaBoost,
    "RTLC": RegularTransferLC
}

METHODS_REQUIRING_YT = {
    "RTLC", "BalancedWeighting", "TrAdaBoost", "IWC", "IWN", "PRED", "BalanceWeighting",
    "NearestNeighborsWeighting", "LDM", "FA"
}

# --- Deep DA helper nets for DANN ---
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Flatten
from tensorflow.keras.optimizers import Adam


def get_encoder(input_shape):
    model = Sequential(name='Encoder')
    model.add(Input(shape=input_shape))
    model.add(Dense(64, activation='relu', name='enc_dense_1'))
    model.add(Dense(32, activation='relu', name='enc_latent'))
    return model


def get_discriminator(input_shape, latent_dim=32):
    model = Sequential(name='Discriminator')
    model.add(Input(shape=input_shape))
    model.add(Dense(32, activation='relu', name='disc_dense_1'))
    model.add(Dense(1, activation='sigmoid', name='disc_output2'))
    return model


def get_task(input_shape, num_classes=2):
    model = Sequential(name='Task_Classifier')
    model.add(Input(shape=input_shape))
    model.add(Dense(num_classes, activation='softmax', name='task_output'))
    return model


# --- SCRIPT ARGS ---
if len(sys.argv) != 4:
    print("Usage: python searchligh_other.py <subject_index> <n_iterations> <da_method_name>")
    sys.exit(1)

source_domain = 'perception'
target_domain = 'imagery'
radius = 12

subjects = sorted(
    [S.split('/')[-1] for S in glob.glob(str(DATA_ROOT / 'perception' / '*'))],
    key=lambda x: int(x)
)
subject_idx = int(sys.argv[1])
NITER = int(sys.argv[2])
da_method_name = sys.argv[3]

if da_method_name not in DA_METHODS_MAP:
    print(f"Error: Unknown DA method '{da_method_name}'")
    print("Available methods are:", list(DA_METHODS_MAP.keys()))
    sys.exit(1)

subject = subjects[subject_idx]
DA_class = DA_METHODS_MAP[da_method_name]
s = re.sub('[0-9]+_', '', subject).capitalize()

print(f"--- Running CROSS-CONTENT DOMAIN searchlight for Subject: {subject}, Method: {da_method_name}, NITER: {NITER} ---")

# --- OUTPUT DIRS ---
base_out_dir = RESULTS_ROOT / 'searchlight' / 'Cross_Content_Domain_Decoding' / 'Ci2Cj' / subject
base_fig_dir = FIGURES_ROOT / 'searchlight' / 'Cross_Content_Domain_Decoding' / 'Ci2Cj' / subject
out_dir = base_out_dir / da_method_name
savefig_dir = base_fig_dir / da_method_name

os.makedirs(out_dir, exist_ok=True)
os.makedirs(savefig_dir, exist_ok=True)

# BASELINE / NAIVE / DA CSVs
pbase = os.path.join(base_out_dir, f'map_{NITER}iter_{radius}mm_BASELINE.csv')
pnaive = os.path.join(base_out_dir, f'map_{NITER}iter_{radius}mm_NAIVE.csv')
pda   = os.path.join(out_dir,       f'map_{NITER}iter_{radius}mm_{da_method_name}.csv')


# ----------------------------------------------------------------------------------
# --- SEARCHLIGHT FUNCTION (same as your version) ---
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
    DA=None,
    X_tgt=None,
    y_tgt=None,
    n_jobs=-1,
    verbose=0,
    i_iter=0
):
    if DA is not None and getattr(DA, "__name__", "") == "DANN":
        n_jobs_limit = 1
    else:
        n_jobs_limit = cpu_count() - 2 if n_jobs == -1 else n_jobs

    n_jobs_actual = min(n_jobs_limit, os.cpu_count() or 1)

    group_iter = GroupIterator(A.shape[0], n_jobs)
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
                DA,
                X_tgt,
                y_tgt,
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
    estimator,
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
    t0 = time.time()

    sphere_results = []

    # You can set this to inf if you want all spheres
    #MAX_SPHERES_TO_RUN = 50
    MAX_SPHERES_TO_RUN = float('inf')

    for i, row in enumerate(list_rows):
        if i >= MAX_SPHERES_TO_RUN:
            print(f"\nStopping: Processed requested {MAX_SPHERES_TO_RUN} spheres.")
            break

        metrics = {
            'neg_j_score': np.nan,
            'cov_distance': np.nan,
            'linear_discrepancy': np.nan,
            # 'normalized_frechet_distance': np.nan,
            # 'frechet_distance': np.nan,
            'domain_classifier': np.nan
        }
        da_attributes = {}

        if DA is not None:

            if getattr(DA, "__name__", "") == "DANN":
                # ---------------- DANN PATH ----------------
                n_features = X[:, row].shape[1]
                unique_labels = np.unique(y)
                n_classes = len(unique_labels)
                LATENT_DIM = 32
                BATCH_SIZE = 16

                feature_extractor = get_encoder(input_shape=(n_features,))
                task_estimator = get_task(input_shape=(32,), num_classes=n_classes)
                domain_estimator = get_discriminator(input_shape=(32,))

                da_instance = DANN(
                    encoder=feature_extractor,
                    task=task_estimator,
                    discriminator=domain_estimator,
                    Xt=X_tgt[:, row],
                    random_state=0,
                    verbose=0
                )

                # One-hot labels
                label_binarizer = LabelBinarizer()
                y_indices = label_binarizer.fit_transform(y)
                y_one_hot = np.zeros((y.size, n_classes))
                if n_classes == 2 and y_indices.ndim == 2:
                    y_indices = y_indices.flatten()
                y_one_hot[np.arange(y.size), y_indices] = 1

                X_float = X[:, row].astype(np.float32)
                X_tgt_float = X_tgt[:, row].astype(np.float32)
                y_float = y_one_hot.astype(np.float32)

                clf = da_instance.fit(
                    X_float, y_float,
                    Xt=X_tgt_float,
                    batch_size=BATCH_SIZE,
                    epochs=10,
                    verbose=0
                )

                y_pred_proba = clf.predict(X_test[:, row].astype(np.float32))
                y_pred = np.argmax(y_pred_proba, axis=1)
                fitted_estimator = clf

                subject_id = subject
                da_name = DA.__name__
                estimators_dir = RESULTS_ROOT / 'searchlight' / 'Cross_Content_Domain_Decoding' / 'Ci2Cj' / subject_id / da_name / 'searchlight_estimators'
                os.makedirs(estimators_dir, exist_ok=True)

                task_filename = str(estimators_dir / f'task_iter{i_iter}_sphere{i}.h5')
                encoder_filename = str(estimators_dir / f'encoder_iter{i_iter}_sphere{i}.h5')
                discriminator_filename = str(estimators_dir / f'discriminator_iter{i_iter}_sphere{i}.h5')
                fitted_estimator.task_.save(task_filename)
                fitted_estimator.encoder_.save(encoder_filename)
                fitted_estimator.discriminator_.save(discriminator_filename)
                da_attributes['task_path'] = task_filename
                da_attributes['estimator_path'] = encoder_filename
                da_attributes['discriminator_path'] = discriminator_filename

            else:
                # --------- generic ADAPT methods path (CORAL, SA, etc.) ---------
                da_instance = DA(estimator=estimator())

                if y_tgt is None:
                    clf = da_instance.fit(X[:, row], y, Xt=X_tgt[:, row])
                else:
                    clf = da_instance.fit(X[:, row], y, Xt=X_tgt[:, row], yt=y_tgt)

                fitted_estimator = clf
                y_pred = clf.predict(X_test[:, row])

            # metrics after DA
            Xs_trans = fitted_estimator.transform(X[:, row]) if hasattr(fitted_estimator, 'transform') else X[:, row]
            Xt_trans = fitted_estimator.transform(X_tgt[:, row]) if hasattr(fitted_estimator, 'transform') else X_tgt[:, row]

            metrics['cov_distance'] = cov_distance(Xs_trans, Xt_trans)
            metrics['linear_discrepancy'] = normalized_linear_discrepancy(Xs_trans, Xt_trans)
            # metrics['normalized_frechet_distance'] = np.nan  # normalized_frechet_distance disabled

            from adapt.feature_based import FA, CORAL, PRED, TCA, SA
            if isinstance(fitted_estimator, (FA, SA, CORAL, PRED, TCA)):
                subject_id = subject
                da_name = DA.__name__
                # NOTE: estimator .pkl saving disabled for faster processing.
                # Uncomment to re-enable saving searchlight estimators.
                # estimators_dir = RESULTS_ROOT / 'searchlight' / 'Cross_Content_Domain_Decoding' / 'Ci2Cj' / subject_id / da_name / 'searchlight_estimators'
                # os.makedirs(estimators_dir, exist_ok=True)
                # estimator_filename = str(
                #     estimators_dir / f'{da_name.lower()}_iter{i_iter}_sphere{i}.pkl'
                # )
                # joblib.dump(fitted_estimator, estimator_filename)
                # da_attributes['estimator_path'] = estimator_filename

        else:
            # no DA – pure baseline
            fitted_estimator = estimator().fit(X[:, row], y)
            y_pred = fitted_estimator.predict(X_test[:, row])
            metrics['cov_distance'] = cov_distance(X[:, row], X_test[:, row])
            metrics['linear_discrepancy'] = normalized_linear_discrepancy(X[:, row], X_test[:, row])
            # metrics['normalized_frechet_distance'] = np.nan  # normalized_frechet_distance disabled

        par_scores[i] = scoring(y_test, y_pred)

        sphere_data = {
            'score': par_scores[i],
            'cov_distance': metrics['cov_distance'],
            'linear_discrepancy': metrics['linear_discrepancy'],
            # 'normalized_frechet_distance': metrics['normalized_frechet_distance'],
            'da_attributes': da_attributes,
            'voxel_indices': row
        }
        sphere_results.append(sphere_data)

    return pd.Series([str(r) for r in sphere_results])


# --- Parsing helpers (unchanged) ---
def parse_and_extract_metric(data_string, metric_keys):
    if isinstance(data_string, str):
        for key in metric_keys:
            try:
                pattern = fr'["\']?{key}["\']?\s*:\s*([0-9\.\-]+)'
                match = re.search(pattern, data_string)
                if match:
                    return float(match.group(1))
            except Exception:
                continue
    return np.nan


def parse_and_extract_score(data_string):
    return parse_and_extract_metric(data_string, ['score', 'accuracy'])


def parse_and_extract_j_score(data_string):
    return parse_and_extract_metric(data_string, ['neg_j_score', 'j_score', 'jscore'])


def parse_and_extract_cov_distance(data_string):
    return parse_and_extract_metric(data_string, ['cov_distance', 'cov_dist'])


def parse_and_extract_lin_disc(data_string):
    return parse_and_extract_metric(data_string, ['linear_discrepancy', 'lin_disc'])

# def parse_and_extract_norm_frechet(data_string):
    # return parse_and_extract_metric(
        # data_string,
        # [
            # 'normalized_frechet_distance',
            # 'norm_frechet_distance',
            # 'normalized_frechet',
        # ],
    # )
# 
# 
# def parse_and_extract_frechet(data_string):
    # return parse_and_extract_metric(data_string, ['frechet_distance', 'frechet'])
# 
# 
def parse_and_extract_domain_clf(data_string):
    return parse_and_extract_metric(data_string, ['domain_classifier', 'domain_clf', 'domain_acc'])


def extract_sphere_ids(path_string):
    if isinstance(path_string, str):
        match = re.search(r'_(\d+)_(\d+)\.pkl$', path_string)
        if match:
            return int(match.group(1)), int(match.group(2))
    return np.nan, np.nan


def extract_metadata(data_string, key):
    if isinstance(data_string, str) and data_string.startswith('{'):
        processed_string = data_string.replace('\n', ' ').replace("'", '"')

        if key == 'estimator_path':
            match = re.search(r'"estimator_path":\s*"(.*?)"', processed_string)
            return match.group(1) if match else "N/A"

        elif key == 'voxel_indices':
            match = re.search(r'"voxel_indices":\s*(\[.*?\])', processed_string)
            return match.group(1) if match else "N/A"
    return "N/A_Error"


def load_and_parse_searchlight(pfile, NITER, metric_parser):
    df = pd.DataFrame()
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
    return not df.empty and df.iloc[:, 0].notna().any()


# --- MAIN LOGIC ---
searchlight, searchlight_naive, searchlight_DA = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

is_file = {
    'baseline': Path(pbase).is_file(),
    'naive': Path(pnaive).is_file(),
    'da': Path(pda).is_file()
}
compute_baseline_models = not Path(pbase).is_file() or not Path(pnaive).is_file()

if all(is_file.values()):
    print(f'All results files found for subject {subject} and method {da_method_name}. Nothing to do.')
else:
    # --- Load data ---
    bold_data, events, masker = load_data(source_domain, subject)
    tgt_data, tgt_events, tgt_masker = load_data(target_domain, subject)

    # Sphere data
    X, A, _, _ = get_sphere_data(masker, bold_data, radius)
    X_tgt, A_tgt, _, _ = get_sphere_data(tgt_masker, tgt_data, radius)

    # group IDs (run + trial) for leakage-safe splitting
    # g = events.run + "trial" + events.trial_idx.astype(str)
    # g_tgt = tgt_events.run + "trial" + tgt_events.trial_idx.astype(str)

    # --- CONTENT AND DOMAIN LABELS ---
    # Content labels (e.g. living vs nonliving)
    content_P = events.target_category.values
    content_I = tgt_events.target_category.values

    # Domain labels: 0 = P, 1 = I
    dom_P = np.zeros(len(events), dtype=int)
    dom_I = np.ones(len(tgt_events), dtype=int)

    g = (events.run + 'trial' + events.trial_idx.astype(str)).reset_index(drop=True)
    g_tgt = (tgt_events.run + 'trial' + tgt_events.trial_idx.astype(str)).reset_index(drop=True)
    # Unique contents across P+I
    all_contents = np.unique(np.concatenate([content_P, content_I]))
    if len(all_contents) != 2:
        raise ValueError(f"Expected exactly 2 content categories for cross-content decoding, got: {all_contents}")

    content1, content2 = all_contents[0], all_contents[1]
    print(f"Cross-CONTENT domain decoding: train on content1 = {content1}, test on content2 = {content2}")

    # ---- Ci -> Cj means: train on content1, test on content2 ----
    train_content = content1   # Ci
    test_content  = content2   # Cj
    print(f"Cross-CONTENT domain decoding: train on Ci = {train_content}, test on Cj = {test_content}")

    # Build TRAIN pool (Ci)
    idx_P_train = np.where(content_P == train_content)[0]
    idx_I_train = np.where(content_I == train_content)[0]
    X_train_pool = np.vstack([X[idx_P_train], X_tgt[idx_I_train]])
    y_train_pool = np.concatenate([dom_P[idx_P_train], dom_I[idx_I_train]])
    g_train_pool = pd.concat([g.iloc[idx_P_train], g_tgt.iloc[idx_I_train]], ignore_index=True)


    # Build TEST/ADAPT pool (Cj)
    idx_P_test = np.where(content_P == test_content)[0]
    idx_I_test = np.where(content_I == test_content)[0]
    X_test_pool = np.vstack([X[idx_P_test], X_tgt[idx_I_test]])
    y_test_pool = np.concatenate([dom_P[idx_P_test], dom_I[idx_I_test]])
    g_test_pool = pd.concat([g.iloc[idx_P_test], g_tgt.iloc[idx_I_test]], ignore_index=True)


    # Leakage-safe splits (train pool vs test pool)
    Source_train_is, Target_train_is = searchlight_cv_DA(
        y_train_pool,
        y_test_pool,
        g_train_pool,
        g_test_pool,
        NITER
    )
    
    # For mapping to 3D later
    try:
        process_mask, process_mask_affine = masking.load_mask_img(tgt_masker.mask_img)
    except Exception:
        process_mask, process_mask_affine = masking._load_mask_img(tgt_masker.mask_img)

    # --- ITERATIONS ---
    for i in tqdm(range(NITER), desc=f"Processing {da_method_name} (cross-content domain)"):
        # ----- TRAINING SET (content1, P+I) -----
        src_idx = Source_train_is[i]      # indices into X_c1
        tgt_idx = Target_train_is[i]      # indices into X_c2

        # held-out target test = content2 pooled trials not in tgt_idx
        tgt_test_mask = np.ones(len(y_test_pool), dtype=bool)
        tgt_test_mask[tgt_idx] = False

        Xs_train = X_train_pool[src_idx]
        ys_train = y_train_pool[src_idx]

        Xt_adapt = X_test_pool[tgt_idx]
        yt_adapt = y_test_pool[tgt_idx]

        Xt_test = X_test_pool[tgt_test_mask]
        yt_test = y_test_pool[tgt_test_mask]


        # For this script, we keep "baseline" and "naive" identical:
        # both = train on content1 (P+I), test on content2 (P+I)
        if compute_baseline_models:
            X_train_naive = np.vstack([X_train_pool[src_idx], X_test_pool[tgt_idx]])
            y_train_naive = np.concatenate([y_train_pool[src_idx], y_test_pool[tgt_idx]])
            # BASELINE
            if not is_file['baseline']:
                searchlight[i] = search_light_ALI(
                    Xs_train, ys_train,
                    LogisticRegression, A,
                    Xt_test, yt_test,
                    scoring=balanced_accuracy_score,
                    verbose=0,
                    i_iter=i
                )

            # NAIVE (same design here, just kept to not break downstream code)
            if not is_file['naive']:
                searchlight_naive[i] = search_light_ALI(
                    X_train_naive, y_train_naive,
                    LogisticRegression, A,
                    Xt_test, yt_test,
                    scoring=balanced_accuracy_score,
                    verbose=0,
                    i_iter=i
    )

        # ----- DA: align content1 → content2, task = P vs I -----
        if not is_file['da']:
            da_kwargs = {'DA': DA_class, 'X_tgt': Xt_adapt, 'i_iter': i}
            if da_method_name in METHODS_REQUIRING_YT:
                da_kwargs['y_tgt'] = yt_adapt
            else:
                da_kwargs['y_tgt'] = None

            searchlight_DA[i] = search_light_ALI(
                Xs_train, ys_train,
                LogisticRegression, A,
                Xt_test, yt_test,
                scoring=balanced_accuracy_score,
                verbose=0,
                **da_kwargs
            )

    # --- SAVE RAW DATA ---
    if not searchlight.empty:
        searchlight.to_csv(pbase, index=False)
    if not searchlight_DA.empty:
        searchlight_DA.to_csv(pda, index=False)
    if not searchlight_naive.empty:
        searchlight_naive.to_csv(pnaive, index=False)


# =====================================================================
#   BELOW: your original parsing + summary + NIfTI writing is unchanged
#   except that it now reflects CROSS-CONTENT DOMAIN decoding.
# =====================================================================

# Load score DataFrames
searchlight      = load_and_parse_searchlight(pbase, NITER, parse_and_extract_score)
searchlight_DA   = load_and_parse_searchlight(pda,   NITER, parse_and_extract_score)
searchlight_naive = load_and_parse_searchlight(pnaive, NITER, parse_and_extract_score)

searchlight_cov      = load_and_parse_searchlight(pbase, NITER, parse_and_extract_cov_distance)
searchlight_DA_cov   = load_and_parse_searchlight(pda,   NITER, parse_and_extract_cov_distance)
searchlight_naive_cov = load_and_parse_searchlight(pnaive, NITER, parse_and_extract_cov_distance)

searchlight_lin_disc      = load_and_parse_searchlight(pbase, NITER, parse_and_extract_lin_disc)
searchlight_DA_lin_disc   = load_and_parse_searchlight(pda,   NITER, parse_and_extract_lin_disc)
searchlight_naive_lin_disc = load_and_parse_searchlight(pnaive, NITER, parse_and_extract_lin_disc)

# searchlight_norm_frechet = load_and_parse_searchlight(pbase, NITER, parse_and_extract_norm_frechet)
# searchlight_DA_norm_frechet = load_and_parse_searchlight(pda, NITER, parse_and_extract_norm_frechet)
# searchlight_naive_norm_frechet = load_and_parse_searchlight(pnaive, NITER, parse_and_extract_norm_frechet)


if is_numeric_df(searchlight) and is_numeric_df(searchlight_DA):
    print("--- Generating plots: DA vs Baseline (cross-content domain) ---")

    searchlight_mean     = searchlight.mean(axis=1)
    searchlight_DA_mean  = searchlight_DA.mean(axis=1)
    searchlight_naive_mean = searchlight_naive.mean(axis=1)

    searchlight_std      = searchlight.std(axis=1)
    searchlight_DA_std   = searchlight_DA.std(axis=1)
    searchlight_naive_std = searchlight_naive.std(axis=1)

    base_cov_mean   = searchlight_cov.mean(axis=1)
    da_cov_mean     = searchlight_DA_cov.mean(axis=1)
    naive_cov_mean  = searchlight_naive_cov.mean(axis=1)

    base_cov_std    = searchlight_cov.std(axis=1)
    da_cov_std      = searchlight_DA_cov.std(axis=1)
    naive_cov_std   = searchlight_naive_cov.std(axis=1)

    base_lin_disc_mean   = searchlight_lin_disc.mean(axis=1)
    da_lin_disc_mean     = searchlight_DA_lin_disc.mean(axis=1)
    naive_lin_disc_mean  = searchlight_naive_lin_disc.mean(axis=1)

    base_lin_disc_std    = searchlight_lin_disc.std(axis=1)
    da_lin_disc_std      = searchlight_DA_lin_disc.std(axis=1)
    naive_lin_disc_std   = searchlight_naive_lin_disc.std(axis=1)

    # normalized frechet distance disabled; keep NaN columns
    nf_nan = pd.Series(np.nan, index=searchlight_mean.index)
    # base_nf_mean = nf_nan
    # da_nf_mean = nf_nan
    # naive_nf_mean = nf_nan

    # base_nf_std = nf_nan
    # da_nf_std = nf_nan
    # naive_nf_std = nf_nan

    raw_base_df  = load_raw_data_for_metadata(pbase)
    raw_da_df    = load_raw_data_for_metadata(pda)
    raw_naive_df = load_raw_data_for_metadata(pnaive)

    # --- DA Summary CSV ---
    if not raw_da_df.empty:
        path_series_da = raw_da_df.iloc[:, 0].apply(lambda x: extract_metadata(x, 'estimator_path'))
        ids_df = path_series_da.apply(lambda x: pd.Series(extract_sphere_ids(x), index=['sphere_group_id', 'sphere_index_id']))

        summary_df_da = pd.DataFrame({
            'score_mean': searchlight_DA_mean,
            'score_std': searchlight_DA_std,
            'cov_distance_mean': da_cov_mean,
            'cov_distance_std': da_cov_std,
            'linear_discrepancy_mean': da_lin_disc_mean,
            'linear_discrepancy_std': da_lin_disc_std,
            'method': da_method_name,
            'estimator_path': path_series_da,
            'voxel_indices': raw_da_df.iloc[:, 0].apply(lambda x: extract_metadata(x, 'voxel_indices'))
        })
        pda_summary = os.path.join(out_dir, f'map_summary_{NITER}iter_{radius}mm_{da_method_name}.csv')
        summary_df_da.to_csv(pda_summary, index=True, index_label='row')
        print(f"✅ DA Summary CSV saved to: {pda_summary}")

    # --- Baseline Summary CSV ---
    if not raw_base_df.empty:
        path_series_base = raw_base_df.iloc[:, 0].apply(lambda x: extract_metadata(x, 'estimator_path'))
        ids_df_base = path_series_base.apply(lambda x: pd.Series(extract_sphere_ids(x), index=['sphere_group_id', 'sphere_index_id']))

        summary_df_base = pd.DataFrame({
            'score_mean': searchlight_mean,
            'score_std': searchlight_std,
            'cov_distance_mean': base_cov_mean,
            'cov_distance_std': base_cov_std,
            'linear_discrepancy_mean': base_lin_disc_mean,
            'linear_discrepancy_std': base_lin_disc_std,
            'method': 'baseline',
            'estimator_path': path_series_base,
            'voxel_indices': raw_base_df.iloc[:, 0].apply(lambda x: extract_metadata(x, 'voxel_indices'))
        })
        pbase_summary = os.path.join(base_out_dir, f'map_summary_{NITER}iter_{radius}mm_BASELINE.csv')
        summary_df_base.to_csv(pbase_summary, index=True, index_label='row')
        print(f"✅ Baseline Summary CSV saved to: {pbase_summary}")

    # --- Naive Summary CSV (same design as baseline here) ---
    if not raw_naive_df.empty:
        path_series_naive = raw_naive_df.iloc[:, 0].apply(lambda x: extract_metadata(x, 'estimator_path'))
        ids_df_naive = path_series_naive.apply(lambda x: pd.Series(extract_sphere_ids(x), index=['sphere_group_id', 'sphere_index_id']))

        summary_df_naive = pd.DataFrame({
            'score_mean': searchlight_naive_mean,
            'score_std': searchlight_naive_std,
            'cov_distance_mean': naive_cov_mean,
            'cov_distance_std': naive_cov_std,
            'linear_discrepancy_mean': naive_lin_disc_mean,
            'linear_discrepancy_std': naive_lin_disc_std,
            'method': 'naive',
            'estimator_path': path_series_naive,
            'voxel_indices': raw_naive_df.iloc[:, 0].apply(lambda x: extract_metadata(x, 'voxel_indices'))
        })
        pnaive_summary = os.path.join(base_out_dir, f'map_summary_{NITER}iter_{radius}mm_NAIVE.csv')
        summary_df_naive.to_csv(pnaive_summary, index=True, index_label='row')
        print(f"✅ Naive Summary CSV saved to: {pnaive_summary}")

    # --- NIfTI maps (accuracy & metrics) – same as your code, now for domain decoding ---
    scores_3D = {}
    i = 0

        # --- Ensure we have tgt_masker/masker even if computation was skipped ---
    if 'tgt_masker' not in locals():
        _, _, tgt_masker = load_data(target_domain, subject)

    if 'masker' not in locals():
        _, _, masker = load_data(source_domain, subject)

    # --- Ensure process_mask exists ---
    if 'process_mask' not in locals():
        try:
            process_mask, process_mask_affine = masking.load_mask_img(tgt_masker.mask_img)
        except Exception:
            process_mask, process_mask_affine = masking._load_mask_img(tgt_masker.mask_img)

    for mean in (
        searchlight_mean,
        searchlight_DA_mean,
        searchlight_DA_mean / searchlight_mean,
        searchlight_mean / searchlight_DA_mean,
        ((searchlight_DA_mean - searchlight_mean) / (searchlight_DA_mean + searchlight_mean)),
        (searchlight_DA_mean - searchlight_mean) / searchlight_DA_mean
    ):
        scores_3D[i] = np.zeros(process_mask.shape)
        if not mean.empty:
            scores_3D[i][process_mask] = mean.values
        i += 1

    img_baseline = new_img_like(tgt_masker.mask_img, scores_3D[0], copy_header=False)
    img_da = new_img_like(tgt_masker.mask_img, scores_3D[1], copy_header=False)
    img_da_div_base = new_img_like(tgt_masker.mask_img, scores_3D[2], copy_header=False)
    img_base_div_da = new_img_like(tgt_masker.mask_img, scores_3D[3], copy_header=False)
    img_da_m_base_div_da_p_base = new_img_like(tgt_masker.mask_img, scores_3D[4], copy_header=False)
    img_da_m_base_div_base = new_img_like(tgt_masker.mask_img, scores_3D[5], copy_header=False)

    img_baseline.to_filename(os.path.join(base_out_dir, f'map_acc_{NITER}iter_{radius}mm_BASELINE.nii.gz'))
    img_da.to_filename(os.path.join(out_dir, f'map_acc_{NITER}iter_{radius}mm_{da_method_name}.nii.gz'))
    img_da_div_base.to_filename(os.path.join(out_dir, f'map_acc_{NITER}iter_{radius}mm_{da_method_name}BASE.nii.gz'))
    img_base_div_da.to_filename(os.path.join(out_dir, f'map_acc_{NITER}iter_{radius}mm_BASE{da_method_name}.nii.gz'))
    img_da_m_base_div_da_p_base.to_filename(os.path.join(out_dir, f'map_acc_{NITER}iter_{radius}mm_{da_method_name}_minus_BASE_d_{da_method_name}_plus_BASE.nii.gz'))
    img_da_m_base_div_base.to_filename(os.path.join(out_dir, f'map_acc_{NITER}iter_{radius}mm_{da_method_name}_minus_BASE_d_BASE.nii.gz'))

    # ACCURACY STD PLOTTING (BASELINE vs DA)

    std_score_3D = {}
    i=0

    for std in (searchlight_std, searchlight_DA_std, searchlight_DA_std/searchlight_std, searchlight_std/searchlight_DA_std, ((searchlight_DA_std - searchlight_std)/(searchlight_DA_std + searchlight_std)),(searchlight_DA_std - searchlight_std)/searchlight_DA_std):
        std_score_3D[i] = np.zeros(process_mask.shape)
        if not std.empty:
            std_score_3D[i][process_mask] = std.values
        i+=1
    

    std_img_baseline = new_img_like(tgt_masker.mask_img, std_score_3D[0], copy_header=False)
    std_img_da = new_img_like(tgt_masker.mask_img, std_score_3D[1], copy_header=False)
    std_img_da_div_base = new_img_like(tgt_masker.mask_img, std_score_3D[2], copy_header=False)
    std_img_base_div_da = new_img_like(tgt_masker.mask_img, std_score_3D[3], copy_header=False)
    std_img_da_m_base_div_da_p_base = new_img_like(tgt_masker.mask_img, std_score_3D[4], copy_header=False)
    std_img_da_m_base_div_base = new_img_like(tgt_masker.mask_img, std_score_3D[5], copy_header=False)

    # Save NIfTI images
    std_img_baseline.to_filename(os.path.join(base_out_dir, f'map_std_{NITER}iter_{radius}mm_BASELINE.nii.gz'))
    std_img_da.to_filename(os.path.join(out_dir, f'map_std_{NITER}iter_{radius}mm_{da_method_name}.nii.gz'))
    std_img_da_div_base.to_filename(os.path.join(out_dir, f'map_std_{NITER}iter_{radius}mm_{da_method_name}BASE.nii.gz'))
    std_img_base_div_da.to_filename(os.path.join(out_dir, f'map_std_{NITER}iter_{radius}mm_BASE{da_method_name}.nii.gz'))
    std_img_da_m_base_div_da_p_base.to_filename(os.path.join(out_dir, f'map_std_{NITER}iter_{radius}mm_{da_method_name}_minus_BASE_d_{da_method_name}_plus_BASE.nii.gz'))
    std_img_da_m_base_div_base.to_filename(os.path.join(out_dir, f'map_std_{NITER}iter_{radius}mm_{da_method_name}_minus_BASE_d_BASE.nii.gz'))


    # Plotting (omitted for brevity, but retained in full script)
    # fig, ax = plt.subplots(4,1,figsize=(12,16))
    # plotting.plot_stat_map(img_baseline, bg_img=masker.mask_img, cut_coords=(0,-5,-4), threshold=0.5, title=f'Baseline > 0.5 - {s}', axes=ax[0])
    # plotting.plot_stat_map(img_da, bg_img=masker.mask_img, cut_coords=(0,-5,-4), threshold=0.5, title=f'{da_method_name} > 0.5 - {s}', axes=ax[1])
    # plotting.plot_stat_map(img_da_div_base, bg_img=masker.mask_img, cut_coords=(0,-5,-4), threshold=1, title=f'{da_method_name}/Baseline > 1 - {s}', axes=ax[2])
    # plotting.plot_stat_map(img_base_div_da, bg_img=masker.mask_img, cut_coords=(0,-5,-4), threshold=1, title=f'Baseline/{da_method_name} > 1 - {s}', axes=ax[3])
    # fig.savefig(os.path.join(savefig_dir, f'fig_{da_method_name}_vs_baseline_{NITER}iter_{radius}mm.png'))
    # plt.close(fig)

    # COVARIANCE PLOTTING (BASELINE vs DA)
    scores_cov_3D = {}
    i=0

    for mean_cov in (base_cov_mean, da_cov_mean, da_cov_mean/base_cov_mean, base_cov_mean/da_cov_mean, ((da_cov_mean - base_cov_mean)/(da_cov_mean + base_cov_mean)),(da_cov_mean - base_cov_mean)/(base_cov_mean)):
        scores_cov_3D[i] = np.zeros(process_mask.shape)
        if not mean_cov.empty:
            scores_cov_3D[i][process_mask] = mean_cov.values
        i+=1
    
    img_baseline_cov = new_img_like(tgt_masker.mask_img, scores_cov_3D[0], copy_header=False)
    img_da_cov = new_img_like(tgt_masker.mask_img, scores_cov_3D[1], copy_header=False)
    img_da_div_base_cov = new_img_like(tgt_masker.mask_img, scores_cov_3D[2], copy_header=False)
    img_base_div_da_cov = new_img_like(tgt_masker.mask_img, scores_cov_3D[3], copy_header=False)
    img_da_m_base_div_da_p_base_cov = new_img_like(tgt_masker.mask_img, scores_cov_3D[4], copy_header=False)
    img_da_m_base_div_base_cov = new_img_like(tgt_masker.mask_img, scores_cov_3D[5], copy_header=False)

    # Save NIfTI images
    img_baseline_cov.to_filename(os.path.join(base_out_dir, f'map_cov_{NITER}iter_{radius}mm_BASELINE.nii.gz'))
    img_da_cov.to_filename(os.path.join(out_dir, f'map_cov_{NITER}iter_{radius}mm_{da_method_name}.nii.gz'))
    img_da_div_base_cov.to_filename(os.path.join(out_dir, f'map_cov_{NITER}iter_{radius}mm_{da_method_name}BASE.nii.gz'))
    img_base_div_da_cov.to_filename(os.path.join(out_dir, f'map_cov_{NITER}iter_{radius}mm_BASE{da_method_name}.nii.gz'))
    img_da_m_base_div_da_p_base_cov.to_filename(os.path.join(out_dir, f'map_cov_{NITER}iter_{radius}mm_{da_method_name}_minus_BASE_d_{da_method_name}_plus_BASE.nii.gz'))
    img_da_m_base_div_base_cov.to_filename(os.path.join(out_dir, f'map_cov_{NITER}iter_{radius}mm_{da_method_name}_minus_BASE_d_BASE.nii.gz'))


       # COVARIANCE STD PLOTTING (BASELINE vs DA)
    std_cov_3D = {}
    i=0

    for std_cov in (base_cov_std, da_cov_std, da_cov_std/base_cov_std, base_cov_std/da_cov_std, ((da_cov_std - base_cov_std)/(da_cov_std + base_cov_std)),((da_cov_std - base_cov_std)/base_cov_std)):
        std_cov_3D[i] = np.zeros(process_mask.shape)
        if not std_cov.empty:
            std_cov_3D[i][process_mask] = std_cov.values
        i+=1
    
    std_img_baseline_cov = new_img_like(tgt_masker.mask_img, std_cov_3D[0], copy_header=False)
    std_img_da_cov = new_img_like(tgt_masker.mask_img, std_cov_3D[1], copy_header=False)
    std_img_da_div_base_cov = new_img_like(tgt_masker.mask_img, std_cov_3D[2], copy_header=False)
    std_img_base_div_da_cov = new_img_like(tgt_masker.mask_img, std_cov_3D[3], copy_header=False)
    std_img_da_m_base_div_da_p_base_cov = new_img_like(tgt_masker.mask_img, std_cov_3D[4], copy_header=False)
    std_img_da_m_base_div_base_cov = new_img_like(tgt_masker.mask_img, std_cov_3D[5], copy_header=False)

    # Save NIfTI images
    std_img_baseline_cov.to_filename(os.path.join(base_out_dir, f'map_cov_std_{NITER}iter_{radius}mm_BASELINE.nii.gz'))
    std_img_da_cov.to_filename(os.path.join(out_dir, f'map_cov_std_{NITER}iter_{radius}mm_{da_method_name}.nii.gz'))
    std_img_da_div_base_cov.to_filename(os.path.join(out_dir, f'map_cov_std_{NITER}iter_{radius}mm_{da_method_name}BASE.nii.gz'))
    std_img_base_div_da_cov.to_filename(os.path.join(out_dir, f'map_cov_std_{NITER}iter_{radius}mm_BASE{da_method_name}.nii.gz'))
    std_img_da_m_base_div_da_p_base_cov.to_filename(os.path.join(out_dir, f'map_cov_std_{NITER}iter_{radius}mm_{da_method_name}_minus_BASE_d_{da_method_name}_plus_BASE.nii.gz'))
    std_img_da_m_base_div_base_cov.to_filename(os.path.join(out_dir, f'map_cov_std_{NITER}iter_{radius}mm_{da_method_name}_minus_BASE_d_BASE.nii.gz'))



    # lin_disc PLOTTING (BASELINE vs DA)
    scores_lin_disc_3D = {}
    i=0

    for mean_lin_disc in (base_lin_disc_mean, da_lin_disc_mean, da_lin_disc_mean/base_lin_disc_mean, base_lin_disc_mean/da_lin_disc_mean, ((da_lin_disc_mean - base_lin_disc_mean)/(da_lin_disc_mean + base_lin_disc_mean)),(da_lin_disc_mean - base_lin_disc_mean)/(base_lin_disc_mean)):
        scores_lin_disc_3D[i] = np.zeros(process_mask.shape)
        if not mean_lin_disc.empty:
            scores_lin_disc_3D[i][process_mask] = mean_lin_disc.values
        i+=1
    
    img_baseline_lin_disc = new_img_like(tgt_masker.mask_img, scores_lin_disc_3D[0], copy_header=False)
    img_da_lin_disc = new_img_like(tgt_masker.mask_img, scores_lin_disc_3D[1], copy_header=False)
    img_da_div_base_lin_disc = new_img_like(tgt_masker.mask_img, scores_lin_disc_3D[2], copy_header=False)
    img_base_div_da_lin_disc = new_img_like(tgt_masker.mask_img, scores_lin_disc_3D[3], copy_header=False)
    img_da_m_base_div_da_p_base_lin_disc = new_img_like(tgt_masker.mask_img, scores_lin_disc_3D[4], copy_header=False)
    img_da_m_base_div_base_lin_disc = new_img_like(tgt_masker.mask_img, scores_lin_disc_3D[5], copy_header=False)

    # Save NIfTI images
    img_baseline_lin_disc.to_filename(os.path.join(base_out_dir, f'map_lin_disc_{NITER}iter_{radius}mm_BASELINE.nii.gz'))
    img_da_lin_disc.to_filename(os.path.join(out_dir, f'map_lin_disc_{NITER}iter_{radius}mm_{da_method_name}.nii.gz'))
    img_da_div_base_lin_disc.to_filename(os.path.join(out_dir, f'map_lin_disc_{NITER}iter_{radius}mm_{da_method_name}BASE.nii.gz'))
    img_base_div_da_lin_disc.to_filename(os.path.join(out_dir, f'map_lin_disc_{NITER}iter_{radius}mm_BASE{da_method_name}.nii.gz'))
    img_da_m_base_div_da_p_base_lin_disc.to_filename(os.path.join(out_dir, f'map_lin_disc_{NITER}iter_{radius}mm_{da_method_name}_minus_BASE_d_{da_method_name}_plus_BASE.nii.gz'))
    img_da_m_base_div_base_lin_disc.to_filename(os.path.join(out_dir, f'map_lin_disc_{NITER}iter_{radius}mm_{da_method_name}_minus_BASE_d_BASE.nii.gz'))

   # lin_disc STD PLOTTING (BASELINE vs DA)
    std_lin_disc_3D = {}
    i=0

    for std_lin_disc in (base_lin_disc_std, da_lin_disc_std, da_lin_disc_std/base_lin_disc_std, base_lin_disc_std/da_lin_disc_std, ((da_lin_disc_std - base_lin_disc_std)/(da_lin_disc_std + base_lin_disc_std)),((da_lin_disc_std - base_lin_disc_std)/base_lin_disc_std)):
        std_lin_disc_3D[i] = np.zeros(process_mask.shape)
        if not std_lin_disc.empty:
            std_lin_disc_3D[i][process_mask] = std_lin_disc.values
        i+=1
    
    std_img_baseline_lin_disc = new_img_like(tgt_masker.mask_img, std_lin_disc_3D[0], copy_header=False)
    std_img_da_lin_disc = new_img_like(tgt_masker.mask_img, std_lin_disc_3D[1], copy_header=False)
    std_img_da_div_base_lin_disc = new_img_like(tgt_masker.mask_img, std_lin_disc_3D[2], copy_header=False)
    std_img_base_div_da_lin_disc = new_img_like(tgt_masker.mask_img, std_lin_disc_3D[3], copy_header=False)
    std_img_da_m_base_div_da_p_base_lin_disc = new_img_like(tgt_masker.mask_img, std_lin_disc_3D[4], copy_header=False)
    std_img_da_m_base_div_base_lin_disc = new_img_like(tgt_masker.mask_img, std_lin_disc_3D[5], copy_header=False)

    # Save NIfTI images
    std_img_baseline_lin_disc.to_filename(os.path.join(base_out_dir, f'map_lin_disc_std_{NITER}iter_{radius}mm_BASELINE.nii.gz'))
    std_img_da_lin_disc.to_filename(os.path.join(out_dir, f'map_lin_disc_std_{NITER}iter_{radius}mm_{da_method_name}.nii.gz'))
    std_img_da_div_base_lin_disc.to_filename(os.path.join(out_dir, f'map_lin_disc_std_{NITER}iter_{radius}mm_{da_method_name}BASE.nii.gz'))
    std_img_base_div_da_lin_disc.to_filename(os.path.join(out_dir, f'map_lin_disc_std_{NITER}iter_{radius}mm_BASE{da_method_name}.nii.gz'))
    std_img_da_m_base_div_da_p_base_lin_disc.to_filename(os.path.join(out_dir, f'map_lin_disc_std_{NITER}iter_{radius}mm_{da_method_name}_minus_BASE_d_{da_method_name}_plus_BASE.nii.gz'))
    std_img_da_m_base_div_base_lin_disc.to_filename(os.path.join(out_dir, f'map_lin_disc_std_{NITER}iter_{radius}mm_{da_method_name}_minus_BASE_d_BASE.nii.gz'))

    # normalized frechet distance (mean/std)
    # scores_nf_3D = {}
    # i = 0
    # for mean_nf in (
        # base_nf_mean,
        # da_nf_mean,
        # da_nf_mean / base_nf_mean,
        # base_nf_mean / da_nf_mean,
        # ((da_nf_mean - base_nf_mean) / (da_nf_mean + base_nf_mean)),
        # (da_nf_mean - base_nf_mean) / base_nf_mean,
    # ):
        # scores_nf_3D[i] = np.zeros(process_mask.shape)
        # if not mean_nf.empty:
            # scores_nf_3D[i][process_mask] = mean_nf.values
        # i += 1

    # img_baseline_nf = new_img_like(tgt_masker.mask_img, scores_nf_3D[0])
    # img_da_nf = new_img_like(tgt_masker.mask_img, scores_nf_3D[1])
    # img_da_div_base_nf = new_img_like(tgt_masker.mask_img, scores_nf_3D[2])
    # img_base_div_da_nf = new_img_like(tgt_masker.mask_img, scores_nf_3D[3])
    # img_da_m_base_div_da_p_base_nf = new_img_like(tgt_masker.mask_img, scores_nf_3D[4])
    # img_da_m_base_div_base_nf = new_img_like(tgt_masker.mask_img, scores_nf_3D[5])


    # std_nf_3D = {}
    # i = 0
    # for std_nf in (
        # base_nf_std,
        # da_nf_std,
        # da_nf_std / base_nf_std,
        # base_nf_std / da_nf_std,
        # ((da_nf_std - base_nf_std) / (da_nf_std + base_nf_std)),
        # (da_nf_std - base_nf_std) / base_nf_std,
    # ):
        # std_nf_3D[i] = np.zeros(process_mask.shape)
        # if not std_nf.empty:
            # std_nf_3D[i][process_mask] = std_nf.values
        # i += 1

    # std_img_baseline_nf = new_img_like(tgt_masker.mask_img, std_nf_3D[0])
    # std_img_da_nf = new_img_like(tgt_masker.mask_img, std_nf_3D[1])
    # std_img_da_div_base_nf = new_img_like(tgt_masker.mask_img, std_nf_3D[2])
    # std_img_base_div_da_nf = new_img_like(tgt_masker.mask_img, std_nf_3D[3])
    # std_img_da_m_base_div_da_p_base_nf = new_img_like(tgt_masker.mask_img, std_nf_3D[4])
    # std_img_da_m_base_div_base_nf = new_img_like(tgt_masker.mask_img, std_nf_3D[5])



print(f"--- Finished CROSS-CONTENT DOMAIN processing for Subject: {subject}, Method: {da_method_name} ---")
