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
Cross-domain content decoding with domain adaptation (I2P).

- Labels: object category (dog vs. scissors).
- Domains: imagery (training/source) and perception (test/target).
- Computes baseline (no DA) and DA searchlight maps, plus covariance-distance metrics.
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
from nilearn.decoding.searchlight import search_light, GroupIterator,_group_iter_search_light
from sklearn.exceptions import ConvergenceWarning
import time # <-- Add this line

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

# Import all DA methods and metrics
from adapt.parameter_based import RegularTransferLR, RegularTransferLC
from adapt.feature_based import PRED, FA, CORAL, SA, fMMD , DANN , TCA
from adapt.instance_based import LDM, KLIEP, KMM, ULSIF, RULSIF, NearestNeighborsWeighting, IWC, IWN, BalancedWeighting, TrAdaBoost


# --- REQUIRED METRIC IMPORTS ---
from adapt.metrics import (
    neg_j_score, 
    cov_distance,
    # normalized_frechet_distance,
    normalized_linear_discrepancy,
    # frechet_distance,
    domain_classifier
)
# -----------------------------

# NOTE: We keep the original import, but we OVERRIDE the function later.
from algorithms.searchlight_DA_save import get_sphere_data, searchlight_cv_DA, search_light
from dataManipulation.whole_brain import load_data

# --- DICTIONARY OF AVAILABLE DA METHODS ---
DA_METHODS_MAP = {
    "PRED": PRED, "FA": FA, "CORAL": CORAL, "SA": SA, "fMMD": fMMD, "DANN": DANN, "TCA": TCA,
    "LDM": LDM, "KLIEP": KLIEP, "KMM": KMM, "ULSIF": ULSIF, "RULSIF": RULSIF,
    "NearestNeighborsWeighting": NearestNeighborsWeighting, "IWC": IWC, "IWN": IWN,
    "BalancedWeighting": BalancedWeighting, "TrAdaBoost": TrAdaBoost,
    "RTLC": RegularTransferLC
}

METHODS_REQUIRING_YT = {
    "RTLC", "BalancedWeighting", "TrAdaBoost", "IWC", "IWN","PRED","BalanceWeighting", 
    "NearestNeighborsWeighting", "LDM", "FA" 
}

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Flatten
from tensorflow.keras.optimizers import Adam

# --- HELPER FUNCTIONS FOR DEEP DA MODELS ---
def get_encoder(input_shape):
    """Creates the shared feature extractor (Encoder)."""
    # Note: input_shape must be a tuple (v,) where v is the number of voxels in the sphere
    model = Sequential(name='Encoder')
    model.add(Input(shape=input_shape))
    model.add(Dense(64, activation='relu', name='enc_dense_1'))
    model.add(Dense(32, activation='relu', name='enc_latent')) # Latent space size 
    return model

def get_discriminator(input_shape, latent_dim=32):
    """Creates the Domain Discriminator (binary classification)."""
    model = Sequential(name='Discriminator')
    model.add(Input(shape=input_shape)) # Input is the latent space output by the encoder (32 dims)
    model.add(Dense(32, activation='relu', name='disc_dense_1'))
    #model.add(Dense(100, activation='sigmoid', name='disc_output')) # Binary output (Source vs. Target)
    model.add(Dense(1, activation='sigmoid', name='disc_output2')) # Binary output (Source vs. Target)
    return model

def get_task(input_shape, num_classes=2):
    """Creates the Task Classifier (Content Decoding)."""
    # Assuming binary classification (Face vs. Scene) so num_classes=2
    model = Sequential(name='Task_Classifier')
    model.add(Input(shape=input_shape)) # Input is the latent space output by the encoder (32 dims)
    model.add(Dense(num_classes, activation='softmax', name='task_output'))
    return model


# --- SCRIPT PARAMETERS AND SETUP ---
if len(sys.argv) != 4:
    print("Usage: python searchligh_other.py <subject_index> <n_iterations> <da_method_name>")
    sys.exit(1)

source_domain = 'imagery'
target_domain = 'perception'
radius = 12

#subjects = sorted([S.split('/')[-1] for S in glob.glob(os.path.join('../data','perception','*'))])
subjects = sorted(
    [S.split('/')[-1] for S in glob.glob(str(DATA_ROOT / source_domain / '*'))],
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
s = re.sub('[0-9]+_','',subject).capitalize()

print(f"--- Running searchlight for Subject: {subject}, Method: {da_method_name}, NITER: {NITER} ---")

# --- DYNAMIC OUTPUT DIRECTORIES ---
base_out_dir = RESULTS_ROOT / 'searchlight' / 'Cross_Domain_Content_Decoding' / 'I2P' / subject
base_fig_dir = FIGURES_ROOT / 'searchlight' / 'Cross_Domain_Content_Decoding' / 'I2P' / subject
out_dir = base_out_dir / da_method_name
savefig_dir = base_fig_dir / da_method_name

os.makedirs(out_dir, exist_ok=True)
os.makedirs(savefig_dir, exist_ok=True)

# Define file paths for results

pbase = os.path.join(base_out_dir, f'map_{NITER}iter_{radius}mm_BASELINE.csv')
#pbase = os.path.join(out_dir, f'map_{NITER}iter_{radius}mm_BASELINE.csv')
pnaive = os.path.join(base_out_dir, f'map_{NITER}iter_{radius}mm_NAIVE.csv') # Naive path restored
#pnaive = os.path.join(out_dir, f'map_{NITER}iter_{radius}mm_NAIVE.csv') # Naive path restored

pda = os.path.join(out_dir, f'map_{NITER}iter_{radius}mm_{da_method_name}.csv')

# ----------------------------------------------------------------------------------
# --- CUSTOM SEQUENTIAL SEARCHLIGHT FUNCTION (OVERRIDE) ---
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
    #n_jobs=1,
    verbose=0,
    i_iter=0 # <--- ADD THIS LINE
):
    """Compute a search_light.

    Parameters
    ----------
    X : array-like of shape at least 2D
        data to fit.

    y : array-like
        target variable to predict.

    estimator : estimator object implementing 'fit'
        object to use to fit the data

    A : scipy sparse matrix.
        adjacency matrix. Defines for each feature the neigbhoring features
        following a given structure of the data.

    groups : array-like, optional, (default None)
        group label for each sample for cross validation.

        .. note::
            This will have no effect for scikit learn < 0.18

    scoring : string or callable, optional
        The scoring strategy to use. See the scikit-learn documentation
        for possible values.
        If callable, it takes as arguments the fitted estimator, the
        test data (X_test) and the test target (y_test) if y is
        not None.

    cv : cross-validation generator, optional
        A cross-validation generator. If None, a 3-fold cross
        validation is used or 3-fold stratified cross-validation
        when y is supplied.
    %(n_jobs_all)s
    %(verbose0)s
    
    DA : ADAPT method, optional
        An ADAPT method. If None, no domain adaptation is performed
        and X_tgt, y_tgt are ignored.
        
    X_tgt : array-like of shape at least 2D, optional
        Instances from the target domain to perform domain adaptation on.
        Must be supplied if a DA method is supplied.
    
    y_tgt : array-like, optional
        Labels from the target domain to perform domain adaptation on.
        Must be supplied if a SUPERVISED DA method is supplied.

    Returns
    -------
    scores : array-like of shape (number of rows in A)
        search_light scores
    """
    #n_jobs = cpu_count()-2 if n_jobs==-1 else n_jobs
    # FIX: Get CPU count from environment or safely limit it
    # Use the logic you had for shallow methods, but apply it to ALL jobs now
    if DA is not None and DA.__name__ == "DANN":
        # DANN/Keras is often unstable in high parallelism. 
        # Set a low limit here, or let the cluster handle it via env variable.
        n_jobs_limit = 1
    else:
        n_jobs_limit = cpu_count() - 2 if n_jobs == -1 else n_jobs
        n_jobs_actual = max(1, min(n_jobs_limit, os.cpu_count() or 1))

    # Use n_jobs=1 if the limit is exceeded (safer for DANN on HPC)
    #n_jobs_actual = min(n_jobs_limit, os.cpu_count() or 1)
    #group_iter = GroupIterator(A.shape[0], n_jobs)
    group_iter = GroupIterator(A.shape[0], n_jobs_actual)

    with warnings.catch_warnings():  # might not converge
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
                i_iter # <--- FIX 2: PASS IT TO THE WORKER
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
    # --- FIX 2: ADD ITERATION ID ---
    i_iter=0
):
    """Perform grouped iterations of search_light."""
    par_scores = np.zeros(len(list_rows))
    t0 = time.time()

    sphere_results = []
    
    # --- ⚠️ SET MAX NUMBER OF SPHERES TO RUN ⚠️ ---
    #MAX_SPHERES_TO_RUN = 50
    MAX_SPHERES_TO_RUN = float('inf')

    for i, row in enumerate(list_rows):
        # --- This is the new, robust logic block ---
        
                # --- BREAK CONDITION ---
        if i >= MAX_SPHERES_TO_RUN:
            print(f"\nStopping: Processed requested {MAX_SPHERES_TO_RUN} spheres.")
            break
        # -----------------------
        metrics = {
            'neg_j_score': np.nan,
            'cov_distance': np.nan,
            'linear_discrepancy': np.nan,
            # 'normalized_frechet_distance': np.nan,
            # 'frechet_distance': np.nan,
            'domain_classifier': np.nan
        }
        learned_patterns = None
        da_attributes = {} 

        if DA is not None:
            # Check if the DA method is parameter-based (like RTLC)
            #is_param_based = 'parameter_based' in DA.__module__

            # if is_param_based:
            #     # --- Path for SUPERVISED, parameter-based methods (e.g., RTLC) ---
            #     # 1. Pre-fit the base estimator on the source data.
            #     source_estimator = estimator().fit(X[:, row], y)
            #     assert hasattr(source_estimator, 'coef_'), "Estimator needs 'coef_' for parameter-based DA"

            #     # 2. Initialize DA method with the PRE-FITTED estimator and fit on TARGET data.
            #     da_instance = DA(source_estimator)
            #     clf = da_instance.fit(Xt=X_tgt[:, row], yt=y_tgt)
            #     fitted_estimator = clf 
            
            if hasattr(DA, "__name__") and DA.__name__ == "DANN":
            # Determine input shape dynamically from the data
                n_features = X[:, row].shape[1]
                
                unique_labels = np.unique(y)
                n_classes = len(unique_labels)
                LATENT_DIM = 32
                BATCH_SIZE = 16
                # 1. Instantiate the Keras models
                feature_extractor = get_encoder(input_shape=(n_features,))
                task_estimator = get_task(input_shape=(32,), num_classes=len(np.unique(y)))
                domain_estimator = get_discriminator(input_shape=(32,))
                
                # 2. Initialize DANN with the Keras models
                da_instance = DANN(
                    encoder=feature_extractor,
                    task=task_estimator,
                    discriminator=domain_estimator,
                    # Pass a dummy estimator if the base estimator is required but ignored by DANN's wrapper
                    #estimator=estimator(),
                    Xt=X_tgt[:, row],
                    random_state=0,
                    verbose=0
                )
                # 3. Prepare One-Hot Labels (CRITICAL FIX FOR SHAPE)
        
                # Manually create a robust one-hot encoding matrix (n_samples x n_classes)
                # 3a. Get integer representation of labels (e.g., convert 'A', 'B' to 0, 1)
                label_binarizer = LabelBinarizer()
                y_indices = label_binarizer.fit_transform(y)
                
                # 3b. Create the n_samples x n_classes matrix (e.g., n_samples x 2)
                y_one_hot = np.zeros((y.size, n_classes))
                
                # Convert n_samples x 1 output of fit_transform to n_samples x n_classes indices
                # This handles the case where fit_transform for binary returns (N, 1)
                if n_classes == 2 and y_indices.ndim == 2:
                    y_indices = y_indices.flatten()
                
                # Populate the one-hot matrix
                y_one_hot[np.arange(y.size), y_indices] = 1

                # 4. Data Type and Shaping (Crucial for TensorFlow)
                X_float = X[:, row].astype(np.float32)
                X_tgt_float = X_tgt[:, row].astype(np.float32)
                y_float = y_one_hot.astype(np.float32)


                # 3. Fit DANN (Unsupervised/Semi-Supervised Deep Learning)
                # DANN generally only uses source labels (y) and both feature sets (X, Xt)
                clf = da_instance.fit(
                    X_float, y_float, 
                    Xt=X_tgt_float, 
                    batch_size=BATCH_SIZE,  
                    epochs=10, 
                    verbose=0
                )   
                
                y_pred_proba = clf.predict(X_test[:, row].astype(np.float32)) # Ensure test data is float32
                y_pred = np.argmax(y_pred_proba, axis=1)
                fitted_estimator = clf
                
                subject_id = subject
                da_name = DA.__name__
                estimators_dir = RESULTS_ROOT / 'searchlight' / 'Cross_Domain_Content_Decoding' / 'I2P' / subject_id / da_name / 'searchlight_estimators'
                os.makedirs(estimators_dir, exist_ok=True)

                task_filename = str(estimators_dir / f'task_iter{i_iter}_sphere{i}.h5')
                encoder_filename = str(estimators_dir / f'encoder_iter{i_iter}_sphere{i}.h5')
                discriminator_filename = str(estimators_dir / f'discriminator_iter{i_iter}_sphere{i}.h5')
                fitted_estimator.task_.save(task_filename)
                fitted_estimator.encoder_.save(encoder_filename)
                fitted_estimator.discriminator_.save(discriminator_filename)
                da_attributes['task_path'] = task_filename
                # Optional: Store Encoder path as well
                da_attributes['estimator_path'] = encoder_filename
                da_attributes['discriminator_path'] = discriminator_filename


            else:
                # --- Path for UNSUPERVISED, feature/instance-based methods (e.g., CORAL, SA, fMMD) ---
                # 1. Initialize DA method with an UN-FITTED estimator.
                da_instance = DA(estimator=estimator())
                
                # 2. Fit DA method on BOTH source and target data.
                if y_tgt is None: # For unsupervised methods
                    clf = da_instance.fit(X[:, row], y, Xt=X_tgt[:, row])
                else: # For supervised instance-based methods (e.g., TrAdaBoost)
                    clf = da_instance.fit(X[:, row], y, Xt=X_tgt[:, row], yt=y_tgt)
                fitted_estimator = clf 
                y_pred = clf.predict(X_test[:, row])
            # Prediction step is the same for all adapted classifiers
            #y_pred = clf.predict(X_test[:, row])


            Xs_trans = fitted_estimator.transform(X[:, row]) if hasattr(fitted_estimator, 'transform') else X[:, row]
            Xt_trans = fitted_estimator.transform(X_tgt[:, row]) if hasattr(fitted_estimator, 'transform') else X_tgt[:, row]
                
            #metrics['neg_j_score'] = neg_j_score(Xs_trans, Xt_trans)
            metrics['cov_distance'] = cov_distance(Xs_trans, Xt_trans)
            metrics['linear_discrepancy'] = normalized_linear_discrepancy(Xs_trans, Xt_trans)
            # metrics['normalized_frechet_distance'] = np.nan  # normalized_frechet_distance disabled

            if isinstance(fitted_estimator, (FA, SA, CORAL, PRED, TCA)): # Check for FA method
                subject_id = subject
                da_name = DA.__name__

                # NOTE: estimator .pkl saving disabled for faster processing.
                # Uncomment to re-enable saving searchlight estimators.
                # estimators_dir = RESULTS_ROOT / 'searchlight' / 'Cross_Domain_Content_Decoding' / 'I2P' / subject_id / da_name / 'searchlight_estimators'
                # os.makedirs(estimators_dir, exist_ok=True)
                #
                # --- FIX 1: INCLUDE ITERATION NUMBER IN FILENAME ---
                # i is the sphere index (local index in the chunk), i_iter is the cross-validation iteration index
                # estimator_filename = str(estimators_dir /
                #                                 f'{da_name.lower()}_iter{i_iter}_sphere{i}.pkl') 
                #
                # joblib.dump(fitted_estimator, estimator_filename)
                # da_attributes['estimator_path'] = estimator_filename       
        else:
            # Standard non-DA searchlight
            fitted_estimator = estimator().fit(X[:, row], y)
            y_pred = fitted_estimator.predict(X_test[:, row])
            X_ref = X_tgt if X_tgt is not None else X_test   # fallback for old calls (e.g., naive)
            metrics['cov_distance'] = cov_distance(X[:, row], X_ref[:, row])
            metrics['linear_discrepancy'] = normalized_linear_discrepancy(X[:, row], X_ref[:, row])
            # metrics['normalized_frechet_distance'] = np.nan  # normalized_frechet_distance disabled
            # metrics['cov_distance'] = cov_distance(X[:, row], X_tgt[:, row])
            # metrics['linear_discrepancy'] = normalized_linear_discrepancy(X[:, row], X_tgt[:, row])
            # metrics['cov_distance'] = 0
            # metrics['linear_discrepancy'] = 0      
        # Scoring step is the same for all paths
        par_scores[i] = scoring(y_test, y_pred)
            # --- Capture and store results ---
        sphere_data = {
            'score': par_scores[i],
                #'neg_j_score': metrics['neg_j_score'],      
            'cov_distance': metrics['cov_distance'],    
            'linear_discrepancy': metrics['linear_discrepancy'],
            # 'normalized_frechet_distance': metrics['normalized_frechet_distance'],
                #'frechet_distance': metrics['frechet_distance'],      
                #'domain_classifier': metrics['domain_classifier'],    
                #'learned_patterns': learned_patterns,
            'da_attributes': da_attributes,
            'voxel_indices': row
        }
        sphere_results.append(sphere_data)

    #     # Verbose printing logic (unchanged)
    #     if verbose > 0:
    #         step = 11 - min(verbose, 10)
    #         if i % step == 0:
    #             crlf = "\r" if total == len(list_rows) else "\n"
    #             percent = float(i) / len(list_rows)
    #             percent = round(percent * 100, 2)
    #             dt = time.time() - t0
    #             remaining = (100.0 - percent) / max(0.01, percent) * dt
    #             sys.stderr.write(
    #                 f"Job #{thread_id}, processed {i}/{len(list_rows)} voxels "
    #                 f"({percent:0.2f}%, {remaining} seconds remaining){crlf}"
    #             )
    # return par_scores

    return pd.Series([str(r) for r in sphere_results])


# --- Helper functions for ROBUST PARSING ---
def parse_and_extract_metric(data_string, metric_keys):
    """
    Parses complex string using regex to extract a numerical metric 
    by checking a list of potential keys.
    """
    if isinstance(data_string, str):
        for key in metric_keys:
            try:
                # Pattern searches for the metric name followed by a number (handles negative signs)
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

# def parse_and_extract_frechet(data_string):
    # return parse_and_extract_metric(data_string, ['frechet_distance', 'frechet'])
# 
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
def parse_and_extract_domain_clf(data_string):
    return parse_and_extract_metric(data_string, ['domain_classifier', 'domain_clf', 'domain_acc'])

def extract_sphere_ids(path_string):
    """
    Extracts two numbers (group_id and index_id) from the PKL path.
    e.g., '..._1_0.pkl' -> (1, 0)
    """
    if isinstance(path_string, str):
        match = re.search(r'_(\d+)_(\d+)\.pkl$', path_string)
        if match:
            return int(match.group(1)), int(match.group(2))
    return np.nan, np.nan 

def extract_metadata(data_string, key):
    """Parses the complex object string to safely extract metadata."""
    if isinstance(data_string, str) and data_string.startswith('{'):
        processed_string = data_string.replace('\n', ' ').replace('\'', '"') 

        if key == 'estimator_path':
            match = re.search(r'"estimator_path":\s*"(.*?)"', processed_string)
            return match.group(1) if match else "N/A"
        
        elif key == 'voxel_indices':
            match = re.search(r'"voxel_indices":\s*(\[.*?\])', processed_string)
            return match.group(1) if match else "N/A"
    return "N/A_Error"

def load_and_parse_searchlight(pfile, NITER, metric_parser):
    """Loads and parses the CSV file into a numerical score DataFrame for a single metric."""
    df = pd.DataFrame()
    if Path(pfile).is_file():
        try:
            df = pd.read_csv(pfile, dtype={col: str for col in range(NITER)})
            for col in df.columns:
                df[col] = df[col].apply(metric_parser)
            return df
        except Exception as e:
            # print(f"Could not load or parse metric from {pfile}. Error: {e}")
            return pd.DataFrame()
    return pd.DataFrame()

def load_raw_data_for_metadata(pfile):
    """Loads the first column (iteration 0) of the raw CSV file for metadata extraction."""
    if Path(pfile).is_file():
        return pd.read_csv(pfile, usecols=[0], dtype={0: str})
    return pd.DataFrame()
# --- END OF HELPER FUNCTIONS ---

# --- MAIN LOGIC ---
searchlight, searchlight_naive, searchlight_DA = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

is_file = {'baseline': Path(pbase).is_file(), 'naive': Path(pnaive).is_file(), 'da': Path(pda).is_file()}
compute_baseline_models = not Path(pbase).is_file() or not Path(pnaive).is_file()

if all(is_file.values()):
    print(f'All results files found for subject {subject} and method {da_method_name}. Nothing to do.')
else:
    # --- Data Loading (Only run if computation is needed) ---
    bold_data, events, masker = load_data(source_domain, subject)
    tgt_data, tgt_events, tgt_masker = load_data(target_domain, subject)
    
    g = events.run+'trial'+events.trial_idx.astype(str)
    g_tgt = tgt_events.run+'trial'+tgt_events.trial_idx.astype(str)
    
    X, A, _, _ = get_sphere_data(masker, bold_data, radius)
    X_tgt, A_tgt, _, _ = get_sphere_data(tgt_masker, tgt_data, radius)
    
    Source_train_is, Target_train_is = searchlight_cv_DA(events.target_category.values, tgt_events.target_category.values, g, g_tgt, NITER)
    
    try:
        process_mask, process_mask_affine = masking.load_mask_img(tgt_masker.mask_img)
    except: 
        process_mask, process_mask_affine = masking._load_mask_img(tgt_masker.mask_img)

    for i in tqdm(range(NITER), desc=f"Processing {da_method_name}"):
        Target_test_is = np.ones(tgt_events.target_category.values.size, dtype=bool)
        Target_test_is[Target_train_is[i]] = False
        
        xtrain = X[Source_train_is[i]]
        ytrain = events.target_category.values[Source_train_is[i]]
        xtest = X_tgt[Target_test_is]
        ytest = tgt_events.target_category.values[Target_test_is]
        xtrain_tgt = X_tgt[Target_train_is[i]]
        ytrain_tgt = tgt_events.target_category.values[Target_train_is[i]]
        
        # --- Run Baseline and Naive (Using the NEW sequential function) ---
        if compute_baseline_models:
            xtrain_naive = pd.concat([pd.DataFrame(X[Source_train_is[i]]), pd.DataFrame(X_tgt[Target_train_is[i]])]).to_numpy()
            ytrain_naive = pd.concat([pd.Series(ytrain), pd.Series(ytrain_tgt)]).to_numpy()
            if not is_file['baseline']:
            # ADD i_iter=i
                searchlight[i] = search_light_ALI(xtrain, ytrain, LogisticRegression, A, xtest, ytest, scoring=balanced_accuracy_score, X_tgt=xtrain_tgt, y_tgt=None, verbose=0, **{'i_iter': i}) 
            if not is_file['naive']: 
                # ADD i_iter=i
                searchlight_naive[i] = search_light_ALI(xtrain_naive, ytrain_naive, LogisticRegression, A, xtest, ytest, scoring=balanced_accuracy_score, verbose=0, **{'i_iter': i})            #if not is_file['baseline']: searchlight[i] = search_light_sequential(xtrain, ytrain, LogisticRegression, A, xtest, ytest, scoring=balanced_accuracy_score, X_tgt=xtrain_tgt, y_tgt=ytrain_tgt, verbose=0) 
            #if not is_file['naive']: searchlight_naive[i] = search_light_sequential(xtrain_naive, ytrain_naive, LogisticRegression, A, xtest, ytest, scoring=balanced_accuracy_score, X_tgt=xtrain_tgt, y_tgt=ytrain_tgt, verbose=0) 

        # --- Run the selected DA Method (Using the NEW sequential function) ---
        if not is_file['da']:
            da_kwargs = {'DA': DA_class, 'X_tgt': xtrain_tgt, 'i_iter': i} # ADD i_iter: i
            if da_method_name in METHODS_REQUIRING_YT:
                da_kwargs['y_tgt'] = ytrain_tgt
            else:
                da_kwargs['y_tgt'] = None

            searchlight_DA[i] = search_light_ALI(xtrain, ytrain, LogisticRegression, A, xtest, ytest,
                                             scoring=balanced_accuracy_score, verbose=0, **da_kwargs)
    
    # --- SAVE RAW DATA ---
    if not searchlight.empty: 
        searchlight.to_csv(pbase, index=False)
    if not searchlight_DA.empty: 
        searchlight_DA.to_csv(pda, index=False)
    if not searchlight_naive.empty: 
        searchlight_naive.to_csv(pnaive, index=False)


# --- ROBUST LOADING AND PARSING FOR ANALYSIS ---

# Load Score DataFrames
searchlight = load_and_parse_searchlight(pbase, NITER, parse_and_extract_score)
searchlight_DA = load_and_parse_searchlight(pda, NITER, parse_and_extract_score)
searchlight_naive = load_and_parse_searchlight(pnaive, NITER, parse_and_extract_score)

# Load Metric DataFrames
#searchlight_jscore = load_and_parse_searchlight(pbase, NITER, parse_and_extract_j_score)
#searchlight_DA_jscore = load_and_parse_searchlight(pda, NITER, parse_and_extract_j_score)
#searchlight_naive_jscore = load_and_parse_searchlight(pnaive, NITER, parse_and_extract_j_score)

searchlight_cov = load_and_parse_searchlight(pbase, NITER, parse_and_extract_cov_distance)
searchlight_DA_cov = load_and_parse_searchlight(pda, NITER, parse_and_extract_cov_distance)
searchlight_naive_cov = load_and_parse_searchlight(pnaive, NITER, parse_and_extract_cov_distance)

searchlight_lin_disc = load_and_parse_searchlight(pbase, NITER, parse_and_extract_lin_disc)
searchlight_DA_lin_disc = load_and_parse_searchlight(pda, NITER, parse_and_extract_lin_disc)
searchlight_naive_lin_disc = load_and_parse_searchlight(pnaive, NITER, parse_and_extract_lin_disc)

# searchlight_norm_frechet = load_and_parse_searchlight(pbase, NITER, parse_and_extract_norm_frechet)
# searchlight_DA_norm_frechet = load_and_parse_searchlight(pda, NITER, parse_and_extract_norm_frechet)
# searchlight_naive_norm_frechet = load_and_parse_searchlight(pnaive, NITER, parse_and_extract_norm_frechet)

#searchlight_domain_clf = load_and_parse_searchlight(pbase, NITER, parse_and_extract_domain_clf)
#searchlight_DA_domain_clf = load_and_parse_searchlight(pda, NITER, parse_and_extract_domain_clf)
#searchlight_naive_domain_clf = load_and_parse_searchlight(pnaive, NITER, parse_and_extract_domain_clf)


def is_numeric_df(df):
    return not df.empty and df.iloc[:,0].notna().any()

# --- ANALYSIS AND PLOTTING ---

# 1. DA vs Baseline Plotting/Saving
if is_numeric_df(searchlight) and is_numeric_df(searchlight_DA):
    print("--- Generating plots: DA vs Baseline ---")
    
    searchlight_mean = searchlight.mean(axis=1) 
    searchlight_DA_mean = searchlight_DA.mean(axis=1)
    searchlight_naive_mean = searchlight_naive.mean(axis=1)

# --- NEW: CALCULATE std ---
    searchlight_std = searchlight.std(axis=1) 
    searchlight_DA_std = searchlight_DA.std(axis=1)
    searchlight_naive_std = searchlight_naive.std(axis=1)

    # Calculate means for new metrics
    #base_jscore_mean = searchlight_jscore.mean(axis=1)
    #da_jscore_mean = searchlight_DA_jscore.mean(axis=1)
    #naive_jscore_mean = searchlight_naive_jscore.mean(axis=1)

    base_cov_mean = searchlight_cov.mean(axis=1)
    da_cov_mean = searchlight_DA_cov.mean(axis=1)
    naive_cov_mean = searchlight_naive_cov.mean(axis=1)

# --- NEW: CALCULATE COV std ---
    base_cov_std = searchlight_cov.std(axis=1)
    da_cov_std = searchlight_DA_cov.std(axis=1)
    naive_cov_std = searchlight_naive_cov.std(axis=1)

    base_lin_disc_mean = searchlight_lin_disc.mean(axis=1)
    da_lin_disc_mean = searchlight_DA_lin_disc.mean(axis=1)
    naive_lin_disc_mean = searchlight_naive_lin_disc.mean(axis=1)

    base_lin_disc_std = searchlight_lin_disc.std(axis=1)
    da_lin_disc_std = searchlight_DA_lin_disc.std(axis=1)
    naive_lin_disc_std = searchlight_naive_lin_disc.std(axis=1)
    # normalized frechet distance disabled; keep NaN columns
    nf_nan = pd.Series(np.nan, index=searchlight_mean.index)
    # base_nf_mean = nf_nan
    # da_nf_mean = nf_nan
    # naive_nf_mean = nf_nan

    # base_nf_std = nf_nan
    # da_nf_std = nf_nan
    # naive_nf_std = nf_nan

    #base_domain_clf_mean = searchlight_domain_clf.mean(axis=1)
    #da_domain_clf_mean = searchlight_DA_domain_clf.mean(axis=1)
    #naive_domain_clf_mean = searchlight_naive_domain_clf.mean(axis=1)

    # --- METADATA EXTRACTION AND SUMMARY CSV SAVING (DA vs Baseline) ---
    raw_base_df = load_raw_data_for_metadata(pbase)
    raw_da_df = load_raw_data_for_metadata(pda)
    raw_naive_df = load_raw_data_for_metadata(pnaive)

    # 1. Save DA Summary CSV 
    if not raw_da_df.empty:
        path_series_da = raw_da_df.iloc[:, 0].apply(lambda x: extract_metadata(x, 'estimator_path'))
        ids_df = path_series_da.apply(lambda x: pd.Series(extract_sphere_ids(x), index=['sphere_group_id', 'sphere_index_id']))
        
        summary_df_da = pd.DataFrame({
            'score_mean': searchlight_DA_mean,
            'score_std': searchlight_DA_std, # <-- ADDED
            #'jscore_mean': da_jscore_mean,
            'cov_distance_mean': da_cov_mean,
            'cov_distance_std': da_cov_std, # <-- ADDED
            'linear_discrepancy_mean': da_lin_disc_mean,
            'linear_discrepancy_std': da_lin_disc_std,  
            #'domain_classifier_mean': da_domain_clf_mean,
            #'sphere_group_id': ids_df['sphere_group_id'],  
            #'sphere_index_id': ids_df['sphere_index_id'], 
            'method': da_method_name,
            'estimator_path': path_series_da,
            'voxel_indices': raw_da_df.iloc[:, 0].apply(lambda x: extract_metadata(x, 'voxel_indices'))
        })
        pda_summary = os.path.join(out_dir, f'map_summary_{NITER}iter_{radius}mm_{da_method_name}.csv')
        summary_df_da.to_csv(pda_summary, index=True, index_label='row') 
        print(f"✅ DA Summary CSV saved to: {pda_summary}")
        
    # 2. Save Baseline Summary CSV
    if not raw_base_df.empty:
        path_series_base = raw_base_df.iloc[:, 0].apply(lambda x: extract_metadata(x, 'estimator_path'))
        ids_df_base = path_series_base.apply(lambda x: pd.Series(extract_sphere_ids(x), index=['sphere_group_id', 'sphere_index_id']))
        
        summary_df_base = pd.DataFrame({
            'score_mean': searchlight_mean,
            'score_std': searchlight_std, # <-- ADDED
            #'jscore_mean': base_jscore_mean,
            'cov_distance_mean': base_cov_mean,
            'cov_distance_std': base_cov_std, # <-- ADDED
            'linear_discrepancy_mean': base_lin_disc_mean,
            'linear_discrepancy_std': base_lin_disc_std,  
            #'domain_classifier_mean': base_domain_clf_mean,
            #'sphere_group_id': ids_df_base['sphere_group_id'],  
            #'sphere_index_id': ids_df_base['sphere_index_id'],  
            'method': 'baseline',
            'estimator_path': path_series_base,
            'voxel_indices': raw_base_df.iloc[:, 0].apply(lambda x: extract_metadata(x, 'voxel_indices'))
        })
        pbase_summary = os.path.join(base_out_dir, f'map_summary_{NITER}iter_{radius}mm_BASELINE.csv')
        #pbase_summary = os.path.join(out_dir, f'map_summary_{NITER}iter_{radius}mm_BASELINE.csv')
        summary_df_base.to_csv(pbase_summary, index=True, index_label='row')
        print(f"✅ Baseline Summary CSV saved to: {pbase_summary}")

    # 3. Save Naive Summary CSV
    if not raw_naive_df.empty:
        path_series_naive = raw_naive_df.iloc[:, 0].apply(lambda x: extract_metadata(x, 'estimator_path'))
        ids_df_naive = path_series_naive.apply(lambda x: pd.Series(extract_sphere_ids(x), index=['sphere_group_id', 'sphere_index_id']))
        
        summary_df_base = pd.DataFrame({
            'score_mean': searchlight_naive_mean,
            'score_std': searchlight_naive_std, # <-- ADDED
            #'jscore_mean': naive_jscore_mean,
            'cov_distance_mean': naive_cov_mean,
            'cov_distance_std': naive_cov_std, # <-- ADDED
            'linear_discrepancy_mean': naive_lin_disc_mean,
            'linear_discrepancy_std': naive_lin_disc_std,  
            #'domain_classifier_mean': naive_domain_clf_mean,
            #'sphere_group_id': ids_df_naive['sphere_group_id'],  
            #'sphere_index_id': ids_df_naive['sphere_index_id'],  
            'method': 'naive',
            'estimator_path': path_series_naive,
            'voxel_indices': raw_naive_df.iloc[:, 0].apply(lambda x: extract_metadata(x, 'voxel_indices'))
        })
        pbase_summary = os.path.join(base_out_dir, f'map_summary_{NITER}iter_{radius}mm_NAIVE.csv')
        #pbase_summary = os.path.join(out_dir, f'map_summary_{NITER}iter_{radius}mm_NAIVE.csv')
        summary_df_base.to_csv(pbase_summary, index=True, index_label='row')
        print(f"✅ Naive Summary CSV saved to: {pbase_summary}")
    # --- END OF SUMMARY CSV SAVING ---
    
    scores_3D = {}
    i=0
    
    # --- Setup Masker/Affine ---
    if 'process_mask' not in locals():
        _, tgt_events, tgt_masker = load_data(target_domain, subject)
        try:
            process_mask, process_mask_affine = masking.load_mask_img(tgt_masker.mask_img)
        except: 
            process_mask, process_mask_affine = masking._load_mask_img(tgt_masker.mask_img)
    if 'masker' not in locals():
         bold_data, events, masker = load_data(source_domain, subject)
    # ---------------------------
   
    # ACCURACY PLOTTING (BASELINE vs DA)

    for mean in (searchlight_mean, searchlight_DA_mean, searchlight_DA_mean/searchlight_mean, searchlight_mean/searchlight_DA_mean, ((searchlight_DA_mean - searchlight_mean)/(searchlight_DA_mean + searchlight_mean)),(searchlight_DA_mean - searchlight_mean)/searchlight_DA_mean):
        scores_3D[i] = np.zeros(process_mask.shape)
        if not mean.empty:
            scores_3D[i][process_mask] = mean.values
        i+=1
    

    img_baseline = new_img_like(tgt_masker.mask_img, scores_3D[0], copy_header=False)
    img_da = new_img_like(tgt_masker.mask_img, scores_3D[1], copy_header=False)
    img_da_div_base = new_img_like(tgt_masker.mask_img, scores_3D[2], copy_header=False)
    img_base_div_da = new_img_like(tgt_masker.mask_img, scores_3D[3], copy_header=False)
    img_da_m_base_div_da_p_base = new_img_like(tgt_masker.mask_img, scores_3D[4], copy_header=False)
    img_da_m_base_div_base = new_img_like(tgt_masker.mask_img, scores_3D[5], copy_header=False)

    # Save NIfTI images
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



    # NIFTI AND PLOTTING - ACCURACY
    scores_3D_naive = {}
    i=0 

    for mean in (searchlight_naive_mean, searchlight_DA_mean, searchlight_DA_mean/searchlight_naive_mean, searchlight_naive_mean/searchlight_DA_mean, ((searchlight_DA_mean - searchlight_naive_mean)/(searchlight_DA_mean + searchlight_naive_mean))):
        scores_3D_naive[i] = np.zeros(process_mask.shape)
        if not mean.empty:
            scores_3D_naive[i][process_mask] = mean.values
        i+=1

    img_naive = new_img_like(tgt_masker.mask_img, scores_3D_naive[0], copy_header=False)
    img_da = new_img_like(tgt_masker.mask_img, scores_3D_naive[1], copy_header=False)
    img_da_div_naive = new_img_like(tgt_masker.mask_img, scores_3D_naive[2], copy_header=False)
    img_naive_div_da = new_img_like(tgt_masker.mask_img, scores_3D_naive[3], copy_header=False)
    img_da_m_naive_div_da_p_naive = new_img_like(tgt_masker.mask_img, scores_3D_naive[4], copy_header=False)

    # Save NIfTI images for naive comparison
    img_naive.to_filename(os.path.join(base_out_dir, f'map_{NITER}iter_{radius}mm_NAIVE.nii.gz'))
    
    img_da_div_naive.to_filename(os.path.join(out_dir, f'map_{NITER}iter_{radius}mm_{da_method_name}NAIVE.nii.gz'))
    img_naive_div_da.to_filename(os.path.join(out_dir, f'map_{NITER}iter_{radius}mm_NAIVE{da_method_name}.nii.gz'))
    img_da_m_naive_div_da_p_naive.to_filename(os.path.join(out_dir, f'map_{NITER}iter_{radius}mm_{da_method_name}_minus_NAIVE_d_{da_method_name}_plus_NAIVE.nii.gz'))

    # Plotting (omitted for brevity, but retained in full script)
    fig, ax = plt.subplots(4,1,figsize=(12,16))
    plotting.plot_stat_map(img_naive, bg_img=masker.mask_img, cut_coords=(0,-5,-4), threshold=0.5, title=f'Naive > 0.5 - {s}', axes=ax[0])
    plotting.plot_stat_map(img_da, bg_img=masker.mask_img, cut_coords=(0,-5,-4), threshold=0.5, title=f'{da_method_name} > 0.5 - {s}', axes=ax[1])
    plotting.plot_stat_map(img_da_div_naive, bg_img=masker.mask_img, cut_coords=(0,-5,-4), threshold=1, title=f'{da_method_name}/Naive > 1 - {s}', axes=ax[2])
    plotting.plot_stat_map(img_naive_div_da, bg_img=masker.mask_img, cut_coords=(0,-5,-4), threshold=1, title=f'Naive/{da_method_name} > 1 - {s}', axes=ax[3])
    fig.savefig(os.path.join(savefig_dir, f'fig_{da_method_name}_vs_naive_{NITER}iter_{radius}mm.png'))
    plt.close(fig)

    # NIFTI AND PLOTTING - COVARIANCE
    scores_cov_3D_naive = {}
    i=0 # <--- INDEX RESET HERE
    
    for mean_cov in (naive_cov_mean, da_cov_mean, da_cov_mean/naive_cov_mean, naive_cov_mean/da_cov_mean , ((da_cov_mean - naive_cov_mean)/(da_cov_mean + naive_cov_mean))):
        scores_cov_3D_naive[i] = np.zeros(process_mask.shape)
        if not mean_cov.empty:
            scores_cov_3D_naive[i][process_mask] = mean_cov.values
        i+=1
    
    img_naive_cov = new_img_like(tgt_masker.mask_img, scores_cov_3D_naive[0], copy_header=False)
    img_da_div_naive_cov = new_img_like(tgt_masker.mask_img, scores_cov_3D_naive[2], copy_header=False)
    img_naive_div_da_cov = new_img_like(tgt_masker.mask_img, scores_cov_3D_naive[3], copy_header=False)
    img_da_m_naive_div_da_p_naive_cov = new_img_like(tgt_masker.mask_img, scores_cov_3D_naive[4], copy_header=False)

    # Save NIfTI images (Covariance)
    img_naive_cov.to_filename(os.path.join(base_out_dir, f'map_cov_{NITER}iter_{radius}mm_NAIVE.nii.gz'))
    img_da_div_naive_cov.to_filename(os.path.join(out_dir, f'map_cov_{NITER}iter_{radius}mm_{da_method_name}NAIVE.nii.gz'))
    img_naive_div_da_cov.to_filename(os.path.join(out_dir, f'map_cov_{NITER}iter_{radius}mm_NAIVE{da_method_name}.nii.gz'))
    img_da_m_naive_div_da_p_naive_cov.to_filename(os.path.join(out_dir, f'map_cov_{NITER}iter_{radius}mm_{da_method_name}_minus_NAIVE_d_{da_method_name}_plus_NAIVE.nii.gz'))



# Note: plotting of the naive model was removed; only baseline and DA results are plotted.

print(f"--- Finished processing for Subject: {subject}, Method: {da_method_name} ---")
