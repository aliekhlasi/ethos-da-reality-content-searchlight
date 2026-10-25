#!/bin/bash
# ------------------------------------------------------------------------------
# Part of: "Is This the Real Life? Is This Just Fantasy? A Domain Adaptation
#          Framework for Identifying Higher-Order Reality Representations"
# Edited by: Ali Ekhlasi, Postdoctoral Researcher,
#            Department of Cognitive Sciences, University of California, Irvine
#            https://github.com/aliekhlasi
# License: GNU General Public License v3.0 (see LICENSE)
# ------------------------------------------------------------------------------
# Run the full per-subject searchlight pipeline:
#   1. baseline searchlights (within-domain and within-content decoding)
#   2. cross-domain / cross-content searchlights, baseline and Subspace Alignment (SA)
#   3. Cohen's d maps (SA vs baseline)
#   4. warp every map to MNI space (FSL applywarp)
#
# Run it from the repository root, either directly:
#     bash run_searchlight_pipeline.sh
# or as a Slurm job (add your own account, partition and e-mail lines below):
#     sbatch run_searchlight_pipeline.sh
#
# Settings can be overridden with environment variables, e.g.
#     SUBJECTS="0 1 2" NITER=10 bash run_searchlight_pipeline.sh
#
# Each subject takes many CPU hours, so on a cluster submit one job per subject
# (or a few subjects per job) with the SUBJECTS variable.
#
# --- Slurm settings (edit for your cluster) ---
#SBATCH --job-name=ccn2026_searchlight
#SBATCH --output=searchlight_%j.out
#SBATCH --error=searchlight_%j.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --time=40:00:00
#SBATCH --mem=0
##SBATCH --account=YOUR_ACCOUNT
##SBATCH --partition=YOUR_PARTITION

set -euo pipefail

echo "========================================================"
echo "Job started: $(date)   Node: $(hostname)"
echo "========================================================"

# ----------------------------
# Environment
# ----------------------------
# On clusters with environment modules, load FSL (needed for applywarp).
# Activate your Python environment (see README.md) before running this script.
if command -v module >/dev/null 2>&1; then
  module load fsl 2>/dev/null || true
fi

echo "Python: $(command -v python)"
python - <<'PY'
import importlib.util, sys
needed = ("numpy", "scipy", "pandas", "sklearn", "nilearn", "nibabel", "joblib", "tqdm", "adapt")
missing = [p for p in needed if importlib.util.find_spec(p) is None]
if missing:
    sys.exit("ERROR: missing Python packages: " + ", ".join(missing) + "\nInstall with: pip install -r requirements.txt")
PY

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$PROJECT_DIR/DA_searchlight"
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"

# ----------------------------
# Run configuration
# ----------------------------
NITER="${NITER:-10}"                    # resampling iterations (paper: 10)
RADIUS=12                               # searchlight radius in mm (fixed in the scripts)
read -r -a SUBJECTS <<< "${SUBJECTS:-$(seq -s ' ' 0 17)}"   # positions 0-17 = subjects 1-18
DA_METHODS=(SA)                         # Subspace Alignment, as in the paper

# MNI reference for the warp. Default: FSL's 2 mm brain mask.
FSL_REF="${FSL_REF:-${FSLDIR:-}/data/standard/MNI152_T1_2mm_brain_mask.nii.gz}"

RESULTS_ROOT=$(python -c "from da_paths import RESULTS_ROOT; print(RESULTS_ROOT)")
RESULTS_ROOT_CROSS_DOMAIN="$RESULTS_ROOT/searchlight/Cross_Domain_Content_Decoding"
RESULTS_ROOT_CROSS_CONTENT="$RESULTS_ROOT/searchlight/Cross_Content_Domain_Decoding"

WARP_SCRIPT="$PROJECT_DIR/DA_searchlight/searchlight_normalisation/scripts/Warping_maps_to_mni_DA.py"
COHEN_SCRIPT="Compute_Cohen.py"

CROSS_DOMAIN_ALL_TASKS=(P2P I2I P2I I2P)        # baseline maps exist for all of these
CROSS_CONTENT_ALL_TASKS=(Ci2Ci Ci2Cj Cj2Ci Cj2Cj)
CROSS_DOMAIN_DA_TASKS=(P2I I2P)                 # DA maps exist only for cross tasks
CROSS_CONTENT_DA_TASKS=(Ci2Cj Cj2Ci)
CROSS_CONTENT_PER_CONTENT_TASKS=(Ci2Ci Cj2Cj)   # stored under content_*/ subfolders

echo "SEARCHLIGHT: NITER=${NITER}, SUBJECTS=${SUBJECTS[*]}, METHODS=${DA_METHODS[*]}"
echo "RESULTS_ROOT=${RESULTS_ROOT}"
echo "FSL_REF=${FSL_REF}"

# Safety checks
command -v applywarp >/dev/null 2>&1 || { echo "ERROR: applywarp (FSL) not found in PATH"; exit 1; }
[[ -f "$FSL_REF" ]] || { echo "ERROR: MNI reference not found: $FSL_REF (set FSL_REF)"; exit 1; }
[[ -f "$WARP_SCRIPT" ]] || { echo "ERROR: warp script not found: $WARP_SCRIPT"; exit 1; }

array_contains () {
  local needle="$1"; shift
  local x
  for x in "$@"; do [[ "$x" == "$needle" ]] && return 0; done
  return 1
}

# ----------------------------
# Main loop
# ----------------------------
for subj in "${SUBJECTS[@]}"; do
  echo "--------------------------------------------------------"
  echo "--- Subject position ${subj} ---"

  subject_name=$(SUBJ_IDX="$subj" python - <<'PY'
from da_paths import DATA_ROOT
import os
subjects = sorted([p.name for p in (DATA_ROOT / "perception").iterdir() if p.is_dir()], key=int)
print(subjects[int(os.environ["SUBJ_IDX"])])
PY
  )
  echo "Position ${subj} -> subject folder ${subject_name}"

  # 1. Baseline: within-domain content decoding (P2P, I2I)
  python searchlight_Cross_Domain_Content_Decoding_P2P.py "$subj" "$NITER"
  python searchlight_Cross_Domain_Content_Decoding_I2I.py "$subj" "$NITER"

  # 1. Baseline: within-content reality decoding (Ci2Ci, Cj2Cj)
  python searchlight_Cross_Content_Domain_Decoding_Ci2Ci.py "$subj" "$NITER"
  python searchlight_Cross_Content_Domain_Decoding_Cj2Cj.py "$subj" "$NITER"

  # 2. Cross-domain and cross-content decoding, baseline + DA
  for method in "${DA_METHODS[@]}"; do
    python searchlight_DA_methods_Cross_Content_Domain_Decoding_Ci2Cj.py "$subj" "$NITER" "$method"
    python searchlight_DA_methods_Cross_Content_Domain_Decoding_Cj2Ci.py "$subj" "$NITER" "$method"
    python searchlight_DA_methods_Cross_Domain_Content_Decoding_I2P.py "$subj" "$NITER" "$method"
    python searchlight_DA_methods_Cross_Domain_Content_Decoding_P2I.py "$subj" "$NITER" "$method"
  done

  # 3. Cohen's d (DA vs baseline), computed in native space before warping
  for method in "${DA_METHODS[@]}"; do
    python "$COHEN_SCRIPT" --subject "$subject_name" --target-domain imagery   --mode P2I   \
      --da-method "$method" --niter "$NITER" --radius "$RADIUS" --results-root "$RESULTS_ROOT_CROSS_DOMAIN"
    python "$COHEN_SCRIPT" --subject "$subject_name" --target-domain perception --mode I2P  \
      --da-method "$method" --niter "$NITER" --radius "$RADIUS" --results-root "$RESULTS_ROOT_CROSS_DOMAIN"
    python "$COHEN_SCRIPT" --subject "$subject_name" --target-domain imagery   --mode Ci2Cj \
      --da-method "$method" --niter "$NITER" --radius "$RADIUS" --results-root "$RESULTS_ROOT_CROSS_CONTENT"
    python "$COHEN_SCRIPT" --subject "$subject_name" --target-domain imagery   --mode Cj2Ci \
      --da-method "$method" --niter "$NITER" --radius "$RADIUS" --results-root "$RESULTS_ROOT_CROSS_CONTENT"
  done

  # 4a. Warp baseline maps to MNI (all tasks)
  echo "---- Warping BASELINE maps ----"
  for t in "${CROSS_DOMAIN_ALL_TASKS[@]}"; do
    python "$WARP_SCRIPT" BASELINE "$subject_name" "$FSL_REF" "$RESULTS_ROOT_CROSS_DOMAIN" "$t"
  done
  for t in "${CROSS_CONTENT_ALL_TASKS[@]}"; do
    subj_task_dir="$RESULTS_ROOT_CROSS_CONTENT/$t/$subject_name"
    if array_contains "$t" "${CROSS_CONTENT_PER_CONTENT_TASKS[@]}"; then
      shopt -s nullglob
      content_dirs=("$subj_task_dir"/content_*)
      shopt -u nullglob
      if (( ${#content_dirs[@]} == 0 )); then
        echo "ERROR: no content_* folders in $subj_task_dir"; exit 1
      fi
      for cdir in "${content_dirs[@]}"; do
        python "$WARP_SCRIPT" BASELINE "$subject_name" "$FSL_REF" "$RESULTS_ROOT_CROSS_CONTENT" "$t" "$(basename "$cdir")"
      done
    else
      python "$WARP_SCRIPT" BASELINE "$subject_name" "$FSL_REF" "$RESULTS_ROOT_CROSS_CONTENT" "$t"
    fi
  done

  # 4b. Warp DA maps to MNI (cross tasks only)
  echo "---- Warping DA maps ----"
  for method in "${DA_METHODS[@]}"; do
    for t in "${CROSS_DOMAIN_DA_TASKS[@]}"; do
      python "$WARP_SCRIPT" "$method" "$subject_name" "$FSL_REF" "$RESULTS_ROOT_CROSS_DOMAIN" "$t"
    done
    for t in "${CROSS_CONTENT_DA_TASKS[@]}"; do
      python "$WARP_SCRIPT" "$method" "$subject_name" "$FSL_REF" "$RESULTS_ROOT_CROSS_CONTENT" "$t"
    done
  done
done

echo "========================================================"
echo "Job finished: $(date)"
echo "========================================================"
