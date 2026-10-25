# ------------------------------------------------------------------------------
# Part of: "Is This the Real Life? Is This Just Fantasy? A Domain Adaptation
#          Framework for Identifying Higher-Order Reality Representations"
# Edited by: Ali Ekhlasi, Postdoctoral Researcher,
#            Department of Cognitive Sciences, University of California, Irvine
#            https://github.com/aliekhlasi
# License: GNU General Public License v3.0 (see LICENSE)
# ------------------------------------------------------------------------------
"""Folder locations used by every script in this repository.

Set these environment variables before running anything:

    DA_DATA_ROOT     folder with the preprocessed dataset (required; layout in README.md)
    DA_RESULTS_ROOT  where searchlight maps are written   (default: <repo>/outputs/results)
    DA_FIGURES_ROOT  where per-subject figures are written (default: <repo>/outputs/figures)

Example:

    export DA_DATA_ROOT=/path/to/olza2025_dataset
    export DA_RESULTS_ROOT=/path/to/outputs/results
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

# Scripts in subfolders import `algorithms` and `dataManipulation` from the repo root.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _env_path(name: str, default: Path | None) -> Path:
    value = os.environ.get(name, "").strip()
    if value:
        return Path(value).expanduser().resolve()
    if default is None:
        raise RuntimeError(
            f"{name} is not set. Point it at the dataset folder, e.g.\n"
            f"    export {name}=/path/to/olza2025_dataset\n"
            "See README.md for the expected folder layout."
        )
    return default


DATA_ROOT = _env_path("DA_DATA_ROOT", None)
RESULTS_ROOT = _env_path("DA_RESULTS_ROOT", REPO_ROOT / "outputs" / "results")
FIGURES_ROOT = _env_path("DA_FIGURES_ROOT", REPO_ROOT / "outputs" / "figures")
