# ------------------------------------------------------------------------------
# Part of: "Is This the Real Life? Is This Just Fantasy? A Domain Adaptation
#          Framework for Identifying Higher-Order Reality Representations"
# Edited by: Ali Ekhlasi, Postdoctoral Researcher,
#            Department of Cognitive Sciences, University of California, Irvine
#            https://github.com/aliekhlasi
# License: GNU General Public License v3.0 (see LICENSE)
# ------------------------------------------------------------------------------

from nibabel import load as load_fmri
import numpy as np
import pandas as pd
import os
from nilearn.maskers import NiftiMasker
from nilearn.image import index_img
from pathlib import Path

from da_paths import DATA_ROOT


def load_data(domain, subject, data_root=DATA_ROOT, drop_resting_state=True):
    # Loads the complete whole-brain dataset for a domain and subject
    data_root = Path(data_root)
    data_dir = data_root / "whole_brain" / domain / subject
    event_dir = data_root / domain / subject / "events.csv"
    events = pd.read_csv(event_dir, usecols=["trial_idx", "run", "target_category"])
    bold_data = load_fmri(
        data_dir / f"{domain}_detrended_zscored_stacked_vols_of_interest_searchlight.nii.gz"
    )
    if drop_resting_state:
        white_noise_events_idx = np.where(events.target_category==2)[0]
        events = events.drop(white_noise_events_idx)
        # I discard the scans corresponding to white noise
        bold_data = index_img(bold_data,events.index)
    
    example_func = (
        data_root
        / "whole_brain"
        / "perception"
        / subject
        / "example_func_deoblique_brainmask.nii"
    )

    ex_f = load_fmri(example_func)
    masker = NiftiMasker(ex_f).fit()
    
    return(bold_data, events, masker)
