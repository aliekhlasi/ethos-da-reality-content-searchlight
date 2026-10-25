#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
# Part of: "Is This the Real Life? Is This Just Fantasy? A Domain Adaptation
#          Framework for Identifying Higher-Order Reality Representations"
# Edited by: Ali Ekhlasi, Postdoctoral Researcher,
#            Department of Cognitive Sciences, University of California, Irvine
#            https://github.com/aliekhlasi
# License: GNU General Public License v3.0 (see LICENSE)
# ------------------------------------------------------------------------------

import subprocess
from pathlib import Path
import sys
# SET FILE STRUCTURE
algorithm   = sys.argv[1]
subject_id  = sys.argv[2]
fsl_location     = sys.argv[3]  # (this should be a ref NIfTI like MNI152_T1_2mm.nii.gz)
results_root= Path(sys.argv[4])  # e.g. outputs/results/searchlight/Cross_Domain_Content_Decoding
task        = sys.argv[5]        # "P2I"
content = sys.argv[6] if len(sys.argv) > 6 else None

exp_dir = results_root / task
sys.path.append(exp_dir)
print(exp_dir)
script_dir = Path(__file__).resolve().parent
searchlight_maps = []
# --- 3. DYNAMIC PATH SELECTION ---

# Define the folder inside exp_dir where the maps are located (e.g., '10')
# We assume the maps are stored under a subject ID (e.g., '10') inside exp_dir
# We'll use a specific participant ID (e.g., '10') for the search base:
search_base_dir = exp_dir / subject_id  # CRITICAL: Adjust '10' to be dynamic or loop over subjects if necessary

if algorithm == 'BASELINE':
    # 🎯 Case 1: BASELINE maps are in the main subject folder (e.g., '10/').
    if content is not None:
        search_path = search_base_dir / content
    else:
        search_path = search_base_dir
    
    search_pattern = f'map_*12mm_BASELINE*.nii.gz'
    
    # Files are directly in the subject folder, so the participant directory needs to be set up higher.
    # We set the participant ID to the subject folder name for transformation file lookup.
    participant_id_for_transform = search_base_dir.name
    
    # Search for files and add them to the list
    searchlight_maps.extend(search_path.glob(search_pattern))

elif algorithm == 'NAIVE':
    # NAIVE maps are generated alongside BASELINE maps in the subject folder.
    if content is not None:
        search_path = search_base_dir / content
    else:
        search_path = search_base_dir
    search_pattern = 'map_*12mm_NAIVE*.nii.gz'
    participant_id_for_transform = search_base_dir.name
    searchlight_maps.extend(search_path.glob(search_pattern))

else:
    # DA maps are stored inside a subdirectory named after the algorithm,
    # for example subject/PRED, subject/RTLC, or subject/BalancedWeighting.
    subfolder_path = search_base_dir / algorithm
    if subfolder_path.is_dir():
        search_pattern = f'map_*12mm_{algorithm}*.nii.gz'
        participant_id_for_transform = search_base_dir.name

        if content is not None:
            search_path = subfolder_path / content
        else:
            search_path = subfolder_path

        searchlight_maps.extend(search_path.glob(search_pattern))
    else:
        if content is not None:
            search_path = search_base_dir / content
        else:
            search_path = search_base_dir
        search_pattern = f'map_*12mm_{algorithm}*.nii.gz'
        participant_id_for_transform = search_base_dir.name
        searchlight_maps.extend(search_path.glob(search_pattern))

print(f"Searching in {search_path} for files matching: {search_pattern}")

if not searchlight_maps:
    print(f"ERROR: No NIfTI files found matching the pattern '{search_pattern}' in {search_path}.")
    sys.exit(1)

# Skip files that have already been warped to MNI.
searchlight_maps = [p for p in searchlight_maps if not p.name.endswith('_mni.nii.gz')]

if not searchlight_maps:
    print(f"All matching files are already warped to MNI in {search_path}. Nothing to do.")
    sys.exit(0)

# --- 4. PROCESSING LOOP (Unchanged Logic, Adjusted Path Variables) ---
for result in searchlight_maps:
    
    # Use the consistent participant ID derived from the subject folder ('10')
    participant = participant_id_for_transform
    print(f"Processing: {result.name} from Subject: {participant}")
    
    # Path logic remains dependent on the main subject directory for transformation files
    participant_dir = script_dir.parent / 'participants' / participant
   
    anat_dir = participant_dir / 'perception/raw/anat'
    mni_example_func_dir = participant_dir / 'perception/preprocessed/mni/example_func'
    
    nonlinear_trans_nii = anat_dir / 'nonlinear_trans.nii.gz'
    output_mat_file = mni_example_func_dir / 'examplefunc2struct.mat'
    
    name = result.name.split('.nii.gz')[0]
    # Output file name changed slightly to ensure unique saving
    searchlight_to_mni_file = result.parent / f'{name}_mni.nii.gz' 
    
    if not nonlinear_trans_nii.exists() or not output_mat_file.exists():
        print(f"WARNING: Skipping {result.name}. Missing transformation files for subject {participant}.")
        continue

    # Execute applywarp (using the preferred list format)
    command_list = [
        "applywarp",
        f"--ref={fsl_location}",
        f"--in={result}",
        f"--warp={nonlinear_trans_nii}",
        f"--premat={output_mat_file}",
        f"--out={searchlight_to_mni_file}"
    ]
    subprocess.run(command_list)
