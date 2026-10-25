# Is This the Real Life? Is This Just Fantasy? A Domain Adaptation Framework for Identifying Higher-Order Reality Representations

**Ali Ekhlasi, Emil Olsson, Michaela Klímová, Xueyi Huang, Angela Shen, Nadine Dijkstra,
Jorge Morales, Megan A. K. Peters**

Code maintained by [Ali Ekhlasi](https://github.com/aliekhlasi), Postdoctoral Researcher,
Department of Cognitive Sciences, University of California, Irvine.

Part of the [ETHOS](https://www.arc-ethos.org/) adversarial collaboration, funded by the
Templeton World Charity Foundation (see [Funding and acknowledgments](#funding-and-acknowledgments)).

## Documents

| File | What it is |
|---|---|
| [`docs/Ekhlasi_et_al_CCN2026_extended_abstract.pdf`](docs/Ekhlasi_et_al_CCN2026_extended_abstract.pdf) | Extended abstract presented at the 9th Conference on Cognitive Computational Neuroscience (CCN 2026), New York (CC BY 4.0) |

## Overview

The analysis uses domain adaptation (DA) to ask where fMRI patterns can be aligned
**across perception and imagery** (content decoding: dog vs. scissors) and **across
object categories** (reality decoding: perception vs. imagery). Whole-brain searchlights
(12 mm spheres) are run in each subject's native space with a logistic-regression
classifier, with and without Subspace Alignment (SA). The maps are then warped to MNI
space and combined across the 18 subjects.

## Repository layout

| Path | What it does |
|---|---|
| `run_searchlight_pipeline.sh` | Runs the whole per-subject pipeline (steps 1–4 below) |
| `DA_searchlight/searchlight_Cross_Domain_Content_Decoding_P2P.py`, `..._I2I.py` | Content decoding within perception / within imagery |
| `DA_searchlight/searchlight_Cross_Content_Domain_Decoding_Ci2Ci.py`, `..._Cj2Cj.py` | Reality decoding within each object category |
| `DA_searchlight/searchlight_DA_methods_Cross_Domain_Content_Decoding_P2I.py`, `..._I2P.py` | Content decoding across perception and imagery, baseline and SA |
| `DA_searchlight/searchlight_DA_methods_Cross_Content_Domain_Decoding_Ci2Cj.py`, `..._Cj2Ci.py` | Reality decoding across object categories, baseline and SA |
| `DA_searchlight/Compute_Cohen.py` | Cohen's d maps (SA vs. baseline) for accuracy and covariance distance |
| `DA_searchlight/searchlight_normalisation/scripts/Warping_maps_to_mni_DA.py` | Warps maps to MNI space with FSL `applywarp` |
| `DA_searchlight/searchlight_normalisation/participants/` | Each subject's native-to-MNI transforms used by the warp (downloaded from OSF, see section 2) |
| `algorithms/searchlight_DA_save.py` | Searchlight, data splitting and DA fitting |
| `dataManipulation/whole_brain.py` | Loads a subject's preprocessed data |
| `da_paths.py` | Data, results and figure folders (set via environment variables) |
| `CCN2026_GitHub_Results.ipynb` | Group analysis: Figures 2–3 and Tables S1–S8 of the full paper |
| `docs/` | Extended abstract (PDF) |
| `LICENSE` | GNU General Public License v3.0 |

Naming: `P` = perception, `I` = imagery, `Ci` / `Cj` = the two object categories.
`P2I` means "train on perception, test on imagery".

## 1. Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

[FSL](https://fsl.fmrib.ox.ac.uk) is also needed for the MNI warp (`applywarp` and the
`MNI152_T1_2mm_brain_mask.nii.gz` template).

## 2. Data

The analysis uses the public perception/imagery fMRI dataset (18 participants, dogs vs.
scissors) shared by Olza, Soto & Santana (2025), *Brain Informatics* 12, 17
(https://doi.org/10.1186/s40708-025-00263-0); task and acquisition details are in
Margolles et al. (2024), *J. Neurosci.* 44(2). The data are already preprocessed by the
original authors (motion correction, linear detrending, peak volumes 5–18 s after trial
onset, z-scoring). The data are **not** included in this repository.

### 2.1 Download

All files are on the Open Science Framework: **https://osf.io/audmx/**

| OSF file | Size | Contents |
|---|---|---|
| `data/perception.zip` | 431 MB | `perception/<subject>/events.csv` (+ ROI `.npy` files, not used here) |
| `data/imagery.zip` | 217 MB | `imagery/<subject>/events.csv` (+ ROI `.npy` files, not used here) |
| `data/whole_brain.zip` | 4.6 GB | `whole_brain/{perception,imagery}/<subject>/` whole-brain data used by the searchlights |
| `participants.zip` | 2 MB | each subject's native-to-MNI transforms, used by the MNI warp |

### 2.2 Where to put the files

Unzip the three `data/*.zip` files into a folder called `data/` at the top of this
repository, and unzip `participants.zip` into `DA_searchlight/searchlight_normalisation/`:

```bash
cd <this repository>
mkdir -p data
unzip perception.zip  -d data/
unzip imagery.zip     -d data/
unzip whole_brain.zip -d data/
unzip participants.zip -d DA_searchlight/searchlight_normalisation/
```

Afterwards the repository should look like this (only the files the code uses are shown):

```
<this repository>/
├── data/                                          <- DA_DATA_ROOT
│   ├── perception/<subject>/events.csv            columns: trial_idx, run, target_category
│   ├── imagery/<subject>/events.csv
│   └── whole_brain/
│       ├── perception/<subject>/perception_detrended_zscored_stacked_vols_of_interest_searchlight.nii.gz
│       ├── perception/<subject>/example_func_deoblique_brainmask.nii
│       └── imagery/<subject>/imagery_detrended_zscored_stacked_vols_of_interest_searchlight.nii.gz
├── DA_searchlight/
│   └── searchlight_normalisation/
│       ├── participants/<subject>/perception/
│       │   ├── preprocessed/mni/example_func/examplefunc2struct.mat
│       │   └── raw/anat/nonlinear_trans.nii.gz
│       └── scripts/Warping_maps_to_mni_DA.py
└── outputs/                                       <- created by the pipeline
    ├── results/searchlight/...                    searchlight maps (DA_RESULTS_ROOT)
    └── ccn2026/...                                figures and tables from the notebook
```

`<subject>` is `1` … `18`. Events with `target_category == 2` (white noise) are dropped
when the data are loaded. `data/`, `participants/` and `outputs/` are listed in
`.gitignore`, so they are never committed.

### 2.3 Tell the code where the data are

```bash
export DA_DATA_ROOT=$PWD/data                  # required
export DA_RESULTS_ROOT=$PWD/outputs/results    # optional, this is the default
```

The data can also live anywhere else (for example on a cluster's scratch disk); just
point `DA_DATA_ROOT` there. The `participants/` transforms must stay inside
`DA_searchlight/searchlight_normalisation/`, because the warp script looks for them there.

## 3. Run the searchlights (per subject)

```bash
bash run_searchlight_pipeline.sh                 # all 18 subjects, one after another
SUBJECTS="0 1" bash run_searchlight_pipeline.sh  # only subjects 1 and 2
sbatch run_searchlight_pipeline.sh               # as a Slurm job (edit the #SBATCH lines)
```

`SUBJECTS` takes positions in the sorted subject list, so `0` is subject `1` and `17` is
subject `18`. Each subject takes many CPU hours; on a cluster, submit one job per subject.

For each subject the script:

1. runs the baseline searchlights (P2P, I2I, Ci2Ci, Cj2Cj),
2. runs the cross-domain and cross-content searchlights, baseline and SA
   (P2I, I2P, Ci2Cj, Cj2Ci), with 10 resampling iterations,
3. computes the Cohen's d maps,
4. warps all maps to MNI space.

Data splits are seeded by iteration number, so reruns give the same splits.

## 4. Group results: figures and tables

Open `CCN2026_GitHub_Results.ipynb` and run all cells. It reads the MNI maps from
`$DA_RESULTS_ROOT/searchlight` (override with `CCN_RESULTS_ROOT`) and writes everything
to `outputs/ccn2026/` (override with `CCN_OUTPUT_ROOT`):

| Output | Paper |
|---|---|
| `figures/Fig2a_content_within.png` … `Fig2f_reality_da.png` | Figure 2 |
| `figures/Fig3a_content_independent.png` … `Fig3f_reality_dependent.png` | Figure 3 |
| `tables/Table_S1_*.csv` … `Table_S8_*.csv` | Tables S1–S8 |
| `overlap/*.nii.gz` | Group overlap maps (MNI) |
| `subject_masks/<subject>/*.nii.gz` | Per-subject category masks |

The notebook documents the rule behind every panel and table.

## Funding and acknowledgments

This work is part of **[ETHOS](https://www.arc-ethos.org/)** (*Empirical Tests of
Higher-Order theories of consciousnesS*), an adversarial collaboration testing the
predictions of competing higher-order theories of consciousness with open-science
methods.

It was supported by the [Templeton World Charity Foundation, Inc.](https://www.templetonworldcharity.org/)
(funder DOI 501100011730) under the grant https://doi.org/10.54224/22032 (to J. Morales,
M. A. K. Peters and N. Dijkstra), and in part by the Canadian Institute for Advanced
Research (CIFAR Fellowship in Brain, Mind, & Consciousness, to M. A. K. Peters).

## Citation

If you use this code, please cite the extended abstract:

> Ekhlasi, A., Olsson, E., Klímová, M., Huang, X., Shen, A., Dijkstra, N., Morales, J., &
> Peters, M. A. K. (2026). Is This the Real Life? Is This Just Fantasy? A Domain Adaptation
> Framework for Identifying Higher-Order Reality Representations. *Extended abstract,
> 9th Conference on Cognitive Computational Neuroscience (CCN 2026)*, New York, NY.

Full-paper citation details will be added after publication. The searchlight and
data-splitting code in `algorithms/` builds on code by Olza, Soto & Santana (2025),
*Brain Informatics* 12, 17 (https://doi.org/10.1186/s40708-025-00263-0).

## License

This code is released under the [GNU General Public License v3.0](LICENSE). It builds on
the GPL-3.0 code of Olza, Soto & Santana
([AlexOlza/DA-enhanced-searchlight](https://github.com/AlexOlza/DA-enhanced-searchlight)),
so the same license applies. The dataset on OSF and the PDF in `docs/` are not covered by
this license: the data follow the terms on https://osf.io/audmx/, and the extended
abstract is licensed CC BY 4.0.

## Contact

Ali Ekhlasi · https://github.com/aliekhlasi
