"""Sanity-check a cropped, parcellated fif from beamform_parcellate_schaefer100.py.

Prints the annotation summary and crop timings, then opens the interactive
MNE browser with annotations shown as coloured spans.

Run with the tokeniser_220926 conda env:
    conda activate tokeniser_220926
    python check_parcellated.py sub-001_ses-003_task-braille_run-001
"""

import sys

import mne
import numpy as np

deriv = "/rdrives/DRS-foundation-brain/zoe_data/BIDS/derivatives_anna_01042026"
id = sys.argv[1] if len(sys.argv) > 1 else "sub-001_ses-001_task-braille_run-001"
#parc_fif = f"{deriv}/osl_schaefer100_noorthog/{id}/lcmv-parc-raw.fif"
#parc_fif = f"{deriv}/osl/{id}/lcmv-parc-raw.fif"
parc_fif = f"{deriv}/osl_correct_annotations/{id}/lcmv-parc-raw.fif"
#parc_fif = f"{deriv}/hmm_8state_correct_annotations/opm/stc/{id}_stc-raw.fif"

raw = mne.io.read_raw_fif(parc_fif, preload=True)
print(raw)
print(raw.annotations)
print(dict(zip(*np.unique(raw.annotations.description, return_counts=True))))

#onsets = raw.annotations.onset[raw.annotations.description == "on"] - raw.first_time
#print(f"n trials: {len(onsets)} (expect 80)")
#print(f"first onset at {onsets.min():.4f}s (expect ~1 sample = {1 / raw.info['sfreq']:.4f}s)")
#print(f"data ends {raw.times[-1] - onsets.max():.2f}s after last onset (expect 14)")

# Interactive browser: parcels + any stim chans, with annotations shown as coloured spans
raw.plot(duration=30, n_channels=20, scalings="auto", block=True)
