"""Beamform and parcellate OPM/SQUID MEG data with the Glasser52 atlas,
symmetric orthogonalisation, with correct trial annotations.

Adapted from EphysTokenizer/neuralfingerprinting_data/beamform_parcellate_schaefer100.py:
- Glasser52 atlas instead of Schaefer100
- symmetric orthogonalisation (as in annas_pipeline/beamforming_and_parcellating.ipynb)
- reads the trigger-cleaned files from preprocessed_cleaned/
  (run clean_duplicate_triggers.py first)
- fix_parc_annotations is applied whenever the annotations have no orig_time
  (the case where osl-dynamics' save_as_fif shifts them by first_time)
- every output is checked for cropping/annotation problems, and the results are
  written to parcellation_timing_report.txt next to this script

Run with the tokeniser_220926 conda env:
    conda activate tokeniser_220926
    python beamform_parcellate_glasser52.py
"""

import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir("/home/anna-beer/Documents/anna_phd/Canonical-HMM-Networks")

import mne
import numpy as np
from osl_dynamics.meeg import rhino, source_recon, parcellation
from osl_dynamics.utils.filenames import OSLFilenames

N_PARCELS = 52

SQUID_STIM_CHANNELS = {  # need this to keep trial timings
    'Right_trial': 'stim',
    'Left_trial': 'stim',
    'Braille_on': 'stim',
    'key_press': 'stim',
    'key_right': 'stim',
    'key_left': 'stim',
    'braille_data': 'stim',
    'braille_clock': 'stim',
    'braille_stop': 'stim',
}


def fix_parc_annotations(parcfif, raw):
    """Re-apply raw's annotations to the saved parcel fif.

    osl_dynamics' save_as_fif adds first_time to annotation onsets a second
    time when meas_date is None, shifting them away from the stim channels.
    """
    parc = mne.io.read_raw_fif(parcfif, preload=True)
    a = raw.annotations
    parc.set_annotations(mne.Annotations(a.onset - raw.first_time, a.duration, a.description))
    parc.save(parcfif, overwrite=True)


def check_parc_timing(parcfif, raw, trial_channels):
    """Check the parcel fif isn't cropped or shifted relative to raw.

    Returns (problems, summary): problems is empty if everything is fine,
    summary has the trial count and first onset per trial channel.
    """
    parc = mne.io.read_raw_fif(parcfif, preload=True, verbose=False)
    sfreq = raw.info["sfreq"]
    problems = []

    if parc.n_times != raw.n_times:
        problems.append(f"n_times {parc.n_times} != raw {raw.n_times}")
    if parc.first_samp != raw.first_samp:
        problems.append(f"first_samp {parc.first_samp} != raw {raw.first_samp}")

    n_parcels = sum(t != "stim" for t in parc.get_channel_types())
    if n_parcels != N_PARCELS:
        problems.append(f"{n_parcels} parcel channels (expected {N_PARCELS})")

    first_onsets = []
    for ch in trial_channels:
        raw_ev = mne.find_events(raw, stim_channel=ch, verbose=False)
        parc_ev = mne.find_events(parc, stim_channel=ch, verbose=False)
        if not np.array_equal(raw_ev[:, 0], parc_ev[:, 0]):
            problems.append(f"{ch}: stim events differ ({len(parc_ev)} vs raw {len(raw_ev)})")

        stim_t = (parc_ev[:, 0] - parc.first_samp) / sfreq
        a = parc.annotations
        ann_t = np.sort(a.onset[a.description == ch] - parc.first_time)
        if len(ann_t) != len(stim_t):
            problems.append(f"{ch}: {len(ann_t)} annotations vs {len(stim_t)} stim events")
        elif len(stim_t) and np.max(np.abs(ann_t - stim_t)) > 1 / sfreq:
            problems.append(f"{ch}: annotations shifted "
                            f"(first: {ann_t[0]:.3f} s vs stim {stim_t[0]:.3f} s)")
        if len(ann_t):
            first_onsets.append(f"{ch}: n={len(ann_t)}, first={ann_t[0]:.3f}s")

    print(f"{parcfif}: " + "; ".join(first_onsets))
    return problems, first_onsets


def source_space(id, preprocfile, outdirsurf, outdir_osl, parcfif, parcellationfile,
                 trial_channels, posfile=None, mode=None, psdfile=None):
    """Filter, beamform, parcellate and save one session.

    Returns (problems, info) from check_parc_timing.
    """
    raw = mne.io.read_raw_fif(preprocfile, preload=True)

    if mode == "squid":
        # set before filtering so the trigger channels are never filtered
        raw = raw.set_channel_types(SQUID_STIM_CHANNELS)

    raw = raw.filter(l_freq=4, h_freq=45, method="iir", iir_params={"order": 5, "ftype": "butter"})

    if mode == "opm":

        fns = OSLFilenames(
            outdir=outdir_osl,
            id=id,
            preproc_file=preprocfile,
            surfaces_dir=outdirsurf,  # use "mni152_surfaces" for the standard brain,
        )

        rhino.extract_fiducials_and_headshape_from_fif(fns)
        rhino.coregister_head_and_mri(fns, show=False)

        rhino.forward_model(fns, model="Single Layer", gridstep=8)
        source_recon.lcmv_beamformer(fns, raw, chantypes="mag", rank={"mag": 100})

    if mode == "squid":

        fns = OSLFilenames(
            outdir=outdir_osl,
            id=id,
            preproc_file=preprocfile,
            surfaces_dir=outdirsurf,  # use "mni152_surfaces" for the standard brain
            pos_file=posfile,
        )

        rhino.extract_fiducials_and_headshape_from_pos(fns)
        rhino.coregister_head_and_mri(
            fns,
            use_headshape=True,  # If you don't have a .pos file, pass False
            use_nose=True,
            allow_mri_scaling=False,  # Note: if you're using a standard brain, pass True
            show=False,
        )

        rhino.forward_model(fns, model="Single Layer", gridstep=8)
        source_recon.lcmv_beamformer(fns, raw, chantypes="mag", rank={"mag": 120})

    voxel_data, voxel_coords = source_recon.apply_lcmv_beamformer(fns, raw)

    parcel_data = parcellation.parcellate(
        fns,
        voxel_data,
        voxel_coords,
        method="spatial_basis",
        orthogonalisation="symmetric",
        parcellation_file=parcellationfile,
    )

    parcellation.save_as_fif(
        parcel_data,
        raw,
        extra_chans="stim",
        filename=parcfif,
    )

    # osl-dynamics only shifts annotations when there's no meas_date (OPM)
    if raw.annotations.orig_time is None:
        fix_parc_annotations(parcfif, raw)

    problems, info = check_parc_timing(parcfif, raw, trial_channels)

    parcellation.plot_psds(parcfif, parcellation_file=parcellationfile, filename=psdfile)

    return problems, info


def append_report(report_file, id, status, notes):
    new = not os.path.exists(report_file)
    with open(report_file, "a") as f:
        if new:
            f.write("id\tstatus\tnotes\n")
        f.write(f"{id}\t{status}\t{notes}\n")


def main():
    # defining universal file locations
    base = "/rdrives/DRS-foundation-brain/zoe_data/BIDS"
    deriv = f"{base}/derivatives_anna_01042026"
    indir = f"{deriv}/preprocessed_cleaned"
    outdir_osl = f"{deriv}/osl_correct_annotations"
    os.makedirs(outdir_osl, exist_ok=True)
    report_file = os.path.join(SCRIPT_DIR, "parcellation_timing_report.txt")

    parcellation_file = "atlas-Glasser_nparc-52_space-MNI_res-8x8x8.nii.gz"

    # full subject/session lists, matching the notebook's batch exactly
    ALL_SUBJECTS = ['001', '002', '003', '004', '005', '006', '007', '009',
                    '010', '011', '012', '013', '014', '015', '016']
    OPM_SESSIONS = ["001", "002"]
    SQUID_SESSIONS = ["003", "004"]
    tasks = ["braille"]
    runs = ['001']

    # which subjects to actually process this run -- change to ALL_SUBJECTS
    # to run the full batch once this has been verified on one subject.
    SUBJECTS_TO_RUN = ALL_SUBJECTS

    jobs = []
    for subject in SUBJECTS_TO_RUN:
        for session in OPM_SESSIONS + SQUID_SESSIONS:
            for task in tasks:
                for run in runs:
                    jobs.append((subject, session, task, run))
    # OPMs first, then SQUIDs, as in the notebook
    jobs.sort(key=lambda j: j[1] in SQUID_SESSIONS)

    for subject, session, task, run in jobs:

        mode = "squid" if session in SQUID_SESSIONS else "opm"
        trial_channels = ["Left_trial", "Right_trial"] if mode == "squid" else ["Lef", "Rig"]
        posfile = (f"{base}/sub-{subject}/ses-{session}/meg/sub-{subject}_headshape.pos"
                   if mode == "squid" else None)

        outdir_surf = f"{deriv}/anat_surfaces/sub-{subject}"
        id = f"sub-{subject}_ses-{session}_task-{task}_run-{run}"
        preproc_file = f"{indir}/{id}/{id}_preproc-raw_cleaned.fif"
        parc_fif = f"{outdir_osl}/{id}/lcmv-parc-raw.fif"
        psd_file = f"{outdir_osl}/{id}/parc_psds.png"

        if not os.path.exists(preproc_file):
            print(f"MISSING: {preproc_file} (run clean_duplicate_triggers.py first)")
            append_report(report_file, id, "MISSING_FILE", preproc_file)
            continue

        try:
            problems, info = source_space(id, preproc_file, outdir_surf, outdir_osl, parc_fif,
                                          parcellation_file, trial_channels, posfile=posfile,
                                          mode=mode, psdfile=psd_file)
        except Exception as e:
            print(f"ERROR in {id}: {e!r}")
            append_report(report_file, id, "ERROR", repr(e))
            continue

        if problems:
            print(f"WARNING {id}: " + "; ".join(problems))
            append_report(report_file, id, "WARNING", "; ".join(problems + info))
        else:
            append_report(report_file, id, "OK", "; ".join(info))


if __name__ == "__main__":
    main()
