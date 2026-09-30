"""Apply the canonical 8-state Glasser52 HMM to the OPM and SQUID parcel data
in osl_correct_annotations/.

Adapted from annas_pipeline/apply_chmm.ipynb, with fixes:
- per-session free energy labels come from the file paths (the notebook
  computed them from the loop index, which is wrong from sub-008 onwards
  because there is no sub-008)
- alpha/stc fif annotations are rebuilt from the parc file, since osl-dynamics'
  convert_to_mne_raw shifts them by first_time when there is no meas_date (OPM)
- alpha and stc are saved per session as npy and fif (alpha also as a pkl)
- every alpha/stc fif is checked against the parc file (samples, stim events,
  trial annotations) and the results are written to timing_report.txt

Outputs go to hmm_8state_correct_annotations/{opm,squid}/.

Run with the annasenv1 conda env (GPU):
    conda activate annasenv1
    python apply_chmm.py
"""

import os
import sys

REPO = "/home/anna-beer/Documents/anna_phd/Canonical-HMM-Networks"
os.chdir(REPO)  # so ./models resolves
sys.path.insert(0, REPO)  # so modules/ imports when run as a script from any folder

import pickle
from glob import glob

import mne
import numpy as np
import tensorflow as tf
from modules import hmm
from osl_dynamics import analysis, inference
from osl_dynamics.data import Data

for gpu in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)  # must happen first

N_STATES = 8
SFREQ = 250
TRIAL_CHANNELS = {"opm": ["Lef", "Rig"], "squid": ["Left_trial", "Right_trial"]}


def fix_annotations(hmm_raw, parc_raw):
    """Copy the parc file's annotations onto an alpha/stc Raw.

    osl_dynamics' convert_to_mne_raw adds first_time to annotation onsets a
    second time when meas_date is None. Onsets relative to the data start with
    orig_time=None are handled correctly with or without meas_date.
    """
    a = parc_raw.annotations
    hmm_raw.set_annotations(mne.Annotations(a.onset - parc_raw.first_time, a.duration, a.description))


def check_hmm_timing(hmm_fif, parc_raw, trial_channels):
    """Check a saved alpha/stc fif isn't cropped or shifted relative to the parc file.

    Returns (problems, summary): problems is empty if everything is fine,
    summary has the trial count and first onset per trial channel.
    """
    hmm_raw = mne.io.read_raw_fif(hmm_fif, preload=True, verbose=False)
    sfreq = parc_raw.info["sfreq"]
    problems, summary = [], []

    if hmm_raw.n_times != parc_raw.n_times:
        problems.append(f"n_times {hmm_raw.n_times} != parc {parc_raw.n_times}")
    if hmm_raw.first_samp != parc_raw.first_samp:
        problems.append(f"first_samp {hmm_raw.first_samp} != parc {parc_raw.first_samp}")

    n_states = sum(t != "stim" for t in hmm_raw.get_channel_types())
    if n_states != N_STATES:
        problems.append(f"{n_states} state channels (expected {N_STATES})")

    for ch in trial_channels:
        parc_ev = mne.find_events(parc_raw, stim_channel=ch, verbose=False)
        hmm_ev = mne.find_events(hmm_raw, stim_channel=ch, verbose=False)
        if not np.array_equal(parc_ev[:, 0], hmm_ev[:, 0]):
            problems.append(f"{ch}: stim events differ ({len(hmm_ev)} vs parc {len(parc_ev)})")

        stim_t = (hmm_ev[:, 0] - hmm_raw.first_samp) / sfreq
        a = hmm_raw.annotations
        ann_t = np.sort(a.onset[a.description == ch] - hmm_raw.first_time)
        if len(ann_t) != len(stim_t):
            problems.append(f"{ch}: {len(ann_t)} annotations vs {len(stim_t)} stim events")
        elif len(stim_t) and np.max(np.abs(ann_t - stim_t)) > 1 / sfreq:
            problems.append(f"{ch}: annotations shifted "
                            f"(first: {ann_t[0]:.3f} s vs stim {stim_t[0]:.3f} s)")
        if len(ann_t):
            summary.append(f"{ch}: n={len(ann_t)}, first={ann_t[0]:.3f}s")

    return problems, summary


def save_hmm_fif(time_course, parc_fif, parc_raw, n_embeddings, filename, trial_channels):
    """Convert an alpha/stc time course to a Raw, fix its annotations, save and check it."""
    hmm_raw = inference.modes.convert_to_mne_raw(
        time_course,
        parc_fif,  # single file (lcmv-parc-raw.fif), not a list
        n_embeddings=n_embeddings,
    )
    fix_annotations(hmm_raw, parc_raw)
    hmm_raw.save(filename, overwrite=True)
    return check_hmm_timing(filename, parc_raw, trial_channels)


def run_group(group, files, out_dir, model):
    """Apply the HMM to one group of sessions and save everything.

    Returns a list of (label, free energy) with the group total first.
    """
    ids = [f.split("/")[-2] for f in files]
    trial_channels = TRIAL_CHANNELS[group]

    # CREATE OUTPUT DIRECTORIES -------------------------------------------------
    alp_dir = f"{out_dir}/alp"
    alp_dir_pkl = f"{out_dir}/alp_pkl"
    stc_dir = f"{out_dir}/stc"
    summary_stats_dir = f"{out_dir}/summary_stats"
    for d in (alp_dir, alp_dir_pkl, stc_dir, summary_stats_dir):
        os.makedirs(d, exist_ok=True)

    with open(f"{out_dir}/sessions.txt", "w") as f:
        f.write("\n".join(ids) + "\n")

    # LOAD IN DATA -------------------------------------------------------------
    data = Data(files, picks="misc", reject_by_annotation="omit", n_jobs=8)
    data = hmm.prepare_data_for_canonical_hmm(data, parcellation="Glasser52")

    # FREE ENERGY --------------------------------------------------------------
    free_energy = [(f"{group.upper()} (all sessions)", model.free_energy(data))]
    for i, id in enumerate(ids):
        free_energy.append((id, model.free_energy(data[i])))

    with open(f"{out_dir}/free_energy.txt", "w") as f:
        f.write(f"variational_free_energy = {free_energy[0][1]}\n")
        for label, fe in free_energy[1:]:
            f.write(f"{label} = {fe}\n")

    # ESTIMATE STATE PROBABILITIES ---------------------------------------------
    alp = model.get_alpha(data)
    if not isinstance(alp, list):  # a single session comes back as one array
        alp = [alp]
    pickle.dump(alp, open(f"{alp_dir_pkl}/alp.pkl", "wb"))
    stc = inference.modes.argmax_time_courses(alp)

    # SAVE ALPS AND STCS AS NPY AND FIF FILES ----------------------------------
    report = []
    for i, (parc_fif, id) in enumerate(zip(files, ids)):
        np.save(f"{alp_dir}/{id}_alpha.npy", alp[i])
        np.save(f"{stc_dir}/{id}_stc.npy", stc[i])

        parc_raw = mne.io.read_raw_fif(parc_fif, preload=True, verbose=False)
        for kind, time_course, d in (("alpha", alp[i], alp_dir), ("stc", stc[i], stc_dir)):
            problems, summary = save_hmm_fif(time_course, parc_fif, parc_raw, data.n_embeddings,
                                             f"{d}/{id}_{kind}-raw.fif", trial_channels)
            status = "WARNING" if problems else "OK"
            if problems:
                print(f"WARNING {id} {kind}: " + "; ".join(problems))
            report.append(f"{id}\t{kind}\t{status}\t" + "; ".join(problems + summary))

        print(f"saved {id}")

    with open(f"{out_dir}/timing_report.txt", "w") as f:
        f.write("id\tfile\tstatus\tnotes\n" + "\n".join(report) + "\n")

    # CALCULATE MULTITAPER -----------------------------------------------------
    trimmed_data = data.trim_time_series(sequence_length=model.config.sequence_length, prepared=False)

    f, psd, coh, w = analysis.spectral.multitaper_spectra(
        data=trimmed_data,
        alpha=alp,
        sampling_frequency=SFREQ,
        frequency_range=[0.5, 45],
        return_weights=True,
    )
    np.save(f"{summary_stats_dir}/f.npy", f)
    np.save(f"{summary_stats_dir}/psd.npy", psd)
    np.save(f"{summary_stats_dir}/coh.npy", coh)
    np.save(f"{summary_stats_dir}/w.npy", w)

    # CALC POW_MAPS, COH_NETS AND SUMMARY STATS AND SAVE AS NPYS ----------------
    pow_maps = analysis.power.variance_from_spectra(f, psd)
    np.save(f"{summary_stats_dir}/pow_maps.npy", pow_maps)

    coh_nets = analysis.connectivity.mean_coherence_from_spectra(f, coh)
    np.save(f"{summary_stats_dir}/coh_nets.npy", coh_nets)

    fo = analysis.post_hoc.fractional_occupancies(stc)
    lt = analysis.post_hoc.mean_lifetimes(stc, sampling_frequency=SFREQ)
    intv = analysis.post_hoc.mean_intervals(stc, sampling_frequency=SFREQ)
    sr = analysis.post_hoc.switching_rates(stc, sampling_frequency=SFREQ)
    np.save(f"{summary_stats_dir}/fo.npy", fo)
    np.save(f"{summary_stats_dir}/lt.npy", lt)
    np.save(f"{summary_stats_dir}/intv.npy", intv)
    np.save(f"{summary_stats_dir}/sr.npy", sr)

    data.delete_dir()
    return free_energy


def main():
    # defining universal file locations
    base = "/rdrives/DRS-foundation-brain/zoe_data/BIDS"
    deriv = f"{base}/derivatives_anna_01042026"
    parc_dir = f"{deriv}/osl_correct_annotations"
    hmm_dir = f"{deriv}/hmm_8state_correct_annotations"
    os.makedirs(hmm_dir, exist_ok=True)

    ALL_SUBJECTS = ['001', '002', '003', '004', '005', '006', '007', '009',
                    '010', '011', '012', '013', '014', '015', '016']
    GROUPS = {"opm": ["001", "002"], "squid": ["003", "004"]}

    # which subjects/groups to actually process this run
    SUBJECTS_TO_RUN = ALL_SUBJECTS
    GROUPS_TO_RUN = ["opm", "squid"]

    model = hmm.load_canonical_hmm(n_states=N_STATES, parcellation="Glasser52")

    free_energy = {}
    for group in GROUPS_TO_RUN:
        files = sorted(
            f for f in glob(f"{parc_dir}/*/lcmv-parc-raw.fif")
            if any(f"ses-{s}" in f for s in GROUPS[group])
            and any(f"sub-{s}_" in f for s in SUBJECTS_TO_RUN)
        )
        expected = len(SUBJECTS_TO_RUN) * len(GROUPS[group])
        if len(files) != expected:
            print(f"WARNING: {group} has {len(files)} parc files (expected {expected})")

        free_energy[group] = run_group(group, files, f"{hmm_dir}/{group}", model)

    # parent free energy file: group totals first, then every session
    with open(f"{hmm_dir}/free_energy.txt", "w") as f:
        for group in GROUPS_TO_RUN:
            label, fe = free_energy[group][0]
            f.write(f"{label} = {fe}\n")
        for group in GROUPS_TO_RUN:
            for label, fe in free_energy[group][1:]:
                f.write(f"{label} = {fe}\n")


if __name__ == "__main__":
    main()
