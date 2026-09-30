"""Remove duplicate trial triggers from the preprocessed OPM and SQUID data
and rebuild the trial annotations with the correct onsets.

Adapted from annas_pipeline/cleaning_duplicate_triggers.ipynb, with fixes:
- annotation onsets no longer have first_time added twice (the notebook used
  find_events samples, which include first_samp, as onsets from data start)
- non-trial annotations (BAD_*, Braille_on, key_* ...) are kept, not replaced
- the whole duplicate pulse is zeroed, not just its first sample
- the duplicate gap is in seconds, not samples
- OPM sessions (Lef/Rig) are cleaned too, not just SQUID
- a report of which sessions had duplicates is written to
  trigger_cleaning_report.txt next to this script

Inputs are read from preprocessed/ and never modified; cleaned files go to
preprocessed_cleaned/.

Run with the tokeniser_220926 conda env:
    conda activate tokeniser_220926
    python clean_duplicate_triggers.py
"""

import os

import mne
import numpy as np

EXPECTED_N_TRIALS = 40  # per trial channel


def trigger_levels(x):
    """Integer trigger levels, as mne.find_events sees them."""
    return np.abs(x.astype(np.int64))


def clean_stim_channel(raw, stim_name, min_gap_s):
    """Zero out any trigger pulse starting < min_gap_s after the last kept one.

    Only modifies the specified stim channel. Returns (n_before, removed_times),
    with removed_times in seconds from the start of the data.
    """
    ch_idx = raw.ch_names.index(stim_name)
    data = raw._data[ch_idx]
    levels = trigger_levels(data)
    min_gap = int(round(min_gap_s * raw.info["sfreq"]))

    # Rising edges: 0 -> >0 (same as the notebook, on find_events' integer levels)
    edges = np.where(np.diff(levels) > 0)[0] + 1
    if len(edges) == 0:
        print(f"No triggers found in {stim_name}.")
        return 0, []

    keep_edges = [edges[0]]  # always keep first rising edge
    removed = []
    for e in edges[1:]:
        if e - keep_edges[-1] > min_gap:
            keep_edges.append(e)
        else:
            # zero the whole duplicate pulse, so it can't re-trigger at e+1
            end = e
            while end < len(levels) and levels[end] > 0:
                end += 1
            data[e:end] = 0
            removed.append(e / raw.info["sfreq"])

    print(f"{stim_name}: {len(removed)} duplicate triggers removed, {len(keep_edges)} kept.")
    return len(edges), removed


def trial_onsets(raw, stim_name):
    """Trial onsets from the stim channel, in seconds from the start of the data."""
    events = mne.find_events(raw, stim_channel=stim_name, verbose=False)
    return (events[:, 0] - raw.first_samp) / raw.info["sfreq"]


def rebuild_trial_annotations(raw, trial_channels):
    """Replace the trial annotations with onsets from the cleaned stim channels.

    All other annotations are kept. Everything is built relative to the data
    start with orig_time=None, which set_annotations handles correctly both
    with and without meas_date (the same as mne's Raw.crop does).
    """
    a = raw.annotations
    keep = ~np.isin(a.description, trial_channels)
    onsets = list(a.onset[keep] - raw.first_time)
    durations = list(a.duration[keep])
    descriptions = list(a.description[keep])

    for ch in trial_channels:
        t = trial_onsets(raw, ch)
        onsets += list(t)
        durations += [0.0] * len(t)
        descriptions += [ch] * len(t)

    order = np.argsort(onsets, kind="stable")
    raw.set_annotations(mne.Annotations(
        np.array(onsets)[order], np.array(durations)[order], np.array(descriptions)[order],
    ))


def check_cleaned(output_file, trial_channels):
    """Reload the saved file and check annotations match the stim channels.

    Returns (counts per channel, list of problems).
    """
    raw = mne.io.read_raw_fif(output_file, preload=True, verbose=False)
    sfreq = raw.info["sfreq"]
    counts, problems = {}, []

    for ch in trial_channels:
        stim_t = trial_onsets(raw, ch)
        a = raw.annotations
        ann_t = np.sort(a.onset[a.description == ch] - raw.first_time)
        counts[ch] = len(stim_t)

        if len(stim_t) != EXPECTED_N_TRIALS:
            problems.append(f"{ch}: {len(stim_t)} stim events (expected {EXPECTED_N_TRIALS})")
        if len(ann_t) != len(stim_t):
            problems.append(f"{ch}: {len(ann_t)} annotations vs {len(stim_t)} stim events")
        elif len(stim_t) and np.max(np.abs(ann_t - stim_t)) > 1 / sfreq:
            problems.append(f"{ch}: annotation onsets differ from stim events "
                            f"(first: {ann_t[0]:.3f} s vs {stim_t[0]:.3f} s)")

    return counts, problems


def remove_duplicate_triggers(preprocfile, trial_channels, min_gap_s, output_file):
    """Clean one session. Returns a dict of results for the report."""
    raw = mne.io.read_raw_fif(preprocfile, preload=True)

    result = {"first_samp": raw.first_samp}
    for ch in trial_channels:
        n_before, removed = clean_stim_channel(raw, ch, min_gap_s)
        result[ch] = {"before": n_before, "removed": removed}

    rebuild_trial_annotations(raw, trial_channels)
    result["n_bad"] = int(sum(d.lower().startswith("bad") for d in raw.annotations.description))

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    raw.save(output_file, overwrite=True)
    print(f"Saved cleaned file to: {output_file}")

    counts, problems = check_cleaned(output_file, trial_channels)
    for ch in trial_channels:
        result[ch]["after"] = counts[ch]
    result["problems"] = problems
    return result


def write_report(report_file, rows):
    """Write one tab-separated row per session plus a summary of duplicates."""
    header = ["id", "modality", "first_samp",
              "left_before", "left_removed", "left_after",
              "right_before", "right_removed", "right_after",
              "n_BAD", "status", "notes"]
    lines = ["\t".join(header)]
    for r in rows:
        lines.append("\t".join(str(r.get(h, "")) for h in header))

    dup = [r for r in rows if r["status"] == "DUPLICATES_REMOVED"
           or (r["status"] == "COUNT_MISMATCH" and (r["left_removed"] or r["right_removed"]))]
    lines += ["", f"Sessions with duplicate triggers ({len(dup)}):"]
    lines += [f"  {r['id']}: left {r['left_removed']}, right {r['right_removed']} removed  "
              f"({r['notes']})" for r in dup]

    bad = [r for r in rows if r["status"] in ("COUNT_MISMATCH", "MISSING_FILE", "ERROR")]
    lines += ["", f"Sessions needing attention ({len(bad)}):"]
    lines += [f"  {r['id']}: {r['status']} {r['notes']}" for r in bad]

    with open(report_file, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nReport written to: {report_file}")


def main():
    # defining universal file locations
    base = "/rdrives/DRS-foundation-brain/zoe_data/BIDS"
    deriv = f"{base}/derivatives_anna_01042026"
    indir = f"{deriv}/preprocessed"
    outdir = f"{deriv}/preprocessed_cleaned"
    report_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "trigger_cleaning_report.txt")

    ALL_SUBJECTS = ['001', '002', '003', '004', '005', '006', '007', '009',
                    '010', '011', '012', '013', '014', '015', '016']
    SESSIONS = {  # session -> (modality, [left trial channel, right trial channel])
        "001": ("opm", ["Lef", "Rig"]),
        "002": ("opm", ["Lef", "Rig"]),
        "003": ("squid", ["Left_trial", "Right_trial"]),
        "004": ("squid", ["Left_trial", "Right_trial"]),
    }
    tasks = ["braille"]
    runs = ['001']

    # trials are >= 16 s apart; duplicates seen so far are ~0.01 s after the real one
    # (the notebook used 1000 samples = 4 s at 250 Hz)
    MIN_GAP_S = 4.0

    SUBJECTS_TO_RUN = ALL_SUBJECTS

    rows = []
    for subject in SUBJECTS_TO_RUN:
        for session, (modality, trial_channels) in SESSIONS.items():
            for task in tasks:
                for run in runs:

                    id = f"sub-{subject}_ses-{session}_task-{task}_run-{run}"
                    preproc_file = f"{indir}/{id}/{id}_preproc-raw.fif"
                    output_file = f"{outdir}/{id}/{id}_preproc-raw_cleaned.fif"
                    row = {"id": id, "modality": modality}

                    if not os.path.exists(preproc_file):
                        print(f"MISSING: {preproc_file}")
                        rows.append({**row, "status": "MISSING_FILE", "notes": preproc_file})
                        continue

                    try:
                        res = remove_duplicate_triggers(preproc_file, trial_channels,
                                                        MIN_GAP_S, output_file)
                    except Exception as e:
                        print(f"ERROR in {id}: {e!r}")
                        rows.append({**row, "status": "ERROR", "notes": repr(e)})
                        continue

                    left, right = (res[ch] for ch in trial_channels)
                    n_removed = len(left["removed"]) + len(right["removed"])
                    removed_times = [f"{ch}@{t:.2f}s" for ch in trial_channels
                                     for t in res[ch]["removed"]]
                    if res["problems"]:
                        status = "COUNT_MISMATCH"
                    elif n_removed:
                        status = "DUPLICATES_REMOVED"
                    else:
                        status = "OK"

                    rows.append({
                        **row,
                        "first_samp": res["first_samp"],
                        "left_before": left["before"], "left_removed": len(left["removed"]),
                        "left_after": left["after"],
                        "right_before": right["before"], "right_removed": len(right["removed"]),
                        "right_after": right["after"],
                        "n_BAD": res["n_bad"],
                        "status": status,
                        "notes": "; ".join(res["problems"] + removed_times),
                    })

    write_report(report_file, rows)


if __name__ == "__main__":
    main()
