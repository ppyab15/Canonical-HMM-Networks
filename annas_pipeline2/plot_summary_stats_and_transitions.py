"""Plot per-subject summary stats and transition probability matrices, with all
four sessions (OPM 1, OPM 2, SQUID 1, SQUID 2) on one figure per subject.

Adapted from annas_pipeline/plotting_summary_stats_fingerprints.ipynb (cells 5
and 11), using the HMM outputs in hmm_8state_correct_annotations/ (run
apply_chmm.py first).

Changes from the notebook:
- one figure of each kind per subject, for every subject (cell 11 only plotted
  one hard-coded subject)
- sessions are matched to subjects by the ids in sessions.txt rather than by
  position (there is no sub-008), and titles show the subject
- summary stat panels have no titles; the metric name is the y-axis label
- summary stat figures are 18:16 (width:height) and use fixed margins (so the axes are the same size for
  every subject) and 2 or 3 y ticks at 1 dp on round steps (0.1, 0.2, 0.5, 1, ...)
- transition probabilities are computed from the stc npys (the same stc the
  summary stats came from); the colour limits are the min/max over the
  subject's four matrices (off-diagonal), rounded out to the nearest 0.1
- transition figures are 16:14 (width:height)
- figures are saved to hmm_8state_correct_annotations/figures/
  {summary_stats_per_subject,transition_probs_per_subject}/

Run with the annasenv1 conda env:
    conda activate annasenv1
    python plot_summary_stats_and_transitions.py
"""

import os
import re

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter

N_STATES = 8
METRICS = {
    # name: y label
    "fo": "Fractional Occupancy",
    "lt": "Mean Lifetime (s)",
    "intv": "Mean Interval (s)",
    "sr": "Switch Rate (Hz)",
}
SESSION_LABELS = {"001": "OPM 1", "002": "OPM 2", "003": "SQUID 1", "004": "SQUID 2"}
SUMMARY_ORDER = ["OPM 1", "SQUID 1", "OPM 2", "SQUID 2"]  # plot order from the notebook (sets colours)
TRANSITION_ORDER = ["OPM 1", "OPM 2", "SQUID 1", "SQUID 2"]

# summary stat figure layout (fixed margins, so every subject's axes are the same size)
SUMMARY_FIGSIZE = (9, 8.25)  # width, height
SUMMARY_LAYOUT = dict(left=0.107, right=0.982, bottom=0.08, top=0.93, wspace=0.35, hspace=0.3)
TICK_STEPS = [0.1, 0.2, 0.3, 0.4, 0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 10]  # y tick steps that read cleanly at 1 dp
MAX_TICK_INTERVALS = 2  # so every panel has 2 or 3 y ticks


# ------------------------------------------------------------------------------
# LOADING
# ------------------------------------------------------------------------------

def load_group(group_dir):
    """Return {session id: {"fo", "lt", "intv", "sr", "trans"}} for one group."""
    ids = open(f"{group_dir}/sessions.txt").read().split()
    stats = {name: np.load(f"{group_dir}/summary_stats/{name}.npy") for name in METRICS}
    for name, arr in stats.items():
        if len(arr) != len(ids):
            raise ValueError(f"{group_dir}: {len(ids)} ids in sessions.txt but {name} has {len(arr)} sessions")

    sessions = {}
    for i, id in enumerate(ids):
        sessions[id] = {name: arr[i] for name, arr in stats.items()}
        sessions[id]["trans"] = transition_probs(np.load(f"{group_dir}/stc/{id}_stc.npy"))
    return sessions


def transition_probs(stc):
    """Off-diagonal transition probabilities from a one-hot (n_samples, n_states) stc."""
    arr = stc.T
    counts = arr[:, :-1] @ arr[:, 1:].T
    np.fill_diagonal(counts, 0)
    return counts / counts.sum(axis=1, keepdims=True)


def group_by_subject(sessions):
    """Return {subject: {label: session data}}, e.g. {"001": {"OPM 1": {...}, ...}}."""
    subjects = {}
    for id, data in sessions.items():
        sub, ses = re.match(r"sub-(\d+)_ses-(\d+)", id).groups()
        subjects.setdefault(sub, {})[SESSION_LABELS[ses]] = data
    return dict(sorted(subjects.items()))


# ------------------------------------------------------------------------------
# PLOTTING
# ------------------------------------------------------------------------------

def set_1dp_yticks(ax, values):
    """Put 2 or 3 y ticks on multiples of a step that shows cleanly at 1 dp,
    choosing the step (from TICK_STEPS) whose ticks span the data most tightly,
    and snap the y-limits to those ticks (plus a small margin)."""
    lo_val, hi_val = np.min(values), np.max(values)
    best = None
    for step in TICK_STEPS:
        lo = np.floor(lo_val / step + 1e-9) * step
        hi = max(np.ceil(hi_val / step - 1e-9) * step, lo + step)  # at least 2 ticks
        if (hi - lo) / step <= MAX_TICK_INTERVALS + 1e-9 and (best is None or hi - lo < best[2] - best[1] - 1e-9):
            best = (step, lo, hi)
    step, lo, hi = best
    ax.set_yticks(np.arange(lo, hi + step / 2, step))
    margin = 0.05 * (hi - lo)
    ax.set_ylim(lo - margin, hi + margin)
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))


def plot_summary_stats(subject, sessions, filename):
    fig, axes = plt.subplots(2, 2, figsize=SUMMARY_FIGSIZE)
    fig.subplots_adjust(**SUMMARY_LAYOUT)  # fixed, so the axes are the same size on every figure
    x = np.arange(N_STATES)

    for ax, (name, y_label) in zip(axes.flat, METRICS.items()):
        for label in SUMMARY_ORDER:
            if label in sessions:
                marker = "o-" if label.startswith("OPM") else "x-"
                ax.plot(x, sessions[label][name], marker, label=label)

        set_1dp_yticks(ax, [sessions[label][name] for label in SUMMARY_ORDER if label in sessions])
        ax.set_ylabel(y_label, fontsize=16)
        ax.set_xlabel("State", fontsize=16)
        ax.set_xticks(x)
        ax.set_xticklabels([str(i + 1) for i in x], fontsize=16)
        ax.tick_params(labelsize=16)

    axes.flat[2].legend(fontsize=12)
    fig.suptitle(f"Subject {subject}", fontsize=18)
    fig.savefig(filename, dpi=300)  # no bbox_inches="tight", which would crop each figure differently
    plt.close(fig)


def plot_transition_probs(subject, sessions, filename):
    mats = {}
    for label in TRANSITION_ORDER:
        if label in sessions:
            mat = sessions[label]["trans"].copy()
            np.fill_diagonal(mat, np.nan)  # white-out diagonal
            mats[label] = mat
    vmin = np.floor(np.nanmin(list(mats.values())) * 10) / 10
    vmax = np.ceil(np.nanmax(list(mats.values())) * 10) / 10

    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color="white")

    fig, axes = plt.subplots(2, 2, figsize=(8, 7), constrained_layout=True)
    for ax, label in zip(axes.flat, TRANSITION_ORDER):
        if label not in mats:
            ax.axis("off")
            continue
        im = ax.imshow(mats[label], cmap=cmap, aspect="equal", vmin=vmin, vmax=vmax)
        ax.set_title(label, fontsize=22, fontweight="bold")
        ax.set_xticks(np.arange(N_STATES))
        ax.set_yticks(np.arange(N_STATES))
        ax.set_xticklabels(np.arange(1, N_STATES + 1), fontsize=16)
        ax.set_yticklabels(np.arange(1, N_STATES + 1), fontsize=16)

    # only label the bottom-left subplot
    axes.flat[2].set_xlabel("State transitioning to", fontsize=22, labelpad=20)
    axes.flat[2].set_ylabel("State transitioning from", fontsize=22, labelpad=20)

    cbar = fig.colorbar(im, ax=axes, location="right", shrink=0.75, pad=0.08)
    cbar.set_label("Transition probability", fontsize=22, labelpad=18)
    cbar.ax.tick_params(labelsize=16)

    fig.suptitle(f"Subject {subject}", fontsize=22)
    fig.savefig(filename, dpi=300)  # no bbox_inches="tight", so the saved image keeps the 8:7 ratio
    plt.close(fig)


# ------------------------------------------------------------------------------

def main():
    # defining universal file locations
    base = "/rdrives/DRS-foundation-brain/zoe_data/BIDS"
    deriv = f"{base}/derivatives_anna_01042026"
    hmm_dir = f"{deriv}/hmm_8state_correct_annotations"
    GROUPS = ["opm", "squid"]

    summary_dir = f"{hmm_dir}/figures/summary_stats_per_subject"
    trans_dir = f"{hmm_dir}/figures/transition_probs_per_subject"
    os.makedirs(summary_dir, exist_ok=True)
    os.makedirs(trans_dir, exist_ok=True)

    sessions = {}
    for group in GROUPS:
        sessions.update(load_group(f"{hmm_dir}/{group}"))
    subjects = group_by_subject(sessions)

    for subject, subject_sessions in subjects.items():
        missing = [label for label in SESSION_LABELS.values() if label not in subject_sessions]
        if missing:
            print(f"WARNING sub-{subject}: missing {', '.join(missing)}")
        plot_summary_stats(subject, subject_sessions, f"{summary_dir}/sub-{subject}_summary_stats.png")
        plot_transition_probs(subject, subject_sessions, f"{trans_dir}/sub-{subject}_transition_probs.png")
        print(f"saved sub-{subject}")

    print(f"saved {len(subjects)} subjects to {summary_dir} and {trans_dir}")


if __name__ == "__main__":
    main()
