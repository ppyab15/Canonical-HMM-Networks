"""Plot the canonical HMM outputs (power maps, coherence networks, summary stat
violins and state PSDs) for the OPM and SQUID groups.

Adapted from annas_pipeline/plotting_chmm_outputs.ipynb, using the HMM outputs
in hmm_8state_correct_annotations/ (run apply_chmm.py first). For each group:
1. group-average state PSD
2. power maps (raw and minus the mean over states)
3. coherence networks (raw and thresholded at the 97th percentile)
4. violin plots of fo/lt/intv/sr across sessions (default and fixed y-limits)
5. state PSDs for every session (as in the notebook's single-subject cell)

Changes from the notebook:
- both groups are plotted in one run, with paths taken from the group folders
- pow_maps/coh_nets are loaded from summary_stats rather than recomputed
- the rescaled violins set the y-limit on their own axis (plt.ylim acted on
  whichever figure was current)
- every PSD figure (group averages and all sessions) matches
  annas_pipeline2/psd_template.png: (7, 4) figure, no title, axis labels 18,
  ticks 16 (y ticks at 2 dp), legend 16 in 2 columns top right. Each figure's y-limit is just
  high enough (at least 5% above its peak) that the legend covers no line.
- figures are saved to hmm_8state_correct_annotations/{opm,squid}/figures/

Run with the annasenv1 conda env:
    conda activate annasenv1
    python plot_chmm_outputs.py
"""

import os

REPO = "/home/anna-beer/Documents/anna_phd/Canonical-HMM-Networks"
os.chdir(REPO)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
from osl_dynamics.analysis import connectivity, power
from osl_dynamics.utils import plotting

# osl-dynamics' connectivity.save passes colorbar as np.bool_, which nilearn>=0.13
# rejects (it only accepts a python bool), so cast it before it reaches nilearn
_nilearn_plot_connectome = connectivity.plotting.plot_connectome
def _plot_connectome(*args, colorbar=True, **kwargs):
    return _nilearn_plot_connectome(*args, colorbar=bool(colorbar), **kwargs)
connectivity.plotting.plot_connectome = _plot_connectome

N_STATES = 8
STATE_COLOURS = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red',
                 'tab:purple', 'tab:brown', '#f084ec', '#444444']
STATE_LABELS = [f"State {i + 1}" for i in range(N_STATES)]
MASK_FILE = "MNI152_T1_8mm_brain.nii.gz"
PARCELLATION_FILE = "Glasser52_binary_space-MNI152NLin6_res-8x8x8.nii.gz"

VIOLINS = {
    # name: (y label, y-limit for the rescaled version)
    "fo": ("Fractional Occupancy", 0.4),
    "lt": ("Mean Lifetime (s)", 0.4),
    "intv": ("Mean Interval (s)", 5.25),
    "sr": ("Switching Rate (Hz)", 5),
}

# PSD figure style (matches annas_pipeline2/psd_template.png, from the
# notebook's single-subject PSD cell: osl-dynamics plot_line's (7, 4) figure)
PSD_FIGSIZE = (7, 4)
LABEL_FONTSIZE = 18
TICK_FONTSIZE = 16
LEGEND_FONTSIZE = 16
LEGEND_NCOL = 2
LEGEND_LOC = "upper right"
LEGEND_PAD_PX = 3  # keep lines at least this far from the legend box
HEADROOM_STEP = 0.05  # y-limit is raised in steps of this fraction of the peak until the legend fits
MAX_HEADROOM = 1.0


# ------------------------------------------------------------------------------
# LOADING
# ------------------------------------------------------------------------------

def load_group(group_dir):
    """Load the session ids and summary stats for one group."""
    ids = open(f"{group_dir}/sessions.txt").read().split()
    stats_dir = f"{group_dir}/summary_stats"
    stats = {name: np.load(f"{stats_dir}/{name}.npy")
             for name in ("f", "psd", "pow_maps", "coh_nets", *VIOLINS)}
    if len(stats["psd"]) != len(ids):
        raise ValueError(f"{group_dir}: {len(ids)} ids in sessions.txt but psd has {len(stats['psd'])} sessions")
    return ids, stats


# ------------------------------------------------------------------------------
# MAPS AND VIOLINS
# ------------------------------------------------------------------------------

def plot_maps(stats, figs_dir):
    p_mean = np.mean(stats["pow_maps"], axis=0)
    power.save(p_mean, mask_file=MASK_FILE, parcellation_file=PARCELLATION_FILE,
               filename=f"{figs_dir}/power_maps.png")
    power.save(p_mean, mask_file=MASK_FILE, parcellation_file=PARCELLATION_FILE,
               subtract_mean=True, filename=f"{figs_dir}/powermaps_minusmean.png")

    mean_c = np.mean(stats["coh_nets"], axis=0)
    mean_c -= np.mean(mean_c, axis=0)
    connectivity.save(mean_c, parcellation_file=PARCELLATION_FILE,
                      filename=f"{figs_dir}/connectivity_networks.png")
    thres_mean_c = connectivity.threshold(mean_c, percentile=97, absolute_value=True)
    connectivity.save(thres_mean_c, parcellation_file=PARCELLATION_FILE,
                      filename=f"{figs_dir}/connectivity_networks_thresholded.png")


def plot_violins(stats, figs_dir):
    for name, (y_label, ylim) in VIOLINS.items():
        fig, ax = plotting.plot_violin(stats[name].T, x_label="State", y_label=y_label)
        fig.savefig(f"{figs_dir}/{name}_violin_plot.png", bbox_inches="tight")
        ax.set_ylim(0, ylim)
        fig.savefig(f"{figs_dir}/{name}_violin_plot-rescaled.png", bbox_inches="tight")
        plt.close(fig)


# ------------------------------------------------------------------------------
# STATE PSDS
# ------------------------------------------------------------------------------

def make_psd_figure(f, state_psds):
    """Plot one line per state (state_psds is (n_states, n_freqs)) in the
    psd_template.png style, with the legend top right."""
    fig, ax = plt.subplots(figsize=PSD_FIGSIZE)
    for state in range(N_STATES):
        ax.plot(f, state_psds[state], color=STATE_COLOURS[state], label=STATE_LABELS[state])
    ax.set_xlim(f[0], f[-1])
    ax.set_xlabel("Frequency (Hz)", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("PSD (a.u.)", fontsize=LABEL_FONTSIZE)
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=3, steps=[1, 2, 5, 10]))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))  # always 2 dp, like the template
    leg = ax.legend(fontsize=LEGEND_FONTSIZE, ncol=LEGEND_NCOL, loc=LEGEND_LOC)
    return fig, ax, leg


def line_points_px(ax, oversample=20):
    """Points along every line in display coordinates, densely sampled so a
    line that crosses the legend between data points is still caught."""
    pts = []
    for line in ax.get_lines():
        x, y = line.get_xdata(), line.get_ydata()
        t = np.linspace(0, len(x) - 1, (len(x) - 1) * oversample + 1)
        xy = np.column_stack([np.interp(t, np.arange(len(x)), x),
                              np.interp(t, np.arange(len(y)), y)])
        pts.append(ax.transData.transform(xy))
    return np.concatenate(pts)


def legend_covers_lines(fig, ax, leg):
    """True if any line passes within LEGEND_PAD_PX of the legend box."""
    fig.canvas.draw()
    box = leg.get_window_extent(fig.canvas.get_renderer())
    points = line_points_px(ax)
    covered = ((points[:, 0] >= box.x0 - LEGEND_PAD_PX) & (points[:, 0] <= box.x1 + LEGEND_PAD_PX)
               & (points[:, 1] >= box.y0 - LEGEND_PAD_PX) & (points[:, 1] <= box.y1 + LEGEND_PAD_PX))
    return covered.any()


def fit_ylim(fig, ax, leg, peak, filename):
    """Raise the top y-limit from 5% above the peak, in HEADROOM_STEP steps,
    until the legend no longer covers a line (up to MAX_HEADROOM)."""
    headroom = 0.05
    while True:
        ax.set_ylim(0, peak * (1 + headroom))
        if not legend_covers_lines(fig, ax, leg):
            return
        if headroom >= MAX_HEADROOM:
            print(f"WARNING {filename}: legend still covers a line at {MAX_HEADROOM:.0%} headroom")
            return
        headroom += HEADROOM_STEP


def save_psd_figure(spec):
    fig, ax, leg = make_psd_figure(spec["f"], spec["state_psds"])
    fit_ylim(fig, ax, leg, spec["state_psds"].max(), spec["filename"])
    fig.savefig(spec["filename"], dpi=300, bbox_inches="tight")
    plt.close(fig)


def psd_specs(ids, stats, figs_dir):
    """One spec per PSD figure: the group average and every session."""
    f = stats["f"]
    session_psds = np.mean(stats["psd"], axis=2)  # average over parcels -> (n_sessions, n_states, n_freqs)
    group_psd = np.mean(session_psds, axis=0)

    specs = [{"f": f, "state_psds": group_psd, "filename": f"{figs_dir}/group_avg_psd.png"}]
    os.makedirs(f"{figs_dir}/psd_sessions", exist_ok=True)
    for id, state_psds in zip(ids, session_psds):
        specs.append({"f": f, "state_psds": state_psds,
                      "filename": f"{figs_dir}/psd_sessions/{id}_state_psd.png"})
    return specs


# ------------------------------------------------------------------------------

def main():
    # defining universal file locations
    base = "/rdrives/DRS-foundation-brain/zoe_data/BIDS"
    deriv = f"{base}/derivatives_anna_01042026"
    hmm_dir = f"{deriv}/hmm_8state_correct_annotations"
    GROUPS = ["opm", "squid"]

    specs = []
    for group in GROUPS:
        group_dir = f"{hmm_dir}/{group}"
        figs_dir = f"{group_dir}/figures_CHMM_outputs"
        os.makedirs(figs_dir, exist_ok=True)

        ids, stats = load_group(group_dir)
        plot_maps(stats, figs_dir)
        plot_violins(stats, figs_dir)
        specs += psd_specs(ids, stats, figs_dir)
        print(f"{group}: maps and violins saved to {figs_dir}")

    for spec in specs:
        save_psd_figure(spec)
    print(f"saved {len(specs)} PSD figures")


if __name__ == "__main__":
    main()
