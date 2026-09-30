"""Within-session subject-by-subject correlations, and how well they replicate across sessions.

Response-to-reviewers version of annas_pipeline2/calc_and_plot_correlations.py,
using the HMM outputs in hmm_8state_correct_annotations/ (run apply_chmm.py first):
1. epoch each session's stc into trials and load summary stats / PSDs exactly as
   in calc_and_plot_correlations.py (its functions are imported, not copied)
2. correlate subjects within each session (OPM1 vs OPM1, OPM2 vs OPM2, SQUID1 vs
   SQUID1, SQUID2 vs SQUID2) using trial timecourses, summary stats, state PSDs
   and transition probabilities; the diagonal (a recording with itself, r = 1)
   is set to NaN
3. correlate the OPM1 vs OPM1 matrix with the OPM2 vs OPM2 matrix (and the same
   for SQUID), using the upper triangle only since the matrices are symmetric

All outputs are saved to
hmm_8state_correct_annotations/response2reviewers_outputs/same_scan_corrs_outputs/
(within_session_correlations_no_sub5.py repeats steps 2-3 without subject 5)

Run with the annasenv1 conda env:
    conda activate annasenv1
    python within_session_correlations.py
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import scipy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from calc_and_plot_correlations import (  # noqa: E402
    MATRIX_TICK_LABELS,
    TRIAL_CHANNELS,
    corr_per_row_mean,
    cross_corr,
    load_session,
    load_summary_stats,
    mean_and_spread,
    trans_rowwise_mean_corr,
)


# ------------------------------------------------------------------------------
# WITHIN-SESSION CORRELATIONS
# ------------------------------------------------------------------------------

def within_session_matrices(opm1, opm2, squid1, squid2, corr_fn):
    """Subject-by-subject matrix within each session, with the diagonal set to NaN."""
    mats = {}
    for label, features in (("OPM1_OPM1", opm1), ("OPM2_OPM2", opm2),
                            ("SQUID1_SQUID1", squid1), ("SQUID2_SQUID2", squid2)):
        mat = cross_corr(features, features, corr_fn)
        np.fill_diagonal(mat, np.nan)
        mats[label] = mat
    return mats


def upper_triangle(mat):
    return mat[np.triu_indices(mat.shape[0], k=1)]


def plot_within_mats(mats, filename, tick_labels=MATRIX_TICK_LABELS, tick_labelsize=12):
    """Plot the 4 within-session matrices side by side with a shared colourbar."""
    all_vals = np.concatenate([m.ravel() for m in mats.values()])
    vmax = np.ceil(np.nanmax(all_vals) * 10) / 10
    vmin = np.floor(np.nanmin(all_vals) * 10) / 10

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.subplots_adjust(wspace=0.4, right=0.88)
    cbar_ax = fig.add_axes([0.9, 0.225, 0.015, 0.5])  # [left, bottom, width, height]

    for ax, (label, mat) in zip(axes, mats.items()):
        im = ax.imshow(mat, cmap='viridis', vmin=vmin, vmax=vmax)  # NaN diagonal is left blank

        off_mean, off_range = mean_and_spread(upper_triangle(mat))
        ax.set_title(f"Between-subject correlation:\n{off_mean:.3f} ± {off_range:.3f}", fontsize=12)
        y_axis, x_axis = label.split("_")
        ax.set_xlabel(x_axis, fontsize=18)
        ax.set_ylabel(y_axis, fontsize=18)

        n = mat.shape[0]
        ax.set_xticks(np.arange(n))
        ax.set_yticks(np.arange(n))
        ax.set_xticklabels(tick_labels)
        ax.set_yticklabels(tick_labels)
        ax.tick_params(axis='both', labelsize=tick_labelsize)

        # mark row-wise maxima
        for i in range(n):
            ax.scatter(np.nanargmax(mat[i, :]), i, color='red', marker='.', s=8, linewidths=2)

    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Pearson r', fontsize=14)
    cbar.ax.tick_params(labelsize=12)

    fig.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {filename}")


# ------------------------------------------------------------------------------
# MATRIX-TO-MATRIX CORRELATION
# ------------------------------------------------------------------------------

def plot_matrix_corrs(pairs, analysis, filename):
    """Scatter the upper triangles of session 1 vs session 2 matrices, one panel per modality."""
    fig, axes = plt.subplots(1, len(pairs), figsize=(5 * len(pairs), 5))
    for ax, (modality, (x, y, r)) in zip(axes, pairs.items()):
        ax.scatter(x, y, s=12, color='tab:blue')
        lims = [min(x.min(), y.min()), max(x.max(), y.max())]
        ax.plot(lims, lims, color='grey', linestyle='--', linewidth=1)
        ax.set_title(f"{modality}\nr = {r:.3f}", fontsize=12)
        ax.set_xlabel(f"{modality}1 vs {modality}1 (r)", fontsize=12)
        ax.set_ylabel(f"{modality}2 vs {modality}2 (r)", fontsize=12)
        ax.tick_params(axis='both', labelsize=11)
    fig.suptitle(analysis, fontsize=14)
    fig.tight_layout()
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {filename}")


def session_replication(mats, name, out_dir):
    """Correlate session 1 and session 2 matrices (upper triangles) per modality.

    Saves the scatter plot and returns the rows for matrix_corr.tsv.
    """
    pairs, rows = {}, []
    for modality in ("OPM", "SQUID"):
        x = upper_triangle(mats[f"{modality}1_{modality}1"])
        y = upper_triangle(mats[f"{modality}2_{modality}2"])
        r = scipy.stats.pearsonr(x, y)[0]
        pairs[modality] = (x, y, r)
        rows.append(f"{name}\t{modality}\t{r:.4f}\t{len(x)}")
        print(f"{name} {modality}: session 1 vs session 2 matrix r = {r:.3f}")
    plot_matrix_corrs(pairs, name, f"{out_dir}/matrix_corr_{name}.png")
    return rows


# ------------------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------------------

def main():
    # defining universal file locations
    base = "/rdrives/DRS-foundation-brain/zoe_data/BIDS"
    deriv = f"{base}/derivatives_anna_01042026"
    hmm_dir = f"{deriv}/hmm_8state_correct_annotations"
    out_dir = f"{hmm_dir}/response2reviewers_outputs/same_scan_corrs_outputs"
    os.makedirs(out_dir, exist_ok=True)

    SUBJECTS = ['001', '002', '003', '004', '005', '006', '007', '009',
                '010', '011', '012', '013', '014', '015', '016']
    SESSIONS = {"opm": ["001", "002"], "squid": ["003", "004"]}  # session 1, session 2
    task, run = "braille", "001"

    def session_id(subject, session):
        return f"sub-{subject}_ses-{session}_task-{task}_run-{run}"

    # LOAD STCS, EPOCH AND AVERAGE OVER TRIALS ---------------------------------
    # everything below is indexed [group][session index 0/1][subject index]
    trial_tcs, trans_probs, fingerprints, psds = {}, {}, {}, {}

    for group, sessions in SESSIONS.items():
        trial_tcs[group], trans_probs[group] = [], []
        fingerprints[group], psds[group] = [], []

        for session in sessions:
            ids = [session_id(s, session) for s in SUBJECTS]
            tcs, tps = [], []
            for id in ids:
                stc_fif = f"{hmm_dir}/{group}/stc/{id}_stc-raw.fif"
                tc, tp, n_events, n_kept, reasons = load_session(stc_fif, TRIAL_CHANNELS[group])
                tcs.append(tc)
                tps.append(tp)
                print(f"{id}: {n_kept}/{n_events} epochs kept {reasons if reasons else ''}")
            trial_tcs[group].append(tcs)
            trans_probs[group].append(tps)

            fp, psd = load_summary_stats(f"{hmm_dir}/{group}", ids)
            fingerprints[group].append(list(fp))
            psds[group].append(list(psd))

    # CORRELATIONS -------------------------------------------------------------
    analyses = {
        # name: (features, correlation function)
        "trial_timecourses": ({g: [[tc.T for tc in s] for s in trial_tcs[g]] for g in SESSIONS},
                              corr_per_row_mean),  # per state over time, mean over states
        "summary_stats": (fingerprints, corr_per_row_mean),  # per measure over states, mean over measures
        "psds": (psds, corr_per_row_mean),  # per state over frequency, mean over states
        "transition_probs": (trans_probs, trans_rowwise_mean_corr),  # per state row, mean over rows
    }

    rows = ["analysis\tmodality\tr\tn_pairs"]
    for name, (features, corr_fn) in analyses.items():
        mats = within_session_matrices(features["opm"][0], features["opm"][1],
                                       features["squid"][0], features["squid"][1], corr_fn)
        for label, mat in mats.items():
            np.save(f"{out_dir}/{name}_{label}.npy", mat)
        plot_within_mats(mats, f"{out_dir}/corr_within_{name}.png")

        # does the between-subject similarity structure replicate across sessions?
        rows += session_replication(mats, name, out_dir)

    with open(f"{out_dir}/matrix_corr.tsv", "w") as f:
        f.write("\n".join(rows) + "\n")
    print(f"Saved {out_dir}/matrix_corr.tsv")


if __name__ == "__main__":
    main()
