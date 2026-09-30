"""Cross-session fingerprinting analysis and permutation p-values without outlier subjects.

Repeats annas_pipeline2/calc_and_plot_correlations.py followed by
annas_pipeline2/diag_diff_pvals.py with some subjects removed (by default subject 5,
sub-005, an outlier in the PSD matrices), reusing their functions:
1. epoch each session's stc into trials and average over trials
2. correlate subjects across sessions (OPM1 vs OPM2, SQUID1 vs SQUID2, OPM vs
   SQUID) using trial timecourses, summary stats, state PSDs and transition
   probabilities, and plot each as three matrices
3. permutation p-values for mean(diagonal) - mean(off diagonal) of each matrix

The trial timecourse plots of calc_and_plot_correlations.py are not remade.

Subjects are given by the paper's 1-15 numbering (position in ALL_SUBJECTS), and
the matrix ticks use that numbering with the removed subjects left out.

All outputs are saved to
hmm_8state_correct_annotations/response2reviewers_outputs/corr_analysis_no_sub<N>/
(e.g. corr_analysis_no_sub5, corr_analysis_no_sub56) with the same layout as the
full analysis (figures/, correlation_matrices/, figures/diag_diffs_with_pvals/,
epoch_counts.txt).

Run with the annasenv1 conda env:
    conda activate annasenv1
    python corr_analysis_no_sub5.py            # drop subject 5
    python corr_analysis_no_sub5.py --drop 5 6 # drop subjects 5 and 6
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from calc_and_plot_correlations import (  # noqa: E402
    TRIAL_CHANNELS,
    corr_per_row_mean,
    fingerprint_matrices,
    load_session,
    load_summary_stats,
    plot_corr_mats,
    trans_rowwise_mean_corr,
)
from diag_diff_pvals import (  # noqa: E402
    ANALYSES,
    COMPARISONS,
    N_SHUFFLES,
    diag_diff,
    plot_diag_diff,
    shuffled_diag_diffs,
    two_sided_pval,
)

ALL_SUBJECTS = ['001', '002', '003', '004', '005', '006', '007', '009',
                '010', '011', '012', '013', '014', '015', '016']


# ------------------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--drop", type=int, nargs="+", default=[5],
                        help="subjects to remove, in the paper's 1-15 numbering (default: 5)")
    drop = sorted(set(parser.parse_args().drop))
    if not all(1 <= n <= len(ALL_SUBJECTS) for n in drop):
        parser.error(f"--drop numbers must be between 1 and {len(ALL_SUBJECTS)}")

    # paper numbering (position 1-15 in ALL_SUBJECTS) with the dropped subjects left out
    SUBJECTS = [s for i, s in enumerate(ALL_SUBJECTS) if i + 1 not in drop]
    TICK_LABELS = [i + 1 for i in range(len(ALL_SUBJECTS)) if i + 1 not in drop]
    print(f"Dropping subjects {drop}: {[ALL_SUBJECTS[n - 1] for n in drop]}")

    # defining universal file locations
    base = "/rdrives/DRS-foundation-brain/zoe_data/BIDS"
    deriv = f"{base}/derivatives_anna_01042026"
    hmm_dir = f"{deriv}/hmm_8state_correct_annotations"
    out_dir = f"{hmm_dir}/response2reviewers_outputs/corr_analysis_no_sub{''.join(map(str, drop))}"
    figs_dir = f"{out_dir}/figures"
    corr_dir = f"{out_dir}/correlation_matrices"
    pval_figs_dir = f"{figs_dir}/diag_diffs_with_pvals"
    for d in (figs_dir, corr_dir, pval_figs_dir):
        os.makedirs(d, exist_ok=True)

    SESSIONS = {"opm": ["001", "002"], "squid": ["003", "004"]}  # session 1, session 2
    task, run = "braille", "001"

    def session_id(subject, session):
        return f"sub-{subject}_ses-{session}_task-{task}_run-{run}"

    # LOAD STCS, EPOCH AND AVERAGE OVER TRIALS ---------------------------------
    # everything below is indexed [group][session index 0/1][subject index]
    trial_tcs, trans_probs, fingerprints, psds = {}, {}, {}, {}
    count_lines = ["id\tn_trials\tn_kept\tn_dropped\tdrop_reasons"]

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
                count_lines.append(f"{id}\t{n_events}\t{n_kept}\t{n_events - n_kept}\t{','.join(reasons)}")
                print(f"{id}: {n_kept}/{n_events} epochs kept {reasons if reasons else ''}")
            trial_tcs[group].append(tcs)
            trans_probs[group].append(tps)

            fp, psd = load_summary_stats(f"{hmm_dir}/{group}", ids)
            fingerprints[group].append(list(fp))
            psds[group].append(list(psd))

    with open(f"{out_dir}/epoch_counts.txt", "w") as f:
        f.write("\n".join(count_lines) + "\n")

    # CORRELATIONS -------------------------------------------------------------
    analyses = {
        # name: (features, correlation function)
        "trial_timecourses": ({g: [[tc.T for tc in s] for s in trial_tcs[g]] for g in SESSIONS},
                              corr_per_row_mean),  # per state over time, mean over states
        "summary_stats": (fingerprints, corr_per_row_mean),  # per measure over states, mean over measures
        "psds": (psds, corr_per_row_mean),  # per state over frequency, mean over states
        "transition_probs": (trans_probs, trans_rowwise_mean_corr),  # per state row, mean over rows
    }
    assert list(analyses) == ANALYSES

    all_mats = {}
    for name, (features, corr_fn) in analyses.items():
        mats = fingerprint_matrices(features["opm"][0], features["opm"][1],
                                    features["squid"][0], features["squid"][1], corr_fn)
        for label, mat in mats.items():
            np.save(f"{corr_dir}/{name}_{label}.npy", mat)
            all_mats[f"{name}_{label}"] = mat
        plot_corr_mats(mats, f"{figs_dir}/corr_{name}.png",
                       tick_labels=TICK_LABELS, tick_labelsize=9)

    # PERMUTATION P-VALUES (as in diag_diff_pvals.py) --------------------------
    rng = np.random.default_rng(0)
    rows = ["analysis\tcomparison\tdiag_diff\tnull_mean\tnull_sd\tp_two_sided"]

    for analysis in ANALYSES:
        print(f"Processing {analysis}")

        # first pass: observed values, nulls and p-values
        results = []
        for comparison in COMPARISONS:
            name = f"{analysis}_{comparison}"
            mat = all_mats[name]
            observed = diag_diff(mat)
            null = shuffled_diag_diffs(mat, N_SHUFFLES, rng)
            p = two_sided_pval(observed, null)
            results.append((name, observed, null, p))
            rows.append(f"{analysis}\t{comparison}\t{observed:.4f}\t{null.mean():.4f}\t{null.std():.4f}\t{p:.2e}")
            print(f"  {comparison}: diag diff = {observed:.3f}, p = {p:.2e}")

        # shared x-range over the analysis' 3 comparisons, padded by 5%
        all_vals = np.concatenate([np.append(null, observed) for _, observed, null, _ in results])
        xmin, xmax = all_vals.min(), all_vals.max()
        pad = 0.05 * (xmax - xmin)
        xlim = (xmin - pad, xmax + pad)

        # second pass: plot
        for name, observed, null, p in results:
            plot_diag_diff(null, observed, p, xlim, name,
                           f"{pval_figs_dir}/{name}_diag_diff_with_pval.png")

    with open(f"{corr_dir}/diag_diff_pvals.tsv", "w") as f:
        f.write("\n".join(rows) + "\n")
    print(f"Saved {corr_dir}/diag_diff_pvals.tsv")


if __name__ == "__main__":
    main()
