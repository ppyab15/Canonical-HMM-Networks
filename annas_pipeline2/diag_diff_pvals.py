"""Permutation p-values for the on/off diagonal difference of the fingerprinting matrices.

Adapted from annas_pipeline/plotting_on_off_diag_diffs.ipynb (cells 1, 2 and 5),
using the matrices saved by calc_and_plot_correlations.py in
hmm_8state_correct_annotations/correlation_matrices/:
1. for each 15x15 correlation matrix, compute mean(diagonal) - mean(off diagonal)
2. build a null distribution by shuffling all matrix entries 1,000,000 times
3. compute a two-sided empirical p-value around the null mean
4. plot the null histogram with the observed value, one figure per matrix, with a
   shared x-range across the 3 comparisons (OPM1_OPM2, SQUID1_SQUID2, OPM_SQUID)
   of each analysis

Changes from the notebook:
- shuffling is vectorised (seeded) instead of a Python loop per shuffle
- matrices are grouped by analysis name rather than the last 4 filename characters
- the margin score and FWHM (computed but unused in the notebook) are dropped
- a summary table is written to correlation_matrices/diag_diff_pvals.tsv
- figures are saved to hmm_8state_correct_annotations/figures/diag_diffs_with_pvals/

Run with the annasenv1 conda env:
    conda activate annasenv1
    python diag_diff_pvals.py
"""

import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

N_SHUFFLES = 1_000_000
ANALYSES = ["trial_timecourses", "summary_stats", "psds", "transition_probs"]
COMPARISONS = ["OPM1_OPM2", "SQUID1_SQUID2", "OPM_SQUID"]


# ------------------------------------------------------------------------------
# ON/OFF DIAGONAL DIFFERENCE AND NULL DISTRIBUTION
# ------------------------------------------------------------------------------

def diag_diff(mat):
    """Mean of the diagonal minus mean of the off-diagonal entries."""
    on_diag = np.eye(mat.shape[0], dtype=bool)
    return mat[on_diag].mean() - mat[~on_diag].mean()


def shuffled_diag_diffs(mat, n_shuffles, rng, chunk=10_000):
    """diag_diff of n_shuffles matrices with all entries (diagonal included) shuffled."""
    flat = mat.ravel()
    on_diag = np.eye(mat.shape[0], dtype=bool).ravel()
    null = np.empty(n_shuffles)
    for start in range(0, n_shuffles, chunk):
        n = min(chunk, n_shuffles - start)
        shuffled = rng.permuted(np.tile(flat, (n, 1)), axis=1)  # each row permuted independently
        null[start:start + n] = shuffled[:, on_diag].mean(axis=1) - shuffled[:, ~on_diag].mean(axis=1)
    return null


def two_sided_pval(observed, null):
    """Two-sided empirical permutation p-value, measured from the null mean."""
    center = np.mean(null)
    return (1 + np.sum(np.abs(null - center) >= abs(observed - center))) / (len(null) + 1)


# ------------------------------------------------------------------------------
# PLOTTING
# ------------------------------------------------------------------------------

def plot_diag_diff(null, observed, p, xlim, name, filename):
    fig = plt.figure(figsize=(8, 6))
    plt.hist(null, bins=1000, range=xlim, label='shuffled diag diff')
    plt.axvline(
        x=observed,
        color='red',
        linestyle='--',
        linewidth=2,
        label=f"Avg on diagonal -\n avg off diagonal = {observed:.3f}",
    )
    plt.plot([], [], ' ', label=f"Two-sided p = {p:.2e}")

    plt.xlim(*xlim)
    plt.legend(prop={'size': 11}, framealpha=1, loc='upper left')
    plt.xlabel("On/off diag correlation difference", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.title(f"{name}\nCorrelation matrix diag diffs")

    fig.savefig(filename, dpi=600, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {filename}")


# ------------------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------------------

def main():
    base = "/rdrives/DRS-foundation-brain/zoe_data/BIDS"
    deriv = f"{base}/derivatives_anna_01042026"
    hmm_dir = f"{deriv}/hmm_8state_correct_annotations"
    corr_dir = f"{hmm_dir}/correlation_matrices"
    figs_dir = f"{hmm_dir}/figures/diag_diffs_with_pvals"
    os.makedirs(figs_dir, exist_ok=True)

    rng = np.random.default_rng(0)
    rows = ["analysis\tcomparison\tdiag_diff\tnull_mean\tnull_sd\tp_two_sided"]

    for analysis in ANALYSES:
        print(f"Processing {analysis}")

        # first pass: observed values, nulls and p-values
        results = []
        for comparison in COMPARISONS:
            name = f"{analysis}_{comparison}"
            mat = np.load(f"{corr_dir}/{name}.npy")
            observed = diag_diff(mat)
            null = shuffled_diag_diffs(mat, N_SHUFFLES, rng)
            p = two_sided_pval(observed, null)
            results.append((name, comparison, observed, null, p))
            rows.append(f"{analysis}\t{comparison}\t{observed:.4f}\t{null.mean():.4f}\t{null.std():.4f}\t{p:.2e}")
            print(f"  {comparison}: diag diff = {observed:.3f}, p = {p:.2e}")

        # shared x-range over the analysis' 3 comparisons, padded by 5%
        all_vals = np.concatenate([np.append(null, observed) for _, _, observed, null, _ in results])
        xmin, xmax = all_vals.min(), all_vals.max()
        pad = 0.05 * (xmax - xmin)
        xlim = (xmin - pad, xmax + pad)

        # second pass: plot
        for name, comparison, observed, null, p in results:
            plot_diag_diff(null, observed, p, xlim, name,
                           f"{figs_dir}/{name}_diag_diff_with_pval.png")

    with open(f"{corr_dir}/diag_diff_pvals.tsv", "w") as f:
        f.write("\n".join(rows) + "\n")
    print(f"Saved {corr_dir}/diag_diff_pvals.tsv")


if __name__ == "__main__":
    main()
