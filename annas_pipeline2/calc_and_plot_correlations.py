"""Trial-locked HMM state timecourses and OPM/SQUID fingerprinting correlations.

Adapted from annas_pipeline/calc_and_plot_correlations.ipynb, using the HMM
outputs in hmm_8state_correct_annotations/ (run apply_chmm.py first):
1. epoch each session's stc into trials (0-14 s, dropping epochs that overlap
   BAD annotations) and average over trials
2. plot the OPM and SQUID group averages, sub-002 and sub-004's sessions, and
   every session
3. correlate subjects across sessions (OPM1 vs OPM2, SQUID1 vs SQUID2, OPM vs
   SQUID) using trial timecourses, summary stats, state PSDs and transition
   probabilities, and plot each as three matrices

Changes from the notebook:
- sessions are paired explicitly by subject/session, not by slicing a sorted list
- sub-002 and sub-004 are chosen by subject ID (the notebook's "sub2" plot used
  list indices 6/7, which is sub-004)
- left/right trials get distinct event ids (OPM used 3 for both)
- correlation colour ranges are the min/max over the 3 matrices, rounded out to 0.1
- all figures are saved to hmm_8state_correct_annotations/figures/

Run with the annasenv1 conda env:
    conda activate annasenv1
    python calc_and_plot_correlations.py
"""

import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mne
import numpy as np
import scipy

N_STATES = 8
TMIN, TMAX = 0, 14
TIME = np.linspace(TMIN, TMAX, 3501)
VLINES_TIMES = [1.17, 2.54, 3.91, 5.28, 6.65]  # times where you want vertical dashed lines
STATE_COLOURS = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red',
                 'tab:purple', 'tab:brown', '#f084ec', '#444444']
STATE_LABELS = [f"State {i + 1}" for i in range(N_STATES)]
TRIAL_CHANNELS = {"opm": ["Lef", "Rig"], "squid": ["Left_trial", "Right_trial"]}
MATRIX_TICK_LABELS = [1, '', 3, '', 5, '', 7, '', 9, '', 11, '', 13, '', 15]  # subjects 1-15


# ------------------------------------------------------------------------------
# LOADING AND EPOCHING
# ------------------------------------------------------------------------------

def load_session(stc_fif, trial_channels):
    """Epoch one session's stc into trials and compute its transition probabilities.

    Returns (trial timecourse (n_times, n_states), transition probs (n_states, n_states),
    n_events, n_kept, drop reasons).
    """
    raw = mne.io.read_raw_fif(stc_fif, preload=True, verbose=False)

    # distinct event ids for left (2) and right (3) trials
    events = []
    for event_value, ch in zip((2, 3), trial_channels):
        ev = mne.find_events(raw, stim_channel=ch, verbose=False)
        ev[:, 2] = event_value
        events.append(ev)
    events = np.concatenate(events, axis=0)
    events = events[np.argsort(events[:, 0])]

    epochs = mne.Epochs(
        raw,
        events=events,
        event_id={"left": 2, "right": 3},
        tmin=TMIN,
        tmax=TMAX,
        baseline=None,
        reject_by_annotation=True,
        preload=True,
        verbose=False,
    )
    epoch_data = epochs.get_data(picks="misc")  # (n_epochs, n_states, n_times)
    trial_tc = np.mean(epoch_data, axis=0).T  # (n_times, n_states)
    drop_reasons = sorted({r for log in epochs.drop_log for r in log})

    # transition probabilities from the whole session's state time course
    stc = raw.get_data(picks="misc")  # (n_states, n_samples)
    transition_counts = stc[:, :-1] @ stc[:, 1:].T
    np.fill_diagonal(transition_counts, 0)
    trans_prob = transition_counts / transition_counts.sum(axis=1, keepdims=True)

    return trial_tc, trans_prob, len(events), len(epochs), drop_reasons


def load_summary_stats(group_dir, ids):
    """Load fo/lt/intv/sr and parcel-averaged PSDs, reordered to match ids."""
    sessions = open(f"{group_dir}/sessions.txt").read().split()
    missing = [id for id in ids if id not in sessions]
    if missing:
        raise ValueError(f"{group_dir}/sessions.txt is missing {missing}")
    order = [sessions.index(id) for id in ids]

    stats_dir = f"{group_dir}/summary_stats"
    fingerprints = np.stack(
        [np.load(f"{stats_dir}/{m}.npy")[order] for m in ("fo", "lt", "intv", "sr")], axis=1,
    )  # (n_sessions, 4 measures, n_states)
    psd = np.load(f"{stats_dir}/psd.npy")[order].mean(axis=2)  # (n_sessions, n_states, n_freqs)
    return fingerprints, psd


# ------------------------------------------------------------------------------
# TRIAL TIMECOURSE PLOTS
# ------------------------------------------------------------------------------

def style_state_axis(ax, ylim):
    for t in VLINES_TIMES:
        ax.axvline(x=t, color='black', linestyle='--', linewidth=1)
    ax.set_ylim(0, ylim)
    step = 0.2 if ylim <= 0.4 else 0.4
    ax.set_yticks(np.arange(0, ylim - 1e-9, step))  # no tick at the top, so stacked rows don't overlap
    ax.set_xlim(TMIN, TMAX)


def round_up(x, step=0.1):
    return np.ceil(x / step) * step


def plot_state_columns(columns, column_titles, filename, figsize, legend=True):
    """Plot one column per timecourse (n_times, n_states), one row per state."""
    n_cols = len(columns)
    ylim = round_up(max(c.max() for c in columns))

    fig, axs = plt.subplots(N_STATES, n_cols, squeeze=False)
    fig.set_size_inches(*figsize)
    fig.supylabel('Probability', fontsize=15)
    fig.subplots_adjust(hspace=0, wspace=0.05)

    for col, tc in enumerate(columns):
        for state in range(N_STATES):
            ax = axs[state, col]
            ax.plot(TIME, tc[:, state], color=STATE_COLOURS[state])
            style_state_axis(ax, ylim)
            if state % 2 == 1:
                ax.set_facecolor('0.9')  # light grey
            if col > 0:
                ax.set_yticklabels([])
            if state < N_STATES - 1:
                ax.tick_params(axis='x', labelbottom=False)
            ax.tick_params(axis='both', labelsize=12)
        axs[0, col].set_title(column_titles[col], fontsize=14)
        axs[-1, col].set_xlabel('Time (s)', fontsize=15)

    if legend:
        for state, ax in enumerate(axs[:, -1]):
            ax.legend([STATE_LABELS[state]], loc='upper right', fontsize=12, frameon=True,
                      fancybox=True, edgecolor='black', handlelength=0, handletextpad=0)

    fig.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {filename}")


def plot_all_sessions(trial_tcs, ids, filename):
    """Plot every session's trial timecourse: one row per session, one column per state."""
    n_sessions = len(trial_tcs)
    fig, axs = plt.subplots(n_sessions, N_STATES)
    fig.set_size_inches(30, 20)
    fig.subplots_adjust(hspace=0)
    ylim = round_up(max(tc.max() for tc in trial_tcs))

    for ses, tc in enumerate(trial_tcs):
        for state in range(N_STATES):
            ax = axs[ses, state]
            ax.plot(TIME, tc[:, state], color=STATE_COLOURS[state])
            style_state_axis(ax, ylim)
            if ses != 0:
                ax.tick_params(labelleft=False)
                ax.set_yticks([])
            if ses != n_sessions - 1:
                ax.tick_params(axis='x', labelbottom=False)
            if ses == 0:
                ax.set_title(STATE_LABELS[state])
            if (ses // 2) % 2 == 1:  # shade alternate subjects (pairs of sessions)
                ax.set_facecolor('0.80')
        axs[ses, 0].set_ylabel(ids[ses].split("_task")[0], rotation=0, ha='right',
                               va='center', fontsize=9)

    axs[-1, 0].set_xlabel('Time (s)', size='large')
    fig.supylabel('Probability of the state being on', size='large')
    fig.savefig(filename, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {filename}")


# ------------------------------------------------------------------------------
# CORRELATIONS
# ------------------------------------------------------------------------------

def pearson(x, y):
    return scipy.stats.pearsonr(np.ravel(x), np.ravel(y))[0]


def corr_per_row_mean(A, B):
    """Pearson r between matching rows of A and B, averaged over rows.

    Used for trial timecourses (rows = states, after transposing), summary
    stats (rows = measures) and PSDs (rows = states).
    """
    return np.mean([pearson(a, b) for a, b in zip(A, B)])


OFF_DIAG = ~np.eye(N_STATES, dtype=bool)


def trans_rowwise_mean_corr(A, B):
    """Correlate each state's off-diagonal outgoing transition probabilities, then average."""
    return np.mean([pearson(A[r][OFF_DIAG[r]], B[r][OFF_DIAG[r]]) for r in range(N_STATES)])


def cross_corr(A_list, B_list, corr_fn):
    """Matrix of corr_fn(A_list[row], B_list[col]) (rows: session A subjects, cols: session B)."""
    mat = np.zeros((len(A_list), len(B_list)))
    for row, a in enumerate(A_list):
        for col, b in enumerate(B_list):
            mat[row, col] = corr_fn(a, b)
    return mat


def fingerprint_matrices(opm1, opm2, squid1, squid2, corr_fn):
    """OPM1 vs OPM2, SQUID1 vs SQUID2, and OPM vs SQUID (mean of the 4 cross pairings)."""
    opm_opm = cross_corr(opm1, opm2, corr_fn)
    squid_squid = cross_corr(squid1, squid2, corr_fn)
    opm_squid = np.mean([
        cross_corr(opm1, squid1, corr_fn),
        cross_corr(opm1, squid2, corr_fn),
        cross_corr(opm2, squid1, corr_fn),
        cross_corr(opm2, squid2, corr_fn),
    ], axis=0)
    return {"OPM1_OPM2": opm_opm, "SQUID1_SQUID2": squid_squid, "OPM_SQUID": opm_squid}


def mean_and_spread(vals):
    """Mean and the largest deviation from the mean."""
    m = vals.mean()
    return m, max(vals.max() - m, m - vals.min())


def plot_corr_mats(mats, filename):
    """Plot the 3 correlation matrices side by side with a shared colourbar."""
    y_axis_labels = ['OPM1', 'SQUID1', 'OPM']
    x_axis_labels = ['OPM2', 'SQUID2', 'SQUID']

    all_vals = np.concatenate([m.ravel() for m in mats.values()])
    vmax = np.ceil(np.nanmax(all_vals) * 10) / 10
    vmin = np.floor(np.nanmin(all_vals) * 10) / 10

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.subplots_adjust(wspace=0.4, right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.225, 0.02, 0.5])  # [left, bottom, width, height]

    for ax, mat, x_axis, y_axis in zip(axes, mats.values(), x_axis_labels, y_axis_labels):
        im = ax.imshow(mat, cmap='viridis', vmin=vmin, vmax=vmax)

        diag_mean, diag_range = mean_and_spread(np.diag(mat))
        off_mean, off_range = mean_and_spread(mat[~np.eye(mat.shape[0], dtype=bool)])
        diag_off_diff = diag_mean - off_mean
        ax.set_title(
            f"Within-subject correlation: {diag_mean:.3f} ± {diag_range:.3f}\n"
            f"Between-subject correlation: {off_mean:.3f} ± {off_range:.3f}\nAverage correlation margin: {diag_off_diff:.3f}",
            fontsize=12,
        )
        ax.set_xlabel(x_axis, fontsize=18)
        ax.set_ylabel(y_axis, fontsize=18)

        n = mat.shape[0]
        ax.set_xticks(np.arange(n))
        ax.set_yticks(np.arange(n))
        ax.set_xticklabels(MATRIX_TICK_LABELS)
        ax.set_yticklabels(MATRIX_TICK_LABELS)
        ax.tick_params(axis='both', labelsize=12)

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
# MAIN
# ------------------------------------------------------------------------------

def main():
    # defining universal file locations
    base = "/rdrives/DRS-foundation-brain/zoe_data/BIDS"
    deriv = f"{base}/derivatives_anna_01042026"
    hmm_dir = f"{deriv}/hmm_8state_correct_annotations"
    figs_dir = f"{hmm_dir}/figures"
    corr_dir = f"{hmm_dir}/correlation_matrices"
    os.makedirs(figs_dir, exist_ok=True)
    os.makedirs(corr_dir, exist_ok=True)

    SUBJECTS = ['001', '002', '003', '004', '005', '006', '007', '009',
                '010', '011', '012', '013', '014', '015', '016']
    SESSIONS = {"opm": ["001", "002"], "squid": ["003", "004"]}  # session 1, session 2
    SUBJECTS_TO_PLOT = ['002', '004']
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

    with open(f"{hmm_dir}/epoch_counts.txt", "w") as f:
        f.write("\n".join(count_lines) + "\n")

    # PLOT TRIAL TIMECOURSES ---------------------------------------------------
    group_means = [np.mean(trial_tcs[g][0] + trial_tcs[g][1], axis=0) for g in ("opm", "squid")]
    plot_state_columns(group_means, ["OPM", "SQUID"],
                       f"{figs_dir}/group_average_timecourses.png", figsize=(5.3, 13))

    for subject in SUBJECTS_TO_PLOT:
        i = SUBJECTS.index(subject)
        columns = [trial_tcs["opm"][0][i], trial_tcs["opm"][1][i],
                   trial_tcs["squid"][0][i], trial_tcs["squid"][1][i]]
        plot_state_columns(columns, ["OPM1", "OPM2", "SQUID1", "SQUID2"],
                           f"{figs_dir}/sub-{subject}_session_timecourses.png",
                           figsize=(10, 7), legend=False)

    for group, sessions in SESSIONS.items():
        # interleave so each subject's two sessions are next to each other
        tcs = [tc for pair in zip(*trial_tcs[group]) for tc in pair]
        ids = [session_id(s, ses) for s in SUBJECTS for ses in sessions]
        plot_all_sessions(tcs, ids, f"{figs_dir}/{group}_all_session_timecourses.png")

    # CORRELATIONS -------------------------------------------------------------
    analyses = {
        # name: (features, correlation function)
        "trial_timecourses": ({g: [[tc.T for tc in s] for s in trial_tcs[g]] for g in SESSIONS},
                              corr_per_row_mean),  # per state over time, mean over states
        "summary_stats": (fingerprints, corr_per_row_mean),  # per measure over states, mean over measures
        "psds": (psds, corr_per_row_mean),  # per state over frequency, mean over states
        "transition_probs": (trans_probs, trans_rowwise_mean_corr),  # per state row, mean over rows
    }

    for name, (features, corr_fn) in analyses.items():
        mats = fingerprint_matrices(features["opm"][0], features["opm"][1],
                                    features["squid"][0], features["squid"][1], corr_fn)
        for label, mat in mats.items():
            np.save(f"{corr_dir}/{name}_{label}.npy", mat)
        plot_corr_mats(mats, f"{figs_dir}/corr_{name}.png")


if __name__ == "__main__":
    main()
