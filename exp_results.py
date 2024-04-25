# %% Packages
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from mne import concatenate_epochs, read_epochs
from scipy.stats import pearsonr, sem, t, ttest_rel
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from statsmodels.api import qqplot

from session_manager import CMD_MAP, FREQS

warnings.simplefilter("ignore", category=(UserWarning, FutureWarning))

sns.set_style("ticks", {"axes.grid": False})
sns.set_context("paper")
sns.set_palette("colorblind")


# %% Functions
def plot_confusion_matrix(cm, labels, ax, cmap="Blues"):
    """Confusion matrix heatmap averaged across participants with standard error.

    Args:
        cm (list of np.array): list of confusion matrices
        labels (list of str): class labels
        ax (axes): figure axes
        cmap (str, optional): colour map. Defaults to "Blues".

    Returns:
        axes: heatmap
    """
    mean_cm = np.nanmean(cm, axis=0)
    if len(cm) == 1:
        annots = np.round(mean_cm, 1).astype(str)
    else:
        sterr_cm = np.std(cm, axis=0) / np.sqrt(len(cm))
        annots = [
            [
                "%.1f\n(%.1f)" % (mean_cm[_r, _c], sterr_cm[_r, _c])
                for _c in range(len(labels))
            ]
            for _r in range(len(labels))
        ]
    sns.heatmap(
        mean_cm,
        annot=annots,
        fmt="s",
        cmap=cmap,
        cbar=False,
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        linewidths=0.5,
        vmax=100,
        vmin=0,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    return ax


def plot_lines(
    data_pts,
    data_xs,
    x,
    y,
    ax,
    hue,
    order,
    col,
    ylabel,
    ylim,
    pt_size=3,
    x_size=5,
    alpha=0.8,
    dodge=False,
):
    """Plot data points for results across different modes connected by lines using
    two different styles for best and worst participants.

    Args:
        data_pts (DataFrame): Results for best participants
        data_xs (DataFrame): Results for worst participants
        x (str): x-axis
        y (str): y-axis
        ax (axes): figure axes
        hue (str): participant id
        order (list of str): order of conditions
        col (tuple): colour
        ylabel (str): y axis label
        ylim (list of float): y axis limits
        pt_size (float, optional): point marker size. Defaults to 3.
        x_size (float, optional): cross marker size . Defaults to 5.
        alpha (float, optional): opacity. Defaults to 0.8.
        dodge (bool, optional): dodge points. Defaults to False.
    """
    for data, marker, ls, size in zip(
        [data_pts, data_xs], ["o", "x"], ["-", "--"], [pt_size, x_size]
    ):
        sns.pointplot(
            data=data,
            x=x,
            y=y,
            ax=ax,
            hue=hue,
            order=order,
            palette=[col for _ in order],
            markers=marker,
            linestyle=ls,
            dodge=dodge,
            markersize=size,
            alpha=alpha,
        )
    ax.set_ylim(ylim)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("")
    ax.legend().remove()


def plot_CI(
    data_pts,
    data_xs,
    x,
    y,
    ax,
    hue,
    order,
    col,
    ylabel,
    ylim,
    pt_size=3,
    x_size=5,
    alpha=0.5,
    ci=95,
):
    """Plot points and confidence interval using two different styles for best and worst participants.

    Args:
        data_pts (DataFrame): Results for best participants
        data_xs (DataFrame): Results for worst participants
        x (str): x-axis
        y (str): y-axis
        ax (axes): figure axes
        hue (str): participant id
        order (list of str): order of conditions
        col (tuple): colour
        ylabel (str): y axis label
        ylim (list of float): y axis limits
        pt_size (float, optional): point marker size. Defaults to 3.
        x_size (float, optional): cross marker size . Defaults to 5.
        alpha (float, optional): opacity. Defaults to 0.8.
        ci (int, optional): size of confidence interval. Defaults to 95.
    """
    # mean and confidence interval
    sns.pointplot(
        data=pd.concat([data_pts, data_xs]),
        x=x,
        y=y,
        ax=ax,
        order=order,
        errorbar=("ci", ci),
        markers="D",
        palette=[col for _ in order],
    )
    # individual data points
    plot_lines(
        data_pts,
        data_xs,
        x,
        y,
        ax,
        hue,
        order,
        col,
        ylabel,
        ylim,
        pt_size=pt_size,
        x_size=x_size,
        alpha=alpha,
        dodge=True,
    )


def plot_reg(
    data_pts, data_xs, x, y, ax, col, xlabel, ylabel, xlim, ylim, pt_size=3, x_size=5
):
    """Plot linear regression lines of y vs x for best and worst participants.

    Args:
        data_pts (DataFrame): Results for best participants
        data_xs (DataFrame): Results for worst participants
        x (str): x-axis
        y (str): y-axis
        ax (axes): figure axes
        col (tuple): colour
        xlabel (str): x-axis label
        ylabel (str): y-axis label
        xlim (list of float): x-axis limits
        ylim (list of float): y-axis limits
        pt_size (float, optional): point marker size. Defaults to 3.
        x_size (float, optional): cross marker size . Defaults to 5.
    """
    cmb_data = pd.concat([data_pts, data_xs])
    for data, marker, fit, size in zip(
        [cmb_data, data_pts, data_xs],
        ["", "o", "x"],
        [True, False, False],
        [pt_size, pt_size, x_size],
    ):
        sns.regplot(
            data=data,
            x=x,
            y=y,
            ax=ax,
            color=col,
            marker=marker,
            fit_reg=fit,
            scatter_kws={"s": size},
        )

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    r, p = pearsonr(cmb_data[x], cmb_data[y])
    print("-" * 40)
    print(f"correlation, r:{r:.3f}, p:{p:.3f}")


def plot_box(data, x, y, ax, order, col, ylabel, ylim, hue=None):
    """_summary_

    Args:
        data (DataFrame): Results
        x (str): x axis
        y (str): y axis
        ax (axes): figure axes
        order (list of str): order of conditions
        col (tuple): colour
        ylabel (str): y-axis label
        ylim (_type_): y-axis limits
        hue (_type_, optional): groups. Defaults to None.

    Returns:
        _type_: _description_
    """
    sns.boxplot(
        data=data,
        x=x,
        y=y,
        whis=(0, 100),
        showmeans=True,
        meanprops={
            "marker": "o",
            "markerfacecolor": "white",
            "markeredgecolor": "black",
            "markersize": "4",
        },
        hue=hue,
        order=order,
        color=col,
        ax=ax,
    )
    ax.set_ylabel(ylabel)
    ax.set_xlabel("")
    ax.set_ylim(ylim)
    return ax


def run_ttest(a, b, ci=0.95, plot_qq=False):
    """Run a paired two-sided t-test.

    Args:
        a (array): group A
        b (array): grpup B
        ci (float, optional): size of confidence interval. Defaults to 0.95.
        plot_qq (bool, optional): plot quantile-quantile plot. Defaults to False.
    """
    if plot_qq:
        qqplot(a - b, line="s")
    ttest = ttest_rel(a, b, alternative="two-sided")
    ci = ttest.confidence_interval()
    print("-" * 40)
    print(
        f"t:{ttest.statistic:.3f}, p:{ttest.pvalue:.4f}, ci:({ci.low:.3f},{ci.high:.3f})"
    )


def get_sample_CI(a, ci=0.95):
    """Compute sample confidence interval.

    Args:
        a (np.array): scores
        ci (float, optional): size of confidence interval. Defaults to 0.95.

    Returns:
        _type_: _description_
    """
    return t.interval(ci, len(a) - 1, loc=np.mean(a), scale=sem(a))


def get_ITR(p, n, t):
    """Calculate information transfer rate in bits/min.

    Args:
        p (float): accuracy
        n (int): number of selections
        t (float): time per selection in minutes

    Returns:
        float: ITR in bits/min
    """
    return (np.log2(n) + p * np.log2(p) + (1 - p) * np.log2((1 - p) / (n - 1))) / t


# %% Constants
CMDS = list(CMD_MAP.keys())
FOLDER = r"C:\Users\Kirill Kokorin\OneDrive - synchronmed.com\SSVEP robot control\Data\Experiment\Processed"
P_IDS = [
    "P2",
    "P3",
    "P5",
    "P6",
    "P7",
    "P8",
    "P9",
    "P10",
    "P11",
    "P13",
    "P14",
    "P15",
    "P16",
    "P17",
    "P18",
    "P1",
    "P12",
    "P4",
]
P_ID_LOW = ["P7", "P9"]
P_ID_HIGH = list(set(P_IDS) - set(P_ID_LOW))
F_LAYOUTS = [
    [7, 13, 11, 8, 9],
    [8, 13, 11, 9, 7],
    [13, 9, 11, 8, 7],
    [13, 7, 8, 9, 11],
    [9, 8, 11, 7, 13],
    [8, 9, 13, 7, 11],
    [11, 9, 8, 7, 13],
    [13, 7, 8, 9, 11],
    [11, 13, 8, 7, 9],
    [9, 13, 11, 7, 8],
    [7, 8, 11, 9, 13],
    [8, 7, 9, 11, 13],
    [9, 11, 8, 13, 7],
    [13, 11, 8, 9, 7],
    [8, 7, 9, 13, 11],
    [8, 7, 13, 11, 9],
    [11, 9, 13, 8, 7],
    [8, 9, 7, 13, 11],
]

# %% Load participant data
reach_blocks = [3, 5, 6, 7]
p_data = pd.concat(
    [
        pd.read_csv(FOLDER + "//tstep//" + p_id + "_results_tstep.csv", index_col=0)
        for p_id in P_IDS
    ]
)

# observation trials
obs_data = p_data[p_data.block == "OBS"].copy().reset_index()

# reaching trials
reach_data = p_data[p_data.block_i.isin(reach_blocks)].copy().reset_index()
reach_data["goal"] = reach_data["goal"].astype(int)
reach_data["pred_obj"] = reach_data["pred_obj"].astype(int)
reach_data["pred_success"] = reach_data.goal == reach_data.pred_obj
reach_trials = reach_data.groupby(["p_id", "block_i", "block", "goal", "trial"])

# %% Decoding performance
fig, axs = plt.subplots(1, 2, figsize=(5.6, 2.8))

# by direction
accs = []
cms = []
all_labels = []
for p_id in P_IDS:
    preds = obs_data[obs_data.p_id == p_id].pred
    labels = obs_data[obs_data.p_id == p_id].goal
    all_labels.append(labels.value_counts())
    accs.append(balanced_accuracy_score(labels, preds) * 100)
    cms.append(confusion_matrix(labels, preds, normalize="true", labels=CMDS) * 100)

plot_confusion_matrix(cms, CMDS, axs[0], cmap="Blues")

# by frequency
f_cms = []
for p_id, p_freqs in zip(P_IDS, F_LAYOUTS):
    freq_map = {_d: _f for _f, _d in zip(p_freqs, CMD_MAP.keys())}
    preds = [freq_map[_p] for _p in obs_data[obs_data.p_id == p_id].pred]
    labels = [freq_map[_l] for _l in obs_data[obs_data.p_id == p_id].goal]
    f_cms.append(confusion_matrix(labels, preds, normalize="true", labels=FREQS) * 100)

plot_confusion_matrix(f_cms, FREQS, axs[1], cmap="Purples")

fig.tight_layout()
# plt.savefig(FOLDER + "//Figures//decoding_cms.svg", format="svg")

# %% Offline decoding correlations
fig, axs = plt.subplots(1, 2, figsize=(5, 1.5), sharex=True, sharey=True)

for ax, f_trial in zip(axs.flatten(), [8, 13]):
    rho_trials = pd.read_csv(FOLDER + "//rhos.csv", index_col=0)
    f_bins = rho_trials.columns[:-32]

    rho_counts = []
    for p_id in P_IDS:
        rho_13 = rho_trials[
            (rho_trials.p_id == p_id) & (rho_trials.ep_freq == f_trial)
        ][f_bins].values
        max_freqs = np.array(f_bins)[np.argmax(rho_13, axis=1)]
        freq_counts = {np.round(float(_f), 1): np.sum(max_freqs == _f) for _f in f_bins}
        rho_counts.append(freq_counts | {"p_id": p_id})

    rho_count_df = pd.DataFrame.from_dict(rho_counts).melt(id_vars="p_id")
    rho_count_df["pc"] = rho_count_df.value / 70 * 100
    sns.lineplot(
        data=rho_count_df,
        x="variable",
        y="pc",
        ax=ax,
        color=sns.color_palette()[-1],
    )
    ax.axvline(f_trial, c="k", linestyle="--", alpha=0.25)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("")
    ax.set_xlim([7, 16])
    ax.set_ylim([0, 60])

axs[0].set_ylabel("Chunks (%)")
sns.despine()
fig.tight_layout()
# plt.savefig(FOLDER + "//Figures//8_13Hz_offline.svg", format="svg")

# %% PSD
ch_i = 1  # Oz
fmin = 5
fmax = 30
fig, axs = plt.subplots(5, 1, figsize=(5, 2.5), sharex=True, sharey=True)

all_eps = concatenate_epochs(
    [
        read_epochs(FOLDER + "//epochs//" + _p + "_obs-epo.fif", preload=True)
        for _p in P_IDS
    ]
)
for f, ax in zip(FREQS, axs):
    psd = all_eps[str(f)].compute_psd(
        fmin=fmin, fmax=fmax, method="welch", remove_dc=False
    )
    sns.lineplot(
        x=psd.freqs,
        y=np.mean(psd.get_data()[:, ch_i, :], axis=0) / 1e-12,
        ax=ax,
        label=f,
        c=sns.color_palette()[9],
    )
    [ax.axvline(_f, c="k", alpha=0.25) for _f in [f, 2 * f]]
    ax.legend(title="", loc="upper left", bbox_to_anchor=(1, 1))
    ax.set_ylabel("")
    ax.set_ylim([0, 8])

axs[4].set_ylabel("PSD")
axs[4].set_xlabel("Frequency (Hz)")
sns.despine()
fig.tight_layout()
# plt.savefig(FOLDER + "//Figures//Oz_psd.pdf", format="pdf")

# %% Success rates
trial_success = reach_trials.success.first().reset_index()
success = (
    trial_success.groupby(["p_id", "block"]).success.sum() * 100 / 24
).reset_index()
c = sns.color_palette()[2]
fig, axs = plt.subplots(1, 2, figsize=(3.5, 2.5), width_ratios=[2, 1])

# success rate
plot_lines(
    success[success.p_id.isin(P_ID_HIGH)],
    success[success.p_id.isin(P_ID_LOW)],
    "block",
    "success",
    axs[0],
    "p_id",
    ["DC", "SC"],
    c,
    "Success rate (%)",
    [-5, 105],
)

# change in success rate
success_wide = success.pivot(
    index="p_id", columns="block", values="success"
).reset_index()
success_wide["SC-DC"] = success_wide["SC"] - success_wide["DC"]
success = pd.melt(
    success_wide,
    id_vars="p_id",
    value_vars=["DC", "SC", "SC-DC"],
    var_name="block",
    value_name="success",
)
plot_CI(
    success[success.p_id.isin(P_ID_HIGH)],
    success[success.p_id.isin(P_ID_LOW)],
    "block",
    "success",
    axs[1],
    "p_id",
    ["SC-DC"],
    c,
    "$\Delta$ Success rate (%)",
    [-5, 80],
)
sns.despine()
fig.tight_layout()
# plt.savefig(FOLDER + "//Figures//success.svg", format="svg")

# compare success rates
run_ttest(
    success[success.block == "SC"]["success"].values,
    success[success.block == "DC"]["success"].values,
)

# %% Success rate bars
fig, axs = plt.subplots(1, 1, figsize=(1.3, 2.5))
mean_marker = {
    "marker": "o",
    "markerfacecolor": "white",
    "markeredgecolor": "black",
    "markersize": "4",
}
sns.boxplot(
    data=success,
    x="block",
    y="success",
    ax=axs,
    order=["DC", "SC"],
    palette=[c, c],
    showmeans=True,
    whis=(0, 100),
    meanprops=mean_marker,
)
axs.set_ylim([-5, 100])
axs.set_ylabel("Success rate (%)")
axs.set_xlabel("")
axs.legend().remove()

sns.despine()
fig.tight_layout()
# plt.savefig(FOLDER + "//Figures//success_bars.svg", format="svg")

# %% Trajectory lengths (cm)
trial_lens = (reach_trials["dL"].sum() / 10).reset_index()
lens = (
    trial_lens[trial_success.success == 1]
    .groupby(["p_id", "block", "goal"])
    .dL.mean()
    .reset_index()
)

# remove objects that have no successful DC or SC trials
len_valid = lens[lens.block == "DC"].merge(
    lens[lens.block == "SC"], on=["p_id", "goal"]
)
len_valid_wide = len_valid.groupby(["p_id"])[["dL_x", "dL_y"]].mean().reset_index()
len_valid_wide["SC-DC"] = len_valid_wide["dL_y"] - len_valid_wide["dL_x"]
len_valid_wide.columns = ["p_id", "DC", "SC", "SC-DC"]
len_valid = pd.melt(
    len_valid_wide,
    id_vars="p_id",
    value_vars=["DC", "SC", "SC-DC"],
    var_name="block",
    value_name="dL",
)

# length
c = sns.color_palette()[1]
fig, axs = plt.subplots(1, 2, figsize=(3.5, 2.5), width_ratios=[2, 1])
plot_lines(
    len_valid[lens.p_id.isin(P_ID_HIGH)],
    lens[lens.p_id.isin(P_ID_LOW)]
    .groupby(["p_id", "block"])["dL"]
    .mean()
    .reset_index(),
    "block",
    "dL",
    axs[0],
    "p_id",
    ["DC", "SC"],
    c,
    "Trajectory length (cm)",
    [28, 73],
)

# change in length
plot_CI(
    len_valid[len_valid.p_id.isin(P_ID_HIGH)],
    len_valid[len_valid.p_id.isin(P_ID_LOW)],
    "block",
    "dL",
    axs[1],
    "p_id",
    ["SC-DC"],
    c,
    "$\Delta$ Trajectory length (cm)",
    [-20, 0],
)
axs[1].set_yticks(range(-20, 1, 5))
sns.despine()
fig.tight_layout()
# plt.savefig(FOLDER + "//Figures//lengths.svg", format="svg")

# compare trajectory length
run_ttest(
    len_valid[len_valid.block == "SC"]["dL"].values,
    len_valid[len_valid.block == "DC"]["dL"].values,
)

# %% Success rate, trajectory length and predictions by object
fig, axs = plt.subplots(3, 1, figsize=(5, 3.5), sharex=True)
objs = np.arange(9)
obj_labels = ["TL", "TM", "TR", "ML", "M", "MR", "BL", "BM", "BR"]

# success rate
obj_success = (
    trial_success.groupby(["p_id", "block", "goal"]).success.sum() / 6 * 100
).reset_index()
plot_box(
    obj_success[obj_success.block == "DC"],
    "goal",
    "success",
    axs[0],
    objs,
    sns.color_palette()[2],
    "Success\nrate (%)",
    [-5, 105],
)

# trajectory length
plot_box(
    lens[lens.block == "DC"],
    "goal",
    "dL",
    axs[1],
    objs,
    sns.color_palette()[1],
    "Trajectory\nlength (cm)",
    [20, 80],
)

# correct SC predictions
trial_recall = (
    (reach_trials.pred_success.sum() / reach_trials.pred_success.count()) * 100
).reset_index()

plot_box(
    trial_recall[trial_recall.block == "SC"],
    "goal",
    "pred_success",
    axs[2],
    objs,
    sns.color_palette()[7],
    "Correct\npredictions (%)",
    [-5, 105],
)

axs[2].set_xlabel("Object")
for ax in axs:
    ax.set_xticklabels(obj_labels)
sns.despine()
fig.tight_layout()
# plt.savefig(FOLDER + "//Figures//obj_results.svg", format="svg")

# %% Impact of decoding accuracy
fig, axs = plt.subplots(1, 2, figsize=(5, 2), sharex=True)
acc_df = pd.DataFrame({"p_id": P_IDS, "acc": accs, "mode": [""] * len(P_IDS)})

# success rate
acc_vs_dc_df = pd.merge(success[success.block == "DC"], acc_df)
acc_vs_dc_df_best = acc_vs_dc_df[acc_vs_dc_df.p_id.isin(P_ID_HIGH)]
acc_vs_dc_df_worst = acc_vs_dc_df[acc_vs_dc_df.p_id.isin(P_ID_LOW)]
plot_reg(
    acc_vs_dc_df_best,
    acc_vs_dc_df_worst,
    "acc",
    "success",
    axs[0],
    sns.color_palette()[0],
    "Accuracy (%)",
    "DC success rate (%)",
    [25, 100],
    [-5, 100],
    pt_size=10,
    x_size=30,
)

# change in success rate
acc_vs_sc_df = pd.merge(success[success.block == "SC-DC"], acc_df)

acc_vs_sc_df_best = acc_vs_sc_df[acc_vs_sc_df.p_id.isin(P_ID_HIGH)]
acc_vs_sc_df_worst = acc_vs_sc_df[acc_vs_sc_df.p_id.isin(P_ID_LOW)]
plot_reg(
    acc_vs_sc_df_best,
    acc_vs_sc_df_worst,
    "acc",
    "success",
    axs[1],
    sns.color_palette()[0],
    "Accuracy (%)",
    "$\Delta$ Success rate (%)",
    [25, 100],
    [-20, 80],
    pt_size=10,
    x_size=30,
)

fig.tight_layout()
sns.despine()
# plt.savefig(FOLDER + "//Figures//corrs.svg", format="svg")

# %% Offline decoding recall with variable window sizes
window_df = pd.read_csv(FOLDER + "//variable_window.csv", index_col=None)
fig, axs = plt.subplots(1, 2, figsize=(5, 1.5), sharey=True)
for p_id, ax in zip(window_df["p_id"].unique(), axs):
    data = window_df[window_df.p_id == p_id].copy()
    sns.pointplot(
        data=data,
        ax=ax,
        x="window_s",
        y="recall",
        hue="freq",
        join=True,
        markersize=3,
        # markers="x",
    )

    r, p = pearsonr(data.window_s, data.recall)
    print(f"P{p_id} recall vs window size correlation, r:{r:.3f}, p: {p:.3f}")
    # ax.set_title(f"P{p_id}")
    ax.set_xlabel("Window size (s)")
    ax.set_ylim([0, 105])
    ax.axhline(20, linestyle="--", color="k", alpha=0.25)

axs[0].set_ylabel("Recall (%)")
axs[0].legend().remove()
axs[1].legend(title="", ncol=1, loc="upper left", bbox_to_anchor=(1, 1))
axs[1].set_ylabel("")

fig.tight_layout()
sns.despine()
# plt.savefig(FOLDER + "//Figures//P7_P9_window.svg", format="svg")

# %% Summary of decoding performance and failure analysis
fig, axs = plt.subplots(1, 3, figsize=(5, 1.5), width_ratios=[1, 1, 2.5])
result_df = pd.read_csv(FOLDER + "//trial_results.csv", index_col=None)
result_df["bounds"] = result_df["collide"] + result_df["near"]

# decoding accuracy
plot_CI(
    acc_df[acc_df.p_id.isin(P_ID_HIGH)],
    acc_df[acc_df.p_id.isin(P_ID_LOW)],
    "mode",
    "acc",
    axs[0],
    "p_id",
    [""],
    sns.color_palette()[0],
    "Accuracy (%)",
    [0, 100],
)
print("ci:", get_sample_CI(acc_df["acc"].values))

# information transfer rate
acc_df["ITR"] = get_ITR(acc_df["acc"] / 100, 5, 1 / 60)
plot_CI(
    acc_df[acc_df.p_id.isin(P_ID_HIGH)],
    acc_df[acc_df.p_id.isin(P_ID_LOW)],
    "mode",
    "ITR",
    axs[1],
    "p_id",
    [""],
    sns.color_palette()[3],
    "ITR (bits/min)",
    [0, 150],
)
print("ci:", get_sample_CI(acc_df["ITR"].values))

# failures
failures_df = pd.melt(
    result_df, id_vars=["p_id", "mode"], value_vars=["bounds", "long", "wrong_obj"]
)
failures_df = failures_df.groupby(["mode", "variable"])["value"].sum().reset_index()
sns.barplot(
    data=failures_df,
    x="variable",
    y="value",
    hue="mode",
    ax=axs[2],
    order=["bounds", "long", "wrong_obj"],
    palette=[sns.color_palette()[7], sns.color_palette()[8]],
)
axs[2].set_ylim([0, 150])
axs[2].set_ylabel("Total failures")
axs[2].set_xlabel("")
axs[2].set_xticklabels(["Bounds", "Length", "Object"])
axs[2].legend(title="")

sns.despine()
fig.tight_layout()
# plt.savefig(FOLDER + "//Figures//acc_itr_fail.svg", format="svg")

# %% Workload
hf_df = pd.read_csv(FOLDER + "//questionnaire.csv", index_col=None)
fig, axs = plt.subplots(1, 1, figsize=(2, 2))

# factor tally
factors = ["Mental", "Physical", "Temporal", "Performance", "Effort", "Frustration"]
factor_labels = ["MD", "PD", "TD", "P", "E", "F"]
sns.barplot(
    data=hf_df.melt(id_vars=["ID"], value_vars=factors),
    x="variable",
    y="value",
    ax=axs,
    errorbar="se",
    order=factors,
    color=sns.color_palette()[3],
)
axs.set_ylabel("Weight")
axs.set_ylim([0, 5])
axs.set_xticklabels(factor_labels)
axs.set_xlabel("Factor")

# workload
fig, axs = plt.subplots(1, 2, figsize=(3.5, 2.5), width_ratios=[2, 1])
wl_df = hf_df.melt(id_vars=["ID"], value_vars=["DC Total", "SC Total"]).copy()
wl_df["mode"] = wl_df["variable"].apply(lambda x: x.split(" ")[0])
hf_df["dWL"] = hf_df["SC Total"] - hf_df["DC Total"]
hf_df["mode"] = "SC-DC"
c = sns.color_palette()[4]

plot_lines(
    wl_df[wl_df.ID.isin(P_ID_HIGH)],
    wl_df[wl_df.ID.isin(P_ID_LOW)],
    "mode",
    "value",
    axs[0],
    "ID",
    ["DC", "SC"],
    c,
    "Workload",
    [-5, 105],
)
plot_CI(
    hf_df[hf_df.ID.isin(P_ID_HIGH)],
    hf_df[hf_df.ID.isin(P_ID_LOW)],
    "mode",
    "dWL",
    axs[1],
    "ID",
    ["SC-DC"],
    c,
    "$\Delta$ Workload",
    [-30, 30],
)

sns.despine()
fig.tight_layout()
# plt.savefig(FOLDER + "//Figures//workload.svg", format="svg")

run_ttest(hf_df["SC Total"].values, hf_df["DC Total"].values)

# %% Embodiment
fig, axs = plt.subplots(1, 1, figsize=(4, 4))
emb_df = hf_df.melt(
    id_vars=["ID"],
    value_vars=np.array(
        [["SC Q%d" % _i, "DC Q%d" % _i] for _i in range(1, 10)]
    ).flatten(),
).copy()
emb_df["mode"] = emb_df["variable"].apply(lambda x: x.split(" ")[0])
emb_df["Q"] = emb_df["variable"].apply(lambda x: x.split(" ")[1])
emb_df["value"] = (emb_df["value"] - 3) * -1

sns.barplot(
    data=emb_df,
    x="value",
    y="Q",
    orient="h",
    hue="mode",
    errorbar="se",
    ax=axs,
    palette=[sns.color_palette()[5], sns.color_palette()[6]],
)
axs.set_xlim([-2, 2])
axs.set_xticklabels(
    [
        "-2\nStrongly\n disagree",
        "-1",
        "0\nNeutral",
        "1",
        "2\nStrongly\n agree",
    ]
)
axs.set_xlabel("")
axs.legend(title="")
# it seemed like...
axs.set_yticklabels(
    [
        "I was looking directly at my own\narm rather than a robotic arm",
        "the robotic arm began to\nresemble my real arm",
        "the robotic arm belonged\nto me",
        "the robotic arm was my arm",
        "the robotic arm was part of\nmy body",
        "my arm was in the location\nwhere the robotic arm was",
        "I could feel the robotic arm\ntouch the object",
        "I could move the robotic arm\nif I wanted to",
        "I was in control of the\nrobotic arm",
    ]
)
axs.set_ylabel("")
sns.despine()
fig.tight_layout()

# %% Experience and fatigue
fig, axs = plt.subplots(1, 2, figsize=(5, 2), sharey=True)
sns.countplot(
    data=hf_df, x="BCI use", ax=axs[0], order=["0h", "<1h", "<2h", "<4h", ">10h"]
)
sns.countplot(
    data=hf_df,
    x="Fatigue",
    ax=axs[1],
    order=["Very low", "Low", "Medium", "High", "Very high"],
)
sns.despine()
fig.tight_layout()

# %% Plot reaching trajectories
p_trajs = reach_data[reach_data.p_id == "P5"].copy()
objs = [7]
fig = plt.figure(figsize=(5, 5))
ax = plt.axes(projection="3d")

# layout
origin = np.array([0.250, -0.204, -0.276])
obj_0 = np.array([0.430, -0.100, -0.130])
obj_dist = 0.11
n_pts = 10
obj_h = 0.06
obj_r = 0.0225
finger_r = 0  # 0.0075
obj_coords = np.array(
    [
        [[obj_0[0], obj_0[1] - obj_dist * i, obj_0[2] - obj_dist * j] for i in range(3)]
        for j in range(3)
    ]
).reshape((-1, 3))

# objects
for obj_i, coords in enumerate(obj_coords):
    xc, yc, zc = coords - origin
    z = np.linspace(0, obj_h + 2 * finger_r, n_pts) - obj_h / 2 - finger_r + zc
    theta = np.linspace(0, 2 * np.pi, n_pts)
    theta_grid, z_grid = np.meshgrid(theta, z)
    x_grid = (obj_r + finger_r) * np.cos(theta_grid) + xc
    y_grid = (obj_r + finger_r) * np.sin(theta_grid) + yc
    alpha = 0.5 if obj_i in p_trajs.goal.unique() else 0.1
    ax.plot_surface(x_grid, y_grid, z_grid, alpha=alpha, color="grey")

start_pos = []
for group, trial in p_trajs[p_trajs.goal.isin(objs)].groupby(["block_i", "trial"]):
    start_pos.append(np.array(trial[["x", "y", "z"]].values[0]) - origin)
    if trial.block.iloc[0] == "SC":
        col = sns.color_palette()[0] if trial.success.max() else sns.color_palette()[1]
        ls = "-"
    else:
        col = sns.color_palette()[2] if trial.success.max() else sns.color_palette()[3]
        ls = "-"

    # trajectories
    traj = np.array(trial[["x", "y", "z"]]) - origin
    ax.plot(
        traj[:, 0],
        traj[:, 1],
        traj[:, 2],
        c=col,
        alpha=0.75,
        linestyle=ls,
        linewidth=2,
    )
    ax.plot(traj[-1, 0], traj[-1, 1], traj[-1, 2], "x", c=col, markersize=5)

# starting point
start_pos = np.array(start_pos).mean(axis=0)
ax.plot(start_pos[0], start_pos[1], start_pos[2], "o", c="k", markersize=5)

# axes formatting
ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.set_box_aspect([1, 1, 1])
ax.set_xlim([-0.05, 0.25])
ax.set_xticks([0, 0.1, 0.2])
ax.set_xlabel("x (m)")
ax.set_ylim([-0.15, 0.15])
ax.set_yticks([-0.1, 0, 0.1])
ax.set_ylabel("y (m)")
ax.set_zlim([-0.15, 0.15])
ax.set_zticks([-0.1, 0, 0.1])
ax.set_zlabel("z (m)")

plt.show()
