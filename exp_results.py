# %% Packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr, sem, t, ttest_rel
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from statsmodels.api import qqplot

from session_manager import CMD_MAP, FREQS

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
        x (str): x axis
        y (str): y axis
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
        x (str): x axis
        y (str): y axis
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
        x (str): x axis
        y (str): y axis
        ax (axes): figure axes
        col (tuple): colour
        xlabel (str): x axis label
        ylabel (str): y axis label
        xlim (list of float): x axis limits
        ylim (list of float): y axis limits
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

# %% Decoding performance
fig, axs = plt.subplots(1, 2, figsize=(5.6, 2.8))

# by direction
accs = []
cms = []
all_labels = []
for p_id in P_IDS:
    data = pd.read_csv(FOLDER + "//" + p_id + "_results_tstep.csv", index_col=0)
    preds = data[data.block == "OBS"].pred
    labels = data[data.block == "OBS"].goal
    all_labels.append(labels.value_counts())
    accs.append(balanced_accuracy_score(labels, preds) * 100)
    cms.append(confusion_matrix(labels, preds, normalize="true", labels=CMDS) * 100)

plot_confusion_matrix(cms, CMDS, axs[0], cmap="Blues")

# by frequency
f_cms = []
for p_id, p_freqs in zip(P_IDS, F_LAYOUTS):
    data = pd.read_csv(FOLDER + "//" + p_id + "_results_tstep.csv", index_col=0)
    freq_map = {_d: _f for _f, _d in zip(p_freqs, CMD_MAP.keys())}
    preds = [freq_map[_p] for _p in data[data.block == "OBS"].pred]
    labels = [freq_map[_l] for _l in data[data.block == "OBS"].goal]
    f_cms.append(confusion_matrix(labels, preds, normalize="true", labels=FREQS) * 100)

plot_confusion_matrix(f_cms, FREQS, axs[1], cmap="Purples")

fig.tight_layout()
# plt.savefig(FOLDER + "//Figures//decoding_cms.svg", format="svg")

# %% Prediction accuracy by time step in SC trials
fig, axs = plt.subplots(1, 1, figsize=(5, 2))
blocks = [3, 5, 6, 7]
obj_labels = ["TL", "TM", "TR", "ML", "M", "MR", "BL", "BM", "BR"]
objs = np.arange(9)

preds = []
labels = []
trial_objs = []
trial_recall = []
preds_per_trial = []
for p_id in P_IDS:
    data = pd.read_csv(FOLDER + "//" + p_id + "_results_tstep.csv", index_col=0)
    sc_data = data[data.block_i.isin(blocks) & (data.block == "SC")].copy()
    sc_data["pred_success"] = sc_data.goal.astype(int) == sc_data.pred_obj.astype(int)
    trials = sc_data.groupby(["block_i", "trial"])

    trial_objs.append(trials["goal"].first().values)
    trial_recall.append(
        (trials["pred_success"].sum() / trials["pred_success"].count()).values
    )
    preds_per_trial.append(trials["goal"].count().values)

obj_trial_preds = pd.DataFrame(
    {"recall": np.concatenate(trial_recall) * 100, "obj": np.concatenate(trial_objs)}
)
sns.boxplot(
    data=obj_trial_preds,
    x="obj",
    y="recall",
    whis=(0, 100),
    showmeans=True,
    meanprops={
        "marker": "o",
        "markerfacecolor": "white",
        "markeredgecolor": "black",
        "markersize": "4",
    },
    order=objs,
    color=sns.color_palette()[7],
    ax=axs,
)
axs.set_ylabel("Correct predictions\nper trial (%)")
axs.set_xlabel("")
axs.set_xticklabels(obj_labels)
axs.set_ylim([-5, 105])
sns.despine()
fig.tight_layout()
# plt.savefig(FOLDER + "//Figures//object_preds.pdf", format="pdf")

# %% Success rates
# load session data
sessions = []
for p_id in P_IDS:
    sessions.append(
        pd.read_csv(FOLDER + "//" + p_id + "_results_session.csv", index_col=0)
    )

session_df = pd.concat(sessions).reset_index(drop=True)
session_df["p_id"] = np.repeat(P_IDS, 3)

# success rates
c = sns.color_palette()[2]
fig, axs = plt.subplots(1, 2, figsize=(3.5, 2.5), width_ratios=[2, 1])
session_best = session_df[session_df.p_id.isin(P_ID_HIGH)]
session_worst = session_df[session_df.p_id.isin(P_ID_LOW)]
plot_lines(
    session_best,
    session_worst,
    "block",
    "success_rate",
    axs[0],
    "p_id",
    ["DC", "SC"],
    c,
    "Success rate (%)",
    [-5, 105],
)
plot_CI(
    session_best,
    session_worst,
    "block",
    "success_rate",
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

run_ttest(
    session_df[session_df.block == "SC"]["success_rate"].values,
    session_df[session_df.block == "DC"]["success_rate"].values,
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
    data=session_df,
    x="block",
    y="success_rate",
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

# %% Trajectory lengths
# lengths
c = sns.color_palette()[1]
fig, axs = plt.subplots(1, 2, figsize=(3.5, 2.5), width_ratios=[2, 1])
plot_lines(
    session_best,
    session_worst,
    "block",
    "len_cm",
    axs[0],
    "p_id",
    ["DC", "SC"],
    c,
    "Trajectory length (cm)",
    [28, 73],
)
plot_CI(
    session_best,
    session_worst,
    "block",
    "len_cm",
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

run_ttest(
    session_best[session_best.block == "SC"]["len_cm"].values,
    session_best[session_best.block == "DC"]["len_cm"].values,
)

# %% Impact of decoding accuracy
fig, axs = plt.subplots(1, 2, figsize=(5, 2), sharex=True)
acc_df = pd.DataFrame({"p_id": P_IDS, "acc": accs, "mode": [""] * len(P_IDS)})

# success rate
acc_vs_dc_df = pd.merge(session_df[session_df.block == "DC"], acc_df)
acc_vs_dc_df_best = acc_vs_dc_df[acc_vs_dc_df.p_id.isin(P_ID_HIGH)]
acc_vs_dc_df_worst = acc_vs_dc_df[acc_vs_dc_df.p_id.isin(P_ID_LOW)]
plot_reg(
    acc_vs_dc_df_best,
    acc_vs_dc_df_worst,
    "acc",
    "success_rate",
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
acc_vs_sc_df = pd.merge(session_df[session_df.block == "SC-DC"], acc_df)

acc_vs_sc_df_best = acc_vs_sc_df[acc_vs_sc_df.p_id.isin(P_ID_HIGH)]
acc_vs_sc_df_worst = acc_vs_sc_df[acc_vs_sc_df.p_id.isin(P_ID_LOW)]
plot_reg(
    acc_vs_sc_df_best,
    acc_vs_sc_df_worst,
    "acc",
    "success_rate",
    axs[1],
    sns.color_palette()[0],
    "Accuracy (%)",
    "$\Delta$ Success rate (%)",
    [25, 100],
    [-20, 80],
    pt_size=10,
    x_size=30,
)

# change in trajectory length
# sns.regplot(
#     data=acc_vs_sc_df,
#     x="acc",
#     y="len_cm",
#     ax=axs[2],
#     color=sns.color_palette()[1],
#     scatter_kws={"s": 5},
# )
# axs[2].set_ylim([-25, 0])
# axs[2].set_xlabel("Accuracy (%)")
# axs[2].set_ylabel("$\Delta$ Trajectory length (cm)")
# r, p = pearsonr(acc_vs_sc_df_best.acc, acc_vs_sc_df_best.len_cm)
# print("correlation, r:%.3f, p: %.3f" % (r, p))

fig.tight_layout()
sns.despine()
# plt.savefig(FOLDER + "//Figures//corrs.svg", format="svg")

# %% Offline decoding recall with variable window sizes
window_df = pd.read_csv(FOLDER + "//variable_window.csv", index_col=None)
fig, axs = plt.subplots(1, 2, figsize=(5, 2), sharey=True)
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
fig, axs = plt.subplots(1, 3, figsize=(5, 2), width_ratios=[1, 1, 2.5])
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
    [30, 100],
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
    [0, 125],
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
axs[2].set_ylabel("Count")
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
fig, axs = plt.subplots(1, 2, figsize=(6, 3), sharey=True)
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
# %matplotlib qt
plt.close("all")
participant = 5
objs = ["7"]
blocks = [3, 5, 6, 7]
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

# load data
results = "P%d_results_tstep.csv" % participant
online_df = pd.read_csv(FOLDER + "//" + results, index_col=0)
reach_df = online_df[online_df.block_i.isin(blocks) & online_df.goal.isin(objs)].copy()
reach_df["label"] = reach_df.block_i.map(str) + " " + reach_df.trial.map(str)

# objects
for obj_i, coords in enumerate(obj_coords):
    xc, yc, zc = coords - origin
    z = np.linspace(0, obj_h + 2 * finger_r, n_pts) - obj_h / 2 - finger_r + zc
    theta = np.linspace(0, 2 * np.pi, n_pts)
    theta_grid, z_grid = np.meshgrid(theta, z)
    x_grid = (obj_r + finger_r) * np.cos(theta_grid) + xc
    y_grid = (obj_r + finger_r) * np.sin(theta_grid) + yc
    alpha = 0.5 if str(obj_i) in online_df.goal.unique() else 0.1
    ax.plot_surface(x_grid, y_grid, z_grid, alpha=alpha, color="grey")

start_pos = []
for label in reach_df.label.unique():
    trial = reach_df[reach_df.label == label]
    start_pos.append(np.array(trial[["x", "y", "z"]].values[0]) - origin)
    if trial.block.max() == "SC":
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
        linewidth=2.5,
    )

# plot starting point
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
