# %% Packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr, ttest_rel
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

from session_manager import CMD_MAP, FREQS

sns.set_style("ticks", {"axes.grid": False})
sns.set_context("paper")
sns.set_palette("colorblind")

# %% Constants
CMDS = list(CMD_MAP.keys())
FOLDER = r"C:\Users\Kirill Kokorin\OneDrive - synchronmed.com\SSVEP robot control\Data\Experiment\All"
P_IDS = ["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9", "P10", "P11", "P12"]
P_ID_LOW = ["P7", "P9", "P12"]
P_ID_HIGH = list(set(P_IDS) - set(P_ID_LOW))
F_LAYOUTS = [
    [8, 7, 13, 11, 9],
    [7, 13, 11, 8, 9],
    [8, 13, 11, 9, 7],
    [8, 9, 7, 13, 11],
    [13, 9, 11, 8, 7],
    [13, 7, 8, 9, 11],
    [9, 8, 11, 7, 13],
    [8, 9, 13, 7, 11],
    [11, 9, 8, 7, 13],
    [13, 7, 8, 9, 11],
    [11, 13, 8, 7, 9],
    [11, 9, 13, 8, 7],
]
T0S = [
    [0.033, -0.008, -0.013],
    [-0.007, -0.003, -0.038],
    [0.036, 0.010, -0.003],
    [0.002, 0.019, -0.035],
    [0.007, -0.003, 0.034],
    [-0.014, 0.033, 0.007],
    [0.036, -0.010, -0.003],
    [0.002, 0.019, -0.035],
    [0.000, -0.011, -0.039],
    [0.023, 0.009, -0.028],
    [0.035, -0.006, -0.011],
    [0.020, -0.032, 0.009],
]

# %% Starting positions
# %matplotlib qt
n_pts = 10
fig = plt.figure(figsize=(10, 7))
ax = plt.axes(projection="3d")

# mean start positions
for t0 in T0S:
    ax.plot(t0[0], t0[1], t0[2], "kx")
ax.plot(0, 0, 0, "ro")

ax.set_box_aspect([1, 1, 1])
ax.set_xlim([-0.5, 0.5])
ax.set_xlabel("x (m)")
ax.set_ylim([-0.5, 0.5])
ax.set_ylabel("y (m)")
ax.set_zlim([-0.5, 0.5])
ax.set_zlabel("z (m)")
plt.show()

offset_distane_cm = [np.linalg.norm(_t) * 100 for _t in T0S]
print(
    "Offset distance (cm): %.3f+-%0.3f"
    % (np.mean(offset_distane_cm), np.std(offset_distane_cm)),
)

# %% Frequency layouts
fig, axs = plt.subplots(1, 1, figsize=(3, 3))
layout_map = np.zeros((len(CMDS), len(CMDS)))
layout_keys = {_f: _i for _i, _f in enumerate(FREQS)}
for layout in F_LAYOUTS:
    for _r in range(len(CMDS)):
        layout_map[_r, layout_keys[layout[_r]]] += 1

sns.heatmap(
    layout_map,
    annot=True,
    cmap="Greens",
    cbar=False,
    xticklabels=FREQS,
    yticklabels=CMDS,
    ax=axs,
)
axs.set_xlabel("Frequency (Hz)")
axs.set_ylabel("Direction")

# %% Decoding performance
accs = []
cms = []
for p_id in P_IDS:
    data = pd.read_csv(FOLDER + "//" + p_id + "_results_tstep.csv", index_col=0)
    preds = data[data.block == "OBS"].pred
    labels = data[data.block == "OBS"].goal
    accs.append(balanced_accuracy_score(labels, preds) * 100)
    cms.append(confusion_matrix(labels, preds, normalize="true", labels=CMDS) * 100)

# confusion matrix
fig, axs = plt.subplots(1, 1, figsize=(3, 3))
mean_cm = np.mean(cms, axis=0)
sterr_cm = np.std(cms, axis=0) / np.sqrt(len(cms))
annots = [
    ["%.1f\n(%.1f)" % (mean_cm[_r, _c], sterr_cm[_r, _c]) for _c in range(len(CMDS))]
    for _r in range(len(CMDS))
]
sns.heatmap(
    mean_cm,
    annot=annots,
    fmt="s",
    cmap="Blues",
    cbar=False,
    xticklabels=CMDS,
    yticklabels=CMDS,
    ax=axs,
)
axs.set_xlabel("Predicted")
axs.set_ylabel("True")

# %% Decoding performance by frequency
f_cms = []
for p_id, p_freqs in zip(P_IDS, F_LAYOUTS):
    data = pd.read_csv(FOLDER + "//" + p_id + "_results_tstep.csv", index_col=0)
    freq_map = {_d: _f for _f, _d in zip(p_freqs, CMD_MAP.keys())}

    preds = [freq_map[_p] for _p in data[data.block == "OBS"].pred]
    labels = [freq_map[_l] for _l in data[data.block == "OBS"].goal]
    f_cms.append(confusion_matrix(labels, preds, normalize="true", labels=FREQS) * 100)

# confusion matrix
fig, axs = plt.subplots(1, 1, figsize=(3, 3))
mean_cm = np.mean(f_cms, axis=0)
sterr_cm = np.std(f_cms, axis=0) / np.sqrt(len(f_cms))
annots = [
    ["%.1f\n(%.1f)" % (mean_cm[_r, _c], sterr_cm[_r, _c]) for _c in range(len(CMDS))]
    for _r in range(len(CMDS))
]
sns.heatmap(
    mean_cm,
    annot=annots,
    fmt="s",
    cmap="Purples",
    cbar=False,
    xticklabels=FREQS,
    yticklabels=FREQS,
    ax=axs,
)
axs.set_xlabel("Predicted")
axs.set_ylabel("True")

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
c = sns.color_palette()[0]
fig, axs = plt.subplots(1, 2, figsize=(4, 3), width_ratios=[2, 1])
sns.pointplot(
    data=session_df[session_df.p_id.isin(P_ID_HIGH)],
    x="block",
    y="success_rate",
    ax=axs[0],
    hue="p_id",
    order=["DC", "SC"],
    palette=[c, c],
    dodge=False,
)
sns.pointplot(
    data=session_df[session_df.p_id.isin(P_ID_LOW)],
    x="block",
    y="success_rate",
    ax=axs[0],
    hue="p_id",
    order=["DC", "SC"],
    palette=[c, c],
    dodge=False,
    markers="x",
    linestyle="--",
)
axs[0].set_ylim([0, 105])
axs[0].set_ylabel("Success rate (%)")
axs[0].set_xlabel("")
axs[0].legend().remove()

# change in success rates
sns.swarmplot(
    data=session_df[session_df.p_id.isin(P_ID_HIGH)],
    x="block",
    y="success_rate",
    ax=axs[1],
    order=["SC-DC"],
    alpha=0.5,
    palette=[c],
)
sns.pointplot(
    data=session_df,
    x="block",
    y="success_rate",
    ax=axs[1],
    order=["SC-DC"],
    errorbar=("ci", 95),
    markers="D",
    palette=[c],
)
axs[1].set_ylim([0, 105])
axs[1].set_ylabel("$\Delta$ Success rate (%)")
axs[1].set_xlabel("")
fig.tight_layout()
sns.despine()

# compute t-test results
test_frate = ttest_rel(
    session_df[session_df.block == "SC"]["success_rate"],
    session_df[session_df.block == "DC"]["success_rate"],
    alternative="two-sided",
)
ci = test_frate.confidence_interval()
print(
    "t-test (SC-DC), t:%.3f, p: %.4f, ci: (%.3f,%.3f)"
    % (test_frate.statistic, test_frate.pvalue, ci.low, ci.high)
)

# %% Trajectory lengths
# lengths
c = sns.color_palette()[1]
fig, axs = plt.subplots(1, 2, figsize=(4, 3), width_ratios=[2, 1])
sns.pointplot(
    data=session_df[session_df.p_id.isin(P_ID_HIGH)],
    x="block",
    y="len_cm",
    ax=axs[0],
    hue="p_id",
    order=["DC", "SC"],
    palette=[c, c],
    dodge=False,
)
low_len_df = session_df[session_df.p_id.isin(P_ID_LOW)].copy()
sns.pointplot(
    data=low_len_df,
    x="block",
    y="len_cm",
    ax=axs[0],
    hue="p_id",
    order=["DC", "SC"],
    palette=[c, c],
    dodge=False,
    markers="x",
)
axs[0].set_ylim([20, 70])
axs[0].set_ylabel("Trajectory length (cm)")
axs[0].set_xlabel("")
axs[0].legend().remove()

# change in lengths
sns.swarmplot(
    data=session_df,
    x="block",
    y="len_cm",
    ax=axs[1],
    order=["SC-DC"],
    alpha=0.5,
    palette=[c],
)
sns.pointplot(
    data=session_df,
    x="block",
    y="len_cm",
    ax=axs[1],
    order=["SC-DC"],
    errorbar=("ci", 95),
    markers="D",
    palette=[c],
)
axs[1].set_ylim([-30, 30])
axs[1].set_ylabel("$\Delta$ Trajectory length (cm)")
axs[1].set_xlabel("")
fig.tight_layout()
sns.despine()

# compute t-test results
valid_lens = session_df[session_df.p_id.isin(P_ID_HIGH)].copy()
test_length = ttest_rel(
    valid_lens[valid_lens.block == "SC"]["len_cm"],
    valid_lens[valid_lens.block == "DC"]["len_cm"],
    alternative="two-sided",
)
ci = test_length.confidence_interval()
print(
    "t-test (SC-DC), t:%.3f, p: %.4f, ci: (%.3f,%.3f)"
    % (test_length.statistic, test_length.pvalue, ci.low, ci.high)
)

# %% Decoding accuracy vs impact of shared control
fig, axs = plt.subplots(1, 2, figsize=(6, 3), sharex=True)
acc_vs_sc_df = pd.merge(
    session_df[session_df.block == "SC-DC"], pd.DataFrame({"p_id": P_IDS, "acc": accs})
)

# acc vs delta success rate
sns.regplot(
    data=acc_vs_sc_df,
    x="acc",
    y="success_rate",
    ax=axs[0],
    color=sns.color_palette()[0],
    marker="",
)
sns.regplot(
    data=acc_vs_sc_df[acc_vs_sc_df.p_id.isin(P_ID_HIGH)],
    x="acc",
    y="success_rate",
    ax=axs[0],
    color=sns.color_palette()[0],
    marker="o",
    fit_reg=False,
)
sns.regplot(
    data=acc_vs_sc_df[acc_vs_sc_df.p_id.isin(P_ID_LOW)],
    x="acc",
    y="success_rate",
    ax=axs[0],
    color=sns.color_palette()[0],
    marker="x",
    fit_reg=False,
)

axs[0].set_ylim([-100, 100])
axs[0].set_xlim([0, 100])
axs[0].set_xlabel("Accuracy (%)")
axs[0].set_ylabel("$\Delta$ Success rate (%)")
r, p = pearsonr(acc_vs_sc_df.acc, acc_vs_sc_df.success_rate)
print("dSR vs acc correlation, r:%.3f, p: %.3f" % (r, p))

# acc vs delta trajectory length
sns.regplot(
    data=acc_vs_sc_df, x="acc", y="len_cm", ax=axs[1], color=sns.color_palette()[1]
)
axs[1].set_ylim([-50, 50])
axs[1].set_xlabel("Accuracy (%)")
axs[1].set_ylabel("$\Delta$ Trajectory length (cm)")
valid_rel_lens = acc_vs_sc_df.dropna(subset=["len_cm"])
r, p = pearsonr(valid_rel_lens.acc, valid_rel_lens.len_cm)
print("dL vs acc correlation, r:%.3f, p: %.3f" % (r, p))

fig.tight_layout()
sns.despine()

# %% Workload
hf_df = pd.read_csv(FOLDER + "//human_factors.csv", index_col=None)
fig, axs = plt.subplots(1, 3, figsize=(6, 3), width_ratios=[3, 2, 1])

# factor tally
factors = ["Mental", "Physical", "Temporal", "Performance", "Effort", "Frustration"]
factor_labels = ["MD", "PD", "TD", "P", "E", "F"]
sns.barplot(
    data=hf_df.melt(id_vars=["ID"], value_vars=factors),
    x="variable",
    y="value",
    ax=axs[0],
    errorbar="se",
    order=factors,
    color=sns.color_palette()[3],
)
axs[0].set_ylabel("Weight")
axs[0].set_ylim([0, 5])
axs[0].set_xticklabels(factor_labels)
axs[0].set_xlabel("Factor")

# workload
wl_df = hf_df.melt(id_vars=["ID"], value_vars=["DC Total", "SC Total"]).copy()
wl_df["mode"] = wl_df["variable"].apply(lambda x: x.split(" ")[0])
c = sns.color_palette()[4]
sns.pointplot(
    data=wl_df[wl_df.ID.isin(P_ID_HIGH)],
    x="mode",
    y="value",
    hue="ID",
    ax=axs[1],
    order=["DC", "SC"],
    palette=[c, c],
    dodge=False,
)
sns.pointplot(
    data=wl_df[wl_df.ID.isin(P_ID_LOW)],
    x="mode",
    y="value",
    hue="ID",
    ax=axs[1],
    order=["DC", "SC"],
    palette=[c, c],
    dodge=False,
    markers="x",
    linestyles="--",
)
axs[1].set_ylim([0, 105])
axs[1].set_ylabel("Workload")
axs[1].set_xlabel("Mode")
axs[1].legend().remove()

# change in workload
hf_df["dWL"] = hf_df["SC Total"] - hf_df["DC Total"]
sns.swarmplot(
    y=hf_df["SC Total"] - hf_df["DC Total"],
    ax=axs[2],
    alpha=0.75,
    palette=[c],
)
sns.pointplot(
    y=hf_df["SC Total"] - hf_df["DC Total"],
    ax=axs[2],
    errorbar=("ci", 95),
    markers="o",
    palette=[c],
)
axs[2].set_ylim([-30, 30])
axs[2].set_ylabel("$\Delta$ Workload")
axs[2].set_xticklabels(["SC-DC"])
axs[2].set_xlabel("Mode")
fig.tight_layout()
sns.despine()

test_wl = ttest_rel(
    hf_df["SC Total"],
    hf_df["DC Total"],
    alternative="two-sided",
)
ci = test_wl.confidence_interval()
print(
    "t-test (SC-DC), t:%.3f, p: %.3f, ci: (%.3f,%.3f)"
    % (test_wl.statistic, test_wl.pvalue, ci.low, ci.high)
)

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
