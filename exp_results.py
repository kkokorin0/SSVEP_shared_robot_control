# %% Packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import ttest_rel
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

from session_manager import CMD_MAP, FREQS

sns.set_style("ticks", {"axes.grid": False})
sns.set_context("paper")
sns.set_palette("colorblind")

# %% Constants
CMDS = list(CMD_MAP.keys())
FOLDER = r"C:\Users\Kirill Kokorin\OneDrive - synchronmed.com\SSVEP robot control\Data\Experiment\All"
P_IDS = ["P1", "P2"]
F_LAYOUTS = [[8, 7, 13, 11, 9], [7, 13, 11, 8, 9]]
T_NOM = [0.250, -0.203, -0.278]
T0S = [[-0.147, 0.018, -0.048], [-0.187, 0.023, -0.074]]

# %% Starting positions
n_pts = 10
fig = plt.figure(figsize=(10, 7))
ax = plt.axes(projection="3d")

# mean start positions
for t0 in T0S:
    ax.plot(t0[0] - T_NOM[0], t0[1] - T_NOM[1], t0[2] - T_NOM[2], "kx")

ax.set_box_aspect([1, 1, 1])
ax.set_xticks(np.arange(-3, 2) / 10)
ax.set_xlabel("x (m)")
ax.set_yticks(np.arange(-2, 3) / 10)
ax.set_ylabel("y (m)")
ax.set_zticks(np.arange(-2, 3) / 10)
ax.set_zlabel("z (m)")
plt.show()

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
std_cm = np.std(cms, axis=0)
annots = [
    ["%.1f\n(%.1f)" % (mean_cm[_r, _c], std_cm[_r, _c]) for _c in range(len(CMDS))]
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

# %% Failure rates
# load session data
sessions = []
for p_id in P_IDS:
    sessions.append(
        pd.read_csv(FOLDER + "//" + p_id + "_results_session.csv", index_col=0)
    )

session_df = pd.concat(sessions).reset_index(drop=True)
session_df["p_id"] = np.repeat(P_IDS, 3)

# failure rates
c = sns.color_palette()[0]
fig, axs = plt.subplots(1, 2, figsize=(4, 3), width_ratios=[2, 1])
sns.pointplot(
    data=session_df,
    x="block",
    y="fail_rate",
    ax=axs[0],
    hue="p_id",
    order=["DC", "SC"],
    palette=[c, c],
    dodge=True,
)
axs[0].set_ylim([0, 100])
axs[0].set_ylabel("Failure rate (%)")
axs[0].set_xlabel("")
axs[0].legend().remove()

# change in failure rates
sns.swarmplot(
    data=session_df,
    x="block",
    y="fail_rate",
    ax=axs[1],
    order=["SC-DC"],
    alpha=0.75,
    palette=[c],
)
sns.pointplot(
    data=session_df,
    x="block",
    y="fail_rate",
    ax=axs[1],
    order=["SC-DC"],
    errorbar=("ci", 95),
    markers="o",
    palette=[c],
)
axs[1].set_ylim([-100, 0])
axs[1].set_ylabel("$\Delta$ Failure rate (%)")
axs[1].set_xlabel("")
fig.tight_layout()
sns.despine()

# compute t-test results
test_frate = ttest_rel(
    session_df[session_df.block == "SC"]["fail_rate"],
    session_df[session_df.block == "DC"]["fail_rate"],
    alternative="two-sided",
)
ci = test_frate.confidence_interval()
print(
    "t-test (SC-DC), t:%.3f, p: %.3f, ci: (%.3f,%.3f)"
    % (test_frate.statistic, test_frate.pvalue, ci.low, ci.high)
)

# %% Trajectory lengths
# lengths
c = sns.color_palette()[1]
fig, axs = plt.subplots(1, 2, figsize=(4, 3), width_ratios=[2, 1])
sns.pointplot(
    data=session_df,
    x="block",
    y="len_cm",
    ax=axs[0],
    hue="p_id",
    order=["DC", "SC"],
    palette=[c, c],
    dodge=True,
)
axs[0].set_ylim([20, 70])
axs[0].set_ylabel("Trajcetory length (cm)")
axs[0].set_xlabel("")
axs[0].legend().remove()

# change in lengths
sns.swarmplot(
    data=session_df,
    x="block",
    y="len_cm",
    ax=axs[1],
    order=["SC-DC"],
    alpha=0.75,
    palette=[c],
)
sns.pointplot(
    data=session_df,
    x="block",
    y="len_cm",
    ax=axs[1],
    order=["SC-DC"],
    errorbar=("ci", 95),
    markers="o",
    palette=[c],
)
axs[1].set_ylim([-30, 30])
axs[1].set_ylabel("$\Delta$ Trajectory length (cm)")
axs[1].set_xlabel("")
fig.tight_layout()
sns.despine()

# compute t-test results
test_frate = ttest_rel(
    session_df[session_df.block == "SC"]["len_cm"],
    session_df[session_df.block == "DC"]["len_cm"],
    alternative="two-sided",
)
ci = test_frate.confidence_interval()
print(
    "t-test (SC-DC), t:%.3f, p: %.3f, ci: (%.3f,%.3f)"
    % (test_frate.statistic, test_frate.pvalue, ci.low, ci.high)
)

# %% Decoding accuracy vs impact of shared control
fig, axs = plt.subplots(1, 2, figsize=(6, 3), sharex=True)
acc_vs_sc_df = pd.merge(
    session_df[session_df.block == "SC-DC"], pd.DataFrame({"p_id": P_IDS, "acc": accs})
)

# acc vs delta failure rate
sns.regplot(
    data=acc_vs_sc_df, x="acc", y="fail_rate", ax=axs[0], color=sns.color_palette()[0]
)
axs[0].set_ylim([-50, 0])
axs[0].set_xlim([0, 100])
axs[0].set_xlabel("Accuracy (%)")
axs[0].set_ylabel("$\Delta$ Failure rate (%)")

# acc vs delta trajectory length
sns.regplot(
    data=acc_vs_sc_df, x="acc", y="len_cm", ax=axs[1], color=sns.color_palette()[1]
)
axs[1].set_ylim([-50, 0])
axs[1].set_xlim([0, 100])
axs[1].set_xlabel("Accuracy (%)")
axs[1].set_ylabel("$\Delta$ Trajectory length (cm)")


fig.tight_layout()
sns.despine()
