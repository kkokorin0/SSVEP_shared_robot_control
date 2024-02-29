# %% Packages
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

from session_manager import CMD_MAP

sns.set_style("ticks", {"axes.grid": False})
sns.set_context("paper")
sns.set_palette("colorblind")

# %% Constants
CH_NAMES = [
    "O1",
    "Oz",
    "O2",
    "PO7",
    "PO3",
    "POz",
    "PO4",
    "PO8",
    "Pz",
    "CPz",
    "C1",
    "Cz",
    "C2",
    "FC1",
    "FCz",
    "FC2",
]
SSVEP_CHS = CH_NAMES[:9]
CMDS = list(CMD_MAP.keys())
FOLDER = r"C:\Users\Kirill Kokorin\OneDrive - synchronmed.com\SSVEP robot control\Data\Pilot\CMB"

# %% Load participant data
p_data = []
for file in os.listdir(FOLDER):
    p_data.append(pd.read_csv(FOLDER + "//" + file, index_col=0))

p_data = pd.concat(p_data, axis=0).reset_index(drop=True)
p_data["label"] = list(
    "P"
    + p_data.p_id.map(str)
    + " "
    + p_data.block
    + " B"
    + p_data.block_i.map(str)
    + " T"
    + p_data.trial.map(str)
    + " G"
    + p_data.goal,
)
p_data.head()

# %% Decoding performance
obs_df = p_data[p_data.block.isin(["OBS", "OBSF", "obs w/o fb"])].copy()
fig, axs = plt.subplots(1, 1, figsize=(3, 3))
decoding_acc = []
conf_mat = np.zeros((len(CMDS), len(CMDS)))
for p_id in p_data.p_id.unique():
    y_true = obs_df[obs_df.p_id == p_id]["goal"]
    y_pred = obs_df[obs_df.p_id == p_id]["pred"]
    conf_mat += confusion_matrix(
        y_true,
        y_pred,
        normalize="true",
        labels=CMDS,
    )
    decoding_acc.append(balanced_accuracy_score(y_true, y_pred))

sns.heatmap(
    conf_mat / len(p_data.p_id.unique()),
    annot=True,
    fmt=".2f",
    cmap="Blues",
    cbar=False,
    xticklabels=CMDS,
    yticklabels=CMDS,
    ax=axs,
)
axs.set_title("%.2f$\pm$%.2f" % (np.mean(decoding_acc), np.std(decoding_acc)))
axs.set_xlabel("Predicted")
axs.set_ylabel("True")

# %% Get reaching block labels
reach_blocks = p_data[p_data.block.isin(["DC", "SC"])].copy()
reach_blocks["p_block"] = reach_blocks.label.map(lambda x: x[0:9])

for p_id in reach_blocks.p_id.unique():
    print(reach_blocks[reach_blocks.p_id == p_id].p_block.unique())

# %% Extract test blocks
test_blocks = [
    "P96 SC B4",
    "P96 DC B6",
    "P97 DC B5",
    "P97 SC B7",
    "P98 DC B3",
    "P98 SC B5",
    "P98 SC B6",
    "P98 SC B7",
    "P98 DC B8",
]
test_df = (
    reach_blocks[reach_blocks.p_block.isin(test_blocks)].copy().reset_index(drop=True)
)

# %% Failure rate
fig, axs = plt.subplots(1, 2, figsize=(4, 3), width_ratios=[2, 1])
successes = pd.DataFrame(test_df.groupby(["label"])["success"].max()).reset_index()
successes["block"] = successes.label.str[0:9]
failure_rate = pd.DataFrame(
    100
    - successes.groupby(["block"])["success"].sum()
    / successes.groupby(["block"])["success"].count()
    * 100,
).reset_index()
failure_rate["p_id"] = failure_rate.block.str[1:3]
failure_rate["mode"] = failure_rate.block.str[4:6]
failure_rate = failure_rate.rename(columns={"success": "rate"})
failure_rate.head()

# total failure rate
sns.pointplot(
    data=failure_rate,
    x="mode",
    y="rate",
    hue="p_id",
    order=["DC", "SC"],
    errorbar=None,
    ax=axs[0],
)
axs[0].set_ylim([0, 100])
axs[0].set_ylabel("Failure rate (%)")
axs[0].set_xlabel("Block")
axs[0].legend().remove()

# difference in failure rate
fail_dif = failure_rate.pivot_table(
    index=["p_id"], columns="mode", values="rate"
).reset_index()
fail_dif["dF"] = fail_dif["SC"] - fail_dif["DC"]
sns.pointplot(
    data=fail_dif,
    y="dF",
    # hue="p_id",
    errorbar="ci",
    ax=axs[1],
)
axs[1].set_ylim([-100, 0])
axs[1].set_ylabel("$\Delta$ Failure rate (%)")
axs[1].set_xlabel("SC-DC")

fig.tight_layout()
sns.despine()

# %% Calculate step length
trial_i = 0
dLs = []
for i, row in test_df.iterrows():
    if row.trial != trial_i:
        trial_i = row.trial
        dLs.append(0)
    else:
        dLs.append(
            np.linalg.norm(
                (row[["x", "y", "z"]] - test_df.iloc[i - 1][["x", "y", "z"]])
            )
            * 1000
        )
test_df["dL"] = dLs
sns.boxplot(data=test_df, x="block", y="dL")

# %% Trajectory length (successful only)
fig, axs = plt.subplots(1, 2, figsize=(4, 3), width_ratios=[2, 1])
lengths = pd.DataFrame(
    test_df[test_df.success == 1].groupby(["label"])["dL"].sum() / 10
).reset_index()
lengths["block"] = lengths.label.str[0:9]
lengths["goal"] = lengths.label.str[-1]

# combine by goal block
goal_lengths = pd.DataFrame(
    lengths.groupby(["block", "goal"])["dL"].mean()
).reset_index()
goal_lengths["p_id"] = goal_lengths.block.str[1:3]
goal_lengths["mode"] = goal_lengths.block.str[4:6]

# total trajectories
sns.pointplot(
    data=goal_lengths,
    x="mode",
    y="dL",
    hue="p_id",
    order=["DC", "SC"],
    errorbar=None,
    ax=axs[0],
)
axs[0].set_ylabel("Trajectory length (cm)")
axs[0].set_ylim([0, 60])
axs[0].set_xlabel("Block")
axs[0].legend().remove()

# difference in trajectory length
traj_dif = goal_lengths.pivot_table(
    index=["p_id"], columns="mode", values="dL"
).reset_index()
traj_dif["dL"] = traj_dif["SC"] - traj_dif["DC"]
sns.pointplot(
    data=traj_dif,
    y="dL",
    # hue="p_id",
    errorbar="ci",
    ax=axs[1],
)
axs[1].set_ylim([-30, 0])
axs[1].set_ylabel("$\Delta$ Trajectory length (cm)")
axs[1].set_xlabel("SC-DC")

sns.despine()
fig.tight_layout(