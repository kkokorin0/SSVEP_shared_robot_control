# %% Packages
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

from decoding import extract_events, load_recording
from session_manager import CMD_MAP, FS, OBJ_COORDS

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

# %% Extract online results
folder = r"C:\Users\Kirill Kokorin\OneDrive - synchronmed.com\SSVEP robot control\Data\Experiment\P1"

block_i = 0
online_results = []
for file in os.listdir(folder):
    if file.endswith(".xdf"):
        raw, events = load_recording(CH_NAMES, folder, file)
        session_events = extract_events(
            events, ["freqs", "start run", "end run", "go", "pred", "reach", "rest"]
        )

        p_id, freq_str = session_events[0][-1].split(" ")
        freqs = [int(_f) for _f in freq_str.strip("freqs:").split(",")]
        for ts, _, label in session_events:
            if "start run" in label:
                # new block
                block = label.strip("start run: ")
                block_i += 1
                trial_i = 0
            elif "go:" in label:
                # new trial
                goal = label[-1]
                reached = None
                trial_i += 1
                trial_start_row = len(online_results)
                active = True
            elif "rest" in label:
                active = False
            elif ("pred" in label) and active:
                # grab outputs for each time
                pred = label[-1]
                tokens = label.split(" ")
                coords = [float(_c) for _c in tokens[0][2:].split(",")]
                pred = tokens[1][-1]

                # test run only
                if block in ["DC", "SC"]:
                    pred_obj = int(tokens[2][-1])
                    confidence = float(tokens[3][5:])
                    alpha = float(tokens[4][6:])
                    u_robot = [float(_c) for _c in tokens[5][8:].split(",")]
                    u_cmb = [float(_c) for _c in tokens[6][6:].split(",")]
                    success = 0  # assume fail
                else:
                    pred_obj = np.nan
                    confidence = np.nan
                    alpha = np.nan
                    u_robot = [np.nan, np.nan, np.nan]
                    u_cmb = [np.nan, np.nan, np.nan]
                    success = int(goal == pred)

                online_results.append(
                    [
                        p_id,
                        block_i,
                        block,
                        trial_i,
                        ts,
                        goal,
                        reached,
                        success,
                        coords[0],
                        coords[1],
                        coords[2],
                        pred,
                        pred_obj,
                        confidence,
                        alpha,
                        u_robot[0],
                        u_robot[1],
                        u_robot[2],
                        u_cmb[0],
                        u_cmb[1],
                        u_cmb[2],
                    ]
                )
            # reaching results
            elif "reach" in label:
                reached_obj = label.split(" ")[1][-1]
                for row in online_results[trial_start_row:]:
                    row[6] = reached_obj
                    row[7] = int(reached_obj == goal)

# store data
online_df = pd.DataFrame(
    online_results,
    columns=[
        "p_id",
        "block_i",
        "block",
        "trial",
        "ts",
        "goal",
        "reached",
        "success",
        "x",
        "y",
        "z",
        "pred",
        "pred_obj",
        "conf",
        "alpha",
        "u_robot_x",
        "u_robot_y",
        "u_robot_z",
        "u_cmb_x",
        "u_cmb_y",
        "u_cmb_z",
    ],
)
online_df.head()
online_df.to_csv(
    folder
    + "//%s_results_tstep_%s.csv" % (p_id, datetime.now().strftime("%Y%m%d_%H%M%S"))
)

# %% Load data
results = "P1_results_tstep.csv"

online_df = pd.read_csv(folder + "//" + results, index_col=0)
online_df["label"] = (
    online_df.block
    + " B"
    + online_df.block_i.map(str)
    + " T"
    + online_df.trial.map(str)
    + " G"
    + online_df.goal
)
print(online_df.groupby("block_i").trial.max())
online_df.head()

# %% Extract step time and length across all blocks
fig, axs = plt.subplots(2, 1, figsize=(4, 4), sharex=True, sharey=False)

trial_i = 0
dts = []
dLs = []
for i, row in online_df.iterrows():
    if row.trial != trial_i:
        trial_i = row.trial
        dts.append(0)
        dLs.append(0)
    else:
        dts.append((row.ts - online_df.iloc[i - 1].ts) / FS)
        dLs.append(
            np.linalg.norm(
                (row[["x", "y", "z"]] - online_df.iloc[i - 1][["x", "y", "z"]])
            )
            * 1000
        )
online_df["dt"] = dts
online_df["dL"] = dLs

# step length
sns.boxplot(data=online_df, x="block", y="dL", ax=axs[0])
axs[0].set_ylabel("Step length (mm)")

# step time
sns.boxplot(data=online_df, x="block", y="dt", ax=axs[1])
axs[1].set_ylabel("Time step (s)")
axs[1].set_ylim([0, 1])
axs[1].set_xlabel("Block")

fig.tight_layout()
sns.despine()
plt.savefig(folder + "//step_time_length.svg", format="svg")

# %% Online decoding performance
obs_df = online_df[online_df.block == "OBS"].copy()

# confusion matrix
fig, axs = plt.subplots(1, 1, figsize=(3, 3))
conf_mat = confusion_matrix(
    obs_df["goal"], obs_df["pred"], normalize="true", labels=CMDS
)
sns.heatmap(
    conf_mat,
    annot=True,
    fmt=".2f",
    cmap="Blues",
    cbar=False,
    xticklabels=["%s (%s)" % (_c, _f) for _c, _f in zip(CMDS, freqs)],
    yticklabels=["%s (%s)" % (_c, _f) for _c, _f in zip(CMDS, freqs)],
    ax=axs,
)
axs.set_title("%.3f" % balanced_accuracy_score(obs_df["goal"], obs_df["pred"]))
axs.set_xlabel("Predicted")
axs.set_ylabel("True")

fig.tight_layout()
plt.savefig(folder + "//decoding_acc.svg", format="svg")

# %% Reaching performance
test_blocks = [3, 5, 6, 7]

# DC and SC trials
test_df = online_df[online_df.block_i.isin(test_blocks)].copy()
trial_df = test_df.groupby(by=["label", "block", "goal"])["success"].max().reset_index()
trial_df["len_cm"] = list(test_df.groupby(["label"])["dL"].sum() / 10)

# failure rate
session_df = (
    100
    - trial_df.groupby(["block"])["success"].sum()
    / trial_df.groupby(["block"])["success"].count()
    * 100
).reset_index()
session_df.rename(columns={"success": "fail_rate"}, inplace=True)

# trajectory length (only include objects with >1 successful reach)
len_df = (
    trial_df[trial_df.success == 1].groupby(["block", "goal"])["len_cm"].mean()
).reset_index()
len_valid = len_df[len_df.block == "DC"].merge(len_df[len_df.block == "SC"], on="goal")
session_df["len_cm"] = [len_valid.len_cm_x.mean(), len_valid.len_cm_y.mean()]

# store data
session_df.head()
session_df.to_csv(
    folder
    + "//%s_results_session_%s.csv" % (p_id, datetime.now().strftime("%Y%m%d_%H%M%S"))
)

# %% Plot reaching results
fig, axs = plt.subplots(1, 4, figsize=(5, 2), width_ratios=[2, 1, 2, 1])

# failure rate
sns.barplot(data=session_df, x="block", y="fail_rate", ax=axs[0])
axs[0].set_ylabel("Failure rate (%)")
axs[0].set_ylim([0, 100])

# change in failure rate
sns.barplot(
    y=session_df[session_df.block == "SC"].fail_rate.values
    - session_df[session_df.block == "DC"].fail_rate.values,
    ax=axs[1],
)
axs[1].set_ylabel("$\Delta$ Failure rate (%)")
axs[1].set_ylim([-100, 100])

# trajectory length
sns.barplot(data=session_df, x="block", y="len_cm", ax=axs[2])
axs[2].set_ylabel("Trajectory length (cm)")
axs[2].set_ylim([0, 60])

# change in trajectory length
sns.barplot(
    y=session_df[session_df.block == "SC"].len_cm.values
    - session_df[session_df.block == "DC"].len_cm.values,
    ax=axs[3],
)
axs[3].set_ylabel("$\Delta$ Trajectory length (cm)")
axs[3].set_ylim([-20, 20])

for ax, xlabel in zip(axs, ["", "SC-DC", "", "SC-DC"]):
    ax.set_xlabel(xlabel)
sns.despine()
fig.tight_layout()
plt.savefig(folder + "//reaching_results.svg", format="svg")

# %% 3D reaching trajectories
fig = plt.figure(figsize=(10, 7))
ax = plt.axes(projection="3d")

# plot object locations
radius_m = 0.0225
height_m = 0.06
n_pts = 10

for test_obj in test_df.goal.unique():
    xc, yc, zc = OBJ_COORDS[int(test_obj)]
    z = np.linspace(0, height_m, n_pts) + zc
    theta = np.linspace(0, 2 * np.pi, n_pts)
    theta_grid, z_grid = np.meshgrid(theta, z)
    x_grid = radius_m * np.cos(theta_grid) + xc
    y_grid = radius_m * np.sin(theta_grid) + yc
    ax.plot_surface(x_grid, y_grid, z_grid, alpha=0.8, color="g")

# plot trajectories
for label in test_df.label.unique():
    col = "r" if "DC" in label else "b"

    # successful only
    if sum(test_df[test_df.label == label].success) > 0:
        traj = np.array(test_df[test_df.label == label][["x", "y", "z"]])
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], col)
        ax.plot(traj[0, 0], traj[0, 1], traj[0, 2], "kx")

# ax.view_init(elev=0, azim=0)
ax.set_xlim([0, 0.5])
ax.set_xlabel("x (m)")
ax.set_ylim([-0.5, 0])
ax.set_ylabel("y (m)")
ax.set_zlim([-0.5, 0])
ax.set_zlabel("z (m)")
plt.show()
