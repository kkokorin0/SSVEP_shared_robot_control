# %% Packages
import os
from datetime import datetime

import matplotlib as mpl
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from decoding import BandpassFilter, Decoder, extract_events, load_recording
from session_manager import (
    CMD_MAP,
    FILTER_ORDER,
    FMAX,
    FMIN,
    FS,
    HARMONICS,
    OBJ_COORDS,
    OBJ_H,
    OBJ_R,
    OBS_TRIAL_MS,
    SAMPLE_T_MS,
)

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
T_NOM = np.array([0.250, -0.204, -0.276])
P_ID = 16
FOLDER = (
    r"C:\Users\Kirill Kokorin\OneDrive - synchronmed.com\SSVEP robot control\Data\Experiment\P"
    + str(P_ID)
)

# %% Extract online results
block_i = 0
online_results = []
for file in os.listdir(FOLDER):
    if file.endswith(".xdf"):
        raw, events = load_recording(CH_NAMES, FOLDER, file)
        session_events = extract_events(
            events,
            [
                "freqs",
                "start run",
                "end run",
                "go",
                "pred",
                "reach",
                "success",
                "fail",
                "rest",
            ],
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

                if block in ["DC", "SC"]:
                    # reaching trials
                    pred_obj = int(tokens[2][-1])
                    confidence = float(tokens[3][5:])
                    alpha = float(tokens[4][6:])
                    u_robot = [float(_c) for _c in tokens[5][8:].split(",")]
                    u_cmb = [float(_c) for _c in tokens[6][6:].split(",")]
                    success = 0  # assume fail
                else:
                    # observation trials
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
            # automatic block collision detected
            elif "reach" in label:
                reached_obj = label.split(" ")[0][-1]
                for row in online_results[trial_start_row:]:
                    row[6] = reached_obj
                    row[7] = int(reached_obj == goal)

            # manual button press reach flag
            elif "success" in label:
                for row in online_results[trial_start_row:]:
                    row[6] = goal
                    row[7] = 1

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
print(online_df.groupby("block_i")["trial"].max())  # trials/block
online_df.to_csv(
    FOLDER
    + "//P%s_results_tstep_%s.csv" % (P_ID, datetime.now().strftime("%Y%m%d_%H%M%S"))
)

# %% Load data
results = "P%d_results_tstep.csv" % P_ID

online_df = pd.read_csv(FOLDER + "//" + results, index_col=0)
online_df["label"] = (
    online_df.block
    + " B"
    + online_df.block_i.map(str)
    + " T"
    + online_df.trial.map(str)
    + " G"
    + online_df.goal
)
online_df.groupby("block_i")["trial"].max()

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
plt.savefig(FOLDER + "//step_time_length.svg", format="svg")

# %% Online decoding performance
obs_df = online_df[online_df.block == "OBS"].copy()

# confusion matrix
fig, axs = plt.subplots(1, 1, figsize=(3, 3))
conf_mat = confusion_matrix(
    obs_df["goal"], obs_df["pred"], normalize="true", labels=CMDS
)
sns.heatmap(
    conf_mat * 100,
    annot=True,
    fmt=".1f",
    cmap="Blues",
    cbar=False,
    xticklabels=["%s (%s)" % (_c, _f) for _c, _f in zip(CMDS, freqs)],
    yticklabels=["%s (%s)" % (_c, _f) for _c, _f in zip(CMDS, freqs)],
    ax=axs,
)
axs.set_title("%.1f" % (100 * balanced_accuracy_score(obs_df["goal"], obs_df["pred"])))
axs.set_xlabel("Predicted")
axs.set_ylabel("True")

fig.tight_layout()
plt.savefig(FOLDER + "//decoding_acc.svg", format="svg")

# %% Reaching performance
test_blocks = [3, 5, 6, 7]

# DC and SC trials
test_df = online_df[online_df.block_i.isin(test_blocks)].copy()
trial_df = test_df.groupby(by=["label", "block", "goal"])["success"].max().reset_index()
trial_df["len_cm"] = list(test_df.groupby(["label"])["dL"].sum() / 10)

# success rate
session_df = (
    trial_df.groupby(["block"])["success"].sum()
    / trial_df.groupby(["block"])["success"].count()
    * 100
).reset_index()
session_df.rename(columns={"success": "success_rate"}, inplace=True)

# trajectory length (only include objects with >1 successful reach)
len_df = (
    trial_df[trial_df.success == 1].groupby(["block", "goal"])["len_cm"].mean()
).reset_index()
len_valid = len_df[len_df.block == "DC"].merge(len_df[len_df.block == "SC"], on="goal")
session_df["len_cm"] = [len_valid.len_cm_x.mean(), len_valid.len_cm_y.mean()]

# %% Plot reaching results
fig, axs = plt.subplots(1, 4, figsize=(5, 2), width_ratios=[2, 1, 2, 1])

# success rate
sns.barplot(data=session_df, x="block", y="success_rate", ax=axs[0])
axs[0].set_ylabel("Success rate (%)")
axs[0].set_ylim([0, 100])

# change in success rate
dF = (
    session_df[session_df.block == "SC"].success_rate.values
    - session_df[session_df.block == "DC"].success_rate.values
)
sns.barplot(y=dF, ax=axs[1])
axs[1].set_ylabel("$\Delta$ Success rate (%)")
axs[1].set_ylim([-100, 100])

# trajectory length
sns.barplot(data=session_df, x="block", y="len_cm", ax=axs[2])
axs[2].set_ylabel("Trajectory length (cm)")
axs[2].set_ylim([0, 80])

# change in trajectory length
dL = (
    session_df[session_df.block == "SC"].len_cm.values
    - session_df[session_df.block == "DC"].len_cm.values
)
sns.barplot(y=dL, ax=axs[3])
axs[3].set_ylabel("$\Delta$ Trajectory length (cm)")
axs[3].set_ylim([-20, 20])

for ax, xlabel in zip(axs, ["", "SC-DC", "", "SC-DC"]):
    ax.set_xlabel(xlabel)
sns.despine()
fig.tight_layout()
plt.savefig(FOLDER + "//reaching_results.svg", format="svg")

# store data
session_df = pd.concat(
    [session_df, pd.DataFrame({"block": "SC-DC", "success_rate": dF, "len_cm": dL})]
)
session_df.to_csv(
    FOLDER
    + "//P%s_results_session_%s.csv" % (P_ID, datetime.now().strftime("%Y%m%d_%H%M%S"))
)
session_df.head()

# %% 3D reaching trajectories
%matplotlib qt
shelf_dims = {'x':0.430, 'y':-0.220, 'z':[-0.160, -0.270, -0.380], 'w':0.400, 'd':0.100, 'h':0.006}
support_dims = {'x':0.430, 'y':[-0.415, -0.025], 'z':-0.270, 'w':0.010, 'd':0.100, 'h':0.220}

def plot_box(x, y, z, ax, alpha, col):
    x_grid, y_grid, z_grid = np.meshgrid(x, y, z)
    ax.plot_surface(x_grid[0, :, :], y_grid[0, :, :], z_grid[0, :, :], alpha=alpha, color=col)
    ax.plot_surface(x_grid[1, :, :], y_grid[1, :, :], z_grid[1, :, :], alpha=alpha, color=col)
    ax.plot_surface(x_grid[:, 0, :], y_grid[:, 0, :], z_grid[:, 0, :], alpha=alpha, color=col)
    ax.plot_surface(x_grid[:, 1, :], y_grid[:, 1, :], z_grid[:, 1, :], alpha=alpha, color=col)
    ax.plot_surface(x_grid[:, :, 0], y_grid[:, :, 0], z_grid[:, :, 0], alpha=alpha, color=col)
    ax.plot_surface(x_grid[:, :, 1], y_grid[:, :, 1], z_grid[:, :, 1], alpha=alpha, color=col)    

n_pts = 10
mpl.rcParams.update(mpl.rcParamsDefault)
start_poss = []
for label in test_df.label.unique():
    trial = test_df[test_df.label == label]
    col = "r"  # collision
    if trial.success.max():
        col = "g"  # success

    fig = plt.figure(figsize=(10, 7))
    ax = plt.axes(projection="3d")
    ax.set_title(label + f' {sum(trial["dt"]):.1f}s')

    # trajectories
    traj = np.array(trial[["x", "y", "z"]]) - T_NOM
    ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], col, alpha=0.7)
    ax.plot(traj[0, 0], traj[0, 1], traj[0, 2], "kx")
    start_poss.append(traj[0, :])

    # cylinder objects
    for test_obj in test_df.goal.unique():
        xc, yc, zc = OBJ_COORDS[int(test_obj)] - T_NOM
        z = np.linspace(0, OBJ_H, n_pts) - OBJ_H / 2 + zc
        theta = np.linspace(0, 2 * np.pi, n_pts)
        theta_grid, z_grid = np.meshgrid(theta, z)
        x_grid = OBJ_R * np.cos(theta_grid) + xc
        y_grid = OBJ_R * np.sin(theta_grid) + yc
        obj_col = "g" if test_obj == trial.goal.max() else "r"
        ax.plot_surface(x_grid, y_grid, z_grid, alpha=0.8, color=obj_col)
        
    # shelf racks
    for shelf_z in shelf_dims['z']:
        xc, yc, zc = np.array([shelf_dims['x'], shelf_dims['y'], shelf_z]) - T_NOM
        x = [xc - shelf_dims['d'] / 2, xc + shelf_dims['d'] / 2]
        y = [yc - shelf_dims['w'] / 2, yc + shelf_dims['w'] / 2]
        z = [zc - shelf_dims['h'], zc]
        plot_box(x, y, z, ax, alpha=0.2, col='grey')
    
    # shelf sides
    for support_y in support_dims['y']:
        xc, yc, zc = np.array([support_dims['x'], support_y, support_dims['z']]) - T_NOM
        x = [xc - support_dims['d'] / 2, xc + support_dims['d'] / 2]
        y = [yc - support_dims['w'] / 2, yc + support_dims['w'] / 2]
        z = [zc-support_dims['h']/2, zc + support_dims['h']/2]
        plot_box(x, y, z, ax, alpha=0.2, col='grey')

    ax.set_box_aspect([1, 1, 1])
    ax.set_xlim([-0.3, 0.3])
    ax.set_xlabel("x (m)")
    ax.set_ylim([-0.3, 0.3])
    ax.set_ylabel("y (m)")
    ax.set_zlim([-0.3, 0.3])
    ax.set_zlabel("z (m)")

print("T0 mean: %s " % ["%.3f" % _x for _x in np.mean(start_poss, axis=0)])
print("T0 std: %s" % ["%.3f" % _x for _x in np.std(start_poss, axis=0)])
plt.show()

# %% Offline decoding with variable window size
raw, events = load_recording(CH_NAMES, FOLDER, f"P{P_ID}_S1_R1.xdf")
obs_events = extract_events(events, [f"go:{_c}" for _c in CMDS])
freqs = [int(_f) for _f in extract_events(events, ["freqs"])[0][2][-11:].split(",")]
bp_filt = BandpassFilter(FILTER_ORDER, FS, FMIN, FMAX, len(SSVEP_CHS))

# load observation run data
Nch = len(SSVEP_CHS)
filt = mne.io.RawArray(
    bp_filt.filter(raw.get_data(SSVEP_CHS)), mne.create_info(Nch, FS, ["eeg"] * Nch)
)
epochs = mne.Epochs(
    filt,
    [[_e[0], _e[1], ord(_e[2].split(":")[-1])] for _e in obs_events],
    tmin=0,
    tmax=OBS_TRIAL_MS / 1000,
    baseline=None,
)

fig, axs = plt.subplots(1, 5, figsize=(15, 3))
for window_s, ax in zip([1, 1.5, 2, 2.5, 3], axs):
    decoder = Decoder(window_s, FS, HARMONICS, freqs)

    # offline decoder prediction
    y_actual = []
    y_preds = []
    N_w = int(window_s * FS)
    N_c = int(SAMPLE_T_MS / 1000 * FS)
    for trial_i, data in enumerate(epochs.get_data()):
        label = obs_events[trial_i][2][-1]
        for sample_i in range(N_w, data.shape[1], N_c):
            y_actual.append(label)
            pred = decoder.predict(data[:, sample_i - N_w : sample_i])
            y_preds.append(list(CMD_MAP.keys())[pred])

    conf_mat = confusion_matrix(y_actual, y_preds, normalize="true", labels=CMDS)

    # confusion matrix
    sns.heatmap(
        conf_mat * 100,
        annot=True,
        fmt=".1f",
        cmap="Blues",
        cbar=False,
        xticklabels=["%s (%s)" % (_c, _f) for _c, _f in zip(CMDS, freqs)],
        yticklabels=["%s (%s)" % (_c, _f) for _c, _f in zip(CMDS, freqs)],
        ax=ax,
    )
    ax.set_title(
        f"{100 * balanced_accuracy_score(y_actual, y_preds):.1f} ({window_s:.1f}s)"
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

fig.tight_layout()
plt.savefig(FOLDER + "//offline_decoding_acc.svg", format="svg")
