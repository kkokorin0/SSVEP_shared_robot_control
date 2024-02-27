# %% Packages
import os
from datetime import datetime

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import seaborn as sns
from mpl_toolkits import mplot3d
from sklearn.metrics import confusion_matrix

from decoding import BandpassFilter, Decoder, extract_events, load_recording, signal
from session_manager import CMD_MAP, FS, HARMONICS, SAMPLE_T_MS, WINDOW_S

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
folder = r"C:\Users\Kirill Kokorin\OneDrive - synchronmed.com\SSVEP robot control\Data\Pilot\P96_S01"

block_i = 0
online_results = []
for file in os.listdir(folder):
    if file.endswith(".xdf"):
        raw, events = load_recording(CH_NAMES, folder, file)
        session_events = extract_events(
            events, ["freqs", "start run", "end run", "go", "pred", "reach"]
        )

        p_id, freq_str = session_events[0][-1].split(" ")
        freqs = [int(_f) for _f in freq_str.strip("freqs:").split(",")]
        for ts, _, label in session_events:
            if "start run" in label:
                block = label.strip("start run: ")
                block_i += 1
                trial_i = 0
            elif "go:" in label:
                goal = label[-1]
                trial_i += 1
                trial_start_row = len(online_results)
            elif "pred" in label:
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
                    success = 0  # assume fail unless reached flag found
                else:
                    pred_obj = np.nan
                    confidence = np.nan
                    alpha = np.nan
                    u_robot = [np.nan, np.nan, np.nan]
                    u_cmb = [np.nan, np.nan, np.nan]
                    success = int(goal == pred)

                online_results.append(
                    [
                        block_i,
                        block,
                        trial_i,
                        ts,
                        goal,
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
            elif "reach" in label:
                if label.split(" ")[1][-1] == goal:
                    for row in online_results[trial_start_row : len(online_results)]:
                        row[5] = 1

online_df = pd.DataFrame(
    online_results,
    columns=[
        "block_i",
        "block",
        "trial",
        "ts",
        "goal",
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
online_df.to_csv(folder + "//results_%s.csv" % datetime.now().strftime("%Y%m%d_%H%M%S"))

# %% Load data
results = "results_20240227_101820.csv"
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
axs[1].set_ylim([0, 0.5])

fig.tight_layout()
sns.despine()

# %% Online decoding performance
fig, axs = plt.subplots(2, 3, figsize=(6, 3), sharex=True, sharey=True)
obs_df = online_df[online_df.block == "OBSF"].copy()

# summed
correct_preds = np.array(obs_df.groupby(["label"])["success"].sum())
total_preds = np.array(obs_df.groupby(["label"])["success"].count())
ax = sns.histplot(
    correct_preds / total_preds * 100, stat="count", ax=axs[0, 0], binwidth=10
)
ax.legend(["all"])
ax.set_ylabel("")
ax.set_xlim([0, 100])

# split by direction
for letter, ax in zip(CMDS, axs.flatten()[1:]):
    letter_df = obs_df[obs_df.goal == letter]
    correct_preds = np.array(letter_df.groupby(["label"])["success"].sum())
    total_preds = np.array(letter_df.groupby(["label"])["success"].count())
    sns.histplot(
        correct_preds / total_preds * 100,
        stat="count",
        ax=ax,
        binwidth=10,
    )
    ax.set_ylabel("Trials")
    ax.legend(letter)

axs[1, 0].set_xlabel("Time steps with correct \npredictions (%)")
sns.despine()
fig.tight_layout()

# overall accuracy
fig, axs = plt.subplots(1, 1, figsize=(2, 2))
acc_rate = (
    obs_df.groupby(["goal"])["success"].sum()
    / obs_df.groupby(["goal"])["success"].count()
)
acc_rate_df = pd.DataFrame(acc_rate).reset_index()
sns.barplot(data=acc_rate_df, x="goal", y="success", ax=axs, order=CMDS)
axs.set_ylabel("Correct time steps (%)")
axs.set_xlabel("Direction")
axs.set_ylim([0, 1])
sns.despine()
fig.tight_layout()

# confusion matrix
fig, axs = plt.subplots(1, 1, figsize=(3, 3))
conf_mat = confusion_matrix(
    obs_df["goal"], obs_df["pred"], labels=CMDS, normalize="true"
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
axs.set_xlabel("Predicted")
axs.set_ylabel("True")

# %% Reaching performance
fig, axs = plt.subplots(1, 2, figsize=(6, 3), sharex=True, sharey=False)
test_df = online_df[online_df.block.isin(["DC", "SC"])].copy()

# failure rate
successes = pd.DataFrame(test_df.groupby(["label"])["success"].max()).reset_index()
successes["block"] = successes.label.str[0:5]
failure_rate = pd.DataFrame(
    100
    - successes.groupby(["block"])["success"].sum()
    / successes.groupby(["block"])["success"].count()
    * 100,
).reset_index()
failure_rate = failure_rate.rename(columns={"success": "rate"})
sns.pointplot(data=failure_rate, x="block", y="rate", ax=axs[0])
axs[0].set_ylim([0, 100])
axs[0].set_ylabel("Failure rate (%)")
axs[0].set_xlabel("Block")

# trajectory length (successful only)
lengths = pd.DataFrame(
    test_df[test_df.success == 1].groupby(["label"])["dL"].sum() / 1000
).reset_index()
lengths["block"] = lengths.label.str[0:5]
lengths["goal"] = lengths.label.str[-1]
goal_lengths = pd.DataFrame(
    lengths.groupby(["block", "goal"])["dL"].mean()
).reset_index()
sns.pointplot(data=goal_lengths, x="block", y="dL", ax=axs[1])
axs[1].set_ylabel("Trajectory length (m)")
axs[1].set_ylim([0, 1])
axs[1].set_xlabel("Block")

sns.despine()
fig.tight_layout()

# %% End-effector starting position in reaching trials
start_pos_df = test_df[test_df.dL == 0]

fig = plt.figure(figsize=(10, 7))
ax = plt.axes(projection="3d")

for label in test_df.label.unique():
    col = "r" if "DC" in label else "b"
    traj = np.array(test_df[test_df.label == label][["x", "y", "z"]])
    ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], col)
    ax.plot(traj[0, 0], traj[0, 1], traj[0, 2], "kx")

ax.set_xlim([0, 0.5])
ax.set_xlabel("x (m)")
ax.set_ylim([-0.5, 0])
ax.set_ylabel("y (m)")
ax.set_zlim([-0.5, 0])
ax.set_zlabel("z (m)")
plt.show()

# %% DC - SC
fig, axs = plt.subplots(1, 2, figsize=(2.5, 2), sharex=True, sharey=False)
dc_block = "DC B6"
sc_block = "SC B4"

# failure rate
sns.barplot(
    y=failure_rate[failure_rate.block == sc_block].rate.values
    - failure_rate[failure_rate.block == dc_block].rate.values,
    ax=axs[0],
)
axs[0].set_ylim([-100, 100])
axs[0].set_ylabel("$\Delta$ Failure rate (%)")
axs[0].set_xlabel("SC-DC")

# trajectory length
sns.barplot(
    y=[
        goal_lengths[goal_lengths.block == sc_block]["dL"].mean()
        - goal_lengths[goal_lengths.block == dc_block]["dL"].mean()
    ],
    ax=axs[1],
)
axs[1].set_ylim([-0.4, 0.4])
axs[1].set_ylabel("$\Delta$ Trajectory length  (m)")
axs[1].set_xlabel("SC-DC")

sns.despine()
fig.tight_layout()

# %% Extract observation run data
files = ["P96_S1_R1.xdf"]
plot_signals = False
fmin, fmax = 1, 40
ep_tmin, ep_tmax = 0, 3.6

X, y = [], []
for file in files:
    # load labrecorder file
    raw, events = load_recording(CH_NAMES, folder, file)
    raw = raw.filter(l_freq=fmin, h_freq=fmax)

    if plot_signals:
        raw.plot()
        raw.compute_psd(fmin=1, fmax=40).plot()

    # epoch go trials
    go_events = extract_events(events, ["go"])
    epochs = mne.Epochs(
        raw,
        [[_e[0], 0, ord(_e[2][-1])] for _e in go_events],
        baseline=None,
        tmin=ep_tmin,
        tmax=ep_tmax,
        picks="eeg",
        event_id={_d: ord(_d) for _d in CMDS},
    )
    X.append(epochs.get_data(SSVEP_CHS))
    y.append([chr(_e) for _e in epochs.events[:, -1]])

X, y = np.concatenate(X), np.concatenate(y)

# %% Offline decoding predictions
fig, axs = plt.subplots(2, 5, figsize=(10, 4))

decoder = Decoder(WINDOW_S, FS, HARMONICS, freqs)
n_window = int(WINDOW_S * FS)
n_chunk = int(SAMPLE_T_MS / 1000 * FS)

scores = []
preds = []
for X_i, y_i in zip(X, y):
    y_pred_i = []
    score_i = []
    for ti_min in range(n_window, X_i.shape[1] - n_chunk, n_chunk):
        X_slice = X_i[:, ti_min - n_window : ti_min]
        score_i.append(decoder.score(X_slice))
        y_pred_i.append(CMDS[np.argmax(score_i[-1])])

    preds.append(y_pred_i)
    scores.append(score_i)

# predictions
correct_preds = np.sum([np.array(_p) == _l for _l, _p in zip(y, preds)], axis=1)
n_slices = len(preds[0])

# scores
score_df = pd.DataFrame(np.array(scores).reshape((-1, len(CMDS))))
score_df["label"] = np.repeat(y, n_slices)
score_df = pd.melt(score_df, id_vars="label")
score_df["direction"] = [CMDS[_i] for _i in score_df["variable"]]

for l_i, letter in enumerate(CMDS):

    # predictions
    sns.histplot(
        (correct_preds / n_slices * 100)[y == letter],
        stat="count",
        ax=axs[0, l_i],
        binwidth=10,
    )
    axs[0, l_i].set_title(letter)
    axs[0, l_i].set_xlim([0, 100])
    axs[0, l_i].set_ylim([0, 20])

    # correlations
    sns.ecdfplot(
        data=score_df[score_df.label == letter],
        x="value",
        hue="direction",
        hue_order=CMDS,
        stat="proportion",
        palette={
            _l: sns.color_palette("colorblind", len(CMDS))[l_i]
            for l_i, _l in enumerate(CMDS)
        },
        ax=axs[1, l_i],
    )
    axs[1, l_i].set_xlim([0, 1])
    axs[1, l_i].set_ylim([0, 1])
    axs[1, l_i].set_xlabel("Correlation")
    if l_i != 4:
        axs[1, l_i].get_legend().remove()

axs[0, 0].set_xlabel("Time steps with correct\npredictions (%)")

sns.despine()
fig.tight_layout()

# %% Confusion matrix
fig, axs = plt.subplots(1, 1, figsize=(3, 3))
conf_mat = confusion_matrix(
    np.repeat(y, n_slices), np.concatenate(preds), labels=CMDS, normalize="true"
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
axs.set_xlabel("Predicted")
axs.set_ylabel("True")

# %% Filtering comparison
recording_len = 30 * FS
filter_order = 4
file = r"P099_S3_OBSFB_R5_eeg.xdf"

raw, events = load_recording(CH_NAMES, folder, file)
recording = raw.get_data(SSVEP_CHS)

# online filter
bandpass = BandpassFilter(filter_order, FS, fmin, fmax, len(SSVEP_CHS))
X_filt = []
for ti_min in range(n_chunk, recording_len, n_chunk):
    X_slice = recording[:, ti_min - n_chunk : ti_min]
    X_filt.append(bandpass.filter(X_slice))
X_filt_online = np.concatenate(X_filt, axis=1)

# offline filter
offline_filter = signal.butter(
    N=filter_order, Wn=[fmin, fmax], fs=FS, btype="bandpass", output="sos"
)
X_filt_offline = signal.sosfilt(
    offline_filter, recording[:, : X_filt_online.shape[1]], axis=1
)

fig, axs = plt.subplots(1, 3, figsize=(10, 3), sharex=True, sharey=True)
axs[0].plot(X_filt_offline.T * 1e6)
axs[1].plot(X_filt_online.T * 1e6)
axs[2].plot((X_filt_offline - X_filt_online).T * 1e6)
[axs[_i].set_title(_t) for _i, _t in enumerate(["Offline", "Online", "Difference"])]
axs[0].set_ylim([1000, -1000])
axs[0].set_xlim([0, 10 * FS])
axs[0].axvline(768, color="k", linestyle="--")
axs[0].set_ylabel("Amplitude (uV)")
axs[0].set_xlabel("Time (samples)")
sns.despine()
fig.tight_layout()
