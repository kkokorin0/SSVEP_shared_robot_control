# %% Packages
import os

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import seaborn as sns

from decoding import BandpassFilter, Decoder, extract_events, load_recording, signal
from session_manager import CMD_MAP, FREQS, FS, HARMONICS, SAMPLE_T_MS, WINDOW_S

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
folder = r"C:\Users\Kirill Kokorin\OneDrive - synchronmed.com\SSVEP robot control\Data\Pilot\P99_S05"
file = r"P99_S5_R1.xdf"

raw, events = load_recording(CH_NAMES, folder, file)
session_events = extract_events(
    events, ["Freqs", "start run", "end run", "go", "pred", "reach"]
)

p_id = session_events[0][-1].split(" ")[0]
online_results = []
block_i = 0
for ts, _, label in session_events:
    if "start run" in label:
        block = label.strip("start run: ")
        block_i += 1
        trial_i = 0
    elif "go" in label:
        goal = label[-1]
        trial_i += 1
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
            success = goal == pred

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
            # TODO iterate backwards and set success to 1 for just this trial
            pass

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

# summed
correct_preds = np.array(online_df.groupby(["trial"])["result"].sum())
total_preds = np.array(online_df.groupby(["trial"])["result"].count())
ax = sns.histplot(
    correct_preds / total_preds * 100, stat="count", ax=axs[0, 0], binwidth=10
)
ax.legend(["all"])
ax.set_ylabel("")
ax.set_xlim([0, 100])

# split by direction
for letter, ax in zip(CMDS, axs.flatten()[1:]):
    letter_df = online_df[online_df.goal == letter]
    correct_preds = np.array(letter_df.groupby(["trial"])["result"].sum())
    total_preds = np.array(letter_df.groupby(["trial"])["result"].count())
    sns.histplot(
        correct_preds / total_preds * 100,
        stat="count",
        ax=ax,
        # binwidth=10,
    )
    ax.set_ylabel("Trials")
    ax.legend(letter)

axs[1, 0].set_xlabel("Time steps with correct \npredictions (%)")
sns.despine()
fig.tight_layout()

# %% Process data offline
plot_signals = False
fmin, fmax = 1, 40
ep_tmin, ep_tmax = 0, 3.6
fig, axs = plt.subplots(4, 2, figsize=(6, 8), sharey=False)

X, y = [], []
for file in os.listdir(folder):
    # load labrecorder file
    if file.endswith(".xdf"):
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

# predict trial stimulus
decoder = Decoder(WINDOW_S, FS, HARMONICS, FREQS)
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

# correlations
ax = axs[0, 0]
sns.histplot(
    np.array(scores).reshape((-1, len(FREQS))),
    stat="count",
    ax=ax,
)
ax.set_xlabel("Correlation score")
ax.set_xlim([0, 1])
ax.legend(FREQS)
axs[0, 1].remove()

# predictions
correct_preds = np.sum([np.array(_p) == _l for _l, _p in zip(y, preds)], axis=1)
n_slices = len(preds[0])

# combined
ax = axs[1, 0]
sns.histplot(
    correct_preds / n_slices * 100,
    stat="count",
    ax=ax,
    binwidth=100 / n_slices,
)
ax.legend(["all"])
ax.set_xlabel("")
ax.set_xlim([0, 100])
ax.set_ylabel("")

# by direction
for letter, ax in zip(CMDS, axs.flatten()[3:]):
    sns.histplot(
        (correct_preds / n_slices * 100)[y == letter],
        stat="count",
        ax=ax,
        # binwidth = 10
    )
    ax.set_xlim([0, 100])
    ax.set_ylabel("")
    ax.legend(letter)

axs[3, 0].set_xlim([0, 100])
axs[3, 0].set_xlabel(
    "Percentage of time steps with\ncorrect predictions ({0:.1f}-{1:.1f}s)".format(
        n_window / FS, (X_i.shape[1] - n_chunk) / FS
    )
)
axs[3, 0].set_ylabel("Trials")
sns.despine()
fig.tight_layout()

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
