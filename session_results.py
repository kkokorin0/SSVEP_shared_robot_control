# %% Packages
import os
from datetime import datetime

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
P_ID = 1
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

# calculate step length
trial_i = 0
dLs = []
for i, row in online_df.iterrows():
    if row.trial != trial_i:
        trial_i = row.trial
        dLs.append(0)
    else:
        dLs.append(
            np.linalg.norm(
                (row[["x", "y", "z"]] - online_df.iloc[i - 1][["x", "y", "z"]])
            )
            * 1000
        )
online_df["dL"] = dLs

# save results
online_df.to_csv(
    FOLDER
    + "//P%s_results_tstep_%s.csv" % (P_ID, datetime.now().strftime("%Y%m%d_%H%M%S"))
)

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
