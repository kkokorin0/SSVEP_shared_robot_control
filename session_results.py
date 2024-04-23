# %% Packages
import os
from datetime import datetime

import numpy as np
import pandas as pd

from decoding import extract_events, load_recording

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
P_ID = 1
FOLDER = ""

# %% Extract results from folder with participant recordings
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
        freqs = [
            int(_f) for _f in freq_str.strip("freqs:").split(",")
        ]  # up, down, left, right, forward
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
