# %% Packages
import mne
import numpy as np
import pandas as pd
from sklearn.metrics import recall_score

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
FOLDER = ""

# %% Load, filter and epoch recording
p_id = "P7"

# load recording
raw, events = load_recording(CH_NAMES, FOLDER, f"//{p_id}_S1_R1.xdf")
obs_events = extract_events(events, [f"go:{_c}" for _c in CMDS])
freqs = [int(_f) for _f in extract_events(events, ["freqs"])[0][2][-11:].split(",")]
c_to_f = {_c: _f for _c, _f in zip(CMDS, freqs)}
bp_filt = BandpassFilter(FILTER_ORDER, FS, FMIN, FMAX, len(SSVEP_CHS))

# epoch observation run data
Nch = len(SSVEP_CHS)
filt = mne.io.RawArray(
    bp_filt.filter(raw.get_data(SSVEP_CHS)), mne.create_info(Nch, FS, ["eeg"] * Nch)
)
epochs = mne.Epochs(
    filt,
    [[_e[0], _e[1], c_to_f[_e[2].split(":")[-1]]] for _e in obs_events],
    tmin=0,
    tmax=OBS_TRIAL_MS / 1000,
    baseline=None,
)
epochs.save(FOLDER + f"//{p_id}_obs-epo.fif")

# %% Offline decoding with variable window size
windows = [1, 1.5, 2, 2.5, 3]

recall = []
for window_s in windows:
    decoder = Decoder(window_s, FS, HARMONICS, freqs)

    # offline decoder prediction
    y_actual = []
    y_preds = []
    N_w = int(window_s * FS)
    N_c = int(SAMPLE_T_MS / 1000 * FS)
    for trial_i, data in enumerate(epochs.get_data()):
        label = epochs.events[trial_i][2]
        for sample_i in range(N_w, data.shape[1], N_c):
            y_actual.append(label)
            pred = decoder.predict(data[:, sample_i - N_w : sample_i])
            y_preds.append(freqs[pred])

    recall.append(recall_score(y_actual, y_preds, labels=freqs, average=None) * 100)

# plot recall vs window size
recall_df = pd.DataFrame(np.array(recall).T, columns=windows)
recall_df["freqs"] = freqs
recall_df.to_csv(FOLDER, f"//{p_id}_recall.csv", index=False)

# %% Correlation scores across different freqs
targets = np.arange(7, 26, 0.2)
window_s = 1

# build decoder
decoder = Decoder(window_s, FS, HARMONICS, targets)
N_w = int(window_s * FS)
N_c = int(SAMPLE_T_MS / 1000 * FS)

# get scores for each window
rhos = []
labels = []
window_eps = []
for trial_i, data in enumerate(epochs.get_data()):
    label = epochs.events[trial_i][2]
    for sample_i in range(N_w, data.shape[1], N_c):
        window = data[:, sample_i - N_w : sample_i]
        window_eps.append(window)
        rhos.append(decoder.score(window))
        labels.append(label)

rho_df = pd.DataFrame(rhos, columns=targets)
rho_df["p_id"] = p_id
rho_df["ep_freq"] = labels
rho_df.to_csv(FOLDER, f"//{p_id}_rho.csv", index=False)
