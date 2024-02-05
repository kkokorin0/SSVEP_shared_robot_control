# %% Packages
import os

import matplotlib.pyplot as plt
import mne
import numpy as np
import seaborn as sns

from decoding import Decoder, load_recording

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
]  # TODO check these
SSVEP_CHS = CH_NAMES[:9]
DIRECTIONS = ["u", "d", "l", "r", "f"]
STIM_FS = [7, 8, 9, 11, 13]
FS = 256
SAMPLE_T = 0.2
WINDOW_T = 1
HARMONICS = [1, 2]

# %% Load and epoch data
folder = r"C:\Users\Kirill Kokorin\OneDrive - synchronmed.com\SSVEP robot control\Data\Observation pilot\P99_S01"
plot_signals = False
fmin, fmax = 1, 40
ep_tmin, ep_tmax = 0, 2.5

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
        go_events = {"go " + _d for _d in DIRECTIONS}
        filtered_events = [
            [e[0], e[1], ord(e[2][-1])] for e in events if e[2] in go_events
        ]

        epochs = mne.Epochs(
            raw,
            filtered_events,
            baseline=None,
            tmin=ep_tmin,
            tmax=ep_tmax,
            picks="eeg",
            event_id={_d: ord(_d) for _d in DIRECTIONS},
        )
        X.append(epochs.get_data(SSVEP_CHS))
        y.append([chr(_e) for _e in epochs.events[:, -1]])

X, y = np.concatenate(X), np.concatenate(y)

# %% Predict trial stimulus
decoder = Decoder(WINDOW_T, FS, HARMONICS, STIM_FS)
n_window = int(WINDOW_T * FS)
n_chunk = int(SAMPLE_T * FS)

scores = []
preds = []
for X_i, y_i in zip(X, y):
    y_pred_i = []
    score_i = []
    for ti_min in range(n_window, X_i.shape[1] - n_chunk, n_chunk):
        X_slice = X_i[:, ti_min - n_window : ti_min]
        score_i.append(decoder.score(X_slice))
        y_pred_i.append(DIRECTIONS[np.argmax(score_i[-1])])

    preds.append(y_pred_i)
    scores.append(score_i)

# %% Results summary
fig, axs = plt.subplots(1, 2, figsize=(6, 2))

# predictions
correct_preds = np.sum([np.array(_p) == _l for _l, _p in zip(y, preds)], axis=1)
n_slices = len(preds[0])

ax = axs[0]
sns.histplot(
    correct_preds / n_slices * 100,
    stat="count",
    ax=ax,
    binwidth=100 / n_slices,
)
ax.set_xlabel(
    "Percentage of time steps with\ncorrect predictions ({0:.1f}-{1:.1f}s)".format(
        n_window / FS, (X_i.shape[1] - n_chunk) / FS
    )
)
ax.set_xlim([0, 100])
ax.set_ylabel("Trials")

# correlations
ax = axs[1]
sns.histplot(
    np.array(scores).reshape((-1, len(STIM_FS))),
    stat="count",
    ax=ax,
)
ax.set_xlabel("Correlation score")
ax.set_xlim([0, 1])
ax.legend(STIM_FS)

sns.despine()
fig.tight_layout()

# %% Result by direction
fig, axs = plt.subplots(3, 2, figsize=(6, 4), sharex=False, sharey=True)

for letter, ax in zip(DIRECTIONS, axs.flatten()):
    sns.histplot(
        (correct_preds / n_slices * 100)[y == letter],
        stat="count",
        ax=ax,
        binwidth=100 / n_slices,
    )
    ax.set_xlim([0, 100])
    ax.set_ylabel("Trials")
    ax.legend(letter)

axs.flatten()[4].set_xlim([0, 100])
axs.flatten()[4].set_xlabel(
    "Percentage of time steps with\ncorrect predictions ({0:.1f}-{1:.1f}s)".format(
        n_window / FS, (X_i.shape[1] - n_chunk) / FS
    )
)
axs.flatten()[5].remove()
sns.despine()
fig.tight_layout()
