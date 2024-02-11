# %% Packages
import os

import matplotlib.pyplot as plt
import mne
import numpy as np
import seaborn as sns

from decoding import BandpassFilter, Decoder, extract_events, load_recording, signal

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
DIRECTIONS = ["u", "d", "l", "r", "f"]
STIM_FS = [7, 8, 9, 11, 13]
FS = 256
SAMPLE_T = 0.2
WINDOW_T = 1
HARMONICS = [1, 2]

# %% Load and epoch data
folder = r"C:\Users\Kirill Kokorin\OneDrive - synchronmed.com\SSVEP robot control\Data\Observation pilot\P99_S02"
plot_signals = False
fmin, fmax = 1, 40
ep_tmin, ep_tmax = 0, 3.6

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

# %% Filtering comparison
recording_len = 30 * FS
filter_order = 4
file = r"sub-P99_ses-S2_task-online_run-002_eeg.xdf"

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

# %% Online decoding
online_preds = extract_events(events, ["pred", "go"])
fig, axs = plt.subplots(2, 3, figsize=(6, 2.5), sharex=True, sharey=True)

# get predictions
pc_correct = []
dts = []
trial_label = None
for ts, _, label in online_preds:
    if "go" in label:
        if trial_label:
            pc_correct.append(
                np.sum(np.array(trial_preds)[:, 1] == trial_label) / len(trial_preds)
            )
            dts.append(
                [_t - trial_preds[_i][0] for _i, (_t, _p) in enumerate(trial_preds[1:])]
            )
        trial_label = label[-1]
        trial_preds = []
    elif "pred" in label:
        trial_preds.append([ts, label[-1]])
pc_correct.append(np.sum(np.array(trial_preds)[:, 1] == trial_label) / len(trial_preds))

# summed
sns.histplot(np.array(pc_correct) * 100, stat="count", ax=axs[0, 0], binwidth=10)
axs[0, 0].legend(["all"])
axs[0, 0].set_ylabel("")

# split by direction
for letter, ax in zip(DIRECTIONS, axs.flatten()[1:]):
    sns.histplot(
        (correct_preds / n_slices * 100)[y == letter],
        stat="count",
        ax=ax,
        binwidth=10,
    )
    ax.legend(letter, loc="upper left")

axs[1, 0].set_xlabel(
    "Percentage of time steps with\ncorrect predictions ({0:.1f}-{1:.1f}s)".format(
        n_window / FS, (X_i.shape[1] - n_chunk) / FS
    )
)
axs[1, 0].set_ylabel("Trials")
axs[1, 0].set_xlim([0, 100])
sns.despine()
fig.tight_layout()

# time between predictions
fig, axs = plt.subplots(1, 1, figsize=(4, 2))
sns.histplot([_x / FS for _xs in dts for _x in _xs], stat="count", ax=axs)
axs.set_xlabel("Time between predictions (s)")

sns.despine()
fig.tight_layout()
