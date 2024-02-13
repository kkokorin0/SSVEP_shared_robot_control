import mne
import numpy as np
import pyxdf
from scipy import signal
from sklearn.cross_decomposition import CCA


def extract_events(events, labels):
    """Extract all events that contain a substring from the start to end of a run

    Args:
        events (list): list of [timestep, 0, event]
        labels (list of str): accepted substrings in the event label

    Returns:
        filtered_events (list): list of [timestep, 0, event]
    """
    filtered_events = []
    valid_events = False
    for ts, _, event in events:
        if valid_events and any([_l in event for _l in labels]):
            filtered_events.append([ts, 0, event])
        elif event == "start run":
            valid_events = True
        elif event == "end run":
            valid_events = False
    return filtered_events


def load_recording(ch_names, folder, file):
    """Load xdf recordings into mne raw structure and events array

    Args:
        ch_names (list): list of recording channels
        folder (str): recording directory
        file (str): recording file

    Returns:
        raw (mne.raw): mne raw structure
        events (list): list of events in mne format
    """
    streams, header = pyxdf.load_xdf("{0}//{1}".format(folder, file))

    # label streams
    eeg_index = 0 if streams[0]["info"]["type"][0] == "EEG" else 1
    marker_index = len(streams) - 1 - eeg_index
    print("Found {0} streams, eeg stream is {1}".format(len(streams), eeg_index))

    # create raw structure
    Nch = int(streams[eeg_index]["info"]["channel_count"][0])
    Fs = float(streams[eeg_index]["info"]["nominal_srate"][0])
    eeg_data = streams[eeg_index]["time_series"].T * 1e-6  # convert from uV to V
    eeg_start = streams[eeg_index]["time_stamps"][0]

    info = mne.create_info(Nch, Fs, ["eeg"] * Nch)
    raw = mne.io.RawArray(eeg_data, info)

    ch_name_map = {}
    for i in range(len(ch_names)):
        ch_name_map[str(i)] = ch_names[i]
    raw.rename_channels(ch_name_map)
    raw.set_montage("easycap-M1")

    # create events array
    label_names = streams[marker_index]["time_series"]
    label_stamps = streams[marker_index]["time_stamps"]

    events = [
        [raw.time_as_index(_s - eeg_start)[0], 0, _l[0]]
        for _s, _l in zip(label_stamps, label_names)
    ]
    return raw, events


def make_template(window, fs, harmonics, stim_freqs):
    """Create template for CCA decoding

    Args:
        window (float): window length in seconds
        fs (Hz): sample rate
        harmonics (list of int): which harmonics to include
        stim_freqs (list of int): target frequencies to model with template

    Returns:
        np.array (n_frequencies, n_harmonics x 2, n_samples): templates array
    """
    t = np.arange(0, window, 1 / fs)
    template = [
        [
            [np.sin(2 * np.pi * _fs * _h * t), np.cos(2 * np.pi * _fs * _h * t)]
            for _h in harmonics
        ]
        for _fs in stim_freqs
    ]
    return np.asarray(template).reshape(len(stim_freqs), -1, len(t))


class Decoder:
    """CCA-based decoder for SSVEP"""

    def __init__(self, window, fs, harmonics, stim_freqs):
        """Setup CCA template

        Args:
            window (float): window length in seconds
            fs (Hz): sample rate
            harmonics (list of int): which harmonics to include
            stim_freqs (list of int): target frequencies to model with template

        """
        self.fs = fs
        self.template = make_template(window, fs, harmonics, stim_freqs)
        self.n_f, self.n_c, self.n_s = self.template.shape
        self.cca = CCA(n_components=self.n_c, scale=True)

    def score(self, X):
        """Calculate CCA scores for input data

        Args:
            X (np.array): EEG signal (n_channels, n_samples)

        Returns:
            list: first correlation score for each template frequency
        """
        scores = []
        for t_i in self.template:
            self.cca.fit(X.T, t_i.T)
            X_c, Y_c = self.cca.transform(X.T, t_i.T)
            scores.append(np.corrcoef(X_c.T, Y_c.T).diagonal(offset=self.n_c)[0])

        return scores

    def predict(self, X):
        """Find the template with the highest correlation

        Args:
            X (np.array): EEG signal (n_channels, n_samples)

        Returns:
            int: index of best template
        """
        return np.argmax(self.score(X))


class OnlineDecoder(Decoder):
    """Online decoder that pulls data from an EEG stream, filters it and decodes the signal using CCA"""

    buffer = None

    def __init__(
        self,
        window,
        fs,
        harmonics,
        stim_freqs,
        n_ch,
        online_filter,
        eeg_stream,
        max_samples,
        logger,
    ):
        """Setup decoder, filter and EEG stream

        Args:
            window (float): window length in seconds
            fs (Hz): sample rate
            harmonics (list of int): which harmonics to include
            stim_freqs (list of int): target frequencies to model with template
            n_ch (int): number of channels used for decoding
            online_filter (BandpassFilter): filter data between two frequencies
            eeg_stream (StreamInlet): stream of EEG data
            max_samples (int): max number of samples to pull from stream
            logger (logger): send messages to log file
        """
        super().__init__(window, fs, harmonics, stim_freqs)
        self.filter = online_filter
        self.eeg_stream = eeg_stream
        self.n_ch = n_ch
        self.max_samples = max_samples
        self.buffer_size = int(window * fs)
        self.logger = logger

    def flush_stream(self):
        """Flush the stream"""
        self.eeg_stream.flush()
        self.eeg_stream.pull_sample()

    def filter_chunk(self):
        """Pull chunk of data from stream and filter it

        Returns:
            np.array: filtered data (Nch, Ns)
        """
        X_chunk, ts = self.eeg_stream.pull_chunk(max_samples=self.max_samples)
        self.logger.warning("t=%.3fs pulled %d samples" % (ts[-1], len(X_chunk)))
        return self.filter.filter(np.array(X_chunk).T[: self.n_ch, :])

    def clear_buffer(self):
        """Reset buffer"""
        self.buffer = None

    def update_buffer(self):
        """Update buffer with filtered data"""
        X_filtered = self.filter_chunk()
        if self.buffer is None:
            self.buffer = X_filtered
        else:
            self.buffer = np.append(self.buffer, X_filtered, axis=1)

    def predict_online(self):
        """Predict the best template for a window of data in buffer

        Returns:
            int or None: index of best template or None if buffer is too small
        """
        if self.buffer.shape[1] > self.buffer_size:
            buffer_window = self.buffer[:, -self.buffer_size :]
            return self.predict(buffer_window)
        else:
            return None


class BandpassFilter:
    """Online bandpass filter"""

    def __init__(self, order, fs, fmin, fmax, n_ch):
        """Setup filter coefficients and initialise

        Args:
            order (int): filter order
            fs (int): sampling frequency
            fmin (int): lower cutoff
            fmax (int): upper cutoff
            n_ch (int): number of channels
        """
        self.fs = fs
        self.fmin = fmin
        self.fmax = fmax
        self.n_channels = n_ch
        self.filter_sos = signal.butter(
            N=order,
            Wn=[fmin, fmax],
            fs=fs,
            btype="bandpass",
            output="sos",
        )
        self.filter_z = np.zeros((self.filter_sos.shape[0], n_ch, 2))

    def filter(self, X):
        """Filter chunk of data and update filter state

        Args:
            X (np.array): EEG signal (n_channels, n_samples)

        Returns:
            np.array: filtered data (n_channels, n_samples)
        """
        filtered, self.filter_z = signal.sosfilt(self.filter_sos, X, zi=self.filter_z)
        return filtered
