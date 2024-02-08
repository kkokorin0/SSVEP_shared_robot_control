import logging
import socket
import time
import tkinter as tk
from datetime import datetime
from random import randint, sample, seed

import coloredlogs
import numpy as np
import pygame
from pylsl import StreamInfo, StreamInlet, StreamOutlet, resolve_byprop

from decoding import BandpassFilter, Decoder
from robot_control import ReachyRobot
from stimulus import StimController

# session
P_ID = 99  # participant ID
FOLDER = r"C:\Users\kkokorin\OneDrive - The University of Melbourne\Documents\CurrentStudy\Logs"
INIT_MS = 10000  # time to settle at the start of each block

# observation block
OBS_TRIALS = 10  # trials per direction
PROMPT_MS = 2000  # direction prompt
DELAY_MS = 1000  # prompt jitter
OBS_TRIAL_MS = 3600  # observation trial duration
OBS_REST_MS = 2000  # rest period
OFFSET_MS = 1000  # offset in the opposite direction

# reaching block
REACH_TRIALS = 3
N_OBJ = 4
OBJ_COORDS = []

# control
SAMPLE_T_MS = 200
CMDS = ["u", "d", "l", "r", "f"]
CMD_MAP = {
    "u": [0, 0, 1],
    "d": [0, 0, -1],
    "l": [0, 1, 0],
    "r": [0, -1, 0],
    "f": [1, 0, 0],
    "b": [-1, 0, 0],
}

# stimulus
FREQS = [7, 8, 9, 11, 13]  # top, bottom, left, right, middle (Hz)
STIM_DIST = 0.15  # distance from end-effector (m)
HOLOLENS_IP = "192.168.137.228"  # HoloLens

# robot
REACHY_WIRED = "169.254.238.100"  # Reachy
SETUP_POS = [25, 0, 0, -110, 0, -10, 0]  # starting joint angles
REST_POS = [15, 0, 0, -75, 0, -30, 0]  # arm drop position
MOVE_SPEED_S = 1  # arm movement duration
QR_OFFSET = [0.03, -0.01, 0.02]  # fine-tune QR position

# recording/decoding
FS = 256  # sample rate (Hz)
MAX_SAMPLES = 7680  # max samples to pull from stream
N_CH = 9  # number of SSVEP channels
FMIN = 1  # filter lower cutoff (Hz)
FMAX = 40  # filter upper cutoff (Hz)
FILTER_ORDER = 4  # filter order
WINDOW_S = 1  # decoder window length (s)
HARMONICS = [1, 2]  # signals to include in template

global experiment_running


class ExperimentGuiApp:
    port = 12345

    def __init__(self, master=None):
        """Setup GUI for running the experiment blocks, either the (i) observation task or the (ii) reaching
        task in direct/shared control mode.

        Args:
            master (_type_, optional): Root process. Defaults to None.
        """
        pygame.init()  # timer

        # block parameters
        self.shared_control_on = False

        # logging
        session_id = str(P_ID) + "_" + datetime.now().strftime("%Y_%m_%d")
        log_file = FOLDER + "//" + session_id + ".log"
        logging.basicConfig(filename=log_file, level=logging.DEBUG)
        coloredlogs.install(level="WARNING", fmt="%(asctime)s,%(msecs)03d: %(message)s")
        self.logger = logging.getLogger(__name__)

        # comms
        reachy_robot = ReachyRobot(REACHY_WIRED, self.logger)
        reachy_robot.turn_off(REST_POS, MOVE_SPEED_S, safely=True)
        unity_game = StimController(HOLOLENS_IP, self.logger)

        # marker stream
        marker_info = StreamInfo("MarkerStream", "Markers", 1, 0, "string", session_id)
        self.marker_stream = StreamOutlet(marker_info)
        self.logger.critical(
            "Connected to marker stream and set-up lab recorder (y/n)?"
        )
        if input() != "y":
            reachy_robot.turn_off(REST_POS, MOVE_SPEED_S, safely=True)
            self.logger.critical("Streams not set up, exiting")
        unity_game.setup_stim([0, 0, 0, 0, 0], [0, 0, 0], 0)

        # find EEG stream
        self.logger.warning("Looking for EEG stream....")
        eeg_streams = resolve_byprop("type", "EEG")
        self.logger.critical("Resolved EEG stream %s" % str(eeg_streams[0].desc()))
        self.eeg_stream = StreamInlet(eeg_streams[0])

        # build GUI
        self.display_socket = socket.socket()
        self.gui_enabled = False
        self.toplevel = tk.Tk() if master is None else tk.Toplevel(master)
        self.toplevel.title("Experiment")
        self.setup_frame = tk.LabelFrame(self.toplevel)

        # experiment frame
        self.setup_frame.configure(height="150", text="Setup Experiment", width="400")
        self.setup_frame.place(anchor="nw", height="120", width="400", x="10", y="10")
        checkbox_x = 150
        checkbox_y = 10

        # experiment settings
        self.shared_box = tk.Checkbutton(self.setup_frame)
        self.shared_box.configure(
            text="Shared Control (on/off)", command=self.shared_box_cb
        )
        self.shared_box.place(anchor="nw", x=checkbox_x, y=checkbox_y)

        self.test_mode_box = tk.Checkbutton(self.setup_frame)
        self.test_mode_box.configure(
            text="Test Mode (on/off)", command=self.test_mode_cb
        )
        self.test_mode_box.place(anchor="nw", x=checkbox_x, y=checkbox_y + 30)

        # action button design
        button_width = 7
        button_height = 1
        button_x0 = 10
        button_y0 = 10
        button_x1 = 75
        button_y1 = 55
        self.off_bg = "#F2F3F4"
        self.off_txt = "#c0c0c0"

        # start trial
        self.start_button = tk.Button(self.setup_frame)
        self.start_button.configure(
            background="#00ff00",
            height=button_height,
            overrelief="raised",
            state="normal",
            takefocus=False,
            text="Start",
            width=button_width,
            disabledforeground=self.off_txt,
            command=self.start_button_cb,
            activebackground="#00ff00",
        )
        self.start_button.place(anchor="nw", x=button_x0, y=button_y0)

        # stop robot motion
        self.stop_button = tk.Button(self.setup_frame)
        self.stop_button.configure(
            activebackground="#ff0000",
            background=self.off_bg,
            disabledforeground=self.off_txt,
            height=button_height,
            overrelief="raised",
            state="disabled",
            text="Stop",
            width=button_width,
            command=self.stop_button_cb,
        )
        self.stop_button.place(anchor="nw", x=button_x1, y=button_y0)

        # turn off robot
        self.off_button = tk.Button(self.setup_frame)
        self.off_button.configure(
            activebackground="#ff8000",
            background=self.off_bg,
            disabledforeground=self.off_txt,
            height=button_height,
            overrelief="raised",
            state="disabled",
            text="Off",
            width=button_width,
            command=self.off_button_cb,
        )
        self.off_button.place(anchor="nw", x=button_x0, y=button_y1)

        # main widget
        self.toplevel.configure(height="480", width="640")
        self.toplevel.geometry("640x480")
        self.mainwindow = self.toplevel

    def run(self):
        """Run the main GUI loop"""
        self.mainwindow.mainloop()

    def shared_box_cb(self):
        """Toggle shared control"""
        self.shared_control_on = not self.shared_control_on
        self.logger(
            "Toggle shared control %s" % ("On" if self.shared_control_on else "Off")
        )

    def start_button_cb(self):
        self.start_button.configure(state="disabled", background=self.off_bg)

        global experiment_running
        experiment_running = True

        # generate trials
        seed(P_ID)
        self.obs_trials = np.array(
            [sample(CMDS, len(CMDS)) for _i in range(OBS_TRIALS)]
        ).reshape(-1)
        self.logger.warning(self.obs_trials)

        eeg_stream.flush()
        eeg_stream.pull_sample()

        marker_stream.push_sample(["start run"])
        self.logger.warning("Start run")
        pygame.time.delay(INIT_MS)

        # setup online filter and decoder
        bp_filter = BandpassFilter(FILTER_ORDER, FS, FMIN, FMAX, N_CH)
        X_chunk, ts = eeg_stream.pull_chunk(max_samples=MAX_SAMPLES)
        self.logger.warning("t=%.3fs pulled %d samples" % (ts[-1], len(X_chunk)))

        bp_filter.filter(np.array(X_chunk).T[:N_CH, :])
        decoder = Decoder(WINDOW_S, FS, HARMONICS, FREQS)

        while experiment_running:
            self.toplevel.update()

        time.sleep(0.5)
        # self.stop_button_cb()
        # self.init_button.configure(state="normal", background="#00ff00")

    def stop_button_cb(self):
        """Stop the arm at the current position and end the trial"""
        global experiment_running
        experiment_running = False
        self.start_button.configure(state="disabled", background=self.off_bg)

        # self.TestReachyRobot.turn_off(safely=True)
        # self.init_button.configure(state="normal", background="#00ff00")

    def off_button_cb(self):
        """Reset the arm and turn it off"""
        global experiment_running
        experiment_running = False
        # self.start_button.configure(state="disabled", background=self.off_bg)
        # self.init_button.configure(state="normal", background="#00ff00")


if __name__ == "__main__":
    app = ExperimentGuiApp()
    app.run()
