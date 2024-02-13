import logging
import socket
import tkinter as tk
from datetime import datetime
from random import randint, sample, seed

import coloredlogs
import numpy as np
import pygame
from pylsl import StreamInfo, StreamInlet, StreamOutlet, resolve_byprop

from decoding import BandpassFilter, OnlineDecoder
from robot_control import ReachyRobot
from stimulus import StimController

# session
P_ID = 99  # participant ID
FOLDER = r"C:\Users\kkokorin\OneDrive - The University of Melbourne\Documents\CurrentStudy\Logs"
INIT_MS = 5000  # time to settle at the start of each block/trial

# observation block
OBS_TRIALS = 5  # trials per direction
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
CMD_MAP = {
    "u": np.array([0, 0, 1]),
    "d": np.array([0, 0, -1]),
    "l": np.array([0, 1, 0]),
    "r": np.array([0, -1, 0]),
    "f": np.array([1, 0, 0]),
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

global experiment_running  # allow to pause arm movement


class ExperimentGuiApp:
    port = 12345

    # block parameters
    shared_control_on = False
    observation_fb = False
    last_trial = 0

    # outer frame
    frame_h = "150"
    frame_w = "400"

    # checkboxes
    cbox_x = 150
    cbox_y = 10
    cbox_dy = 30

    # buttons
    button_w = 7
    button_h = 1
    button_x0 = 10
    button_y0 = 10
    button_dx = 65
    button_dy = 45
    light_grey = "#f2f3f4"
    silver = "#c0c0c0"
    green = "#00ff00"
    yellow = "#ffff00"
    red = "#ff0000"
    orange = "#ff8000"

    def __init__(
        self,
        logger,
        sample_t,
        obs_block,
        reach_block,
        cmd_map,
        reachy_robot,
        unity_game,
        marker_stream,
        decoder,
        master=None,
    ):
        # experiment parameters
        self.sample_t = sample_t
        self.obs_block = obs_block
        self.reach_trials = reach_block
        self.cmd_map = cmd_map

        # IO streams
        self.logger = logger
        self.reachy_robot = reachy_robot
        self.unity_game = unity_game
        self.marker_stream = marker_stream
        self.decoder = decoder

        # build GUI
        self.display_socket = socket.socket()
        self.gui_enabled = False
        self.toplevel = tk.Tk() if master is None else tk.Toplevel(master)
        self.toplevel.title("Experiment")
        self.setup_frame = tk.LabelFrame(self.toplevel)

        # experiment frame
        self.setup_frame.configure(
            height=self.frame_h, text="Setup Experiment", width=self.frame_w
        )
        self.setup_frame.place(
            anchor="nw", height=self.frame_h, width=self.frame_w, x="0", y="0"
        )

        # experiment settings
        self.shared_box = tk.Checkbutton(self.setup_frame)
        self.shared_box.configure(
            text="Shared Control (on/off)", command=self.shared_box_cb
        )
        self.shared_box.place(anchor="nw", x=self.cbox_x, y=self.cbox_y)

        self.obs_fb_box = tk.Checkbutton(self.setup_frame)
        self.obs_fb_box.configure(
            text="Observation Feedback (on/off)", command=self.obs_feedback_cb
        )
        self.obs_fb_box.place(anchor="nw", x=self.cbox_x, y=self.cbox_y + self.cbox_dy)

        # start reaching trial
        self.start_button = tk.Button(self.setup_frame)
        self.start_button.configure(
            background=self.green,
            height=self.button_h,
            overrelief="raised",
            state="normal",
            takefocus=False,
            text="Start",
            width=self.button_w,
            disabledforeground=self.silver,
            command=self.start_button_cb,
            activebackground=self.green,
        )
        self.start_button.place(anchor="nw", x=self.button_x0, y=self.button_y0)

        # start observation block
        self.obs_button = tk.Button(self.setup_frame)
        self.obs_button.configure(
            background=self.yellow,
            height=self.button_h,
            overrelief="raised",
            state="normal",
            takefocus=False,
            text="Observe",
            width=self.button_w,
            disabledforeground=self.silver,
            command=self.obs_button_cb,
            activebackground=self.yellow,
        )
        self.obs_button.place(
            anchor="nw",
            x=self.button_x0 + self.button_dx,
            y=self.button_y0 + self.button_dy,
        )

        # stop robot motion
        self.stop_button = tk.Button(self.setup_frame)
        self.stop_button.configure(
            activebackground=self.red,
            background=self.light_grey,
            disabledforeground=self.silver,
            height=self.button_h,
            overrelief="raised",
            state="disabled",
            text="Stop",
            width=self.button_w,
            command=self.stop_button_cb,
        )
        self.stop_button.place(
            anchor="nw", x=self.button_x0 + self.button_dx, y=self.button_y0
        )

        # turn off robot
        self.off_button = tk.Button(self.setup_frame)
        self.off_button.configure(
            activebackground=self.orange,
            background=self.light_grey,
            disabledforeground=self.silver,
            height=self.button_h,
            overrelief="raised",
            state="disabled",
            text="Off",
            width=self.button_w,
            command=self.off_button_cb,
        )
        self.off_button.place(
            anchor="nw", x=self.button_x0, y=self.button_y0 + self.button_dy
        )

        # main widget
        self.toplevel.configure(height=self.frame_h, width=self.frame_w)
        self.toplevel.geometry(self.frame_w + "x" + self.frame_h)
        self.mainwindow = self.toplevel

    def run(self):
        """Run the main GUI loop"""
        self.mainwindow.mainloop()

    def shared_box_cb(self):
        """Toggle shared control"""
        self.shared_control_on = not self.shared_control_on
        self.logger.critical(
            "Toggle shared control %s" % ("On" if self.shared_control_on else "Off")
        )

    def obs_feedback_cb(self):
        """Toggle observation block feedback"""
        self.observation_fb = not self.observation_fb
        self.logger.critical(
            "Toggle observation feedback %s" % ("On" if self.observation_fb else "Off")
        )

    def start_button_cb(self):
        global experiment_running
        experiment_running = True
        self.start_button.configure(state="disabled", background=self.light_grey)
        self.obs_button.configure(state="disabled", background=self.light_grey)
        self.stop_button.configure(state="normal", background=self.red)

        # self.stop_button_cb()
        # self.init_button.configure(state="normal", background="#00ff00")

    def obs_button_cb(self):
        """Observe the robotic arm move in a given direction and decode EEG data. With feedback turned on
        decoder outputs will control robot velocity"""
        global experiment_running
        experiment_running = True
        self.obs_button.configure(state="disabled", background=self.light_grey)
        self.start_button.configure(state="disabled", background=self.light_grey)
        self.off_button.configure(state="normal", background=self.orange)

        # initialise stream and decoder
        self.decoder.flush_stream()
        self.marker_stream.push_sample(
            ["start run: obs w/%s fb" % ("" if self.observation_fb else "o")]
        )
        self.logger.critical(
            "Start observation run with%s feedback"
            % ("" if self.observation_fb else "out")
        )
        pygame.time.delay(self.obs_block["init"])
        self.decoder.filter_chunk()

        for t_i, trial in enumerate(self.obs_block["trials"]):
            logger.warning(
                "Trial (%s) %d/%d" % (trial, t_i + 1, len(self.obs_block["trials"]))
            )

            # setup arm in the opposite direction from trial
            direction = self.cmd_map[trial]
            self.reachy_robot.setup()
            ef_pose = self.reachy_robot.translate(
                -direction, self.obs_block["start_offset"]
            )

            # highlight direction
            unity_game.prompt([_c == trial for _c in self.cmd_map.keys()], ef_pose)
            self.marker_stream.push_sample(["prompt:%s" % trial])
            pygame.time.delay(
                randint(
                    self.obs_block["prompt"],
                    self.obs_block["prompt"] + self.obs_block["delay"],
                )
            )

            # start flashing
            unity_game.turn_on_stim(ef_pose)
            self.marker_stream.push_sample(["go:%s" % trial])
            trial_start_ms = pygame.time.get_ticks()
            last_stim_update_ms = trial_start_ms
            last_move_ms = trial_start_ms

            # create online buffer and clear stream
            if self.observation_fb:
                self.decoder.clear_buffer()
                self.decoder.filter_chunk()

            # move the robot continuously
            while last_move_ms - trial_start_ms < obs_block["length"]:
                self.toplevel.update()
                ef_pose = reachy_robot.move_continuously(direction, ef_pose)
                data_msg = "X:" + ",".join(["%.3f" % _x for _x in ef_pose[:3, 3]]) + " "

                if last_move_ms - last_stim_update_ms > self.sample_t:
                    # decode EEG chunk
                    self.decoder.update_buffer()
                    pred_i = self.decoder.predict_online()

                    # predict if enough data in buffer
                    if pred_i is not None:
                        pred = list(self.cmd_map.keys())[pred_i]
                        data_msg += "pred:%s" % (pred)

                        # control the robot
                        if self.observation_fb:
                            direction = self.cmd_map[pred]

                    # save data and update stim
                    self.logger.warning(data_msg)
                    self.marker_stream.push_sample([data_msg])
                    unity_game.move_stim(unity_game.coord_transform(ef_pose))
                    last_stim_update_ms = last_move_ms

                last_move_ms = pygame.time.get_ticks()

            # rest while resetting the arm/stim
            self.unity_game.turn_off_stim()
            self.marker_stream.push_sample(["rest:%s" % trial])
            reachy_robot.turn_off(safely=True)
            pygame.time.delay(obs_block["rest"])

        self.marker_stream.push_sample(
            ["end run: obs w/%s fb" % ("" if self.observation_fb else "o")]
        )
        self.logger.critical(
            "End observation run with%s feedback"
            % ("" if self.observation_fb else "out")
        )
        self.off_button_cb()

    def stop_button_cb(self):
        """Stop the arm at the current position and end the trial"""
        global experiment_running
        experiment_running = False
        self.start_button.configure(state="disabled", background=self.light_grey)
        self.obs_button.configure(state="disabled", background=self.light_grey)

        self.logger.warning("Arm stopped")
        self.off_button.configure(state="normal", background=self.orange)

    def off_button_cb(self):
        """Reset the arm and turn it off"""
        self.stop_button.configure(state="disabled", background=self.light_grey)
        self.off_button.configure(state="disabled", background=self.light_grey)

        self.reachy_robot.turn_off(safely=True)
        self.logger.warning("Arm reset and turned off")

        self.start_button.configure(state="normal", background=self.green)
        self.obs_button.configure(state="normal", background=self.yellow)


if __name__ == "__main__":
    pygame.init()  # timer

    # logging
    session_id = str(P_ID) + "_" + datetime.now().strftime("%Y_%m_%d")
    log_file = FOLDER + "//" + session_id + ".log"
    logging.basicConfig(filename=log_file, level=logging.DEBUG)
    coloredlogs.install(level="WARNING", fmt="%(asctime)s,%(msecs)03d: %(message)s")
    logger = logging.getLogger(__name__)

    # observation block
    seed(P_ID)
    obs_trials = np.array(
        [sample(CMD_MAP.keys(), len(CMD_MAP)) for _i in range(OBS_TRIALS)]
    )
    obs_block = dict(
        trials=obs_trials.reshape(-1),
        init=INIT_MS,
        prompt=PROMPT_MS,
        delay=DELAY_MS,
        length=OBS_TRIAL_MS,
        rest=OBS_REST_MS,
        start_offset=OFFSET_MS,
    )
    logger.warning("Observation trials: %s" % "".join(obs_trials.reshape(-1)))

    # reaching block
    reach_block = None

    # comms
    reachy_robot = ReachyRobot(REACHY_WIRED, logger, SETUP_POS, REST_POS, MOVE_SPEED_S)
    reachy_robot.turn_off(safely=True)
    unity_game = StimController(HOLOLENS_IP, logger, QR_OFFSET, FREQS, STIM_DIST)
    unity_game.turn_off_stim()

    # setup marker stream
    marker_info = StreamInfo("MarkerStream", "Markers", 1, 0, "string", session_id)
    marker_stream = StreamOutlet(marker_info)
    logger.critical("Connected to marker stream and set-up lab recorder (y/n)?")
    if input() != "y":
        logger.critical("Streams not set up, exiting")
    marker_stream.push_sample(["start session"])
    marker_stream.push_sample(
        ["P%d freqs:%s" % (P_ID, ",".join(str(_f) for _f in FREQS))]
    )

    # find EEG stream and build online decoder
    logger.warning("Looking for EEG stream....")
    eeg_streams = resolve_byprop("type", "EEG")
    logger.critical("Resolved EEG stream %s" % str(eeg_streams[0].desc()))
    bp_filter = BandpassFilter(FILTER_ORDER, FS, FMIN, FMAX, N_CH)
    decoder = OnlineDecoder(
        window=WINDOW_S,
        fs=FS,
        harmonics=HARMONICS,
        stim_freqs=FREQS,
        n_ch=N_CH,
        online_filter=bp_filter,
        eeg_stream=StreamInlet(eeg_streams[0]),
        max_samples=MAX_SAMPLES,
        logger=logger,
    )

    # create GUI
    app = ExperimentGuiApp(
        logger=logger,
        sample_t=SAMPLE_T_MS,
        obs_block=obs_block,
        reach_block=reach_block,
        cmd_map=CMD_MAP,
        reachy_robot=reachy_robot,
        unity_game=unity_game,
        marker_stream=marker_stream,
        decoder=decoder,
    )
    app.run()

    # end session
    marker_stream.push_sample(["end session"])
    unity_game.end_run()
