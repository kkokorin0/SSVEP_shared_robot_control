import logging
import socket
import tkinter as tk
from datetime import datetime
from random import randint, sample, seed

import coloredlogs
import numpy as np
import pygame
from pylsl import StreamInfo, StreamInlet, StreamOutlet, resolve_byprop
from scipy.stats import uniform_direction

from decoding import BandpassFilter, OnlineDecoder
from robot_control import ReachyRobot, SharedController
from stimulus import StimController

# session
P_ID = 98  # participant ID
FOLDER = r"C:\Users\kkokorin\OneDrive - The University of Melbourne\Documents\CurrentStudy\Logs"
INIT_MS = 5000  # time to settle at the start of each block/trial

# observation block
OBS_TRIALS = 5  # per direction
PROMPT_MS = 2000  # direction prompt
DELAY_MS = 1000  # prompt jitter
OBS_TRIAL_MS = 3600  # observation trial duration
OBS_REST_MS = 2000  # rest period
OFFSET_MS = 1000  # offset in the opposite direction

# reaching block
REACH_TRIALS = 3  # per object
N_OBJ = 4  # object subset size
OBJ_COORDS = [
    np.array([0.420, -0.080, -0.130]),  # top left
    np.array([0.420, -0.220, -0.130]),  # top middle
    np.array([0.420, -0.330, -0.130]),  # top right
    np.array([0.430, -0.080, -0.240]),  # middle left
    np.array([0.430, -0.230, -0.240]),  # middle middle
    np.array([0.430, -0.340, -0.240]),  # middle right
    np.array([0.450, -0.080, -0.350]),  # bottom left
    np.array([0.450, -0.220, -0.350]),  # bottom middle
    np.array([0.450, -0.340, -0.350]),  # bottom right
]
OBJ_H = 0.06  # object height (m)
OBJ_R = 0.023  # object radius (m)
COLLISION_DIST = 0.02  # object reached distance (m)
REACH_TRIAL_MS = 30000  # max trial duration
REVERSE_OFFSET_MS = 2000  # reverse move duration
INIT_JITTER_MS = 1000  # jitter end-effector on initialisation

# control
SAMPLE_T_MS = 200
CMD_MAP = {
    "u": np.array([0, 0, 1]),
    "d": np.array([0, 0, -1]),
    "l": np.array([0, 1, 0]),
    "r": np.array([0, -1, 0]),
    "f": np.array([1, 0, 0]),
}
ALPHA_MAX = 0.7  # max proportion of robot assistance
ALPHA_C0 = 0.5  # confidence for median assistance
ALPHA_A = 10  # assistance aggressiveness

# stimulus
FREQS = [7, 8, 9, 11, 13]  # top, bottom, left, right, middle (Hz)
STIM_DIST = 0.15  # distance from end-effector (m)
HOLOLENS_IP = "192.168.137.228"  # HoloLens

# robot
REACHY_WIRED = "169.254.238.100"  # Reachy
SETUP_POS = [25, 0, 0, -110, 0, -10, 0]  # starting joint angles
REST_POS = [15, 0, 0, -75, 0, -30, 0]  # arm drop position
MOVE_SPEED_S = 1  # arm movement duration
QR_OFFSET = [0.03, -0.01, 0.01]  # fine-tune QR position

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
    """GUI for running the experiment session that allows for asynchronous control of the robot"""

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
        shared_controller,
        master=None,
    ):
        # experiment parameters
        self.sample_t = sample_t
        self.obs_block = obs_block
        self.reach_block = reach_block
        self.cmd_map = cmd_map

        # data streams
        self.logger = logger
        self.marker_stream = marker_stream

        # robot control
        self.reachy_robot = reachy_robot
        self.shared_controller = shared_controller

        # stimuli and decoder
        self.decoder = decoder
        self.unity_game = unity_game

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
            background=self.orange,
            height=self.button_h,
            overrelief="raised",
            state="normal",
            takefocus=False,
            text="Start",
            width=self.button_w,
            disabledforeground=self.silver,
            command=self.start_button_cb,
            activebackground=self.orange,
        )
        self.start_button.place(anchor="nw", x=self.button_x0, y=self.button_y0)

        # stop robot motion with fail or success flag
        self.fail_button = tk.Button(self.setup_frame)
        self.fail_button.configure(
            activebackground=self.red,
            background=self.light_grey,
            disabledforeground=self.silver,
            height=self.button_h,
            overrelief="raised",
            state="disabled",
            text="Fail",
            width=self.button_w,
            command=self.fail_button_cb,
        )
        self.fail_button.place(
            anchor="nw", x=self.button_x0, y=self.button_y0 + self.button_dy
        )

        self.success_button = tk.Button(self.setup_frame)
        self.success_button.configure(
            activebackground=self.green,
            background=self.light_grey,
            disabledforeground=self.silver,
            height=self.button_h,
            overrelief="raised",
            state="disabled",
            text="Pass",
            width=self.button_w,
            command=self.success_button_cb,
        )
        self.success_button.place(
            anchor="nw", x=self.button_x0, y=self.button_y0 + 2 * self.button_dy
        )

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
            anchor="nw", x=self.button_x0 + self.button_dx, y=self.button_y0
        )

        # turn off robot
        self.off_button = tk.Button(self.setup_frame)
        self.off_button.configure(
            activebackground=self.red,
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
            anchor="nw",
            x=self.button_x0 + self.button_dx,
            y=self.button_y0 + self.button_dy,
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
        self.last_trial = 0
        self.logger.critical(
            "Toggle shared control %s and reset trial index"
            % ("ON" if self.shared_control_on else "OFF")
        )
        self.logger.warning(
            "Reaching trials: %s"
            % ",".join([str(_i) for _i in self.reach_block["trials"]]),
        )

    def obs_feedback_cb(self):
        """Toggle observation block feedback"""
        self.observation_fb = not self.observation_fb
        self.logger.critical(
            "Toggle observation feedback %s" % ("ON" if self.observation_fb else "OFF")
        )

    def start_button_cb(self):
        """Continuously control the robotic arm in shared or direct mode until an object is reached or
        the stop button is pressed"""
        global experiment_running
        experiment_running = True
        self.start_button.configure(state="disabled", background=self.light_grey)
        self.obs_button.configure(state="disabled", background=self.light_grey)
        self.fail_button.configure(state="normal", background=self.red)
        self.success_button.configure(state="normal", background=self.green)

        # first trial in block
        if self.last_trial == 0:
            self.marker_stream.push_sample(
                ["start run: %s" % ("SC" if self.shared_control_on else "DC")]
            )
            self.logger.critical(
                "Start %s control run"
                % ("shared" if self.shared_control_on else "direct")
            )

        # initialise stream and decoder
        self.decoder.flush_stream()
        goal_obj = self.reach_block["trials"][self.last_trial]
        logger.warning(
            "Trial:%d/%d Obj:%d"
            % (self.last_trial + 1, len(self.reach_block["trials"]), goal_obj)
        )
        self.marker_stream.push_sample(["init:obj%d" % goal_obj])
        pygame.time.delay(self.reach_block["init"])
        self.decoder.filter_chunk()

        # setup arm and start flashing
        ef_pose = self.reachy_robot.setup()
        ef_pose = self.reachy_robot.translate(
            self.reach_block["jitter_v"],
            self.reach_block["init_jitter"],
        )
        unity_game.turn_on_stim(ef_pose)
        self.marker_stream.push_sample(["go:obj%d" % goal_obj])
        trial_start_ms = pygame.time.get_ticks()
        last_stim_update_ms = trial_start_ms
        last_move_ms = trial_start_ms

        # create online buffer, clear stream and reset command history
        self.decoder.clear_buffer()
        self.decoder.filter_chunk()
        self.shared_controller.reset(ef_pose[:3, 3])

        # move the robot continuously
        u_cmb = None
        while True:
            # stop button pressed
            if not experiment_running:
                return
            # trial too long
            elif last_move_ms - trial_start_ms > self.reach_block["length"]:
                self.fail_button_cb()

            # update GUI and move the robot
            self.toplevel.update()
            if u_cmb is not None:
                ef_pose = reachy_robot.move_continuously(u_cmb, ef_pose)
            data_msg = "X:" + ",".join(["%.3f" % _x for _x in ef_pose[:3, 3]]) + " "

            # check if an object has been reached
            reached_obj = self.shared_controller.check_collision(ef_pose[:3, 3])
            if reached_obj is not None:
                self.marker_stream.push_sample(
                    ["reach:obj%d goal:obj%d" % (reached_obj, goal_obj)]
                )
                self.logger.critical(
                    "%s: Reached object %d"
                    % ("Success" if reached_obj == goal_obj else "Fail", reached_obj)
                )
                break

            # get new control command every sample_t
            if last_move_ms - last_stim_update_ms > self.sample_t:
                # decode EEG chunk
                self.decoder.update_buffer()
                pred_i = self.decoder.predict_online()

                # predict direction if enough data in buffer
                if pred_i is not None:
                    pred = list(self.cmd_map.keys())[pred_i]
                    u_user = self.cmd_map[pred]

                    # predict obj and get robot velocity
                    pred_obj, u_robot = self.shared_controller.predict_obj(
                        ef_pose[:3, 3], u_user
                    )

                    # calculate confidence
                    confidence = self.shared_controller.get_confidence(
                        pred_obj, ef_pose[:3, 3]
                    )

                    # compute alpha
                    alpha = (
                        self.shared_controller.get_alpha(confidence)
                        if self.shared_control_on
                        else 0
                    )

                    # get combined velocity
                    u_cmb = self.shared_controller.get_cmb_vel(u_user, u_robot, alpha)
                    data_msg += (
                        "pred:%s pred_obj:%d conf:%.2f alpha:%.2f u_robot:%s u_cmb:%s"
                        % (
                            pred,
                            pred_obj,
                            confidence,
                            alpha,
                            ",".join(["%.3f" % _x for _x in u_robot]),
                            ",".join(["%.3f" % _x for _x in u_cmb]),
                        )
                    )

                # save data and update stim
                self.logger.warning(data_msg)
                self.marker_stream.push_sample([data_msg])
                unity_game.move_stim(unity_game.coord_transform(ef_pose))
                last_stim_update_ms = last_move_ms

            last_move_ms = pygame.time.get_ticks()

        self.success_button_cb()

    def obs_button_cb(self):
        """Observe the robotic arm move in a given direction and decode EEG data. With feedback turned on
        decoder outputs will control robot velocity"""
        global experiment_running
        experiment_running = True
        self.obs_button.configure(state="disabled", background=self.light_grey)
        self.start_button.configure(state="disabled", background=self.light_grey)
        self.off_button.configure(state="normal", background=self.red)

        # initialise stream and decoder
        self.decoder.flush_stream()
        self.marker_stream.push_sample(
            ["start run: OBS%s" % ("F" if self.observation_fb else "")]
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
            self.decoder.clear_buffer()
            self.decoder.filter_chunk()

            # move the robot continuously
            while last_move_ms - trial_start_ms < self.obs_block["length"]:
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
            ["end run: OBS%s" % ("F" if self.observation_fb else "")]
        )
        self.logger.critical(
            "End observation run with%s feedback"
            % ("" if self.observation_fb else "out")
        )
        self.off_button_cb()

    def fail_button_cb(self):
        """Stop the arm at the current position and end the trial wtih a fail flag"""
        global experiment_running
        experiment_running = False
        self.start_button.configure(state="disabled", background=self.light_grey)
        self.obs_button.configure(state="disabled", background=self.light_grey)

        self.marker_stream.push_sample(
            ["fail:obj%d" % self.reach_block["trials"][self.last_trial]]
        )
        self.logger.warning("Fail: Arm stopped, reversing")
        self.clean_up_trial()

    def success_button_cb(self):
        """Stop the arm at the current position and end the trial with a success flag"""
        global experiment_running
        experiment_running = False
        self.start_button.configure(state="disabled", background=self.light_grey)
        self.obs_button.configure(state="disabled", background=self.light_grey)

        goal_obj = self.reach_block["trials"][self.last_trial]
        self.marker_stream.push_sample(
            ["reach:obj%d goal:obj%d" % (goal_obj, goal_obj)]
        )
        self.logger.warning("Success: Arm stopped, reversing")
        self.clean_up_trial()

    def clean_up_trial(self):
        """Reset the arm and stimuli"""
        # rest while resetting
        self.unity_game.turn_off_stim()
        self.marker_stream.push_sample(
            ["rest:obj%d" % self.reach_block["trials"][self.last_trial]]
        )
        self.reachy_robot.translate([-1, 0, 0], self.reach_block["reverse_offset"])

        # increment and reset trial if last
        self.last_trial += 1
        if self.last_trial == len(self.reach_block["trials"]):
            self.last_trial = 0
            self.marker_stream.push_sample(
                ["end run: %s" % ("SC" if self.shared_control_on else "DC")]
            )
            self.logger.critical(
                "End %s control run"
                % ("shared" if self.shared_control_on else "direct")
            )

        self.off_button_cb()

    def off_button_cb(self):
        """Reset the arm and turn it off"""
        self.fail_button.configure(state="disabled", background=self.light_grey)
        self.success_button.configure(state="disabled", background=self.light_grey)
        self.off_button.configure(state="disabled", background=self.light_grey)

        self.reachy_robot.turn_off(safely=True)
        self.logger.warning("Arm reset and turned off")

        self.start_button.configure(state="normal", background=self.orange)
        self.obs_button.configure(state="normal", background=self.yellow)


if __name__ == "__main__":
    """Run an experiment session comprising multiple observation and/or reaching blocks.

    - Observation block: the user observes the robotic arm move in a given direction, while the
        system displays SSVEP stimuli in a cross pattern above the robotic arm, and decodes their EEG to
        predict which direction the user wants the arm to move. With feedback turned on, the arm moves
        based on the decoder predictions.

    - Reaching block: the user actively controls the robotic arm to reach a set of objects in a given order.
        In shared control mode, the system predicts which object the user wants to reach and assists them.
    """
    pygame.init()  # timer

    # logging
    session_id = str(P_ID) + "_" + datetime.now().strftime("%Y_%m_%d")
    log_file = FOLDER + "//" + session_id + ".log"
    logging.basicConfig(
        filename=log_file,
        level=logging.DEBUG,
        format="%(asctime)s: %(message)s",
    )
    coloredlogs.install(level="WARNING", fmt="%(asctime)s,%(msecs)03d: %(message)s")
    logger = logging.getLogger(__name__)

    # robot
    reachy_robot = ReachyRobot(REACHY_WIRED, logger, SETUP_POS, REST_POS, MOVE_SPEED_S)
    reachy_robot.turn_off(safely=True)

    # stimuli
    seed(P_ID)
    p_freqs = sample(FREQS, len(FREQS))
    unity_game = StimController(HOLOLENS_IP, logger, QR_OFFSET, p_freqs, STIM_DIST)
    unity_game.turn_off_stim()

    # setup marker stream
    marker_info = StreamInfo("MarkerStream", "Markers", 1, 0, "string", session_id)
    marker_stream = StreamOutlet(marker_info)
    logger.critical("Connected to marker stream and set-up lab recorder (y/n)?")
    if input() != "y":
        logger.critical("Streams not set up, exiting")
    marker_stream.push_sample(["start session"])
    marker_stream.push_sample(
        ["P%d freqs:%s" % (P_ID, ",".join(str(_f) for _f in p_freqs))]
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
        stim_freqs=p_freqs,
        n_ch=N_CH,
        online_filter=bp_filter,
        eeg_stream=StreamInlet(eeg_streams[0]),
        max_samples=MAX_SAMPLES,
        logger=logger,
    )

    # observation block
    obs_trials = np.array(
        [sample(list(CMD_MAP.keys()), len(CMD_MAP)) for _i in range(OBS_TRIALS)]
    ).reshape(-1)
    obs_block = dict(
        trials=obs_trials,
        init=INIT_MS,
        prompt=PROMPT_MS,
        delay=DELAY_MS,
        length=OBS_TRIAL_MS,
        rest=OBS_REST_MS,
        start_offset=OFFSET_MS,
    )
    logger.warning(
        "Observation trials (%d): %s" % (len(obs_trials), ",".join(obs_trials))
    )

    # reaching block
    obj_subset = sorted(sample(range(len(OBJ_COORDS)), N_OBJ))
    reaching_trials = np.array(
        [sample(obj_subset, len(obj_subset)) for _i in range(REACH_TRIALS)]
    ).reshape(-1)
    logger.warning(
        "Reaching trials %dx(%s): %s"
        % (
            len(reaching_trials),
            ",".join([str(_o) for _o in obj_subset]),
            ",".join([str(_i) for _i in reaching_trials]),
        )
    )
    reach_block = dict(
        trials=reaching_trials,
        init=INIT_MS,
        init_jitter=INIT_JITTER_MS,
        jitter_v=uniform_direction.rvs(3, random_state=P_ID),
        length=REACH_TRIAL_MS,
        reverse_offset=REVERSE_OFFSET_MS,
    )

    # create shared controller
    shared_controller = SharedController(
        obj_labels=obj_subset,
        obj_coords=[OBJ_COORDS[_i] for _i in obj_subset],
        obj_h=OBJ_H,
        obj_r=OBJ_R,
        collision_d=COLLISION_DIST,
        max_assistance=ALPHA_MAX,
        median_confidence=ALPHA_C0,
        aggressiveness=ALPHA_A,
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
        shared_controller=shared_controller,
    )
    app.run()

    # end session
    marker_stream.push_sample(["end session"])
    unity_game.end_run()
