# Imports
import logging
from datetime import datetime
from random import randint, sample, seed

import coloredlogs
import numpy as np
import pygame
from pylsl import StreamInfo, StreamInlet, StreamOutlet, resolve_byprop

from decoding import BandpassFilter, Decoder
from robot_control import ReachyRobot, SimRobot
from stimulus import StimController

# experiment parameters
P_ID = 99
ONLINE_DECODING = False
N_TRIALS = 1
SAMPLE_T_MS = 200
INIT_MS = 10000
PROMPT_MS = 2000
DELAY_MS = 1000
TRIAL_MS = 3600
REST_MS = 2000
OFFSET_MS = 1000
CMD_MAP = {
    "u": [0, 0, 1],
    "d": [0, 0, -1],
    "l": [0, 1, 0],
    "r": [0, -1, 0],
    "f": [1, 0, 0],
    "b": [-1, 0, 0],
}
CMDS = ["u", "d", "l", "r", "f"]
FOLDER = r"C:\Users\kkokorin\OneDrive - The University of Melbourne\Documents\CurrentStudy\Logs"
FOLDER = r"C:\Users\Kirill Kokorin\Documents\Data\Robot_control\Logs"

# stimulus
FREQS = [7, 8, 9, 11, 13]  # top, bottom, left, right, middle (Hz)
HOLOLENS_IP = "192.168.137.228"  # HoloLens
HOLOLENS_IP = "127.0.0.1"  # UnitySim
STIM_DIST = 0.15  # distance from end-effector (m)

# robot
REACHY_WIRED = "169.254.238.100"  # Reachy
SETUP_POS = [25, 0, 0, -110, 0, -10, 0]  # starting joint angles
REST_POS = [15, 0, 0, -75, 0, -30, 0]  # arm drop position
MOVE_SPEED_S = 1  # arm movement duration

# recording/decoding
FS = 256  # sample rate (Hz)
MAX_SAMPLES = 7680  # max samples to from stream
N_CH = 9  # number of SSVEP channels
FMIN = 1  # filter lower cutoff (Hz)
FMAX = 40  # filter upper cutoff (Hz)
FILTER_ORDER = 4  # filter order
WINDOW_S = 1  # decoder window length (s)
HARMONICS = [1, 2]  # signals to include in template

if __name__ == "__main__":
    pygame.init()  # timer

    # logging
    session_id = str(P_ID) + "_" + datetime.now().strftime("%Y_%m_%d")
    log_file = FOLDER + "//" + session_id + ".log"
    logging.basicConfig(filename=log_file, level=logging.DEBUG)
    coloredlogs.install(level="WARNING", fmt="%(asctime)s,%(msecs)03d: %(message)s")
    main_logger = logging.getLogger(__name__)

    # comms
    try:
        reachy_robot = ReachyRobot(REACHY_WIRED, main_logger)
        reachy_robot.turn_off(REST_POS, MOVE_SPEED_S, safely=True)
    except Exception as e:
        main_logger.critical("Couldn't connect to Reachy: %s" % e)
        main_logger.critical("Using simulated robot")
        reachy_robot = SimRobot()

    unity_game = StimController(HOLOLENS_IP, main_logger)

    # marker stream
    marker_info = StreamInfo("MarkerStream", "Markers", 1, 0, "string", session_id)
    marker_stream = StreamOutlet(marker_info)
    main_logger.critical("Connected to marker stream and set-up lab recorder (y/n)?")
    if input() != "y":
        reachy_robot.turn_off(REST_POS, MOVE_SPEED_S, safely=True)
        main_logger.critical("Streams not set up, exiting")
    unity_game.setup_stim([0, 0, 0, 0, 0], [0, 0, 0], 0)

    # find EEG stream
    if ONLINE_DECODING:
        main_logger.warning("Looking for EEG stream....")
        eeg_streams = resolve_byprop("type", "EEG")
        main_logger.critical("Resolved EEG stream %s" % str(eeg_streams[0].desc()))
        eeg_stream = StreamInlet(eeg_streams[0])
        eeg_stream.flush()
        eeg_stream.pull_sample()

    marker_stream.push_sample(["start run"])
    main_logger.warning("Start run")
    pygame.time.delay(INIT_MS)

    # setup online filter and decoder
    if ONLINE_DECODING:
        bp_filter = BandpassFilter(FILTER_ORDER, FS, FMIN, FMAX, N_CH)
        X_chunk, ts = eeg_stream.pull_chunk(max_samples=MAX_SAMPLES)
        main_logger.warning("t=%.3fs pulled %d samples" % (ts[-1], len(X_chunk)))

        bp_filter.filter(np.array(X_chunk).T[:N_CH, :])
        decoder = Decoder(WINDOW_S, FS, HARMONICS, FREQS)

    # generate trials
    seed(P_ID)
    trials = np.array([sample(CMDS, len(CMDS)) for _ in range(N_TRIALS)]).reshape(-1)
    main_logger.warning(trials)

    for t_i, trial in enumerate(trials):
        main_logger.warning("Running trial (%s) %d/%d" % (trial, t_i + 1, len(trials)))

        # setup robot
        reachy_robot.turn_on()
        reachy_robot.move_arm_joints(SETUP_POS, MOVE_SPEED_S)
        ef_pose = reachy_robot.get_pose()

        # start with an offset in the opposite direction
        direction = CMD_MAP[trial]
        offset_start_ms = pygame.time.get_ticks()
        while pygame.time.get_ticks() - offset_start_ms < OFFSET_MS:
            ef_pose = reachy_robot.move_continuously(-np.array(direction), ef_pose)
        ef_coords = [-ef_pose[1, 3], -ef_pose[0, 3], ef_pose[2, 3]]

        # highlight direction
        unity_game.setup_stim([_c == trial for _c in CMDS], ef_coords, STIM_DIST)
        marker_stream.push_sample(["prompt %s" % trial])
        pygame.time.delay(randint(PROMPT_MS, PROMPT_MS + DELAY_MS))

        # start flashing
        unity_game.setup_stim(FREQS, ef_coords, STIM_DIST)
        marker_stream.push_sample(["go %s" % trial])
        trial_start_ms = pygame.time.get_ticks()
        last_stim_update_ms = trial_start_ms

        # create online buffer and clear stream
        if ONLINE_DECODING:
            trial_buffer = []
            X_chunk, ts = eeg_stream.pull_chunk(max_samples=MAX_SAMPLES)
            main_logger.warning("t=%.3fs pulled %d samples" % (ts[-1], len(X_chunk)))
            bp_filter.filter(np.array(X_chunk).T[:N_CH, :])

        while pygame.time.get_ticks() - trial_start_ms < TRIAL_MS:
            # move the robot
            ef_pose = reachy_robot.move_continuously(direction, ef_pose)

            # map from Reachy to HoloLens frame
            ef_coords = [-ef_pose[1, 3], -ef_pose[0, 3], ef_pose[2, 3]]

            # update stimuli/decode EEG chunk
            if pygame.time.get_ticks() - last_stim_update_ms > SAMPLE_T_MS:
                if ONLINE_DECODING:
                    X_chunk, ts = eeg_stream.pull_chunk(max_samples=MAX_SAMPLES)
                    main_logger.warning(
                        "t=%.3fs pulled %d samples" % (ts[-1], len(X_chunk))
                    )

                    # add filtered signal to buffer
                    X_filtered = bp_filter.filter(np.array(X_chunk).T[:N_CH, :])
                    trial_buffer.append(X_filtered)

                    # predict direction if buffer is full enough
                    if np.concatenate(trial_buffer, axis=1).shape[1] > WINDOW_S * FS:
                        buffer_window = np.concatenate(trial_buffer, axis=1)[
                            :, -int(WINDOW_S * FS) :
                        ]
                        pred = CMDS[decoder.predict(buffer_window)]
                        main_logger.warning(pred)
                        marker_stream.push_sample(["pred %s" % pred])
                        direction = CMD_MAP[pred]

                unity_game.move_stim(ef_coords)
                last_stim_update_ms = pygame.time.get_ticks()

        unity_game.setup_stim([0, 0, 0, 0, 0], [0, 0, 0], 0)
        marker_stream.push_sample(["rest %s" % trial])
        reachy_robot.turn_off(REST_POS, MOVE_SPEED_S, safely=True)
        pygame.time.delay(REST_MS)

    # close unity app
    unity_game.end_run()
    marker_stream.push_sample(["end run"])
    main_logger.warning("End run")
