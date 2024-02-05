# Imports
import logging
from datetime import datetime
from random import randint, sample, seed

import coloredlogs
import numpy as np
import pygame
from pylsl import StreamInfo, StreamOutlet

from robot_control import ReachyRobot, SimRobot
from stimulus import StimController

# Experiment parameters
P_ID = 99
N_TRIALS = 5
SAMPLE_T_MS = 200
INIT_MS = 10000
PROMPT_MS = 2000
DELAY_MS = 1000
TRIAL_MS = 2500
REST_MS = 2000
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

# Stimulus
FREQS = [
    7,
    8,
    9,
    11,
    13,
]  # top, bottom, left, right, middle (Hz)
HOLOLENS_IP = "10.13.144.90"  # HoloLens
# HOLOLENS_IP = "127.0.0.1"  # UnitySim

# Robot
REACHY_WIRED = "169.254.238.100"  # Reachy
SETUP_POS = [25, 0, 0, -110, 0, -10, 0]  # starting joint angles
REST_POS = [15, 0, 0, -75, 0, -30, 0]  # arm drop position
MOVE_SPEED_S = 1  # arm movement duration

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
        reachy_robot.turn_off()
    except Exception as e:
        main_logger.critical("Couldn't connect to Reachy: %s" % e)
        main_logger.critical("Using simulated robot")
        reachy_robot = SimRobot()

    unity_game = StimController(HOLOLENS_IP, main_logger)

    # markers
    marker_info = StreamInfo("MarkerStream", "Markers", 1, 0, "string", session_id)
    marker_stream = StreamOutlet(marker_info)
    main_logger.critical("Connected to marker stream and set-up lab recorder (y/n)?")
    if input() != "y":
        reachy_robot.turn_off(REST_POS, MOVE_SPEED_S, safely=True)
        main_logger.critical("Streams not set up, exiting")
    unity_game.setup_stim([0, 0, 0, 0, 0], [0, 0, 0])

    marker_stream.push_sample(["start run"])
    main_logger.warning("Start run")
    pygame.time.delay(INIT_MS)

    # generate trials
    seed(P_ID)
    trials = np.array([sample(CMDS, len(CMDS)) for _ in range(N_TRIALS)]).reshape(-1)
    main_logger.warning(trials)

    for t_i, trial in enumerate(trials):
        logging.warning("Running trial (%s) %d/%d" % (trial, t_i + 1, len(trials)))

        # setup robot
        reachy_robot.turn_on()
        reachy_robot.move_arm_joints(SETUP_POS, MOVE_SPEED_S)
        ef_pose = reachy_robot.get_pose()
        ef_coords = [-ef_pose[1, 3], -ef_pose[0, 3], ef_pose[2, 3]]

        # gighlight direction
        unity_game.setup_stim([_c == trial for _c in CMDS], ef_coords)
        marker_stream.push_sample(["prompt %s" % trial])
        pygame.time.delay(randint(PROMPT_MS, PROMPT_MS + DELAY_MS))

        # start flashing
        unity_game.setup_stim(FREQS, ef_coords)
        marker_stream.push_sample(["go %s" % trial])
        trial_start_ms = pygame.time.get_ticks()
        last_stim_update_ms = trial_start_ms
        while pygame.time.get_ticks() - trial_start_ms < TRIAL_MS:
            # move the robot
            direction = CMD_MAP[trial]
            ef_pose = reachy_robot.move_continuously(direction, ef_pose)

            # map from Reachy to HoloLens frame
            ef_coords = [-ef_pose[1, 3], -ef_pose[0, 3], ef_pose[2, 3]]

            # update stimuli
            if pygame.time.get_ticks() - last_stim_update_ms > SAMPLE_T_MS:
                unity_game.move_stim(ef_coords)
                last_stim_update_ms = pygame.time.get_ticks()

        unity_game.setup_stim([0, 0, 0, 0, 0], [0, 0, 0])
        marker_stream.push_sample(["rest %s" % trial])
        reachy_robot.turn_off(REST_POS, MOVE_SPEED_S, safely=True)
        pygame.time.delay(REST_MS)

    # close unity app
    unity_game.end_run()
    marker_stream.push_sample(["end run"])
    main_logger.warning("End run")
