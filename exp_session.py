# Imports
import logging
from datetime import datetime
from random import sample, seed, shuffle

import coloredlogs
import numpy as np
import pygame

from robot_control import ReachyRobot
from stimulus import StimController

# Experiment parameters
P_ID = 99
N_TRIALS = 1
SAMPLE_T_MS = 200
TRIAL_LEN_MS = 3000
CMDS = ["u", "d", "l", "r"]
CMD_MAP = {
    "u": [0, 0, 1],
    "d": [0, 0, -1],
    "l": [0, 1, 0],
    "r": [0, -1, 0],
    "f": [1, 0, 0],
    "b": [-1, 0, 0],
}
FOLDER = r"C:\Users\Kirill Kokorin\Documents\Data\Robot_control\Logs"

# Stimulus
FREQS = [14, 15, 17, 18]  # grab something from literature
HOLOLENS_IP = "10.15.254.106"  # HoloLens
# HOLOLENS_IP = "127.0.0.1"  # UnitySim

# Robot
REACHY_WIRED = "169.254.238.100"
SETUP_POS = [10, 0, 0, -100, 0, -10, 0]  # starting joint angles
REST_POS = [15, 0, 0, -75, 0, -30, 0]  # arm drop position
MOVE_SPEED_S = 1  # arm movement duration

if __name__ == "__main__":
    # Setup timer
    pygame.init()

    # Setup logging
    log_file = "%s//%s_%s.log" % (
        FOLDER,
        P_ID,
        datetime.now().strftime("%Y_%m_%d_%H_%M_%S"),
    )
    logging.basicConfig(filename=log_file, level=logging.DEBUG)
    coloredlogs.install(level="WARNING", fmt="%(asctime)s,%(msecs)03d: %(message)s")
    main_logger = logging.getLogger(__name__)

    # Setup comms
    reachy_robot = ReachyRobot(REACHY_WIRED, main_logger)
    unity_game = StimController(HOLOLENS_IP, main_logger)

    # Generate trials
    trials = np.array([sample(CMDS, len(CMDS)) for _ in range(N_TRIALS)]).reshape(-1)
    main_logger.warning(trials)

    for t_i, trial in enumerate(trials):
        logging.warning("Running trial (%s) %d/%d" % (trial, t_i + 1, len(trials)))

        # Setup robot and start flashing
        reachy_robot.turn_on()
        reachy_robot.move_arm_joints(SETUP_POS, MOVE_SPEED_S)
        unity_game.setup_stim(FREQS)

        trial_start_ms = pygame.time.get_ticks()
        last_stim_update_ms = trial_start_ms
        ef_pose = reachy_robot.get_pose()
        while pygame.time.get_ticks() - trial_start_ms < TRIAL_LEN_MS:
            # move the robot
            direction = CMD_MAP[trial]
            ef_pose = reachy_robot.move_continuously(direction, ef_pose)

            # map from Reachy to HoloLens frame
            ef_coords = [-ef_pose[1, 3], -ef_pose[0, 3], ef_pose[2, 3]]

            # update stimuli
            if pygame.time.get_ticks() - last_stim_update_ms > SAMPLE_T_MS:
                unity_game.move_stim(ef_coords)
                last_stim_update_ms = pygame.time.get_ticks()

        unity_game.setup_stim([0, 0, 0, 0])
        reachy_robot.turn_off(REST_POS, MOVE_SPEED_S, safely=True)

    # Close unity app
    # unity_game.end_run()
