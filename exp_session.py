# Imports
import logging
import time
from datetime import datetime

import coloredlogs

from robot_control import ReachyRobot
from stimulus import StimController

# Experiment parameters
N_TRIALS = 1
SAMPLE_T = 0.2  # s
FOLDER = r"C:\Users\Kirill Kokorin\Documents\Data\Robot_control\Logs"

# Stimulus
FREQS = [14, 15, 17, 18]  # grab something from literature
HOLOLENS_IP = "10.15.254.106"  # HoloLens
# HOLOLENS_IP = "127.0.0.1"  # UnitySim

# Robot
ROBOT_VEL = 0.05  # m/s
REACHY_WIRED = "169.254.238.100"
SETUP_POS = [10, 0, 0, -100, 0, -10, 0]  # starting joint angles
REST_POS = [15, 0, 0, -75, 0, -30, 0]  # arm drop position
MOVE_SPEED_S = 1  # arm movement duration

if __name__ == "__main__":
    # Setup logging
    log_file = "%s//%s.log" % (FOLDER, datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
    logging.basicConfig(filename=log_file, level=logging.DEBUG)
    coloredlogs.install(level="WARNING", fmt="%(asctime)s,%(msecs)03d: %(message)s")
    main_logger = logging.getLogger(__name__)

    # Setup robot
    reachy_robot = ReachyRobot(REACHY_WIRED, main_logger)
    reachy_robot.turn_on()
    reachy_robot.move_arm_joints(SETUP_POS, MOVE_SPEED_S)

    # Setup stim
    unity_game = StimController(HOLOLENS_IP, main_logger)
    for trial in range(N_TRIALS):
        logging.warning("Running trial %d/%d" % (trial + 1, N_TRIALS))
        unity_game.setup_stim(FREQS)  # reset the stimuli

        run_trial = True
        # while run_trial:
        for i in range(10):
            # get user command
            direction = "d"

            # move the robot
            ef_pose = reachy_robot.get_pose()
            ef_coords = [-ef_pose[1, 3], -ef_pose[0, 3], ef_pose[2, 3]]

            # update stimuli
            unity_game.move_stim(ef_coords)
            time.sleep(SAMPLE_T)  # change to use pygame clock

    reachy_robot.turn_off(REST_POS, MOVE_SPEED_S, safely=True)
    unity_game.end_run()
    # reset robot
