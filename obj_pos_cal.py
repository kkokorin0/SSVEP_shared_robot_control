import logging
from datetime import datetime

import coloredlogs
import numpy as np

from robot_control import ReachyRobot, SharedController

# session
P_ID = 99  # participant ID
FOLDER = r"C:\Users\kkokorin\OneDrive - The University of Melbourne\Documents\CurrentStudy\Logs"

# robot
REACHY_WIRED = "169.254.238.100"  # Reachy
SETUP_POS = [25, 0, 0, -110, 0, -10, 0]  # starting joint angles
REST_POS = [15, 0, 0, -75, 0, -30, 0]  # arm drop position
MOVE_SPEED_S = 1  # arm movement duration
QR_OFFSET = [0.03, -0.01, 0.02]  # fine-tune QR position
BACK_VEL = [-1, 0, 0]  # backwards direction
BACK_MOVE_LEN_MS = 2000  # reverse move duration

# environment
OBJ_SELECTION = 8  # object to reach
OBJ_COORDS = [
    np.array([0.420, -0.080, -0.130]),  # top left
    np.array([0.420, -0.220, -0.130]),  # top middle
    np.array([0.420, -0.330, -0.130]),  # top right
    np.array([0.430, -0.080, -0.240]),  # middle left
    np.array([0.430, -0.230, -0.240]),  # middle middle
    np.array([0.430, -0.340, -0.240]),  # middle right
    np.array([0.440, -0.080, -0.350]),  # bottom left
    np.array([0.440, -0.220, -0.350]),  # bottom middle
    np.array([0.440, -0.340, -0.350]),  # bottom right
]
COLLISION_DIST = 0.02  # object reached distance (m)

# shared controller
ALPHA_MAX = 0.7  # max proportion of robot assistance
ALPHA_C0 = 0.5  # confidence for median assistance
ALPHA_A = 10  # assistance aggressiveness

if __name__ == "__main__":
    session_id = str(P_ID) + "_" + datetime.now().strftime("%Y_%m_%d")
    log_file = FOLDER + "//" + session_id + ".log"
    logging.basicConfig(filename=log_file, level=logging.DEBUG)
    coloredlogs.install(level="WARNING", fmt="%(asctime)s,%(msecs)03d: %(message)s")
    logger = logging.getLogger(__name__)

    # create shared controller
    shared_controller = SharedController(
        obj_labels=list(range(len(OBJ_COORDS))),
        obj_coords=OBJ_COORDS,
        collision_d=COLLISION_DIST,
        max_assistance=ALPHA_MAX,
        median_confidence=ALPHA_C0,
        aggressiveness=ALPHA_A,
    )

    # setup robot
    reachy_robot = ReachyRobot(REACHY_WIRED, logger, SETUP_POS, REST_POS, MOVE_SPEED_S)
    reachy_robot.turn_off(safely=True)
    ef_pose = reachy_robot.setup()

    # move to target continuously
    while shared_controller.check_collision(ef_pose[:3, 3]) is None:
        direction = (
            shared_controller.get_obj_vectors(ef_pose[:3, 3])[OBJ_SELECTION]
            / shared_controller.get_obj_distances(ef_pose[:3, 3])[OBJ_SELECTION]
        )
        ef_pose = reachy_robot.move_continuously(direction, ef_pose)
        logger.warning(f"X:{ef_pose[:3, 3]}")

    # move to rest position
    input("Press Enter to move to rest position...")
    reachy_robot.translate(BACK_VEL, BACK_MOVE_LEN_MS)
    reachy_robot.turn_off(safely=True)
