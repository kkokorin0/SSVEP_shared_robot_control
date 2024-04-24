import logging
from datetime import datetime

import coloredlogs
import numpy as np

from robot_control import ReachyRobot, SharedController
from session_manager import (
    ALPHA_A,
    ALPHA_C0,
    ALPHA_MAX,
    COLLISION_DIST,
    FOLDER,
    MOVE_SPEED_S,
    OBJ_COORDS,
    OBJ_H,
    OBJ_R,
    P_ID,
    REACHY_WIRED,
    REST_POS,
    REVERSE_OFFSET_MS,
    SETUP_POS,
)

if __name__ == "__main__":
    """Move to each object position continuously and check for collisions to assess calibrated
    object positions."""

    session_id = str(P_ID) + "_" + datetime.now().strftime("%Y_%m_%d")
    log_file = FOLDER + "//" + session_id + ".log"
    logging.basicConfig(filename=log_file, level=logging.DEBUG)
    coloredlogs.install(level="WARNING", fmt="%(asctime)s,%(msecs)03d: %(message)s")
    logger = logging.getLogger(__name__)

    # create shared controller
    shared_controller = SharedController(
        obj_labels=list(range(len(OBJ_COORDS))),
        obj_coords=OBJ_COORDS,
        obj_h=OBJ_H,
        obj_r=OBJ_R,
        collision_d=COLLISION_DIST,
        max_assistance=ALPHA_MAX,
        median_confidence=ALPHA_C0,
        aggressiveness=ALPHA_A,
    )

    # setup robot
    reachy_robot = ReachyRobot(REACHY_WIRED, logger, SETUP_POS, REST_POS, MOVE_SPEED_S)
    reachy_robot.turn_off(safely=True)

    # move to each target continuously
    for obj_i in range(len(OBJ_COORDS)):
        ef_pose = reachy_robot.setup()
        while shared_controller.check_collision(ef_pose[:3, 3]) is None:
            direction = (
                shared_controller.get_obj_vectors(ef_pose[:3, 3])[obj_i]
                / shared_controller.get_obj_distances(ef_pose[:3, 3])[obj_i]
            )
            ef_pose = reachy_robot.move_continuously(direction, ef_pose)
            logger.warning(f"X:{ef_pose[:3, 3]}")

        # move to rest position
        input("Press Enter to reach the next object...")
        reachy_robot.translate(np.array([-1, 0, 0]), REVERSE_OFFSET_MS)

    reachy_robot.turn_off(safely=True)
