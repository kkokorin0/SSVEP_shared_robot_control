# Imports
import socket
import time

from robot_control import (
    ARM_JOINTS,
    PARALLEL_GRIPPER,
    connect_to_robot,
    get_motor_temp,
    get_pose,
    go_to_coords,
    move_arm_joints,
    turn_off,
    turn_on,
)


class ReachyRobot:
    setup_pos = [10, 0, 0, -100, 0, -10, 0]  # starting joint angles
    rest_pos = [15, 0, 0, -75, 0, -30, 0]  # arm drop position
    fast_move_s = 1  # arm movement duration
    gripper_orientation = PARALLEL_GRIPPER  # gripper orientation
    joint_names = ARM_JOINTS

    def __init__(self, reachy_ip):
        """Connect and turn on

        Args:
            reachy_ip (str): reachy IP
        """
        self.reachy = connect_to_robot(reachy_ip)
        self.start_time = time.time()

    def print_motor_temp(self):
        """Print temperature of each joint motor

        Returns:
            temps (list of int): motor temperatures
        """
        temps = get_motor_temp(self.reachy)
        for t, j in zip(temps, self.joint_names):
            print("%s: %.1f C" % (j, t))
        return temps

    def setup_arm(self):
        """Setup arm at initial position"""
        turn_on(self.reachy)
        move_arm_joints(self.reachy, self.setup_pos, self.fast_move_s)

    def reset_arm(self):
        """Turn off robot motors"""

        turn_off(self.reachy, self.rest_pos, self.fast_move_s, safely=True)

    def move_to_coords(self, coords):
        """Move end-effector to target coords

        Args:
            coords (np.array): 3x1 array of coordinates [x, y, z].T
        """
        turn_on(self.reachy)
        go_to_coords(
            self.reachy,
            coords,
            self.gripper_orientation,
            self.fast_move_s,
        )


class StimController:
    bufsize = 1024
    port = 25001

    def __init__(self, host):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((host, self.port))

    def send_data(self, msg, get_response=True):
        try:
            # connect to server and send msg
            self.socket.sendall(msg.encode("utf-8"))

            if get_response:
                response = self.socket.recv(self.bufsize).decode("utf-8")
                print("Response:", response)

        except Exception as e:
            print(f"An error occurred: {e}")

    def setup_stim(self, freqs):
        self.send_data("reset:%s" % ",".join("%.3f" % f for f in freqs))

    def move_stim(self, ef_coords):
        self.send_data(
            "move:%.3f,%.3f,%.3f" % (ef_coords[0], ef_coords[1], ef_coords[2])
        )

    def end_run(self):
        self.setup_stim([0, 0, 0, 0])
        self.socket.close()


# Constants
N_TRIALS = 10
ROBOT_VEL = 0.05  # m/s
SAMPLE_T = 0.2  # s
STEP_SIZE = ROBOT_VEL * SAMPLE_T  # m
FREQS = [14, 15, 17, 18]  # grab something from literature
HOLOLENS_IP = "10.15.254.106"  # HoloLens
# HOLOLENS_IP = "127.0.0.1"  # UnitySim
REACHY_WIFI = "10.12.25.133"
REACHY_WIRED = "169.254.238.100"

if __name__ == "__main__":
    # reachy_robot = ReachyRobot(REACHY_WIRED)
    # reachy_robot.setup_arm()
    # reachy_robot.move_to_coords([0.378, -0.182, -0.390])
    # pose = get_pose(reachy_robot.reachy)
    # print(pose)
    # reachy_robot.reset_arm()

    # exit(0)
    unity_game = StimController(HOLOLENS_IP)
    for trial in range(N_TRIALS):
        print("Running trial %d/%d" % (trial + 1, N_TRIALS))
        unity_game.setup_stim(FREQS)  # reset the stimuli
        # exit(0)

        run_trial = True
        while run_trial:
            # get user command
            direction = "d"

            # move the robot
            ef_coords = [0.15, 0.10, -0.25]

            # update stimuli
            unity_game.move_stim(ef_coords)
            time.sleep(SAMPLE_T)  # change to use pygame clock

    unity_game.end_run()
    # reset robot
