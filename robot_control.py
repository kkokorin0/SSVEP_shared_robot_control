# Libraries
import numpy as np
from reachy_sdk import ReachySDK
from reachy_sdk.trajectory import goto
from reachy_sdk.trajectory.interpolation import InterpolationMode


class ReachyRobot:
    # Reachy parameters
    temp_limits = (45, 55)  # (fan on, motor shutdown)
    joint_limits = [
        (-180, 90),
        (-180, 15),
        (-90, 90),
        (-135, 5),
        (-100, 100),
        (-45, 45),
        (-45, 45),
    ]
    arm_joints = [
        "r_shoulder_pitch",
        "r_shoulder_roll",
        "r_arm_yaw",
        "r_elbow_pitch",
        "r_forearm_yaw",
        "r_wrist_pitch",
        "r_wrist_roll",
    ]
    gripper_limits = (-20, 50)
    gripper = "r_gripper"
    parallel_gripper = np.array(
        [[-0.153, -0.027, -0.988], [-0.009, 1, -0.026], [0.988, 0.005, -0.153]]
    )  # parallel to table

    def __init__(self, reachy_ip, logger):
        """Create robot object by connecting to IP address

        Args:
            reachy_ip (str): robot IP
            logger (logging): logger object

        Returns:
            ReachySDK: Reachy
        """
        self.reachy = ReachySDK(host=reachy_ip)
        self.logger = logger
        self.logger.warning("Connected to Reachy on %s" % reachy_ip)

    def turn_on(self):
        """Turn on right arm motors"""
        self.reachy.turn_on("r_arm")

    def get_joints(self):
        """Get joint angle measurements in the arm

        Returns:
            list of floats: Joint angles
        """
        return [j.present_position for j in self.reachy.r_arm.joints.values()]

    def get_pose(self):
        """Get robot pose (4x4 pose matrix)
        [R11 R12 R13 Tx
        R21 R22 R23 Ty
        R31 R32 R33 Tz
        0   0   0   1]

        Returns:
            np.array: 4x4 array
        """
        measured_pose = self.fw_kinematics(
            [j.present_position for j in self.reachy.r_arm.joints.values()][0:-1],
        )
        return measured_pose

    def get_motor_temp(self):
        """Get temperature of each motor in the arm

        Returns:
            list of float: Motor temperatures
        """
        temps = [j.temperature for j in self.reachy.r_arm.joints.values()][0:-1]
        for t, joint in zip(temps, self.joint_names):
            if t > self.temp_limits[1]:
                self.logger.critical("shutdown %s at %.1f C" % (joint, t))
            elif t > self.temp_limits[0]:
                self.logger.warning("warning %s at %.1f C" % (joint, t))
        return temps

    def fw_kinematics(self, joint_angles):
        """Compute pose matrix from joint angles using forward kinematics and pre-defined robot model
        [R11 R12 R13 Tx
        R21 R22 R23 Ty
        R31 R32 R33 Tz
        0   0   0   1]

        Args:
            joint_angles (list of float): Joint angles

        Returns:
            np.array: 4x4 pose matrix
        """
        return self.reachy.r_arm.forward_kinematics(joints_position=joint_angles)

    def check_arm_joint_limits(self, joint_angles):
        """Confirm all target joint angles are within limits

        Args:
            joint_angles (list of float): Joint angles

        Returns:
            boolean: whether all angles are within the limits
        """
        for i, (th, (th_min, th_max)) in enumerate(
            zip(joint_angles, self.joint_limits)
        ):
            if (th < th_min) or (th > th_max):
                self.logger.critical(
                    "%s of %.1f exceeds (%.1f-%.1f)"
                    % (self.joint_names[i], th, th_min, th_max)
                )
                return False
        return True

    def move_arm_joints(self, joint_angles, duration):
        """Move arm to target joint angles if they're within limits

        Args:
            joint_angles (list of float): Goal joint angles
            duration (float): Time to complete the move in seconds

        Returns:
            boolean: Whether the move was completed
        """
        if self.check_arm_joint_limits(joint_angles):
            pos_keys = [
                self.reachy.r_arm.r_shoulder_pitch,
                self.reachy.r_arm.r_shoulder_roll,
                self.reachy.r_arm.r_arm_yaw,
                self.reachy.r_arm.r_elbow_pitch,
                self.reachy.r_arm.r_forearm_yaw,
                self.reachy.r_arm.r_wrist_pitch,
                self.reachy.r_arm.r_wrist_roll,
            ]
            goto(
                goal_positions={
                    pos_key: angle for pos_key, angle in zip(pos_keys, joint_angles)
                },
                starting_positions=None,  # less jerk when omitting initial angles
                duration=duration,
                interpolation_mode=InterpolationMode.MINIMUM_JERK,
            )
            return True
        else:
            self.logger.critical("Invalid move")
            return False

    def turn_off(self, pos=None, speed=None, safely=False):
        """Turn off right arm motors

        Args:
            pos (list of float, optional): Joint angles to move to before turning off
            speed (float, optional): Time to complete the move in seconds
            safely (bool, optional): Whether to move to pos before turning off
        """
        if safely:
            self.move_arm_joints(pos, speed)
        self.reachy.turn_off("r_arm")

    def go_to_coords(self, coords, rotation_mat, speed):
        """Move the arm to a target configuration specified by coordinates and rotation matrix

        Args:
            coords (np.array): 3x1 array of target coordinates [x, y, z].T
            rotation_mat (np.array): Goal orientation as a 3x3 rotation matrix
                                    [R11, R12, R13,
                                    R21, R22, R23,
                                    R31, R32, R33]
            speed (float): Move duration in seconds

        Returns:
            boolean: Whether the move was completed
        """
        goal_pose = np.zeros((4, 4))
        goal_pose[0:3, 0:3] = rotation_mat
        goal_pose[0:3, 3] = coords
        goal_pose[3, 3] = 1

        initial_joints = self.get_joints()[0:7]  # ignore gripper
        final_joints = self.reachy.r_arm.inverse_kinematics(goal_pose, initial_joints)
        return self.move_arm_joints(final_joints, speed)
