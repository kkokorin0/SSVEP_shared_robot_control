# Libraries
import time

import numpy as np
from reachy_sdk import ReachySDK
from reachy_sdk.trajectory import goto
from reachy_sdk.trajectory.interpolation import InterpolationMode

# Reachy parameters
TEMP_LIMITS = (45, 55)  # (fan on, motor shutdown)
ARM_LIMITS = [
    (-180, 90),
    (-180, 15),
    (-90, 90),
    (-135, 5),
    (-100, 100),
    (-45, 45),
    (-45, 45),
]
ARM_JOINTS = [
    "r_shoulder_pitch",
    "r_shoulder_roll",
    "r_arm_yaw",
    "r_elbow_pitch",
    "r_forearm_yaw",
    "r_wrist_pitch",
    "r_wrist_roll",
]
GRIPPER_LIMITS = (-20, 50)
GRIPPER_JOINT = "r_gripper"
PARALLEL_GRIPPER = np.array(
    [[-0.153, -0.027, -0.988], [-0.009, 1, -0.026], [0.988, 0.005, -0.153]]
)  # parallel to table
# PARALLEL_GRIPPER = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])


# Reachy control functions
def connect_to_robot(reachy_ip):
    """Create robot object by connecting to IP address

    Args:
        reachy_ip (str): robot IP

    Returns:
        ReachySDK: Reachy
    """
    reachy = ReachySDK(host=reachy_ip)
    print("Connected to Reachy on %s" % reachy_ip)
    return reachy


def turn_on(reachy):
    """Turn on right arm motors

    Args:
        reachy (ReachySDK): Reachy
    """
    reachy.turn_on("r_arm")


def get_joints(reachy):
    """Get joint angle measurements in the arm

    Args:
        reachy (ReachySDK): Reachy

    Returns:
        list of floats: Joint angles
    """
    return [j.present_position for j in reachy.r_arm.joints.values()]


def get_pose(reachy):
    """Get robot pose (4x4 pose matrix)
    [R11 R12 R13 Tx
     R21 R22 R23 Ty
     R31 R32 R33 Tz
     0   0   0   1]

    Args:
        reachy (ReachySDK): Reachy

    Returns:
        np.array: 4x4 array
    """
    measured_pose = fw_kinematics(
        reachy, [j.present_position for j in reachy.r_arm.joints.values()][0:-1]
    )
    return measured_pose


def get_motor_temp(reachy, temp_limits=TEMP_LIMITS, joint_names=ARM_JOINTS):
    """Get temperature of each motor in the arm

    Args:
        reachy (ReachySDK): Reachy
        temp_limits (tuple of int, optional): Warning and shutdown temps
        joint_names (list of str, optional): Arm joint motor names

    Returns:
        list of float: Motor temperatures
    """
    temps = [j.temperature for j in reachy.r_arm.joints.values()][0:-1]
    for t, joint in zip(temps, joint_names):
        if t > temp_limits[1]:
            print("shutdown %s at %.1f C" % (joint, t))
        elif t > temp_limits[0]:
            print("warning %s at %.1f C" % (joint, t))
    return temps


def fw_kinematics(reachy, joint_angles):
    """Compute pose matrix from joint angles using forward kinematics and pre-defined robot model
    [R11 R12 R13 Tx
     R21 R22 R23 Ty
     R31 R32 R33 Tz
     0   0   0   1]

    Args:
        reachy (ReachySDK): Reachy
        joint_angles (list of float): Joint angles

    Returns:
        np.array: 4x4 pose matrix
    """
    return reachy.r_arm.forward_kinematics(joints_position=joint_angles)


def check_arm_joint_limits(
    joint_angles, joint_limits=ARM_LIMITS, joint_names=ARM_JOINTS
):
    """Confirm all target joint angles are within limits

    Args:
        joint_angles (list of float): Joint angles
        joint_limits (list of int, optional): Joint limits (min, max)
        joint_names (list of str, optional): Arm joint motor names

    Returns:
        boolean: whether all angles are within the limits
    """
    for i, (th, (th_min, th_max)) in enumerate(zip(joint_angles, joint_limits)):
        if (th < th_min) or (th > th_max):
            print(
                "%s of %.1f exceeds (%.1f-%.1f)" % (joint_names[i], th, th_min, th_max)
            )
            return False
    return True


def move_arm_joints(reachy, joint_angles, duration):
    """Move arm to target joint angles if they're within limits

    Args:
        reachy (ReachySDK): Reachy
        joint_angles (list of float): Goal joint angles
        duration (float): Time to complete the move in seconds

    Returns:
        boolean: Whether the move was completed
    """
    if check_arm_joint_limits(joint_angles):
        pos_keys = [
            reachy.r_arm.r_shoulder_pitch,
            reachy.r_arm.r_shoulder_roll,
            reachy.r_arm.r_arm_yaw,
            reachy.r_arm.r_elbow_pitch,
            reachy.r_arm.r_forearm_yaw,
            reachy.r_arm.r_wrist_pitch,
            reachy.r_arm.r_wrist_roll,
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
        print("Invalid move")
        return False


def turn_off(reachy, pos=None, speed=None, safely=False):
    """Turn off right arm motors

    Args:
        reachy (ReachySDK): Reachy
        pos (list of float, optional): Joint angles to move to before turning off
        speed (float, optional): Time to complete the move in seconds
        safely (bool, optional): Whether to move to pos before turning off
    """
    if safely:
        move_arm_joints(reachy, pos, speed)
    reachy.turn_off("r_arm")


def move_gripper(
    reachy,
    angle,
    duration,
    gripper_limits=GRIPPER_LIMITS,
    move_mode=InterpolationMode.MINIMUM_JERK,
):
    """Move the gripper to a target angle if it's within limits

    Args:
        reachy (ReachySDK): Reachy
        angle (float): Goal angle
        duration (float): Move duration in seconds
        gripper_limits (tuple of int, optional): Gripper limits (min, max)
        move_mode (InterpolationMode, optional): Optimisation goal of trajectory optimiser

    Returns:
        boolean: Whether the move was completed
    """
    th_min, th_max = gripper_limits
    if (angle < th_min) or (angle > th_max):
        print("%s of %.1f exceeds (%.1f-%.1f)" % ("r_gripper", angle, th_min, th_max))
        return False
    else:
        goto(
            goal_positions={reachy.r_arm.r_gripper: angle},
            starting_positions={reachy.r_arm.r_gripper: get_joints(reachy)[-1]},
            duration=duration,
            interpolation_mode=move_mode,
        )
        return True


def go_to_coords(reachy, coords, rotation_mat, speed):
    """Move the arm to a target configuration specified by coordinates and rotation matrix

    Args:
        reachy (ReachySDK): Reachy
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

    initial_joints = get_joints(reachy)[0:7]  # ignore gripper
    final_joints = reachy.r_arm.inverse_kinematics(goal_pose, initial_joints)
    return move_arm_joints(reachy, final_joints, speed)


def grasp_by_force(reachy, step_len_s, step_angle, max_force):
    """Iteratively close the gripper until the max force or the minimum joint angle is reached

    Args:
        reachy (ReachySDK): Reachy
        step_len_s (float): Move duration in seconds
        step_angle (float): Angle to move the gripper by in each step
        max_force (float): Strain gauge force threshold

    Returns:
        boolean: Whether an object was grasped (i.e. significant force was detected)
    """
    force = reachy.force_sensors.r_force_gripper.force
    move_success = True
    while (force < max_force) and move_success:  # move gripper by step_angle
        move_success = move_gripper(
            reachy, get_joints(reachy)[-1] + step_angle, step_len_s
        )
        force = reachy.force_sensors.r_force_gripper.force
        time.sleep(step_len_s)
    return move_success
