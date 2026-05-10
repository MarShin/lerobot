#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Teleoperate XLeRobot with two Android phones plus keyboard head/base control.

Run the robot-side host first, for example:

    PYTHONPATH=src python -m lerobot.robots.xlerobot_2wheels.xlerobot_2wheels_host \
        --robot.id=my_xlerobot_2wheels

Then run this client-side script. Each Android phone controls one SO101 arm in
Cartesian space. The script converts each phone command to robot-native joint
targets, merges both arms with keyboard head/base commands, and sends the final
action dictionary through XLerobot2WheelsClient.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.model.kinematics import RobotKinematics
from lerobot.processor import (
    RobotProcessorPipeline,
    robot_action_observation_to_transition,
    transition_to_robot_action,
)
from lerobot.robots.so_follower.robot_kinematic_processor import (
    EEBoundsAndSafety,
    EEReferenceAndDelta,
    GripperVelocityToJoint,
    InverseKinematicsEEToJoints,
)
from lerobot.robots.xlerobot_2wheels import XLerobot2WheelsClient, XLerobot2WheelsClientConfig
from lerobot.teleoperators.keyboard.teleop_keyboard import KeyboardTeleop, KeyboardTeleopConfig
from lerobot.teleoperators.phone import Phone, PhoneConfig
from lerobot.teleoperators.phone.config_phone import PhoneOS
from lerobot.teleoperators.phone.phone_processor import MapPhoneActionToRobotAction
from lerobot.types import RobotAction, RobotObservation
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

FPS = 30
REMOTE_IP = "raspberrypi.local"
ROBOT_ID = "my_xlerobot_2wheels"
URDF_PATH = Path("../SO101/so101_new_calib.urdf")
TARGET_FRAME_NAME = "gripper_frame_link"

SO101_MOTOR_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

LEFT_ARM_PREFIX = "left_arm"
RIGHT_ARM_PREFIX = "right_arm"

HEAD_KEYS = {
    "head_motor_1+": "<",
    "head_motor_1-": ">",
    "head_motor_2+": ",",
    "head_motor_2-": ".",
    "reset": "?",
}
HEAD_STEP_DEG_PER_S = 30.0
HEAD_LIMITS_DEG = {
    "head_motor_1": (-90.0, 90.0),
    "head_motor_2": (-45.0, 45.0),
}


ArmSide = Literal["left", "right"]


@dataclass
class HeadKeyboardController:
    target_positions: dict[str, float]

    @classmethod
    def from_observation(cls, observation: RobotObservation) -> HeadKeyboardController:
        return cls(
            target_positions={
                "head_motor_1": float(observation.get("head_motor_1.pos", 0.0)),
                "head_motor_2": float(observation.get("head_motor_2.pos", 0.0)),
            }
        )

    def update(self, pressed_keys: set[str], dt_s: float) -> RobotAction:
        if HEAD_KEYS["reset"] in pressed_keys:
            self.target_positions = {"head_motor_1": 0.0, "head_motor_2": 0.0}

        step = HEAD_STEP_DEG_PER_S * dt_s
        for motor in ["head_motor_1", "head_motor_2"]:
            if HEAD_KEYS[f"{motor}+"] in pressed_keys:
                self.target_positions[motor] += step
            if HEAD_KEYS[f"{motor}-"] in pressed_keys:
                self.target_positions[motor] -= step

            low, high = HEAD_LIMITS_DEG[motor]
            self.target_positions[motor] = float(np.clip(self.target_positions[motor], low, high))

        return {f"{motor}.pos": pos for motor, pos in self.target_positions.items()}


class BaseKeyboardController:
    def __init__(self, robot: XLerobot2WheelsClient):
        self.robot = robot
        self.speed_index = robot.speed_index
        self._previous_pressed_keys: set[str] = set()

    def update(self, pressed_keys: set[str]) -> RobotAction:
        speed_up = self.robot.teleop_keys["speed_up"]
        speed_down = self.robot.teleop_keys["speed_down"]

        if speed_up in pressed_keys and speed_up not in self._previous_pressed_keys:
            self.speed_index = min(self.speed_index + 1, len(self.robot.speed_levels) - 1)
        if speed_down in pressed_keys and speed_down not in self._previous_pressed_keys:
            self.speed_index = max(self.speed_index - 1, 0)

        speed = self.robot.speed_levels[self.speed_index]
        x_cmd = 0.0
        theta_cmd = 0.0

        if self.robot.teleop_keys["forward"] in pressed_keys:
            x_cmd += speed["linear"]
        if self.robot.teleop_keys["backward"] in pressed_keys:
            x_cmd -= speed["linear"]
        if self.robot.teleop_keys["rotate_left"] in pressed_keys:
            theta_cmd += speed["angular"]
        if self.robot.teleop_keys["rotate_right"] in pressed_keys:
            theta_cmd -= speed["angular"]

        self.robot.speed_index = self.speed_index
        self._previous_pressed_keys = set(pressed_keys)
        return {"x.vel": x_cmd, "theta.vel": theta_cmd}


def arm_prefix(side: ArmSide) -> str:
    return LEFT_ARM_PREFIX if side == "left" else RIGHT_ARM_PREFIX


def extract_arm_observation(observation: RobotObservation, side: ArmSide) -> RobotObservation:
    prefix = arm_prefix(side)
    arm_observation = {}
    for name in SO101_MOTOR_NAMES:
        prefixed_key = f"{prefix}_{name}.pos"
        arm_observation[f"{name}.pos"] = float(observation[prefixed_key])
    return arm_observation


def prefix_arm_action(action: RobotAction, side: ArmSide) -> RobotAction:
    prefix = arm_prefix(side)
    prefixed_action = {}
    for name in SO101_MOTOR_NAMES:
        key = f"{name}.pos"
        if key in action:
            prefixed_action[f"{prefix}_{name}.pos"] = float(action[key])
    return prefixed_action


def hold_current_arm_position(observation: RobotObservation, side: ArmSide) -> RobotAction:
    prefix = arm_prefix(side)
    return {f"{prefix}_{name}.pos": float(observation[f"{prefix}_{name}.pos"]) for name in SO101_MOTOR_NAMES}


def make_phone_to_arm_joints_processor(
    *,
    platform: PhoneOS,
    kinematics: RobotKinematics,
) -> RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction]:
    return RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
        steps=[
            MapPhoneActionToRobotAction(platform=platform),
            EEReferenceAndDelta(
                kinematics=kinematics,
                end_effector_step_sizes={"x": 0.5, "y": 0.5, "z": 0.5},
                motor_names=SO101_MOTOR_NAMES,
                use_latched_reference=True,
            ),
            EEBoundsAndSafety(
                end_effector_bounds={"min": [-1.0, -1.0, -1.0], "max": [1.0, 1.0, 1.0]},
                max_ee_step_m=0.10,
            ),
            GripperVelocityToJoint(speed_factor=20.0),
            InverseKinematicsEEToJoints(
                kinematics=kinematics,
                motor_names=SO101_MOTOR_NAMES,
                initial_guess_current_joints=True,
            ),
        ],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )


def process_phone_arm_action(
    *,
    phone_action: RobotAction,
    observation: RobotObservation,
    side: ArmSide,
    processor: RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction],
) -> RobotAction:
    required_phone_keys = {"phone.pos", "phone.rot", "phone.raw_inputs", "phone.enabled"}
    if not required_phone_keys.issubset(phone_action):
        return hold_current_arm_position(observation, side)

    arm_observation = extract_arm_observation(observation, side)
    unprefixed_action = processor((phone_action, arm_observation))
    return prefix_arm_action(unprefixed_action, side)


def print_controls(robot: XLerobot2WheelsClient) -> None:
    print("\nXLeRobot two-phone teleoperation")
    print("Android phone #1: left arm, hold Move to enable")
    print("Android phone #2: right arm, hold Move to enable")
    print("Keyboard base:")
    print(f"  {robot.teleop_keys['forward']}/{robot.teleop_keys['backward']}: forward/backward")
    print(f"  {robot.teleop_keys['rotate_left']}/{robot.teleop_keys['rotate_right']}: rotate left/right")
    print(f"  {robot.teleop_keys['speed_up']}/{robot.teleop_keys['speed_down']}: speed up/down")
    print(f"  {robot.teleop_keys['quit']}: quit")
    print("Keyboard head:")
    print("  </>: head_motor_1 +/-")
    print("  ,/.: head_motor_2 +/-")
    print("  ?: reset head targets to zero\n")


def make_robot() -> XLerobot2WheelsClient:
    camera_config = {
        "left_side": OpenCVCameraConfig(index_or_path=0, width=640, height=480, fps=FPS),
        "right_side": OpenCVCameraConfig(index_or_path=1, width=640, height=480, fps=FPS),
        "head": OpenCVCameraConfig(index_or_path=2, width=640, height=480, fps=FPS),
    }
    robot_config = XLerobot2WheelsClientConfig(
        remote_ip=REMOTE_IP,
        id=ROBOT_ID,
        cameras=camera_config,
    )
    return XLerobot2WheelsClient(robot_config)


def main():
    if not URDF_PATH.exists():
        raise FileNotFoundError(
            f"SO101 URDF not found at {URDF_PATH}. Update URDF_PATH before running this script."
        )

    robot = make_robot()
    keyboard = KeyboardTeleop(KeyboardTeleopConfig())

    left_phone_config = PhoneConfig(phone_os=PhoneOS.ANDROID)
    right_phone_config = PhoneConfig(phone_os=PhoneOS.ANDROID)
    left_phone = Phone(left_phone_config)
    right_phone = Phone(right_phone_config)

    left_kinematics = RobotKinematics(
        urdf_path=str(URDF_PATH),
        target_frame_name=TARGET_FRAME_NAME,
        joint_names=SO101_MOTOR_NAMES,
    )
    right_kinematics = RobotKinematics(
        urdf_path=str(URDF_PATH),
        target_frame_name=TARGET_FRAME_NAME,
        joint_names=SO101_MOTOR_NAMES,
    )
    left_processor = make_phone_to_arm_joints_processor(
        platform=left_phone_config.phone_os,
        kinematics=left_kinematics,
    )
    right_processor = make_phone_to_arm_joints_processor(
        platform=right_phone_config.phone_os,
        kinematics=right_kinematics,
    )

    try:
        robot.connect()
        keyboard.connect()

        print("Connect/calibrate the left Android phone.")
        left_phone.connect()
        print("Connect/calibrate the right Android phone.")
        right_phone.connect()

        init_rerun(session_name="keyboard_phone_to_xlerobot_teleop")
        print_controls(robot)

        observation = robot.get_observation()
        head_controller = HeadKeyboardController.from_observation(observation)
        base_controller = BaseKeyboardController(robot)

        while True:
            loop_start = time.perf_counter()

            observation = robot.get_observation()
            left_phone_action = left_phone.get_action()
            right_phone_action = right_phone.get_action()
            pressed_keys = set(keyboard.get_action().keys())

            if robot.teleop_keys["quit"] in pressed_keys:
                print("Quit requested by keyboard.")
                break

            left_arm_action = process_phone_arm_action(
                phone_action=left_phone_action,
                observation=observation,
                side="left",
                processor=left_processor,
            )
            right_arm_action = process_phone_arm_action(
                phone_action=right_phone_action,
                observation=observation,
                side="right",
                processor=right_processor,
            )

            dt_s = time.perf_counter() - loop_start
            head_action = head_controller.update(pressed_keys, dt_s)
            base_action = base_controller.update(pressed_keys)

            merged_action = {**left_arm_action, **right_arm_action, **head_action, **base_action}
            robot.send_action(merged_action)
            log_rerun_data(observation=observation, action=merged_action)

            precise_sleep(max(1.0 / FPS - (time.perf_counter() - loop_start), 0.0))

    except KeyboardInterrupt:
        print("KeyboardInterrupt received.")
    finally:
        for device in [left_phone, right_phone, keyboard, robot]:
            try:
                if device.is_connected:
                    device.disconnect()
            except Exception as exc:
                print(f"Failed to disconnect {device}: {exc}")
        print("Teleoperation ended.")


if __name__ == "__main__":
    main()
