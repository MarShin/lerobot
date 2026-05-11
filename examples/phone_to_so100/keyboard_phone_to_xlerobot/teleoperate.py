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

import argparse
import time
from dataclasses import dataclass, field
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
LEFT_PHONE_PORT = 4443
RIGHT_PHONE_PORT = 4444
RERUN_LOG_EVERY_N = 10
PROFILE_EVERY_N = 60

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
    "zero": "?",
}
RESET_KEYS = {
    "left_arm": "1",
    "right_arm": "2",
}
HEAD_STEP_DEG_PER_S = 30.0
HEAD_LIMITS_DEG = {
    "head_motor_1": (-90.0, 90.0),
    "head_motor_2": (-45.0, 45.0),
}


ArmSide = Literal["left", "right"]


@dataclass
class LoopProfiler:
    enabled: bool
    report_every_n: int
    target_dt_s: float
    sums: dict[str, float] = field(default_factory=dict)
    maxes: dict[str, float] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)
    loop_count: int = 0

    def record(self, name: str, dt_s: float) -> None:
        if not self.enabled:
            return
        self.sums[name] = self.sums.get(name, 0.0) + dt_s
        self.maxes[name] = max(self.maxes.get(name, 0.0), dt_s)
        self.counts[name] = self.counts.get(name, 0) + 1

    def maybe_report(self) -> None:
        if not self.enabled:
            return

        self.loop_count += 1
        if self.loop_count < self.report_every_n:
            return

        names = [
            "robot_obs",
            "left_phone",
            "right_phone",
            "keyboard",
            "left_arm",
            "right_arm",
            "head_base",
            "send_action",
            "rerun",
            "loop_work",
            "sleep",
        ]
        parts = []
        for name in names:
            count = self.counts.get(name, 0)
            if count == 0:
                continue
            avg_ms = self.sums[name] / count * 1000.0
            max_ms = self.maxes[name] * 1000.0
            parts.append(f"{name}={avg_ms:.1f}/{max_ms:.1f}ms")

        target_ms = self.target_dt_s * 1000.0
        print(
            f"[latency avg/max over {self.loop_count} loops, target={target_ms:.1f}ms] "
            + " | ".join(parts),
            flush=True,
        )
        self.sums.clear()
        self.maxes.clear()
        self.counts.clear()
        self.loop_count = 0


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
        if HEAD_KEYS["zero"] in pressed_keys:
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


def extract_init_arm_position(observation: RobotObservation, side: ArmSide) -> RobotAction:
    return hold_current_arm_position(observation, side)


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
    if not required_phone_keys.issubset(phone_action) or not bool(phone_action["phone.enabled"]):
        processor.reset()
        return hold_current_arm_position(observation, side)

    arm_observation = extract_arm_observation(observation, side)
    unprefixed_action = processor((phone_action, arm_observation))
    return prefix_arm_action(unprefixed_action, side)


def phone_action_enabled(phone_action: RobotAction) -> bool:
    required_phone_keys = {"phone.pos", "phone.rot", "phone.raw_inputs", "phone.enabled"}
    return required_phone_keys.issubset(phone_action) and bool(phone_action["phone.enabled"])


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
    print("  ?: set head targets to zero")
    print("Keyboard reset:")
    print("  1: reset left arm to startup pose")
    print("  2: reset right arm to startup pose\n")


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Teleoperate XLeRobot with two Android phones.")
    parser.add_argument(
        "--enable-rerun",
        action="store_true",
        help="Enable Rerun visualization. Disabled by default for lower teleop latency.",
    )
    parser.add_argument(
        "--rerun-log-every-n",
        type=int,
        default=RERUN_LOG_EVERY_N,
        help="When Rerun is enabled, log one frame every N control-loop ticks.",
    )
    parser.add_argument(
        "--profile-latency",
        action="store_true",
        help="Print rolling control-loop timing to identify latency bottlenecks.",
    )
    parser.add_argument(
        "--profile-every-n",
        type=int,
        default=PROFILE_EVERY_N,
        help="When latency profiling is enabled, print one timing report every N control-loop ticks.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    rerun_log_every_n = max(1, args.rerun_log_every_n)
    profiler = LoopProfiler(
        enabled=args.profile_latency,
        report_every_n=max(1, args.profile_every_n),
        target_dt_s=1.0 / FPS,
    )

    if not URDF_PATH.exists():
        raise FileNotFoundError(
            f"SO101 URDF not found at {URDF_PATH}. Update URDF_PATH before running this script."
        )

    robot = make_robot()
    keyboard = KeyboardTeleop(KeyboardTeleopConfig())

    left_phone_config = PhoneConfig(phone_os=PhoneOS.ANDROID, android_port=LEFT_PHONE_PORT)
    right_phone_config = PhoneConfig(phone_os=PhoneOS.ANDROID, android_port=RIGHT_PHONE_PORT)
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

        print(f"Connect/calibrate the left Android phone on port {LEFT_PHONE_PORT}.")
        left_phone.connect()
        print(f"Connect/calibrate the right Android phone on port {RIGHT_PHONE_PORT}.")
        right_phone.connect()

        if args.enable_rerun:
            init_rerun(session_name="keyboard_phone_to_xlerobot_teleop")
        print_controls(robot)

        observation = robot.get_observation()
        init_left_arm_action = extract_init_arm_position(observation, "left")
        init_right_arm_action = extract_init_arm_position(observation, "right")
        head_controller = HeadKeyboardController.from_observation(observation)
        base_controller = BaseKeyboardController(robot)
        left_reset_to_init = False
        right_reset_to_init = False
        loop_idx = 0

        while True:
            loop_start = time.perf_counter()

            section_start = time.perf_counter()
            observation = robot.get_observation()
            profiler.record("robot_obs", time.perf_counter() - section_start)

            section_start = time.perf_counter()
            left_phone_action = left_phone.get_action()
            profiler.record("left_phone", time.perf_counter() - section_start)

            section_start = time.perf_counter()
            right_phone_action = right_phone.get_action()
            profiler.record("right_phone", time.perf_counter() - section_start)

            section_start = time.perf_counter()
            pressed_keys = set(keyboard.get_action().keys())
            profiler.record("keyboard", time.perf_counter() - section_start)

            if robot.teleop_keys["quit"] in pressed_keys:
                print("Quit requested by keyboard.")
                break

            left_phone_enabled = phone_action_enabled(left_phone_action)
            right_phone_enabled = phone_action_enabled(right_phone_action)
            if left_phone_enabled and RESET_KEYS["left_arm"] not in pressed_keys:
                left_reset_to_init = False
            if right_phone_enabled and RESET_KEYS["right_arm"] not in pressed_keys:
                right_reset_to_init = False

            section_start = time.perf_counter()
            left_arm_action = process_phone_arm_action(
                phone_action=left_phone_action,
                observation=observation,
                side="left",
                processor=left_processor,
            )
            profiler.record("left_arm", time.perf_counter() - section_start)

            section_start = time.perf_counter()
            right_arm_action = process_phone_arm_action(
                phone_action=right_phone_action,
                observation=observation,
                side="right",
                processor=right_processor,
            )
            profiler.record("right_arm", time.perf_counter() - section_start)

            if RESET_KEYS["left_arm"] in pressed_keys:
                left_processor.reset()
                left_reset_to_init = True
            if RESET_KEYS["right_arm"] in pressed_keys:
                right_processor.reset()
                right_reset_to_init = True

            if left_reset_to_init:
                left_arm_action = init_left_arm_action.copy()
            if right_reset_to_init:
                right_arm_action = init_right_arm_action.copy()

            section_start = time.perf_counter()
            dt_s = time.perf_counter() - loop_start
            head_action = head_controller.update(pressed_keys, dt_s)
            base_action = base_controller.update(pressed_keys)
            profiler.record("head_base", time.perf_counter() - section_start)

            merged_action = {**left_arm_action, **right_arm_action, **head_action, **base_action}
            section_start = time.perf_counter()
            robot.send_action(merged_action)
            profiler.record("send_action", time.perf_counter() - section_start)
            if args.enable_rerun and loop_idx % rerun_log_every_n == 0:
                section_start = time.perf_counter()
                log_rerun_data(observation=observation, action=merged_action)
                profiler.record("rerun", time.perf_counter() - section_start)
            loop_idx += 1

            loop_work_s = time.perf_counter() - loop_start
            profiler.record("loop_work", loop_work_s)
            sleep_s = max(1.0 / FPS - loop_work_s, 0.0)
            section_start = time.perf_counter()
            precise_sleep(sleep_s)
            profiler.record("sleep", time.perf_counter() - section_start)
            profiler.maybe_report()

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
