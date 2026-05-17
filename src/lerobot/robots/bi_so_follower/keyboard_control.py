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

import logging
import time
from dataclasses import dataclass
from typing import Any

from lerobot.configs import PipelineFeatureType, PolicyFeature
from lerobot.processor.converters import robot_action_observation_to_transition, transition_to_robot_action
from lerobot.processor.pipeline import ProcessorStep, RobotProcessorPipeline
from lerobot.types import EnvTransition, RobotAction, RobotObservation, TransitionKey

logger = logging.getLogger(__name__)

DEFAULT_JOINT_STEP = 2.0
DEFAULT_ELBOW_STEP = 5.0
DEFAULT_GRIPPER_STEP = 2.0
ACTIVE_KEY_LOG_INTERVAL_S = 0.5

LEFT_KEY_BINDINGS = {
    "q": ("left_shoulder_pan.pos", 1.0),
    "a": ("left_shoulder_pan.pos", -1.0),
    "w": ("left_shoulder_lift.pos", 1.0),
    "s": ("left_shoulder_lift.pos", -1.0),
    "e": ("left_elbow_flex.pos", -1.0),
    "d": ("left_elbow_flex.pos", 1.0),
    "r": ("left_wrist_flex.pos", 1.0),
    "f": ("left_wrist_flex.pos", -1.0),
    "t": ("left_wrist_roll.pos", 1.0),
    "g": ("left_wrist_roll.pos", -1.0),
    "y": ("left_gripper.pos", 1.0),
    "h": ("left_gripper.pos", -1.0),
}

RIGHT_KEY_BINDINGS = {
    "u": ("right_shoulder_pan.pos", 1.0),
    "j": ("right_shoulder_pan.pos", -1.0),
    "i": ("right_shoulder_lift.pos", 1.0),
    "k": ("right_shoulder_lift.pos", -1.0),
    "o": ("right_elbow_flex.pos", -1.0),
    "l": ("right_elbow_flex.pos", 1.0),
    "p": ("right_wrist_flex.pos", 1.0),
    ";": ("right_wrist_flex.pos", -1.0),
    "[": ("right_wrist_roll.pos", 1.0),
    "'": ("right_wrist_roll.pos", -1.0),
    "]": ("right_gripper.pos", 1.0),
    "\\": ("right_gripper.pos", -1.0),
}

KEYBOARD_MAPPING_ROWS = (
    ("left", "shoulder_pan", "a", "q", DEFAULT_JOINT_STEP),
    ("left", "shoulder_lift", "s", "w", DEFAULT_JOINT_STEP),
    ("left", "elbow_flex", "e", "d", DEFAULT_ELBOW_STEP),
    ("left", "wrist_flex", "f", "r", DEFAULT_JOINT_STEP),
    ("left", "wrist_roll", "g", "t", DEFAULT_JOINT_STEP),
    ("left", "gripper", "h", "y", DEFAULT_GRIPPER_STEP),
    ("right", "shoulder_pan", "j", "u", DEFAULT_JOINT_STEP),
    ("right", "shoulder_lift", "k", "i", DEFAULT_JOINT_STEP),
    ("right", "elbow_flex", "o", "l", DEFAULT_ELBOW_STEP),
    ("right", "wrist_flex", ";", "p", DEFAULT_JOINT_STEP),
    ("right", "wrist_roll", "'", "[", DEFAULT_JOINT_STEP),
    ("right", "gripper", "\\", "]", DEFAULT_GRIPPER_STEP),
)


@dataclass
class KeyboardActionToBiSOFollowerAction(ProcessorStep):
    """Convert raw keyboard keys into complete bimanual follower joint targets."""

    action_names: tuple[str, ...]
    joint_step: float = DEFAULT_JOINT_STEP
    elbow_step: float = DEFAULT_ELBOW_STEP
    gripper_step: float = DEFAULT_GRIPPER_STEP

    def __post_init__(self) -> None:
        self._key_bindings = {**LEFT_KEY_BINDINGS, **RIGHT_KEY_BINDINGS}
        self._last_active_key_log_s = 0.0

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        action = transition.get(TransitionKey.ACTION)
        observation = transition.get(TransitionKey.OBSERVATION)
        if not isinstance(action, dict):
            raise ValueError(f"Action should be a RobotAction type (dict), but got {type(action)}")
        if not isinstance(observation, dict):
            raise ValueError(
                f"Observation should be a RobotObservation type (dict), but got {type(observation)}"
            )

        new_transition = transition.copy()
        new_transition[TransitionKey.ACTION] = self._keyboard_to_action(action, observation)
        return new_transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features

    def _keyboard_to_action(self, keyboard_action: RobotAction, observation: RobotObservation) -> RobotAction:
        missing = [name for name in self.action_names if name not in observation]
        if missing:
            raise KeyError(
                "Cannot build bimanual keyboard action because the robot observation is missing "
                f"these action keys: {missing}"
            )

        target_action = {name: float(observation[name]) for name in self.action_names}
        applied = []
        for key in keyboard_action:
            if key is None:
                continue

            binding = self._key_bindings.get(str(key).lower())
            if binding is None:
                continue

            action_name, direction = binding
            if action_name not in target_action:
                continue

            step = self._step_for_action(action_name)
            delta = direction * step
            target_action[action_name] += delta
            applied.append(f"{key}->{action_name} {delta:+g}")

        self._log_applied_keys(applied)

        return target_action

    def _step_for_action(self, action_name: str) -> float:
        if action_name.endswith("gripper.pos"):
            return self.gripper_step
        if action_name.endswith("elbow_flex.pos"):
            return self.elbow_step
        return self.joint_step

    def _log_applied_keys(self, applied: list[str]) -> None:
        if not applied:
            return

        now_s = time.monotonic()
        if now_s - self._last_active_key_log_s < ACTIVE_KEY_LOG_INTERVAL_S:
            return

        logger.info("Bimanual keyboard jog: %s", ", ".join(applied))
        self._last_active_key_log_s = now_s


def format_bi_so_follower_keyboard_mapping() -> str:
    lines = [
        "Bimanual keyboard joint jogger enabled for bi_so_follower",
        (
            f"Joint step: {DEFAULT_JOINT_STEP:g}; elbow step: {DEFAULT_ELBOW_STEP:g}; "
            f"gripper step: {DEFAULT_GRIPPER_STEP:g}"
        ),
        "",
        "Arm    Joint           Decrease  Increase",
        "-----  --------------  --------  --------",
    ]
    for arm, joint, decrease_key, increase_key, _step in KEYBOARD_MAPPING_ROWS:
        lines.append(f"{arm:<5}  {joint:<14}  {decrease_key:<8}  {increase_key:<8}")
    lines.append("")
    lines.append("Idle keyboard input holds the current observed joint positions.")
    return "\n".join(lines)


def maybe_make_bi_so_follower_keyboard_processor(
    *,
    robot: Any,
    teleop: Any,
    fallback: RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction],
) -> RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction]:
    if getattr(robot, "name", None) != "bi_so_follower" or getattr(teleop, "name", None) != "keyboard":
        return fallback

    logger.info("\n%s", format_bi_so_follower_keyboard_mapping())
    return RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
        steps=[KeyboardActionToBiSOFollowerAction(action_names=tuple(robot.action_features))],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )
