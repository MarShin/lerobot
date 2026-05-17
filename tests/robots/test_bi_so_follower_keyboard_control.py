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

from lerobot.robots.bi_so_follower.keyboard_control import (
    KeyboardActionToBiSOFollowerAction,
    format_bi_so_follower_keyboard_mapping,
)
from lerobot.types import TransitionKey

ACTION_NAMES = (
    "left_shoulder_pan.pos",
    "left_shoulder_lift.pos",
    "left_elbow_flex.pos",
    "left_wrist_flex.pos",
    "left_wrist_roll.pos",
    "left_gripper.pos",
    "right_shoulder_pan.pos",
    "right_shoulder_lift.pos",
    "right_elbow_flex.pos",
    "right_wrist_flex.pos",
    "right_wrist_roll.pos",
    "right_gripper.pos",
)


def _observation() -> dict[str, float]:
    return {name: float(i) for i, name in enumerate(ACTION_NAMES)}


def test_idle_keyboard_holds_current_bimanual_positions():
    step = KeyboardActionToBiSOFollowerAction(action_names=ACTION_NAMES)

    output = step(
        {
            TransitionKey.ACTION: {},
            TransitionKey.OBSERVATION: _observation(),
            TransitionKey.REWARD: 0.0,
            TransitionKey.DONE: False,
            TransitionKey.TRUNCATED: False,
            TransitionKey.INFO: {},
            TransitionKey.COMPLEMENTARY_DATA: {},
        }
    )

    assert output[TransitionKey.ACTION] == _observation()


def test_keyboard_keys_jog_left_and_right_arm_targets():
    step = KeyboardActionToBiSOFollowerAction(action_names=ACTION_NAMES)

    output = step(
        {
            TransitionKey.ACTION: {"q": None, "j": None, "e": None, "l": None, "unknown": None},
            TransitionKey.OBSERVATION: _observation(),
            TransitionKey.REWARD: 0.0,
            TransitionKey.DONE: False,
            TransitionKey.TRUNCATED: False,
            TransitionKey.INFO: {},
            TransitionKey.COMPLEMENTARY_DATA: {},
        }
    )

    action = output[TransitionKey.ACTION]
    assert action["left_shoulder_pan.pos"] == _observation()["left_shoulder_pan.pos"] + 2.0
    assert action["right_shoulder_pan.pos"] == _observation()["right_shoulder_pan.pos"] - 2.0
    assert action["left_elbow_flex.pos"] == _observation()["left_elbow_flex.pos"] - 5.0
    assert action["right_elbow_flex.pos"] == _observation()["right_elbow_flex.pos"] + 5.0
    assert action["left_shoulder_lift.pos"] == _observation()["left_shoulder_lift.pos"]


def test_keyboard_mapping_printout_is_human_readable():
    output = format_bi_so_follower_keyboard_mapping()

    assert "Bimanual keyboard joint jogger enabled for bi_so_follower" in output
    assert "Arm    Joint           Decrease  Increase" in output
    assert "left   shoulder_pan    a         q" in output
    assert "left   elbow_flex      e         d" in output
    assert "right  elbow_flex      o         l" in output
    assert "right  gripper         \\         ]" in output
    assert "Idle keyboard input holds the current observed joint positions." in output
