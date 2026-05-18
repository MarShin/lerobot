#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest

fake_zmq = types.ModuleType("zmq")
fake_zmq.Socket = object
sys.modules.setdefault("zmq", fake_zmq)


@pytest.fixture
def robot():
    from lerobot.robots.xlerobot_2wheels import XLerobot2Wheels

    robot = XLerobot2Wheels.__new__(XLerobot2Wheels)
    robot.id = "test_xlerobot_2wheels"
    robot.config = SimpleNamespace(wheel_radius=0.05, wheelbase=0.25)
    return robot


class FakeBus:
    is_connected = True

    def sync_read(self, data_name, motors):
        if data_name == "Present_Velocity":
            return {"base_left_wheel": 0.0, "base_right_wheel": 0.0}
        return dict.fromkeys(motors, 0.0)


class FakeCamera:
    is_connected = True

    def __init__(self):
        self.read_count = 0

    def async_read(self):
        self.read_count += 1
        return np.zeros((2, 2, 3), dtype=np.uint8)


def test_body_to_wheel_raw_uses_standard_differential_drive_signs(robot):
    forward = robot._body_to_wheel_raw(x=0.1, theta=0.0)
    backward = robot._body_to_wheel_raw(x=-0.1, theta=0.0)
    rotate_left = robot._body_to_wheel_raw(x=0.0, theta=30.0)
    rotate_right = robot._body_to_wheel_raw(x=0.0, theta=-30.0)

    assert forward["base_left_wheel"] > 0
    assert forward["base_right_wheel"] > 0
    assert backward["base_left_wheel"] < 0
    assert backward["base_right_wheel"] < 0
    assert rotate_left["base_left_wheel"] < 0
    assert rotate_left["base_right_wheel"] > 0
    assert rotate_right["base_left_wheel"] > 0
    assert rotate_right["base_right_wheel"] < 0


@pytest.mark.parametrize(
    ("x", "theta"),
    [
        (0.1, 0.0),
        (-0.1, 0.0),
        (0.0, 30.0),
        (0.0, -30.0),
        (0.05, 10.0),
    ],
)
def test_base_kinematics_round_trip(robot, x, theta):
    wheel_raw = robot._body_to_wheel_raw(x=x, theta=theta)
    body = robot._wheel_raw_to_body(
        wheel_raw["base_left_wheel"],
        wheel_raw["base_right_wheel"],
    )

    assert body["x.vel"] == pytest.approx(x, abs=1e-4)
    assert body["theta.vel"] == pytest.approx(theta, abs=2e-2)


def test_split_observation_state_path_does_not_read_cameras(robot):
    camera = FakeCamera()
    robot.bus1 = FakeBus()
    robot.bus2 = FakeBus()
    robot.left_arm_motors = ["left_arm_shoulder_pan"]
    robot.right_arm_motors = ["right_arm_shoulder_pan"]
    robot.head_motors = ["head_motor_1"]
    robot.base_motors = ["base_left_wheel", "base_right_wheel"]
    robot.cameras = {"head": camera}

    state_obs = robot.get_state_observation()
    split_obs = robot.get_observation(include_images=False)
    image_obs = robot.get_observation(include_images=True)

    assert "head" not in state_obs
    assert "head" not in split_obs
    assert camera.read_count == 1
    assert image_obs["head"].shape == (2, 2, 3)
