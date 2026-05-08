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

import pytest

sys.modules.setdefault("zmq", types.ModuleType("zmq"))


@pytest.fixture
def robot():
    from lerobot.robots.xlerobot_2wheels import XLerobot2Wheels

    robot = XLerobot2Wheels.__new__(XLerobot2Wheels)
    robot.config = SimpleNamespace(wheel_radius=0.05, wheelbase=0.25)
    return robot


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
