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

"""Record remote XLeRobot demonstrations from two Android phones plus keyboard.

Run the robot-side host first, for example:

    uv run python -m lerobot.robots.xlerobot_2wheels.xlerobot_2wheels_host \
        --robot.id=my_xlerobot_2wheels

Then run this script on the client machine:

    uv run python examples/phone_to_so100/keyboard_phone_to_xlerobot/record.py \
        --repo-id <hf_user>/<dataset_name> \
        --task "Pick up the object"

The control path mirrors teleoperate.py, but each dataset action is the final
robot-native command dictionary sent to XLerobot2WheelsClient.send_action().
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lerobot.common.control_utils import init_keyboard_listener, sanity_check_dataset_robot_compatibility
from lerobot.datasets import LeRobotDataset, VideoEncodingManager
from lerobot.model.kinematics import RobotKinematics
from lerobot.robots.xlerobot_2wheels import XLerobot2WheelsClient, XLerobot2WheelsClientConfig
from lerobot.teleoperators.keyboard.teleop_keyboard import KeyboardTeleop, KeyboardTeleopConfig
from lerobot.teleoperators.phone import Phone, PhoneConfig
from lerobot.teleoperators.phone.config_phone import PhoneOS
from lerobot.utils.constants import ACTION, HF_LEROBOT_HOME, OBS_STR
from lerobot.utils.feature_utils import build_dataset_frame, combine_feature_dicts, hw_to_dataset_features
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging, log_say
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from teleoperate import (  # noqa: E402
    FPS,
    LEFT_PHONE_PORT,
    PROFILE_EVERY_N,
    REMOTE_IP,
    RIGHT_PHONE_PORT,
    ROBOT_ID,
    SO101_MOTOR_NAMES,
    TARGET_FRAME_NAME,
    URDF_PATH,
    BaseKeyboardController,
    HeadKeyboardController,
    LoopProfiler,
    extract_init_arm_position,
    make_phone_to_arm_joints_processor,
    phone_action_enabled,
    print_controls,
    process_phone_arm_action,
    reset_arms_to_initial_pose,
)

DEFAULT_REPO_ID = "marshin68/xlerobot-phone-keyboard-dataset"
DEFAULT_TASK = "XLeRobot phone and keyboard demonstration"
DEFAULT_NUM_EPISODES = 2
DEFAULT_EPISODE_TIME_S = 30
DEFAULT_RESET_TIME_S = 30
RERUN_LOG_EVERY_N = 10
RESET_PAUSE_KEY = "p"
RESET_COUNTDOWN_LAST_S = 10


@dataclass
class RecordingContext:
    left_processor: Any
    right_processor: Any
    head_controller: HeadKeyboardController
    base_controller: BaseKeyboardController
    init_left_arm_action: dict[str, float]
    init_right_arm_action: dict[str, float]
    left_reset_to_init: bool = False
    right_reset_to_init: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote-ip", default=REMOTE_IP, help="Robot-side host IP or hostname.")
    parser.add_argument("--robot-id", default=ROBOT_ID, help="Robot id used for metadata.")
    parser.add_argument(
        "--split-observations",
        action="store_true",
        help="Read robot state from the main observation port and full-resolution cameras from a separate image port.",
    )
    parser.add_argument(
        "--max-camera-age-ms",
        type=float,
        default=250.0,
        help="Fail recording if any camera frame is older than this many ms. Use 0 to disable.",
    )
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID, help="Hugging Face dataset repo id.")
    parser.add_argument("--root", type=Path, default=None, help="Optional local dataset root.")
    parser.add_argument("--task", default=DEFAULT_TASK, help="Task string saved with every frame.")
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=DEFAULT_NUM_EPISODES,
        help="Target total number of episodes. With --resume, recording continues until this total is reached.",
    )
    parser.add_argument("--resume", action="store_true", help="Resume/appends to an existing local dataset.")
    parser.add_argument("--episode-time-s", type=int, default=DEFAULT_EPISODE_TIME_S)
    parser.add_argument("--reset-time-s", type=int, default=DEFAULT_RESET_TIME_S)
    parser.add_argument("--fps", type=int, default=FPS)
    parser.add_argument("--left-phone-port", type=int, default=LEFT_PHONE_PORT)
    parser.add_argument("--right-phone-port", type=int, default=RIGHT_PHONE_PORT)
    parser.add_argument(
        "--no-videos",
        action="store_true",
        help="Store camera frames as images instead of encoding video features.",
    )
    parser.add_argument(
        "--streaming-encoding",
        action="store_true",
        help="Encode videos during capture instead of after each episode.",
    )
    parser.add_argument("--encoder-threads", type=int, default=2)
    parser.add_argument("--image-writer-threads-per-camera", type=int, default=2)
    parser.add_argument("--push-to-hub", action="store_true", help="Push the dataset after recording.")
    parser.add_argument(
        "--enable-rerun",
        action="store_true",
        help="Enable Rerun visualization. Disabled by default to preserve recording FPS.",
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
        help="Print rolling control-loop timing to identify recording bottlenecks.",
    )
    parser.add_argument(
        "--profile-every-n",
        type=int,
        default=PROFILE_EVERY_N,
        help="When latency profiling is enabled, print one timing report every N control-loop ticks.",
    )
    parser.add_argument(
        "--profile-phone-stream",
        action="store_true",
        help="Print Android WebXR callback rate and move=True percentage for each phone.",
    )
    parser.add_argument(
        "--skip-disabled-phone-ik",
        action="store_true",
        help="Skip each arm EE/IK pipeline while its phone is disabled and hold measured joint position.",
    )
    return parser.parse_args()


def print_recording_controls() -> None:
    print("Keyboard recording controls:")
    print("  Right Arrow: finish the current record/reset loop early")
    print("  Left Arrow: discard and re-record the current episode")
    print("  Esc: stop data recording")
    print("Reset period:")
    print(f"  {RESET_PAUSE_KEY}: pause/resume the reset timer\n")


def make_robot(args: argparse.Namespace) -> XLerobot2WheelsClient:
    config = XLerobot2WheelsClientConfig(
        remote_ip=args.remote_ip,
        id=args.robot_id,
        split_observations=args.split_observations,
    )
    return XLerobot2WheelsClient(config)


def make_recording_context(
    *,
    robot: XLerobot2WheelsClient,
    initial_observation: dict[str, Any],
    left_phone_config: PhoneConfig,
    right_phone_config: PhoneConfig,
) -> RecordingContext:
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

    return RecordingContext(
        left_processor=make_phone_to_arm_joints_processor(
            platform=left_phone_config.phone_os,
            kinematics=left_kinematics,
        ),
        right_processor=make_phone_to_arm_joints_processor(
            platform=right_phone_config.phone_os,
            kinematics=right_kinematics,
        ),
        head_controller=HeadKeyboardController.from_observation(initial_observation),
        base_controller=BaseKeyboardController(robot),
        init_left_arm_action=extract_init_arm_position(initial_observation, "left"),
        init_right_arm_action=extract_init_arm_position(initial_observation, "right"),
    )


def make_dataset_features(robot: XLerobot2WheelsClient, *, use_videos: bool) -> dict[str, dict]:
    features = combine_feature_dicts(
        hw_to_dataset_features(robot.observation_features, OBS_STR, use_videos),
        hw_to_dataset_features(robot.action_features, ACTION, use_videos),
    )

    action_names = features[ACTION]["names"]
    expected_action_names = list(robot.action_features)
    if action_names != expected_action_names:
        raise ValueError(
            "Dataset action order does not match XLerobot2WheelsClient.action_features: "
            f"{action_names} != {expected_action_names}"
        )
    return features


def dataset_root_for_recording(args: argparse.Namespace) -> Path | None:
    if args.root is not None:
        return args.root
    if args.resume:
        return HF_LEROBOT_HOME / args.repo_id
    return None


def make_or_resume_dataset(
    *,
    args: argparse.Namespace,
    robot: XLerobot2WheelsClient,
    features: dict[str, dict],
    use_videos: bool,
) -> LeRobotDataset:
    root = dataset_root_for_recording(args)
    image_writer_threads = args.image_writer_threads_per_camera * len(robot.config.cameras)
    if args.resume:
        dataset = LeRobotDataset.resume(
            args.repo_id,
            root=root,
            streaming_encoding=args.streaming_encoding,
            encoder_threads=args.encoder_threads,
            image_writer_threads=image_writer_threads,
        )
        sanity_check_dataset_robot_compatibility(dataset, robot, args.fps, features)
        if dataset.num_episodes >= args.num_episodes:
            logging.warning(
                "Dataset already has %s episodes, which meets or exceeds target --num-episodes=%s.",
                dataset.num_episodes,
                args.num_episodes,
            )
        else:
            logging.info(
                "Resuming dataset at %s with %s/%s episodes saved.",
                dataset.root,
                dataset.num_episodes,
                args.num_episodes,
            )
        return dataset

    return LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=args.fps,
        root=root,
        robot_type=robot.name,
        features=features,
        use_videos=use_videos,
        image_writer_threads=image_writer_threads,
        streaming_encoding=args.streaming_encoding,
        encoder_threads=args.encoder_threads,
    )


# is this correct? should we be checking for specific camera keys instead of all tuple features?
def assert_observation_has_cameras(robot: XLerobot2WheelsClient, observation: dict[str, Any]) -> None:
    missing = [
        name
        for name, feature in robot.observation_features.items()
        if isinstance(feature, tuple) and name not in observation
    ]
    if missing:
        raise KeyError(
            "Missing camera observations from the remote client: "
            f"{missing}. Confirm the Pi host is streaming these cameras before recording."
        )


def assert_camera_freshness(robot: XLerobot2WheelsClient, *, max_age_ms: float) -> None:
    if max_age_ms <= 0 or not robot.split_observations:
        return

    stale = {
        name: age_ms
        for name, age_ms in robot.camera_frame_ages_ms().items()
        if age_ms is None or age_ms > max_age_ms
    }
    if stale:
        formatted = {
            name: "missing" if age_ms is None else f"{age_ms:.0f}ms" for name, age_ms in stale.items()
        }
        raise RuntimeError(
            "Camera stream is stale while recording: "
            f"{formatted}. Check the Pi image stream or increase --max-camera-age-ms."
        )


def wait_for_camera_observation(
    robot: XLerobot2WheelsClient, *, timeout_s: float, max_camera_age_ms: float
) -> dict[str, Any]:
    deadline_s = time.perf_counter() + timeout_s
    last_error: Exception | None = None
    while time.perf_counter() < deadline_s:
        observation = robot.get_observation()
        try:
            assert_observation_has_cameras(robot, observation)
            assert_camera_freshness(robot, max_age_ms=max_camera_age_ms)
            return observation
        except (KeyError, RuntimeError) as exc:
            last_error = exc
            time.sleep(0.05)
    if last_error is not None:
        raise last_error
    raise TimeoutError("Timed out waiting for remote camera observations.")


def compute_merged_action(
    *,
    context: RecordingContext,
    observation: dict[str, Any],
    left_phone_action: dict[str, Any],
    right_phone_action: dict[str, Any],
    pressed_keys: set[str],
    dt_s: float,
    skip_disabled_phone_ik: bool,
    profiler: LoopProfiler,
) -> dict[str, Any]:
    left_phone_enabled = phone_action_enabled(left_phone_action)
    right_phone_enabled = phone_action_enabled(right_phone_action)
    if left_phone_enabled and "1" not in pressed_keys:
        context.left_reset_to_init = False
    if right_phone_enabled and "2" not in pressed_keys:
        context.right_reset_to_init = False

    section_start = time.perf_counter()
    left_arm_action = process_phone_arm_action(
        phone_action=left_phone_action,
        observation=observation,
        side="left",
        processor=context.left_processor,
        skip_disabled_phone_ik=skip_disabled_phone_ik,
    )
    profiler.record("left_arm", time.perf_counter() - section_start)

    section_start = time.perf_counter()
    right_arm_action = process_phone_arm_action(
        phone_action=right_phone_action,
        observation=observation,
        side="right",
        processor=context.right_processor,
        skip_disabled_phone_ik=skip_disabled_phone_ik,
    )
    profiler.record("right_arm", time.perf_counter() - section_start)

    if "1" in pressed_keys:
        context.left_processor.reset()
        context.left_reset_to_init = True
    if "2" in pressed_keys:
        context.right_processor.reset()
        context.right_reset_to_init = True

    if context.left_reset_to_init:
        left_arm_action = context.init_left_arm_action.copy()
    if context.right_reset_to_init:
        right_arm_action = context.init_right_arm_action.copy()

    section_start = time.perf_counter()
    head_action = context.head_controller.update(pressed_keys, dt_s)
    base_action = context.base_controller.update(pressed_keys)
    profiler.record("head_base", time.perf_counter() - section_start)
    return {**left_arm_action, **right_arm_action, **head_action, **base_action}


def record_control_loop(
    *,
    robot: XLerobot2WheelsClient,
    keyboard: KeyboardTeleop,
    left_phone: Phone,
    right_phone: Phone,
    context: RecordingContext,
    events: dict[str, bool],
    fps: int,
    control_time_s: int,
    task: str,
    dataset: LeRobotDataset | None,
    display_data: bool,
    rerun_log_every_n: int,
    skip_disabled_phone_ik: bool,
    max_camera_age_ms: float,
    profiler: LoopProfiler,
    allow_reset_pause: bool = False,
    countdown_last_s: int = 0,
) -> None:
    control_interval_s = 1.0 / fps
    start_episode_t = time.perf_counter()
    paused_total_s = 0.0
    pause_started_t: float | None = None
    reset_paused = False
    previous_pressed_keys: set[str] = set()
    last_countdown_remaining: int | None = None
    loop_idx = 0

    def elapsed_control_time() -> float:
        now = time.perf_counter()
        current_pause_s = now - pause_started_t if reset_paused and pause_started_t is not None else 0.0
        return now - start_episode_t - paused_total_s - current_pause_s

    while elapsed_control_time() < control_time_s:
        loop_start = time.perf_counter()

        if events["exit_early"]:
            events["exit_early"] = False
            break

        section_start = time.perf_counter()
        observation = robot.get_observation()
        if dataset is not None or display_data:
            assert_observation_has_cameras(robot, observation)
            assert_camera_freshness(robot, max_age_ms=max_camera_age_ms)
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

        if (
            allow_reset_pause
            and RESET_PAUSE_KEY in pressed_keys
            and RESET_PAUSE_KEY not in previous_pressed_keys
        ):
            if reset_paused:
                if pause_started_t is not None:
                    paused_total_s += time.perf_counter() - pause_started_t
                pause_started_t = None
                reset_paused = False
                log_say("Reset timer resumed")
            else:
                pause_started_t = time.perf_counter()
                reset_paused = True
                log_say(f"Reset timer paused. Press {RESET_PAUSE_KEY} to resume.")
        previous_pressed_keys = pressed_keys

        if countdown_last_s > 0 and not reset_paused:
            remaining_s = max(0.0, control_time_s - elapsed_control_time())
            remaining_whole_s = math.ceil(remaining_s)
            if 1 <= remaining_whole_s <= countdown_last_s and remaining_whole_s != last_countdown_remaining:
                log_say(f"{remaining_whole_s} seconds")
                last_countdown_remaining = remaining_whole_s

        if robot.teleop_keys["quit"] in pressed_keys:
            print("Quit requested by keyboard.")
            events["stop_recording"] = True
            break

        merged_action = compute_merged_action(
            context=context,
            observation=observation,
            left_phone_action=left_phone_action,
            right_phone_action=right_phone_action,
            pressed_keys=pressed_keys,
            dt_s=time.perf_counter() - loop_start,
            skip_disabled_phone_ik=skip_disabled_phone_ik,
            profiler=profiler,
        )

        section_start = time.perf_counter()
        sent_action = robot.send_action(merged_action)
        profiler.record("send_action", time.perf_counter() - section_start)

        if dataset is not None:
            observation_frame = build_dataset_frame(dataset.features, observation, prefix=OBS_STR)
            action_frame = build_dataset_frame(dataset.features, sent_action, prefix=ACTION)
            dataset.add_frame({**observation_frame, **action_frame, "task": task})

        if display_data and loop_idx % rerun_log_every_n == 0:
            section_start = time.perf_counter()
            log_rerun_data(observation=observation, action=sent_action)
            profiler.record("rerun", time.perf_counter() - section_start)

        loop_idx += 1
        loop_work_s = time.perf_counter() - loop_start
        profiler.record("loop_work", loop_work_s)
        if loop_work_s > control_interval_s:
            logging.warning(
                "Recording loop is running slower (%.1f Hz) than the target FPS (%s Hz).",
                1.0 / loop_work_s,
                fps,
            )

        section_start = time.perf_counter()
        precise_sleep(max(control_interval_s - loop_work_s, 0.0))
        profiler.record("sleep", time.perf_counter() - section_start)
        profiler.maybe_report()


def main() -> None:
    args = parse_args()
    init_logging()

    if not URDF_PATH.exists():
        raise FileNotFoundError(
            f"SO101 URDF not found at {URDF_PATH}. Update URDF_PATH before running this script."
        )

    robot = make_robot(args)
    keyboard = KeyboardTeleop(KeyboardTeleopConfig())
    left_phone_config = PhoneConfig(
        phone_os=PhoneOS.ANDROID,
        android_port=args.left_phone_port,
        android_profile_stream=args.profile_phone_stream,
    )
    right_phone_config = PhoneConfig(
        phone_os=PhoneOS.ANDROID,
        android_port=args.right_phone_port,
        android_profile_stream=args.profile_phone_stream,
    )
    left_phone = Phone(left_phone_config)
    right_phone = Phone(right_phone_config)
    dataset = None
    listener = None
    context = None

    use_videos = not args.no_videos
    profiler = LoopProfiler(
        enabled=args.profile_latency,
        report_every_n=max(1, args.profile_every_n),
        target_dt_s=1.0 / args.fps,
    )

    try:
        robot.connect()
        keyboard.connect()

        print(f"Connect/calibrate the left Android phone on port {args.left_phone_port}.")
        left_phone.connect()
        print(f"Connect/calibrate the right Android phone on port {args.right_phone_port}.")
        right_phone.connect()

        if args.enable_rerun:
            init_rerun(session_name="keyboard_phone_to_xlerobot_record")
        print_controls(robot)
        print_recording_controls()

        initial_observation = wait_for_camera_observation(
            robot,
            timeout_s=5.0,
            max_camera_age_ms=args.max_camera_age_ms,
        )
        context = make_recording_context(
            robot=robot,
            initial_observation=initial_observation,
            left_phone_config=left_phone_config,
            right_phone_config=right_phone_config,
        )

        features = make_dataset_features(robot, use_videos=use_videos)
        dataset = make_or_resume_dataset(
            args=args,
            robot=robot,
            features=features,
            use_videos=use_videos,
        )

        listener, events = init_keyboard_listener()
        with VideoEncodingManager(dataset):
            while dataset.num_episodes < args.num_episodes and not events["stop_recording"]:
                log_say(f"Recording episode {dataset.num_episodes + 1} of {args.num_episodes}")
                record_control_loop(
                    robot=robot,
                    keyboard=keyboard,
                    left_phone=left_phone,
                    right_phone=right_phone,
                    context=context,
                    events=events,
                    fps=args.fps,
                    control_time_s=args.episode_time_s,
                    task=args.task,
                    dataset=dataset,
                    display_data=args.enable_rerun,
                    rerun_log_every_n=max(1, args.rerun_log_every_n),
                    skip_disabled_phone_ik=args.skip_disabled_phone_ik,
                    max_camera_age_ms=args.max_camera_age_ms,
                    profiler=profiler,
                )

                if not events["stop_recording"] and (
                    dataset.num_episodes < args.num_episodes - 1 or events["rerecord_episode"]
                ):
                    log_say("Reset the environment")
                    record_control_loop(
                        robot=robot,
                        keyboard=keyboard,
                        left_phone=left_phone,
                        right_phone=right_phone,
                        context=context,
                        events=events,
                        fps=args.fps,
                        control_time_s=args.reset_time_s,
                        task=args.task,
                        dataset=None,
                        display_data=args.enable_rerun,
                        rerun_log_every_n=max(1, args.rerun_log_every_n),
                        skip_disabled_phone_ik=args.skip_disabled_phone_ik,
                        max_camera_age_ms=args.max_camera_age_ms,
                        profiler=profiler,
                        allow_reset_pause=True,
                        countdown_last_s=RESET_COUNTDOWN_LAST_S,
                    )

                if events["rerecord_episode"]:
                    log_say("Re-recording episode")
                    events["rerecord_episode"] = False
                    events["exit_early"] = False
                    dataset.clear_episode_buffer()
                    continue

                dataset.save_episode()

    finally:
        if robot.is_connected and context is not None:
            try:
                log_say("Resetting both arms to startup pose")
                reset_arms_to_initial_pose(
                    robot,
                    context.init_left_arm_action,
                    context.init_right_arm_action,
                    fps=args.fps,
                )
            except Exception as exc:
                logging.warning("Failed to reset arms before disconnect: %s", exc)
        log_say("Stop recording")
        for device in [left_phone, right_phone, keyboard, robot]:
            try:
                if device.is_connected:
                    device.disconnect()
            except Exception as exc:
                logging.warning("Failed to disconnect %s: %s", device, exc)
        if listener is not None:
            listener.stop()
        if dataset is not None:
            dataset.finalize()

    if args.push_to_hub and dataset is not None and dataset.num_episodes > 0:
        dataset.push_to_hub()


if __name__ == "__main__":
    main()
