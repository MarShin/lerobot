# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

import json
import logging
import time
from typing import Any

import numpy as np
import zmq

# from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.robot_utils import precise_sleep

from .xlerobot_2wheels import XLerobot2Wheels
from .config_xlerobot_2wheels import XLerobot2WheelsConfig, XLerobot2WheelsHostConfig

logger = logging.getLogger(__name__)


class HostProfiler:
    def __init__(self, enabled: bool, report_every_s: float, target_dt_s: float):
        self.enabled = enabled
        self.report_every_s = report_every_s
        self.target_dt_s = target_dt_s
        self.report_start_t = time.perf_counter()
        self.sums: dict[str, float] = {}
        self.maxes: dict[str, float] = {}
        self.counts: dict[str, int] = {}
        self.loop_count = 0
        self.command_count = 0
        self.drained_command_count = 0
        self.watchdog_count = 0
        self.observation_bytes = 0
        self.dropped_observation_count = 0

    def record(self, name: str, dt_s: float) -> None:
        if not self.enabled:
            return
        self.sums[name] = self.sums.get(name, 0.0) + dt_s
        self.maxes[name] = max(self.maxes.get(name, 0.0), dt_s)
        self.counts[name] = self.counts.get(name, 0) + 1

    def increment_command(self) -> None:
        if self.enabled:
            self.command_count += 1

    def add_drained_commands(self, count: int) -> None:
        if self.enabled:
            self.drained_command_count += count

    def increment_watchdog(self) -> None:
        if self.enabled:
            self.watchdog_count += 1

    def add_observation_bytes(self, size_bytes: int) -> None:
        if self.enabled:
            self.observation_bytes += size_bytes

    def increment_dropped_observation(self) -> None:
        if self.enabled:
            self.dropped_observation_count += 1

    def maybe_report(self) -> None:
        if not self.enabled:
            return

        self.loop_count += 1
        now = time.perf_counter()
        elapsed_s = now - self.report_start_t
        if elapsed_s < self.report_every_s:
            return

        names = [
            "cmd_poll",
            "recv_cmd",
            "process_cmd",
            "get_observation",
            "send_observation",
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
        loop_hz = self.loop_count / elapsed_s
        command_hz = self.command_count / elapsed_s
        obs_kbps = self.observation_bytes / elapsed_s / 1024.0
        print(
            f"[host avg/max over {elapsed_s:.1f}s, target={target_ms:.1f}ms] "
            f"loop={loop_hz:.1f}Hz cmd={command_hz:.1f}Hz obs={obs_kbps:.1f}KiB/s "
            f"watchdog={self.watchdog_count} cmd_drop={self.drained_command_count} "
            f"obs_drop={self.dropped_observation_count} | "
            + " | ".join(parts),
            flush=True,
        )

        self.report_start_t = now
        self.sums.clear()
        self.maxes.clear()
        self.counts.clear()
        self.loop_count = 0
        self.command_count = 0
        self.drained_command_count = 0
        self.watchdog_count = 0
        self.observation_bytes = 0
        self.dropped_observation_count = 0


class XLerobot2WheelsHost:
    """
    Host for XLerobot2Wheels that runs on the robot hardware.
    Receives commands via ZMQ and sends them to the robot.
    Sends observations back via ZMQ.
    """

    def __init__(self, robot_config: XLerobot2WheelsConfig, host_config: XLerobot2WheelsHostConfig):
        self.robot_config = robot_config
        self.host_config = host_config
        
        self.robot = XLerobot2Wheels(robot_config)
        
        # ZMQ setup
        self.zmq_context = None
        self.zmq_cmd_socket = None
        self.zmq_observation_socket = None
        
        self._is_running = False
        self.last_cmd_time = time.time()

    def connect(self):
        """Connect to robot hardware and setup ZMQ sockets"""
        logger.info("Connecting to robot hardware...")
        self.robot.connect()
        
        logger.info("Setting up ZMQ sockets...")
        self.zmq_context = zmq.Context()
        
        # Command socket (PULL - receives commands)
        self.zmq_cmd_socket = self.zmq_context.socket(zmq.PULL)
        # Keep command backlog short. The host loop drains this socket each tick
        # and executes only the newest command so old teleop targets are skipped.
        self.zmq_cmd_socket.setsockopt(zmq.RCVHWM, 1)
        self.zmq_cmd_socket.bind(f"tcp://*:{self.host_config.port_zmq_cmd}")
        
        # Observation socket (PUSH - sends observations)
        self.zmq_observation_socket = self.zmq_context.socket(zmq.PUSH)
        # Keep only a tiny outbound queue. If the Mac client is not reading
        # observations during phone calibration/disconnect, drop frames instead
        # of letting the robot-side control loop block behind stale data.
        self.zmq_observation_socket.setsockopt(zmq.SNDHWM, 1)
        self.zmq_observation_socket.setsockopt(zmq.SNDTIMEO, 0)
        self.zmq_observation_socket.bind(f"tcp://*:{self.host_config.port_zmq_observations}")
        
        logger.info(f"ZMQ sockets bound to ports {self.host_config.port_zmq_cmd} and {self.host_config.port_zmq_observations}")
        self._is_running = True

    def run(self):
        """Main control loop"""
        if not self._is_running:
            raise RuntimeError("Host not connected. Call connect() first.")
        
        logger.info("Starting XLerobot2Wheels host control loop...")
        start_time = time.time()
        target_dt = 1.0 / self.host_config.max_loop_freq_hz
        profiler = HostProfiler(
            enabled=self.host_config.profile_diagnostics,
            report_every_s=max(self.host_config.profile_every_s, target_dt),
            target_dt_s=target_dt,
        )
        
        try:
            while self._is_running:
                loop_start = time.perf_counter()
                
                # Check for commands with timeout
                section_start = time.perf_counter()
                has_command = self.zmq_cmd_socket.poll(timeout=1)  # 1ms timeout
                profiler.record("cmd_poll", time.perf_counter() - section_start)
                if has_command:
                    try:
                        section_start = time.perf_counter()
                        # Drain all queued commands and execute only the newest
                        # one. This avoids replaying stale teleop targets after
                        # a brief network or host-side delay.
                        cmd_string, drained_command_count = self._recv_latest_command()
                        cmd = json.loads(cmd_string)
                        profiler.record("recv_cmd", time.perf_counter() - section_start)
                        profiler.add_drained_commands(drained_command_count)

                        section_start = time.perf_counter()
                        self._process_command(cmd)
                        profiler.record("process_cmd", time.perf_counter() - section_start)
                        profiler.increment_command()
                        self.last_cmd_time = time.time()
                    except zmq.Again:
                        pass  # No command available
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to decode command: {e}")
                
                # Check watchdog timeout
                if time.time() - self.last_cmd_time > self.host_config.watchdog_timeout_ms / 1000.0:
                    logger.warning("Watchdog timeout - stopping base motors")
                    self.robot.stop_base()
                    profiler.increment_watchdog()
                    self.last_cmd_time = time.time()  # Reset to avoid spam
                
                # Get observation and send it
                try:
                    section_start = time.perf_counter()
                    obs = self.robot.get_observation()
                    profiler.record("get_observation", time.perf_counter() - section_start)

                    section_start = time.perf_counter()
                    observation_bytes, observation_sent = self._send_observation(obs)
                    profiler.record("send_observation", time.perf_counter() - section_start)
                    if observation_sent:
                        profiler.add_observation_bytes(observation_bytes)
                    else:
                        # Expected when the Mac is not currently consuming observations.
                        # The newest future observation is more useful than blocking here.
                        profiler.increment_dropped_observation()
                except Exception as e:
                    logger.error(f"Failed to get observation: {e}")
                
                # Check if we should stop
                if time.time() - start_time > self.host_config.connection_time_s:
                    logger.info("Connection time limit reached, stopping host")
                    break
                
                # Control loop frequency
                loop_duration = time.perf_counter() - loop_start
                profiler.record("loop_work", loop_duration)
                if loop_duration < target_dt:
                    section_start = time.perf_counter()
                    precise_sleep(target_dt - loop_duration)
                    profiler.record("sleep", time.perf_counter() - section_start)
                profiler.maybe_report()
                
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt, stopping host")
        finally:
            self.stop()

    def _process_command(self, cmd: dict[str, Any]):
        """Process a received command"""
        try:
            # Send command to robot
            self.robot.send_action(cmd)
        except Exception as e:
            logger.error(f"Failed to process command: {e}")

    def _recv_latest_command(self) -> tuple[str, int]:
        """Drain queued commands and return the newest command string."""
        latest_cmd_string = self.zmq_cmd_socket.recv_string(zmq.NOBLOCK)
        drained_count = 0
        while True:
            try:
                # Keep overwriting until the queue is empty; the last message is
                # the command closest to the current phone/keyboard state.
                latest_cmd_string = self.zmq_cmd_socket.recv_string(zmq.NOBLOCK)
                drained_count += 1
            except zmq.Again:
                return latest_cmd_string, drained_count

    def _send_observation(self, obs: dict[str, Any]) -> tuple[int, bool]:
        """Send observation via ZMQ"""
        try:
            # Convert images to base64 for transmission
            obs_for_transmission = {}
            for key, value in obs.items():
                if isinstance(value, np.ndarray) and len(value.shape) == 3:  # Image
                    # Encode image as base64
                    import cv2
                    import base64
                    _, buffer = cv2.imencode('.jpg', value)
                    obs_for_transmission[key] = base64.b64encode(buffer).decode('utf-8')
                else:
                    obs_for_transmission[key] = value
            
            # Send observation
            obs_string = json.dumps(obs_for_transmission)
            obs_size = len(obs_string)
            # Non-blocking send is intentional: teleop should prefer dropping an
            # observation over pausing motor command handling on the Pi.
            self.zmq_observation_socket.send_string(obs_string, flags=zmq.NOBLOCK)
            return obs_size, True
            
        except zmq.Again:
            return 0, False
        except Exception as e:
            logger.error(f"Failed to send observation: {e}")
            return 0, False

    def stop(self):
        """Stop the host and disconnect"""
        logger.info("Stopping XLerobot2Wheels host...")
        self._is_running = False
        
        if self.robot.is_connected:
            self.robot.disconnect()
        
        if self.zmq_cmd_socket:
            self.zmq_cmd_socket.close()
        if self.zmq_observation_socket:
            self.zmq_observation_socket.close()
        if self.zmq_context:
            self.zmq_context.term()
        
        logger.info("XLerobot2Wheels host stopped")


def main():
    """Main function for running the host"""
    import argparse
    
    parser = argparse.ArgumentParser(description="XLerobot2Wheels Host")
    parser.add_argument("--robot.id", dest="robot_id", type=str, default="xlerobot_2wheels", help="Robot ID")
    parser.add_argument("--robot.port1", dest="robot_port1", type=str, default="/dev/ttyACM0", help="Port 1")
    parser.add_argument("--robot.port2", dest="robot_port2", type=str, default="/dev/ttyACM1", help="Port 2")
    parser.add_argument(
        "--host.port_zmq_cmd", dest="host_port_zmq_cmd", type=int, default=5555, help="ZMQ command port"
    )
    parser.add_argument(
        "--host.port_zmq_observations",
        dest="host_port_zmq_observations",
        type=int,
        default=5556,
        help="ZMQ observation port",
    )
    parser.add_argument(
        "--host.connection_time_s", dest="host_connection_time_s", type=int, default=3600, help="Connection time limit"
    )
    parser.add_argument(
        "--host.watchdog_timeout_ms", dest="host_watchdog_timeout_ms", type=int, default=500, help="Watchdog timeout"
    )
    parser.add_argument(
        "--host.max_loop_freq_hz", dest="host_max_loop_freq_hz", type=int, default=30, help="Max loop frequency"
    )
    parser.add_argument(
        "--host.profile_diagnostics",
        dest="host_profile_diagnostics",
        action="store_true",
        help="Print rolling host loop timing diagnostics.",
    )
    parser.add_argument(
        "--host.profile_every_s",
        dest="host_profile_every_s",
        type=float,
        default=2.0,
        help="When host diagnostics are enabled, print one report every N seconds.",
    )
    
    args = parser.parse_args()
    
    # Create configs
    robot_config = XLerobot2WheelsConfig(
        id=args.robot_id,
        port1=args.robot_port1,
        port2=args.robot_port2,
    )
    
    host_config = XLerobot2WheelsHostConfig(
        port_zmq_cmd=args.host_port_zmq_cmd,
        port_zmq_observations=args.host_port_zmq_observations,
        connection_time_s=args.host_connection_time_s,
        watchdog_timeout_ms=args.host_watchdog_timeout_ms,
        max_loop_freq_hz=args.host_max_loop_freq_hz,
        profile_diagnostics=args.host_profile_diagnostics,
        profile_every_s=args.host_profile_every_s,
    )
    
    # Create and run host
    host = XLerobot2WheelsHost(robot_config, host_config)
    
    try:
        host.connect()
        host.run()
    except Exception as e:
        logger.error(f"Host error: {e}")
        host.stop()


if __name__ == "__main__":
    main()
