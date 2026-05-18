# Phone to SO100 Archive

This archive holds historical notes moved out of `AGENTS.md` once the XLeRobot teleop and recording pipeline
became usable for human data collection. Keep active commands, contracts, and remaining work in `AGENTS.md`;
put older debugging findings here.

## XLeRobot Completed Implementation Notes

The XLeRobot extension added a full-body control path under
`examples/phone_to_so100/keyboard_phone_to_xlerobot/`:

- `teleoperate.py`: live control from two Android phones plus keyboard head/base controls.
- `record.py`: recording path that mirrors live teleop and saves the final robot-native command dictionary
  returned by `XLerobot2WheelsClient.send_action()`.
- Pi host/client transport in `src/lerobot/robots/xlerobot_2wheels/`, including non-blocking observation
  sends, command draining, split state/image observations, and host diagnostics.

The control loop shape is:

```text
robot.get_observation()
  -> left phone get_action()
  -> right phone get_action()
  -> per-arm EE/IK pipelines
  -> keyboard head/base merge
  -> XLerobot2WheelsClient.send_action()
  -> optional Rerun logging
```

For the full XLeRobot embodiment, keep actions and `observation.state` aligned in robot-native space:
left arm joints, right arm joints, head joints, `x.vel`, and `theta.vel`. Do not make intermediate EE targets
the primary dataset action unless deployment is intentionally changed to EE-space.

## Historical Performance Findings

The early dual-phone loop felt sluggish because the Pi host could spend too much of each tick moving camera
data through the main observation path. Useful findings:

- Non-blocking host observation sends were the highest-impact transport fix.
- Draining queued commands and executing only the newest command was a useful secondary latency guard.
- Rerun should be off or throttled for serious responsiveness testing.
- Skipping disabled-phone IK was kept as an A/B option, but it was not the main bottleneck in measured runs.
- The 2026-05-12 run showed the host could stay near 30 Hz once stale observation sends no longer blocked.
- The 2026-05-18 full-observation run at `640x480` showed `get_observation` around 62 ms, `loop=15Hz`, and
  observation bandwidth around 850 KiB/s, which identified camera capture/serialization on the main host loop.
- The split-transport regression was that `XLerobot2Wheels.get_state_observation()` still read cameras, so
  `--host.split_observations` continued to put base64 JPEGs on the main observation socket while the image
  thread read cameras again.
- The fix was to keep `get_state_observation()` camera-free and leave camera capture in
  `get_camera_observation()` plus the image stream loop.

After that fix, live teleop and recording were reported responsive with three `640x480` camera streams,
`--host.image_send_freq_hz 30`, and `--host.image_jpeg_quality 80`.

## Historical TODOs That Became Non-Blocking

These were useful during profiling but should not block data collection now:

- Debug Android/WebXR stream stalls if they recur; one phone could stop delivering callbacks while the other
  kept streaming.
- Optimize `EEReferenceAndDelta(use_latched_reference=True)` so FK runs only when a reference pose is needed.
- Seed live IK from the previous IK solution after enable instead of measured joints every frame, while still
  resetting from measured joints on enable or after large tracking errors.
- Move remote observation receiving into a background latest-observation cache if Mac-side polling becomes a
  real bottleneck again.
- Avoid sending full hold commands for inactive arms every tick if the robot-side controller can safely keep
  torque/position without repeated command refresh.

## Older Recording Guardrails

The recorder intentionally stays separate from `examples/phone_to_so100/record.py` because the XLeRobot path
has a different dataset contract. It derives dataset features from `XLerobot2WheelsClient.observation_features`
and `XLerobot2WheelsClient.action_features`, validates action-name order at startup, and saves the final
robot-native command returned by `send_action()`.

Recording options that were added during implementation:

- `--split-observations`: state on the main observation socket, images on the image socket.
- `--max-camera-age-ms`: fail recording if cached image frames are stale.
- `--resume`: append until the target total `--num-episodes`.
- `--streaming-encoding`: encode videos during capture.
- `--push-to-hub`: upload after successful finalization.

When diagnosing recording quality, watch `obs_drop`, `cmd_drop`, loop rate, command rate, and camera frame
age. For robust future alignment work, add timestamps or sequence numbers so each saved frame can be tied to
a fresh observation and the newest command intended for that tick.
