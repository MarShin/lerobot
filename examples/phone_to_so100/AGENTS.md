# Phone to SO100 Notes

This folder contains examples for controlling an SO100/SO101 follower arm from a phone and for recording,
replaying, and deploying policies trained from that phone teleoperation data.

## Mental Model

Phone teleoperation is Cartesian-first:

```text
Android/iOS phone pose and buttons
  -> calibrated phone action
  -> robot-frame end-effector delta
  -> absolute end-effector pose
  -> bounded/safety-checked end-effector pose
  -> gripper joint target
  -> full robot joint target
  -> robot.send_action()
```

The phone never directly commands robot joints. It emits a small action dictionary, and
`RobotProcessorPipeline` turns that action into the format required by `robot.send_action()`.

## Phone Action Source

`Phone(PhoneConfig(...))` dispatches to the platform-specific implementation in
`src/lerobot/teleoperators/phone/teleop_phone.py`.

- iOS uses HEBI Mobile I/O. Hold `B1` to enable motion; analog `A3` controls gripper velocity.
- Android uses the `teleop` WebXR package. Hold `Move` to enable motion; buttons `A` and `B` open/close
  the gripper.
- Android calibration is not complete at websocket connection time. After tapping `Start` in the WebXR page,
  press and hold `Hold to Move` once to send a `move=True` pose message and capture calibration.

`get_action()` returns:

```python
{
    "phone.pos": pos_cal,
    "phone.rot": rot_cal,
    "phone.raw_inputs": raw_inputs,
    "phone.enabled": enabled,
}
```

Calibration stores the current phone pose as the reference pose. When enable is pressed again after being
released, the implementation reapplies position calibration so the user can reposition the phone while
disabled without causing a robot jump.

## RobotProcessorPipeline

`RobotProcessorPipeline` is a semantic alias for `DataProcessorPipeline`, not a separate robot-specific
runtime. Its behavior comes from:

- `to_transition`: wraps raw inputs into an `EnvTransition`.
- `steps`: applies processor steps sequentially.
- `to_output`: extracts the final action or observation.

For live phone teleop, the input is usually:

```python
(phone_action, robot_observation)
```

with:

```python
to_transition=robot_action_observation_to_transition
to_output=transition_to_robot_action
```

The robot observation is required because FK, IK initial guesses, and gripper integration need the current
measured joint positions.

## Live Teleoperation Pipeline

`teleoperate.py` builds one pipeline from phone action to joint action:

```python
MapPhoneActionToRobotAction
EEReferenceAndDelta
EEBoundsAndSafety
GripperVelocityToJoint
InverseKinematicsEEToJoints
```

Processor responsibilities:

- `MapPhoneActionToRobotAction`: converts calibrated phone pose/buttons into `target_x`, `target_y`,
  `target_z`, `target_wx`, `target_wy`, `target_wz`, `gripper_vel`, and `enabled`. This is where phone
  axes and signs are mapped into robot-frame deltas.
- `EEReferenceAndDelta`: uses FK on current robot joints to compute the current gripper pose, latches a
  reference pose on enable, and converts target deltas into absolute `ee.*` pose fields.
- `EEBoundsAndSafety`: clips the desired EE position to workspace bounds and raises on excessive EE jumps.
- `GripperVelocityToJoint`: integrates `ee.gripper_vel` from the current observed gripper joint to produce
  `ee.gripper_pos`.
- `InverseKinematicsEEToJoints`: converts the absolute EE target into `{motor_name}.pos` joint commands.

In live teleop and closed-loop evaluation, keep `initial_guess_current_joints=True` so IK starts from the
measured robot state each frame.

## Recording, Replay, and Policy Deployment

The current single-arm SO100/SO101 phone examples are organized around an EE-space dataset contract:
actions and proprioceptive observations are stored as `ee.*` features, while live robot execution still
requires joint commands. Each script chooses where to convert between these spaces.

`teleoperate.py` is the live manual-control smoke test. It does not create a dataset or load a policy. The
loop reads `robot.get_observation()`, reads `phone.get_action()`, runs one combined phone -> EE -> joint
pipeline, and sends the resulting joint action to `robot.send_action()`. Use this script first when changing
phone axis mapping, URDFs, IK settings, bounds, or gripper scaling because failures show up immediately
without touching dataset code.

`record.py` collects demonstrations. It intentionally splits processing into three pipelines:

- phone action -> EE action: phone mapping, EE reference/delta, safety, gripper velocity integration
- EE action -> joint action: IK for robot execution
- joint observation -> EE observation: FK for dataset observations

During recording, the robot is controlled by the joint action from the second pipeline, but the dataset
features are derived from the phone -> EE action pipeline and the joint -> EE observation pipeline. That is
why the saved dataset contains EE-space actions and EE-space state instead of raw joint commands. This is
useful for the single-arm phone examples because the policy learns the same Cartesian command space that
the phone produced.

`replay.py` validates a recorded dataset by executing one saved episode. It reads `action` rows from the
dataset, interprets them as EE-space actions, runs only the EE -> joint IK pipeline, and sends the result to
the robot. It sets `initial_guess_current_joints=False` because replay is open-loop: the IK solver should
continue from the previous IK solution for smoothness instead of reanchoring every frame to measured joints.
Use this after recording to catch feature-order mistakes, bad bounds, or IK instability before training.

`evaluate.py` is a self-contained policy-evaluation-and-recording example. It manually loads an `ACTPolicy`,
creates a new evaluation dataset, builds policy pre/postprocessors from the model and dataset stats, and
runs its own episode loop. Each tick converts joint observations to EE observations before policy inference,
converts the policy EE action to robot joints through IK, sends the joint command, then writes the EE
observation and EE action to the evaluation dataset. At the end it finalizes and pushes that eval dataset.
Use it when you want autonomous policy rollouts saved as new episodes for inspection or comparison.

`rollout.py` is the deployment-style path. It runs a trained EE-space policy through the rollout framework
instead of hand-writing the control loop. It builds the same robot-side processors as `evaluate.py`
(`ForwardKinematicsJointsToEE` for observations and `InverseKinematicsEEToJoints` for actions), then passes
them into `build_rollout_context()` with a `BaseStrategy` and `SyncInferenceConfig`. It does not create or
push a dataset. Use it when you want to run the policy on the robot without evaluation recording, or as the
template for later production rollout strategies.

When extending to XLeRobot, do not copy this EE-space dataset contract blindly. For SmolVLA/pi0.5 on the
full embodiment, prefer recording the final robot-native command dictionary as `action` so policy output can
map directly back to `XLerobot2WheelsClient.send_action()`.

## XLeRobot Extension Target

The intended larger embodiment is full XLeRobot-style control:

- two SO101 arms: left arm and right arm
- one side camera per arm
- one head camera attached to the head motor assembly
- head motors
- differential wheel base with two driven wheels, represented at the policy level as body velocity commands

That is three visual streams total:

```text
observation.images.left_side
observation.images.right_side
observation.images.head
```

Use stable camera names consistently across recording, training, and inference. The exact names can differ,
but the dataset feature keys must match the keys used when building inference frames for SmolVLA/pi0.5.

For this full embodiment, prefer saving the primary dataset action in robot-native command space, not EE
space. A good default schema is:

```text
observation.state:
  left_arm_shoulder_pan.pos
  left_arm_shoulder_lift.pos
  left_arm_elbow_flex.pos
  left_arm_wrist_flex.pos
  left_arm_wrist_roll.pos
  left_arm_gripper.pos
  right_arm_shoulder_pan.pos
  right_arm_shoulder_lift.pos
  right_arm_elbow_flex.pos
  right_arm_wrist_flex.pos
  right_arm_wrist_roll.pos
  right_arm_gripper.pos
  head_motor_1.pos
  head_motor_2.pos
  x.vel
  theta.vel

action:
  left_arm_shoulder_pan.pos
  left_arm_shoulder_lift.pos
  left_arm_elbow_flex.pos
  left_arm_wrist_flex.pos
  left_arm_wrist_roll.pos
  left_arm_gripper.pos
  right_arm_shoulder_pan.pos
  right_arm_shoulder_lift.pos
  right_arm_elbow_flex.pos
  right_arm_wrist_flex.pos
  right_arm_wrist_roll.pos
  right_arm_gripper.pos
  head_motor_1.pos
  head_motor_2.pos
  x.vel
  theta.vel
```

This gives 16 state dimensions and 16 action dimensions, which fits SmolVLA and pi0.5 default padded
dimensions (`max_state_dim=32`, `max_action_dim=32`). It also keeps the policy output directly compatible
with `robot.send_action()` after ordinary postprocessing.

Teleoperation may still be Cartesian internally. For example, phone/leader input can be mapped to
left/right EE targets, IK can convert those targets to joint commands, and base/head controls can be mixed
in. The important recording rule is: save the final robot command dictionary sent to the robot as the
primary `action`, not the intermediate EE target. EE pose can be stored as extra diagnostic state if useful,
but avoid making it the canonical action unless deployment is intentionally EE-space.

This recommendation is especially important for pi0.5 relative actions. Relative action processing assumes
the action vector and `observation.state` vector are aligned enough for `action - state` to be meaningful.
Joint/head/base actions paired with joint/head/base state satisfy that assumption. EE actions paired with
joint state do not, unless relative actions are disabled or the state is converted into the same EE action
space.

## XLeRobot Current Status

Historical profiling notes and completed action items live in
[`AGENTS_ARCHIVE.md`](./AGENTS_ARCHIVE.md). Keep this section limited to current operating guidance.

The XLeRobot teleoperation and recording pipeline is ready for human data collection:

- Host/client split observation transport is responsive with three `640x480` camera streams at
  `--host.image_send_freq_hz 30` and `--host.image_jpeg_quality 80`.
- `examples/phone_to_so100/keyboard_phone_to_xlerobot/teleoperate.py` is the live smoke test for two Android
  phones plus keyboard head/base control.
- `examples/phone_to_so100/keyboard_phone_to_xlerobot/record.py` records the same control path into a
  robot-native 16-dim state/action dataset.
- The next human task is collection quality, not core implementation: record roughly 20-50 good episodes,
  then replay/inspect before training SmolVLA/pi0.5.

Keep the dataset recommendation above as the target contract: teleop may use EE-space internally, but the
recorded primary `action` should be the final robot-native command dictionary returned by
`XLerobot2WheelsClient.send_action()`.

### Host Command

Run this on the Pi before teleop or recording:

```bash
python -m lerobot.robots.xlerobot_2wheels.xlerobot_2wheels_host \
  --robot.id=my_xlerobot_2wheels \
  --host.split_observations \
  --host.stream_images \
  --host.image_send_freq_hz 30 \
  --host.image_jpeg_quality 80 \
  --host.profile_diagnostics
```

Quality guardrail: prefer collecting only when host diagnostics stay near `loop=30Hz`, `cmd=30Hz`, and
`obs_drop=0`. Occasional `cmd_drop` can be intentional stale-command draining, but persistent large values
mean the host is executing fewer distinct commands than the client sent.

### Teleop Command

Run this on the Mac for live control checks:

```bash
python examples/phone_to_so100/keyboard_phone_to_xlerobot/teleoperate.py \
  --split-observations \
  --profile-latency \
  --profile-phone-stream \
  --enable-rerun
```

The script uses `REMOTE_IP=raspberrypi.local`, `ROBOT_ID=my_xlerobot_2wheels`, left phone port `4443`, and
right phone port `4444` by default. Update constants or CLI flags when those values change.

### Record Command

Use the same split-observation host for full-resolution recording:

```bash
python examples/phone_to_so100/keyboard_phone_to_xlerobot/record.py \
  --profile-latency \
  --profile-phone-stream \
  --enable-rerun \
  --repo-id marshin68/xlerobot_pick_n_place4 \
  --task "Pick up the box and put into the gray tray" \
  --streaming-encoding \
  --episode-time-s 30 \
  --reset-time-s 30 \
  --num-episodes 5 \
  --split-observations
```

Use `--resume` to append to an existing dataset. In resume mode, `--num-episodes` is the target total number
of episodes, not the number to add in that run. Use `--push-to-hub` only after local recording finalizes
successfully and the dataset has the expected episodes.

Recording controls come from `init_keyboard_listener()`: Right Arrow finishes the current record/reset loop
early, Left Arrow discards and re-records the current episode, Esc stops recording, and `p` pauses/resumes
the reset timer.

### Remaining Engineering Work

- Add policy rollout/evaluation once data collection is stable.
- Build inference frames with the same three camera keys and 16-dim state used during recording.
- Convert policy output with `make_robot_action()` and send it directly through the XLeRobot client path.
- Only add EE postprocessing at inference if the dataset action space intentionally changes to EE-space.

## Common Edits

- To change motion speed, adjust `end_effector_step_sizes` in `EEReferenceAndDelta`.
- To change reachable workspace or jump tolerance, adjust `end_effector_bounds` and `max_ee_step_m` in
  `EEBoundsAndSafety`.
- To fix inverted or swapped motion, edit the axis/sign mapping in `MapPhoneActionToRobotAction`.
- To tune gripper behavior, adjust `GripperVelocityToJoint(speed_factor=..., clip_min=..., clip_max=...)`.
- Keep `motor_names=list(robot.bus.motors.keys())` aligned with the URDF joint order used by
  `RobotKinematics`; IK output is mapped back to motor names by index.
- For XLeRobot/SmolVLA recording, derive dataset features from the final robot-native action/observation
  schema so `dataset.features["action"]["names"]` matches the order expected by `make_robot_action()`.

## Safety Notes

Use the SO101 URDF referenced in `docs/source/phone_teleop.mdx` unless you have verified another URDF's
frames and joint order. The kinematics target frame should be the gripper frame, usually
`gripper_frame_link`.

Do not remove the bounds/safety step from live teleoperation. If you change phone axis mapping, speed
scales, bounds, or IK initial-guess behavior, test first with conservative speeds and workspace limits.
