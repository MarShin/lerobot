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

`record.py` intentionally splits processing into separate pipelines:

- phone action -> EE action: phone mapping, EE reference/delta, safety, gripper velocity integration
- EE action -> joint action: IK for robot execution
- joint observation -> EE observation: FK for dataset observations

This lets the dataset store EE-space actions and observations rather than low-level joint-space data.

`replay.py` loads recorded EE-space actions from the dataset and only needs the EE -> joint pipeline. It
uses `initial_guess_current_joints=False` because replay is open-loop and should continue from the previous
IK solution instead of constantly reanchoring to measured joints.

`evaluate.py` and `rollout.py` run an EE-space policy. They convert robot joint observations to EE
observations before policy inference, then convert policy EE actions back to joint actions before sending
them to the robot.

## Common Edits

- To change motion speed, adjust `end_effector_step_sizes` in `EEReferenceAndDelta`.
- To change reachable workspace or jump tolerance, adjust `end_effector_bounds` and `max_ee_step_m` in
  `EEBoundsAndSafety`.
- To fix inverted or swapped motion, edit the axis/sign mapping in `MapPhoneActionToRobotAction`.
- To tune gripper behavior, adjust `GripperVelocityToJoint(speed_factor=..., clip_min=..., clip_max=...)`.
- Keep `motor_names=list(robot.bus.motors.keys())` aligned with the URDF joint order used by
  `RobotKinematics`; IK output is mapped back to motor names by index.

## Safety Notes

Use the SO101 URDF referenced in `docs/source/phone_teleop.mdx` unless you have verified another URDF's
frames and joint order. The kinematics target frame should be the gripper frame, usually
`gripper_frame_link`.

Do not remove the bounds/safety step from live teleoperation. If you change phone axis mapping, speed
scales, bounds, or IK initial-guess behavior, test first with conservative speeds and workspace limits.
