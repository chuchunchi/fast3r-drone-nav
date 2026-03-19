# Feature Requirement Specification: Gimbal-Aware Error Reprojection Module

## 1. Module Overview

This module is a small, server-side geometric adapter for the existing visual homing pipeline. Its purpose is to keep the current system behavior unchanged at parallel view (`gimbal_pitch_deg = 0`) while allowing the same controller to operate when the camera is pitched downward by a fixed angle.

The module does **not** redesign the control architecture. It only reprojects the pose error estimated by Fast3R from the camera frame into body-aligned errors that are compatible with the current homing controller path.

### Design Goals

- Preserve current behavior for parallel view flights.
- Avoid major changes to the Android client, protocol, and server control loop.
- Treat gimbal pitch as a fixed flight configuration, not a continuously changing signal.
- Continue using the current single-axis, fixed-velocity homing controller after reprojection.

## 2. Scope and Non-Goals

### In Scope

- One-time configuration of gimbal pitch angle before homing.
- Reprojection of translation and rotation from camera frame to body-aligned control errors.
- Compatibility with the existing waypoint stack, scale calibration, and velocity command generation.

### Out of Scope

- Redesigning the protocol around distance targets.
- Replacing the current velocity-based homing controller.
- Continuous per-frame gimbal synchronization.
- Vertical motion control changes. Altitude remains managed separately.

## 3. Current-System Compatibility

This spec is intentionally aligned with the current implementation:

- Fast3R + pose estimation remain unchanged.
- The module runs on the **server** inside `HomingController.process_homing_frame`, after pose estimation and before command generation.
- The current controller still outputs `pitch_velocity`, `roll_velocity`, `vertical_velocity`, and `yaw_rate` (with vertical currently fixed to `0.0` for safety).
- The Android client continues to consume velocity commands through the existing WebSocket protocol.

In other words, this module changes **how the error is interpreted**, not how commands are transmitted or executed.

## 4. Configuration Handshake

To keep the implementation simple, the gimbal pitch angle is configured once and then held constant for the homing run.

### 4.1 Client Behavior

- The Android client should expose the current gimbal pitch angle to the user.
- Before homing begins, the user may send the selected pitch angle to the server.
- The angle is normalized to positive downward tilt:
  - `alpha_deg = abs(gimbal_pitch_deg)`

### 4.2 Allowed Timing

To match the current system workflow, this configuration should be accepted when the system is in:

- `IDLE`, or
- `ARMED` before `start_homing`

It should not change during an active homing run.

### 4.3 Example Payload

This can remain a simple JSON text command without changing the binary frame protocol:

```json
{
  "type": "init_gimbal_config",
  "gimbal_pitch_deg": 30.0
}
```

### 4.4 Server Behavior

- The server stores `gimbal_pitch_deg` as part of the homing configuration.
- The configured angle remains fixed for the duration of the homing flight.
- The setting is **per flight session only**:
  - on new session start: `gimbal_pitch_deg = 0.0`
  - on reset: `gimbal_pitch_deg = 0.0`
  - no global persistence across server restart
- If no value is provided, the default is:
  - `gimbal_pitch_deg = 0.0`

This default guarantees backward compatibility with the existing parallel-view system.

### 4.5 Integration Point in Existing Command Flow

To align with current implementation, this command is handled in the existing command callback path:

- `WebSocketServer` receives text JSON and forwards it to processor `handle_command`.
- `ProductionFrameProcessor.handle_command` (and optionally `MockFrameProcessor.handle_command`) adds:
  - `init_gimbal_config` command parsing
  - state guard (`IDLE` or `ARMED` only)
  - range validation and explicit rejection (no silent clamp)
  - persistence in controller/session config

### 4.6 Validation and ACK/NACK Policy

`init_gimbal_config` should always return an explicit result message to avoid ambiguity between invalid input and transport issues.

- Valid request:
  - apply setting and return `ok: true`
- Invalid request:
  - reject setting and return `ok: false` with reason
- Do **not** silently clamp values.
- Do **not** reject by silence (no dropped-ack behavior).

### 4.7 Example Response Payloads

Success:

```json
{
  "type": "command_result",
  "command": "init_gimbal_config",
  "ok": true,
  "gimbal_pitch_deg": 30.0
}
```

Out-of-range:

```json
{
  "type": "command_result",
  "command": "init_gimbal_config",
  "ok": false,
  "reason": "out_of_range",
  "allowed_range_deg": [0.0, 89.9],
  "received_gimbal_pitch_deg": 120.0
}
```

Invalid-state:

```json
{
  "type": "command_result",
  "command": "init_gimbal_config",
  "ok": false,
  "reason": "invalid_state",
  "allowed_states": ["IDLE", "ARMED"],
  "current_state": "HOMING"
}
```

## 5. Module I/O Definition

This module is an **internal server-side module**.

### Inputs

- `t_cam`: translation vector from pose estimation, in meters
  - `t_cam = [x_c, y_c, z_c]`
  - Camera convention matches the current system:
    - `x_c`: right
    - `y_c`: down
    - `z_c`: forward
- `R_cam`: relative rotation matrix from pose estimation
- `gimbal_pitch_deg`: fixed downward gimbal tilt for the homing run

### Outputs

- `error_forward_m`: forward/backward planar error in body-aligned space
- `error_lateral_m`: left/right planar error in body-aligned space
- `error_yaw_deg`: yaw correction in degrees
- `distance_planar_m`: planar distance used by waypoint and progress logic

These outputs are consumed by the existing homing controller, which continues to convert them into fixed-velocity commands.

## 6. Algorithm

### Step 1: Convert Pitch Angle

- `alpha_rad = gimbal_pitch_deg * (PI / 180.0)`

### Step 2: Build Camera-to-Body Rotation

Assume:

- Camera frame: `X = right`, `Y = down`, `Z = forward`
- Body frame: `X = forward`, `Y = right`, `Z = down`
- Gimbal effect is pitch only

The camera-to-body rotation with downward pitch `alpha` is:

\[
R_C^B(\alpha) =
\begin{bmatrix}
0 & -\sin\alpha & \cos\alpha \\
1 & 0 & 0 \\
0 & \cos\alpha & \sin\alpha
\end{bmatrix}
\]

This reduces to the current parallel-view axis mapping when `alpha = 0`.

### Step 3: Reproject Translation

Project the pose translation into body-aligned coordinates:

\[
t_{body} = R_C^B(\alpha) \cdot t_{cam}
\]

Expanded:

- `body_x = z_c * cos(alpha_rad) - y_c * sin(alpha_rad)`
- `body_y = x_c`
- `body_z = y_c * cos(alpha_rad) + z_c * sin(alpha_rad)`

For compatibility with the current controller:

- `error_forward_m = body_x`
- `error_lateral_m = body_y`
- `distance_planar_m = sqrt(body_x^2 + body_y^2)`
- Vertical error is ignored for command generation

The module does **not** command vertical motion.

### Step 4: Reproject Rotation and Extract Yaw

Project the relative rotation into the body-aligned frame:

\[
R_{body} = R_C^B(\alpha) \cdot R_{cam} \cdot (R_C^B(\alpha))^T
\]

Extract yaw correction:

- `yaw_rad = atan2(R_body[1, 0], R_body[0, 0])`
- `error_yaw_deg = yaw_rad * 180.0 / PI`

The output unit is degrees to match the current system conventions.

At `alpha = 0`, this reduces to the same yaw extraction behavior currently used by `extract_yaw_error` in the controller path.

## 7. Scale Handling

No new scale stage is introduced in this module.

The current system already computes metric translation using IMU-derived scale during the existing pose estimation pipeline. Therefore:

- `t_cam` should be treated as **already expressed in meters**
- the module must **not** apply IMU scale again

This is important to avoid double-scaling the output errors.

## 8. Integration with Current Homing Controller

The integration should be minimal:

1. Run Fast3R and pose estimation exactly as today.
2. Pass `pose_result.translation` and `pose_result.rotation` into the new module.
3. Replace the current direct camera-frame error extraction with the module outputs:
   - use `error_forward_m` instead of raw `t_cam[2]`
   - use `error_lateral_m` instead of raw `t_cam[0]`
   - use `error_yaw_deg` instead of yaw extracted directly from `R_cam`
4. Use `distance_planar_m` for:
   - waypoint reach check (instead of full `norm(t_cam)`)
   - progress/stuck detection distance metric
5. Keep the existing single-axis control, fixed-velocity logic, confidence scaling, and safety clamps unchanged.

This ensures that only the geometric interpretation changes.

## 9. Constraints

- `gimbal_pitch_deg` must satisfy:
  - `0 <= gimbal_pitch_deg < 90`
- The angle is assumed fixed during a homing run.
- Only pitch is modeled. Gimbal yaw and roll offsets are assumed zero or negligible.
- The module must preserve existing behavior when `gimbal_pitch_deg = 0`.

## 10. Failure Handling and Safety

- If pose estimation fails, the system should keep the existing hover behavior.
- If any module input contains NaN or Inf, output:
  - `error_forward_m = 0`
  - `error_lateral_m = 0`
  - `error_yaw_deg = 0`
  - `distance_planar_m = 0`
- If the reprojected planar error is unreasonably large, downstream controller-side clipping remains responsible for limiting motion.

This keeps safety handling aligned with the current architecture instead of introducing a second control stack.

## 11. Acceptance Criteria

- With `gimbal_pitch_deg = 0`, homing behavior is unchanged from the current working system.
- With moderate downward pitch (for example `20` to `45` degrees), forward/lateral error directions remain physically consistent.
- No protocol redesign is required for normal frame streaming and command execution.
- The Android client only needs a small added command to send the fixed gimbal pitch before homing.
- The existing velocity-based controller remains the sole command generator.
- Waypoint pop and stuck detection remain stable under pitched camera operation because they use planar reprojection distance.

## 12. Implementation Notes (Current Code Mapping)

- `run_server.py`
  - Add storage for `gimbal_pitch_deg` in `ProductionFrameProcessor` (default `0.0`).
  - Handle `init_gimbal_config` in `handle_command`.
  - Include gimbal config in `flight_session.save_config(...)` metadata.
- `visual_homing/server/homing_controller.py`
  - Add a small helper call after pose estimation success:
    - input: `t_cam`, `R_cam`, `gimbal_pitch_deg`
    - output: `error_forward_m`, `error_lateral_m`, `error_yaw_deg`, `distance_planar_m`
  - Keep command synthesis unchanged except replacing raw error extraction and distance metric.
- `visual_homing/server/coordinate_utils.py` (optional)
  - Place reprojection utility here if preferred to keep geometry isolated.
- `visual_homing/tests`
  - Add regression checks:
    - `alpha=0` output equivalence to legacy mapping
    - sign correctness for forward/lateral with nonzero `alpha`
    - waypoint/stuck metric uses planar distance.