# Visual Homing System Design Specification

**Project:** UAV Visual Retracing using Fast3R  
**Version:** 4.0  
**Status:** ready to start implement

---

## 1. Executive Summary

The objective of this system is to enable a DJI UAV to autonomously return to its launch point in GPS-denied environments using a **Server-Centric "Teach and Repeat"** architecture.

### Core Architecture
- **Server (Ground Station):** All heavy computation — Fast3R inference, keyframe management, pose estimation via SVD Procrustes, PID control calculation
- **Client (Android App):** Thin client — streams video/telemetry, executes velocity commands via DJI Mobile SDK (MSDK)

### Key Design Decisions
| Decision | Choice | Rationale |
|----------|--------|-----------|
| Client Platform | Android + DJI MSDK | Easier to prototype, wide device support |
| Pose Estimation | SVD Procrustes (3D-3D) | Fast3R outputs pointmaps, not 2D features; 3D-3D alignment is direct |
| Scale Resolution | IMU velocity integration | Provides metric scale during TEACH phase |
| Communication | Binary WebSocket | Low latency, reliable delivery |

---

## 2. System Architecture

### 2.1 Client Side (Android App with DJI MSDK)

**Role:** Relay & Actuator (Thin Client)

**Hardware Requirements:**
- Android device (API 24+) with USB-C
- DJI Remote Controller (RC-N1, RC-Pro, or similar)
- Compatible DJI drone (Mavic 3, Air 2S, Mini 3 Pro, etc.)

**Software Components:**

```
┌─────────────────────────────────────────────────────────┐
│                    Android App                          │
├─────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐              │
│  │  VideoFeeder    │  │  FlightController│              │
│  │  YuvDataCallback│  │  StateCallback   │              │
│  └────────┬────────┘  └────────┬────────┘              │
│           │                     │                        │
│           ▼                     ▼                        │
│  ┌─────────────────────────────────────────┐            │
│  │         Frame & Telemetry Packager       │            │
│  │  - JPEG compression (quality=80)         │            │
│  │  - Binary packet assembly                │            │
│  └────────────────────┬────────────────────┘            │
│                       │                                  │
│                       ▼                                  │
│  ┌─────────────────────────────────────────┐            │
│  │         WebSocket Client                 │            │
│  │  - Binary frames to server               │            │
│  │  - JSON commands from server             │            │
│  └────────────────────┬────────────────────┘            │
│                       │                                  │
│                       ▼                                  │
│  ┌─────────────────────────────────────────┐            │
│  │         VirtualStick Controller          │            │
│  │  - Velocity command execution            │            │
│  │  - Failsafe watchdog                     │            │
│  └─────────────────────────────────────────┘            │
└─────────────────────────────────────────────────────────┘
```

**Responsibilities:**

1. **Video Extraction:**
   - Use `VideoFeeder.VideoDataListener` for H.264 stream (decode with MediaCodec)
   - OR use `YuvDataObserver` for raw YUV frames (lower latency, higher CPU)
   - Resize to **512×384** (4:3 aspect ratio, matches Fast3R training)
   - Compress to JPEG (quality=80, ~30-50KB per frame)

2. **Telemetry Extraction (10Hz minimum):**
   ```java
   FlightControllerState state = flightController.getState();
   
   // Velocity from IMU (body frame, m/s)
   float vx = state.getVelocityX();  // Forward
   float vy = state.getVelocityY();  // Right  
   float vz = state.getVelocityZ();  // Down
   
   // Attitude
   float yaw = state.getAttitude().yaw;      // degrees
   float pitch = state.getAttitude().pitch;
   float roll = state.getAttitude().roll;
   
   // Height (barometric + ultrasonic fusion)
   float height = state.getUltrasonicHeightInMeters();
   
   // Timestamp
   long timestamp = System.currentTimeMillis();
   ```

3. **Data Transmission:** Binary WebSocket frames (see Section 3)

4. **Actuation:** Execute velocity commands via VirtualStick mode

---

### 2.2 Server Side (Ground Station)

**Role:** Brain & Memory (Fat Server)

**Hardware Requirements:**
- Linux workstation with NVIDIA RTX 4090 GPU
- CUDA 11.8+ and PyTorch 2.0+
- Fast3R model weights loaded in GPU memory

**Software Components:**

```
┌─────────────────────────────────────────────────────────┐
│                    Python Server                         │
├─────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────┐            │
│  │         WebSocket Server                 │            │
│  │  - Async frame reception                 │            │
│  │  - Command dispatch                      │            │
│  └────────────────────┬────────────────────┘            │
│                       │                                  │
│                       ▼                                  │
│  ┌─────────────────────────────────────────┐            │
│  │         State Machine                    │            │
│  │  IDLE → RECORDING → ARMED → HOMING       │            │
│  └────────────────────┬────────────────────┘            │
│                       │                                  │
│           ┌───────────┴───────────┐                      │
│           ▼                       ▼                      │
│  ┌─────────────────┐    ┌─────────────────┐             │
│  │ Keyframe Stack  │    │   Fast3R        │             │
│  │ Manager         │    │   Inference     │             │
│  │ - Scale tracker │    │   Engine        │             │
│  └────────┬────────┘    └────────┬────────┘             │
│           │                       │                      │
│           └───────────┬───────────┘                      │
│                       ▼                                  │
│  ┌─────────────────────────────────────────┐            │
│  │         Pose Estimator                   │            │
│  │  - SVD Procrustes (roma library)         │            │
│  │  - Confidence-weighted alignment         │            │
│  └────────────────────┬────────────────────┘            │
│                       │                                  │
│                       ▼                                  │
│  ┌─────────────────────────────────────────┐            │
│  │         PID Controller                   │            │
│  │  - Velocity command generation           │            │
│  │  - Coordinate frame conversion           │            │
│  └─────────────────────────────────────────┘            │
└─────────────────────────────────────────────────────────┘
```

**Responsibilities:**

1. **Keyframe Stack Management:** 
   - Push/pop keyframes based on visual overlap or temporal spacing
   - Track cumulative IMU-based distance for scale calibration

2. **State Machine:** Manage system phases (see Section 4)

3. **Fast3R Inference:** 
   - Load model once at startup
   - Run inference on [Live_Frame, Keyframe_Target] pairs
   - Output: 3D pointmaps in shared reference frame

4. **Pose Estimation:**
   - Use SVD Procrustes (`roma.rigid_points_registration`) for 3D-3D alignment
   - Apply IMU-derived scale factor to get metric translation

5. **Control Loop:**
   - PID controller for each axis (forward, lateral, vertical, yaw)
   - Convert to DJI VirtualStick commands

---

## 3. Communication Protocol

### 3.1 Control Link (Binary WebSocket)

**Purpose:** Real-time computer vision and control  
**Requirements:** Latency < 200ms end-to-end  
**Port:** 8765 (configurable)

#### 3.1.1 Upstream: Client → Server (Binary Frame)

```
┌────────────────────────────────────────────────────────────────┐
│ Byte Offset │ Size    │ Field            │ Description         │
├─────────────┼─────────┼──────────────────┼─────────────────────┤
│ 0           │ 1       │ message_type     │ 0x01 = Frame+Telem  │
│ 1           │ 4       │ frame_id         │ uint32, sequential  │
│ 5           │ 8       │ timestamp_ms     │ uint64, epoch ms    │
│ 13          │ 4       │ velocity_x       │ float32, m/s        │
│ 17          │ 4       │ velocity_y       │ float32, m/s        │
│ 21          │ 4       │ velocity_z       │ float32, m/s        │
│ 25          │ 4       │ yaw              │ float32, degrees    │
│ 29          │ 4       │ pitch            │ float32, degrees    │
│ 33          │ 4       │ roll             │ float32, degrees    │
│ 37          │ 4       │ height           │ float32, meters     │
│ 41          │ 4       │ image_size       │ uint32, JPEG bytes  │
│ 45          │ N       │ image_data       │ JPEG bytes          │
└────────────────────────────────────────────────────────────────┘
```

**Frame Rate:** 5-10 Hz (configurable based on network bandwidth)

#### 3.1.2 Downstream: Server → Client (JSON)

```json
{
  "type": "control",
  "frame_id": 12345,
  "timestamp_ms": 1704412800000,
  "command": {
    "pitch_velocity": 0.5,
    "roll_velocity": -0.2,
    "vertical_velocity": 0.0,
    "yaw_rate": 0.0
  },
  "status": {
    "state": "HOMING",
    "keyframes_remaining": 5,
    "target_distance_m": 2.3,
    "confidence": 0.85
  }
}
```

**Command Semantics (BODY frame, VELOCITY mode):**
| Field | Unit | Range | Description |
|-------|------|-------|-------------|
| `pitch_velocity` | m/s | [-5, 5] | Forward (+) / Backward (-) |
| `roll_velocity` | m/s | [-5, 5] | Right (+) / Left (-) |
| `vertical_velocity` | m/s | [-3, 3] | Up (+) / Down (-) |
| `yaw_rate` | deg/s | [-100, 100] | Clockwise (+) / Counter-clockwise (-) |

### 3.2 Monitoring Link (RTMP - Optional)

**Purpose:** Human visual monitoring only  
**Implementation:** Standard DJI MSDK `LiveStreamManager.startStream()`  
**Note:** NOT used for Fast3R inference due to buffering latency (~1-2 seconds)

---

## 4. Operational Workflow

### 4.1 State Machine

```
                    ┌──────────┐
                    │   IDLE   │
                    └────┬─────┘
                         │ User starts recording
                         ▼
                    ┌──────────┐
             ┌──────│RECORDING │◄─────────┐
             │      └────┬─────┘          │
             │           │ User stops     │ Keyframe pushed
             │           │ recording      │ (loop)
             │           ▼                │
             │      ┌──────────┐          │
             │      │  ARMED   │──────────┘
             │      └────┬─────┘
             │           │ User triggers return
             │           ▼
             │      ┌──────────┐
             │      │  HOMING  │◄─────────┐
             │      └────┬─────┘          │
             │           │                │ Next keyframe
             │           │ Keyframe       │ (loop)
             │           │ reached        │
             │           ├────────────────┘
             │           │ Stack empty (home reached)
             │           ▼
             │      ┌──────────┐
             └─────►│COMPLETED │
                    └──────────┘
```

### 4.2 Phase 1: RECORDING (Teach Phase)

**Goal:** Build a stack of keyframes with known inter-keyframe distances

**Input:** Continuous stream of frames + telemetry from Client

**Logic:**

```python
class KeyframeStackManager:
    def __init__(self):
        self.stack = []  # List of (image, pointmap, cumulative_distance)
        self.cumulative_distance = 0.0  # meters, from IMU integration
        self.last_telemetry_time = None
        self.keyframe_interval_m = 2.0  # Push keyframe every 2 meters
        self.keyframe_interval_s = 3.0  # OR every 3 seconds (whichever first)
        
    def process_frame(self, frame, telemetry):
        # 1. Update cumulative distance from IMU velocity
        if self.last_telemetry_time is not None:
            dt = (telemetry.timestamp - self.last_telemetry_time) / 1000.0
            velocity_magnitude = sqrt(
                telemetry.vx**2 + telemetry.vy**2 + telemetry.vz**2
            )
            self.cumulative_distance += velocity_magnitude * dt
        self.last_telemetry_time = telemetry.timestamp
        
        # 2. Decide whether to push keyframe
        if len(self.stack) == 0:
            self._push_keyframe(frame)
            return
            
        distance_since_last = self.cumulative_distance - self.stack[-1].cumulative_distance
        time_since_last = telemetry.timestamp - self.stack[-1].timestamp
        
        if distance_since_last > self.keyframe_interval_m:
            self._push_keyframe(frame)
        elif time_since_last > self.keyframe_interval_s * 1000:
            self._push_keyframe(frame)
            
    def _push_keyframe(self, frame):
        keyframe = Keyframe(
            image=frame,
            cumulative_distance=self.cumulative_distance,
            timestamp=current_time_ms()
        )
        self.stack.append(keyframe)
        log.info(f"Pushed keyframe {len(self.stack)} at distance {self.cumulative_distance:.2f}m")
```

**Scale Calibration:**

The cumulative distance from IMU integration provides the **metric scale** for the path. When computing relative poses during HOMING, we use this to convert Fast3R's unitless translations to meters:

```python
# During RECORDING, compute scale factor for each keyframe pair
for i in range(1, len(stack)):
    imu_distance = stack[i].cumulative_distance - stack[i-1].cumulative_distance
    
    # Run Fast3R on consecutive keyframes
    pts3d_i, pts3d_j = fast3r_inference(stack[i-1].image, stack[i].image)
    _, t_fast3r, s = roma.rigid_points_registration(pts3d_i, pts3d_j)
    fast3r_distance = np.linalg.norm(t_fast3r.numpy())
    
    # Compute scale factor: meters per Fast3R unit
    scale_factor = imu_distance / fast3r_distance
    stack[i].scale_factor = scale_factor
    
# Average scale factor for robustness
global_scale_factor = np.median([kf.scale_factor for kf in stack[1:]])
```

### 4.3 Phase 2: HOMING (Visual Retracing)

**Trigger:** User sends "Return" command

**Loop:**

```python
class HomingController:
    def __init__(self, keyframe_stack, fast3r_model, scale_factor):
        self.stack = keyframe_stack
        self.model = fast3r_model
        self.scale = scale_factor  # meters per Fast3R unit
        self.target_idx = len(self.stack) - 1  # Start from last keyframe
        
        # PID controllers
        self.pid_forward = PIDController(kp=0.5, ki=0.01, kd=0.1)
        self.pid_lateral = PIDController(kp=0.5, ki=0.01, kd=0.1)
        self.pid_vertical = PIDController(kp=0.3, ki=0.01, kd=0.05)
        self.pid_yaw = PIDController(kp=1.0, ki=0.0, kd=0.2)
        
        # Waypoint threshold
        self.waypoint_threshold_m = 0.8  # Pop keyframe when within 0.8m
        
    def process_frame(self, live_frame, telemetry):
        if self.target_idx < 0:
            return {"state": "COMPLETED", "command": hover_command()}
            
        # 1. Get current target keyframe
        target_keyframe = self.stack[self.target_idx]
        
        # 2. Run Fast3R inference
        output = fast3r_inference(live_frame, target_keyframe.image)
        pts3d_live = output[0]['pts3d_in_other_view']  # Live frame points in target frame
        pts3d_target = output[1]['pts3d_in_other_view']  # Target points in target frame
        conf_live = output[0]['conf']
        
        # 3. Compute relative pose via SVD Procrustes
        R, t, s = self.compute_relative_pose(pts3d_live, pts3d_target, conf_live)
        
        # 4. Apply scale to get metric translation
        t_meters = t.numpy() * self.scale * s
        
        # 5. Extract pose error (in camera frame)
        #    Camera frame: X=right, Y=down, Z=forward
        error_forward = t_meters[2]   # Z = forward distance to target
        error_lateral = t_meters[0]   # X = right offset
        error_vertical = t_meters[1]  # Y = down offset
        error_yaw = self.extract_yaw_error(R)
        
        # 6. PID control
        cmd_forward = self.pid_forward.compute(error_forward)
        cmd_lateral = self.pid_lateral.compute(error_lateral)
        cmd_vertical = self.pid_vertical.compute(-error_vertical)  # Invert: down→up
        cmd_yaw = self.pid_yaw.compute(error_yaw)
        
        # 7. Check if we've reached the waypoint
        distance_to_target = np.linalg.norm(t_meters)
        if distance_to_target < self.waypoint_threshold_m:
            self.target_idx -= 1  # Pop to previous keyframe
            log.info(f"Reached keyframe, {self.target_idx + 1} remaining")
            self.pid_forward.reset()
            self.pid_lateral.reset()
            
        # 8. Build command (see Section 5 for coordinate mapping)
        return {
            "state": "HOMING",
            "command": {
                "pitch_velocity": np.clip(cmd_forward, -2.0, 2.0),
                "roll_velocity": np.clip(cmd_lateral, -2.0, 2.0),
                "vertical_velocity": np.clip(cmd_vertical, -1.0, 1.0),
                "yaw_rate": np.clip(cmd_yaw, -30.0, 30.0)
            },
            "target_distance_m": distance_to_target,
            "keyframes_remaining": self.target_idx + 1
        }
        
    def compute_relative_pose(self, pts3d_live, pts3d_target, conf):
        """
        Compute rigid transformation from live frame to target frame
        using SVD Procrustes (3D-3D point alignment).
        
        Returns:
            R: 3x3 rotation matrix
            t: 3D translation vector (live_origin → target_origin in target frame)
            s: scale factor
        """
        # Flatten pointmaps
        pts_live_flat = pts3d_live.reshape(-1, 3)
        pts_target_flat = pts3d_target.reshape(-1, 3)
        conf_flat = conf.reshape(-1)
        
        # Filter by confidence
        conf_threshold = torch.quantile(conf_flat, 0.5)  # Use top 50%
        mask = conf_flat >= conf_threshold
        
        if mask.sum() < 100:  # Not enough confident points
            return torch.eye(3), torch.zeros(3), 1.0
            
        pts_live_good = pts_live_flat[mask]
        pts_target_good = pts_target_flat[mask]
        weights = conf_flat[mask]
        
        # SVD Procrustes: find R, t, s such that
        # pts_target ≈ s * (pts_live @ R.T) + t
        R, t, s = roma.rigid_points_registration(
            pts_live_good, 
            pts_target_good, 
            weights=weights,
            compute_scaling=True
        )
        
        return R, t, s
        
    def extract_yaw_error(self, R):
        """Extract yaw angle from rotation matrix (assuming small roll/pitch)."""
        # Yaw = atan2(R[1,0], R[0,0]) for Z-up convention
        # For camera frame (Y-down, Z-forward):
        yaw_rad = torch.atan2(R[0, 2], R[2, 2])
        return float(yaw_rad) * 180 / np.pi  # Convert to degrees
```

---

## 5. Coordinate Frame Mapping

### 5.1 Coordinate Frames Reference

```
CAMERA FRAME (OpenCV/Fast3R convention):
         Y (down)
         │
         │
         └───── X (right)
        ╱
       ╱
      Z (forward, into scene)

DJI BODY FRAME (with camera pointing forward):
         Z (down)
         │
         │
         └───── Y (right)
        ╱
       ╱
      X (forward)

DJI VIRTUALSTICK (BODY mode, VELOCITY mode):
    pitch_velocity  → X axis (forward +, backward -)
    roll_velocity   → Y axis (right +, left -)
    vertical_velocity → Z axis (up +, down -)  [INVERTED from body frame!]
    yaw_rate        → rotation around Z (clockwise +)
```

### 5.2 Transformation: Fast3R → DJI Commands

```python
def fast3r_to_dji_command(t_cam, yaw_error_deg, pid_gains):
    """
    Convert Fast3R pose error (camera frame) to DJI VirtualStick commands.
    
    Args:
        t_cam: [tx, ty, tz] translation in camera frame (meters)
               tx = right offset, ty = down offset, tz = forward distance
        yaw_error_deg: yaw error in degrees
        pid_gains: dict with forward/lateral/vertical/yaw gains
        
    Returns:
        dict: DJI VirtualStick velocity commands
    """
    # Map camera frame axes to DJI body frame
    # Camera Z (forward) → DJI pitch (forward velocity)
    # Camera X (right) → DJI roll (right velocity)  
    # Camera Y (down) → DJI vertical (INVERTED: up velocity)
    
    error_forward = t_cam[2]   # Camera Z = how far forward to go
    error_lateral = t_cam[0]   # Camera X = how far right to go
    error_vertical = -t_cam[1] # Camera Y inverted = how far up to go
    
    # Apply PID (simplified as proportional control here)
    cmd = {
        "pitch_velocity": pid_gains['kp_forward'] * error_forward,
        "roll_velocity": pid_gains['kp_lateral'] * error_lateral,
        "vertical_velocity": pid_gains['kp_vertical'] * error_vertical,
        "yaw_rate": pid_gains['kp_yaw'] * yaw_error_deg
    }
    
    # Clamp to safe limits
    cmd["pitch_velocity"] = np.clip(cmd["pitch_velocity"], -2.0, 2.0)
    cmd["roll_velocity"] = np.clip(cmd["roll_velocity"], -2.0, 2.0)
    cmd["vertical_velocity"] = np.clip(cmd["vertical_velocity"], -1.0, 1.0)
    cmd["yaw_rate"] = np.clip(cmd["yaw_rate"], -30.0, 30.0)
    
    return cmd
```

### 5.3 DJI VirtualStick Configuration (Android)

```java
// In your FlightController setup
FlightController fc = aircraft.getFlightController();

// Enable Virtual Stick mode
fc.setVirtualStickModeEnabled(true, error -> {
    if (error == null) {
        Log.d(TAG, "Virtual Stick enabled");
    }
});

// Configure control modes
fc.setRollPitchControlMode(RollPitchControlMode.VELOCITY);  // m/s
fc.setYawControlMode(YawControlMode.ANGULAR_VELOCITY);       // deg/s
fc.setVerticalControlMode(VerticalControlMode.VELOCITY);     // m/s
fc.setRollPitchCoordinateSystem(FlightCoordinateSystem.BODY); // CRITICAL!

// Send velocity commands (called at ~10Hz)
void sendVelocityCommand(float pitch, float roll, float yaw, float throttle) {
    // FlightControlData(pitch, roll, yaw, throttle)
    // pitch = forward velocity (m/s, positive = forward)
    // roll = right velocity (m/s, positive = right)
    // yaw = rotation rate (deg/s, positive = clockwise)
    // throttle = vertical velocity (m/s, positive = up)
    
    FlightControlData data = new FlightControlData(pitch, roll, yaw, throttle);
    fc.sendVirtualStickFlightControlData(data, error -> {
        if (error != null) {
            Log.e(TAG, "Control error: " + error.getDescription());
        }
    });
}
```

---

## 6. Fast3R Integration

### 6.1 Model Loading

```python
import torch
from fast3r.models.fast3r import Fast3R
from fast3r.dust3r.inference_multiview import inference
from fast3r.dust3r.utils.image import load_images

class Fast3REngine:
    def __init__(self, checkpoint="jedyang97/Fast3R_ViT_Large_512", device="cuda"):
        self.device = torch.device(device)
        self.model = Fast3R.from_pretrained(checkpoint).to(self.device)
        self.model.eval()
        
    @torch.no_grad()
    def infer_pair(self, image1: np.ndarray, image2: np.ndarray):
        """
        Run Fast3R on a pair of images.
        
        Args:
            image1, image2: RGB images as numpy arrays (H, W, 3), uint8
            
        Returns:
            dict: Contains pts3d_in_other_view, conf for each view
        """
        # Prepare images in the format Fast3R expects
        imgs = self._prepare_images([image1, image2])
        
        # Run inference
        output, _ = inference(
            imgs, 
            self.model, 
            self.device,
            dtype=torch.float32,
            verbose=False,
            profiling=False
        )
        
        return output['preds']
        
    def _prepare_images(self, images):
        """Convert numpy images to Fast3R format."""
        # Similar to load_images but from memory
        prepared = []
        for img in images:
            # Resize to 512x384 if needed
            if img.shape[:2] != (384, 512):
                img = cv2.resize(img, (512, 384))
            
            # Convert to tensor and normalize
            img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            img_tensor = img_tensor.unsqueeze(0)  # Add batch dim
            
            prepared.append({
                'img': img_tensor.to(self.device),
                'true_shape': torch.tensor([[384, 512]]).to(self.device)
            })
            
        return prepared
```

### 6.2 Inference Latency Budget

| Stage | Expected Time | Notes |
|-------|---------------|-------|
| Image decode (JPEG) | ~5ms | On server CPU |
| Fast3R forward pass | 100-200ms | RTX 4090, 512×384 input |
| SVD Procrustes | ~10ms | roma library, GPU |
| PID computation | <1ms | CPU |
| **Total** | **~120-220ms** | Meets <200ms target |

### 6.3 Asynchronous Inference Pipeline

To avoid blocking the control loop, run inference asynchronously:

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

class AsyncInferenceEngine:
    def __init__(self, fast3r_engine):
        self.engine = fast3r_engine
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.latest_result = None
        self.latest_timestamp = 0
        
    async def submit_inference(self, live_frame, target_frame, timestamp):
        """Submit inference job, non-blocking."""
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            self.executor,
            self.engine.infer_pair,
            live_frame, 
            target_frame
        )
        self.latest_result = result
        self.latest_timestamp = timestamp
        return result
        
    def get_latest_result(self):
        """Get most recent inference result for control."""
        return self.latest_result, self.latest_timestamp
```

---

## 7. Safety & Failsafes

### 7.1 Heartbeat Mechanism

**Client → Server:** Send frame+telemetry at minimum 5Hz  
**Server → Client:** Send command/status at minimum 5Hz

**Timeout Actions:**

| Condition | Timeout | Action |
|-----------|---------|--------|
| Server stops receiving frames | 500ms | Log warning, continue with last known state |
| Server stops receiving frames | 2000ms | Abort homing, send HOVER command |
| Client stops receiving commands | 500ms | Ignore, use last command |
| Client stops receiving commands | 1000ms | **Disable VirtualStick, enter GPS hover** |

### 7.2 Client-Side Watchdog (Android)

```java
class VirtualStickWatchdog {
    private static final long COMMAND_TIMEOUT_MS = 1000;
    private long lastCommandTime = 0;
    private Handler handler = new Handler(Looper.getMainLooper());
    
    public void onCommandReceived() {
        lastCommandTime = System.currentTimeMillis();
    }
    
    private Runnable watchdogTask = () -> {
        if (System.currentTimeMillis() - lastCommandTime > COMMAND_TIMEOUT_MS) {
            // Emergency: disable virtual stick
            flightController.setVirtualStickModeEnabled(false, null);
            Log.w(TAG, "WATCHDOG: Virtual stick disabled due to timeout");
        }
        handler.postDelayed(watchdogTask, 100);
    };
    
    public void start() {
        handler.post(watchdogTask);
    }
}
```

### 7.3 Additional Safety Measures

1. **Velocity Limits:** Never exceed 2 m/s horizontal, 1 m/s vertical
2. **Confidence Threshold:** If Fast3R confidence < 0.3, enter HOVER and wait
3. **Geofencing:** Track cumulative displacement; abort if > 1.5× outbound distance
4. **Battery Monitor:** If battery < 25%, abort homing and trigger RTH
5. **Height Limits:** Maintain height within ±2m of recorded path
6. **Obstacle Avoidance:** If DJI obstacle sensors trigger, pause and hover

### 7.4 Emergency Stop

**Physical:** DJI Remote Controller's pause button always works  
**Software:** Client listens for "EMERGENCY_STOP" WebSocket message

---

## 8. Implementation Roadmap

### Phase 1: Server Proof of Concept (Week 1-2)
- [ ] Set up Fast3R model loading and inference
- [ ] Implement SVD Procrustes pose estimation
- [ ] Test with static image pairs
- [ ] Validate coordinate frame conversions with synthetic data
- **Deliverable:** Python script that takes 2 images, outputs relative pose in meters

### Phase 2: Communication Layer (Week 2-3)
- [ ] Implement WebSocket server (Python/asyncio)
- [ ] Implement Android WebSocket client
- [ ] Test binary frame encoding/decoding
- [ ] Measure round-trip latency
- **Deliverable:** Working bidirectional communication with <100ms latency

### Phase 3: Android Video Streaming (Week 3-4)
- [ ] Integrate DJI MSDK video feed
- [ ] Implement JPEG compression pipeline
- [ ] Stream to server, verify image quality
- [ ] Extract and stream telemetry
- **Deliverable:** Android app streaming 10 FPS to server

### Phase 4: VirtualStick Control (Week 4-5)
- [ ] Implement VirtualStick setup on Android
- [ ] Test manual velocity commands from server
- [ ] Implement watchdog and safety stops
- [ ] Tethered hover test
- **Deliverable:** Server can command drone to move in all directions

### Phase 5: Keyframe Recording (Week 5-6)
- [ ] Implement keyframe stack manager
- [ ] IMU-based distance tracking
- [ ] Scale factor computation
- [ ] Test record/playback of keyframe sequences
- **Deliverable:** Record a path, visualize keyframes with distances

### Phase 6: Homing Integration (Week 6-8)
- [ ] Integrate Fast3R inference in control loop
- [ ] Implement PID controllers
- [ ] Test on simple paths (straight line, square)
- [ ] Tune PID parameters
- **Deliverable:** Drone can retrace a recorded path

### Phase 7: Robustness & Testing (Week 8-10)
- [ ] Add all safety features
- [ ] Test in various lighting conditions
- [ ] Test with occlusions/changes in scene
- [ ] Extended flight testing
- **Deliverable:** Robust visual homing system

---

## 9. Appendix

### A. Dependencies

**Server (Python):**
```
torch>=2.0.0
roma>=1.4.0
websockets>=12.0
opencv-python>=4.8.0
numpy>=1.24.0
fast3r  # This repository
```

**Client (Android):**
```gradle
implementation 'com.dji:dji-sdk:4.16.4'
implementation 'org.java-websocket:Java-WebSocket:1.5.4'
```

### B. Key Equations

**SVD Procrustes Alignment:**
Given corresponding points $P_1, P_2 \in \mathbb{R}^{N \times 3}$, find $R, t, s$ minimizing:
$$\sum_{i=1}^{N} w_i \| P_2^{(i)} - (s \cdot R \cdot P_1^{(i)} + t) \|^2$$

Solution via SVD of weighted covariance matrix.

**PID Control:**
$$u(t) = K_p e(t) + K_i \int_0^t e(\tau) d\tau + K_d \frac{de}{dt}$$

Where $e(t)$ is the pose error in each axis.

### C. Testing Checklist

- [ ] Unit test: SVD Procrustes with known transformation
- [ ] Unit test: Coordinate frame conversion
- [ ] Integration test: Fast3R on consecutive video frames
- [ ] System test: Full loop with simulated drone (AirSim)
- [ ] Field test: Tethered indoor flight
- [ ] Field test: Untethered outdoor flight (GPS-denied)
