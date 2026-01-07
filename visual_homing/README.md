# Visual Homing System

A server-centric "Teach and Repeat" architecture for enabling DJI UAVs to autonomously return to their launch point in GPS-denied environments using Fast3R.

## Overview

This system enables a UAV to:
1. **TEACH Phase**: Record a path by flying manually, capturing keyframes with IMU-based distance tracking
2. **REPEAT Phase**: Autonomously retrace the path using visual matching with Fast3R

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Server (Ground Station)                 │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │ Fast3R       │  │ Pose         │  │ Keyframe         │   │
│  │ Engine       │  │ Estimator    │  │ Manager          │   │
│  └──────────────┘  └──────────────┘  └──────────────────┘   │
│                           │                                  │
│                           ▼                                  │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              Homing Controller                        │   │
│  │  - State Machine (IDLE→RECORDING→ARMED→HOMING)       │   │
│  │  - PID Control                                        │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                              ▲
                              │ WebSocket
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   Client (Android + DJI MSDK)                │
│  - Video streaming                                           │
│  - Telemetry extraction                                      │
│  - VirtualStick control                                      │
└─────────────────────────────────────────────────────────────┘
```

## Project Structure

```
visual_homing/
├── server/                     # Server-side components
│   ├── config.py              # Configuration constants
│   ├── fast3r_engine.py       # Fast3R model wrapper
│   ├── pose_estimator.py      # SVD Procrustes pose estimation
│   ├── coordinate_utils.py    # Coordinate frame conversions
│   ├── keyframe_manager.py    # Keyframe stack management
│   ├── state_machine.py       # System state management
│   ├── pid_controller.py      # PID control
│   ├── homing_controller.py   # Main controller
│   └── websocket_server.py    # Async WebSocket server
├── protocol/                   # Communication protocol
│   ├── messages.py            # Message definitions
│   └── binary_codec.py        # Binary encoding/decoding
├── tests/                      # Unit tests
│   ├── test_pose_estimator.py
│   ├── test_coordinate_utils.py
│   ├── test_websocket_integration.py  # WebSocket integration tests
│   └── test_websocket_client.py       # Test client (simulates Android)
├── scripts/                    # Demo and utility scripts
│   ├── demo_pose_estimation.py
│   ├── demo_websocket.py      # WebSocket server demo
│   └── test_coordinate_frames.py
├── ANDROID_CLIENT_SPEC.md     # Detailed Android client specification
├── requirements.txt
└── README.md
```

## Installation

```bash
# From the fast3r-drone-nav root directory
cd visual_homing
pip install -r requirements.txt
```

## Quick Start

### Phase 1: Pose Estimation Demo

Test the core pose estimation functionality:

```bash
# Using demo images from the repository
python scripts/demo_pose_estimation.py --demo

# Using custom images
python scripts/demo_pose_estimation.py --image1 path/to/img1.jpg --image2 path/to/img2.jpg

# With visualization
python scripts/demo_pose_estimation.py --demo --visualize
```

### Validate Coordinate Frames

```bash
python scripts/test_coordinate_frames.py
```

### Run Unit Tests

```bash
pytest tests/ -v
```

### Run Server for Android Development

Start the server in **mock mode** (no GPU required) while developing the Android app:

```bash
# Mock mode - responds to frames without Fast3R model
python run_server.py --mock

# Production mode - loads Fast3R model (requires GPU)
python run_server.py

# With verbose logging
python run_server.py --mock -v
```

The server will display connection status and frame counts as the Android app connects.

### Phase 2: WebSocket Communication Demo

Test the WebSocket server with simulated frames:

```bash
# Run server and client together (recommended for testing)
python scripts/demo_websocket.py --mode both --duration 10 --rate 10

# Run server only (for connecting real Android app)
python scripts/demo_websocket.py --mode server --port 8765

# Run client only (connect to existing server)
python scripts/demo_websocket.py --mode client --url ws://localhost:8765
```

Expected output shows latency well under 100ms:
```
Round-trip Latency (Client Measured):
  Min:  1.0 ms
  Avg:  1.1 ms
  Max:  1.2 ms
  P50:  1.1 ms
  P95:  1.2 ms

✓ PASS: Average round-trip latency < 100ms
```

## Key Components

### Fast3R Engine
Wraps the Fast3R model for efficient inference on image pairs:
```python
from visual_homing.server import Fast3REngine

engine = Fast3REngine()
engine.load_model()
result = engine.infer_pair(image1, image2)
# result contains 3D pointmaps and confidence maps
```

### Pose Estimator
Uses SVD Procrustes for 3D-3D point alignment:
```python
from visual_homing.server import PoseEstimator

estimator = PoseEstimator(metric_scale=1.0)
pose = estimator.estimate_pose(pts3d_source, pts3d_target, confidence)
# pose contains rotation, translation, scale, confidence
```

### Coordinate Conversion
Convert Fast3R camera frame to DJI VirtualStick commands:
```python
from visual_homing.server import fast3r_to_dji_command

command = fast3r_to_dji_command(
    t_cam=[0, 0, 2],  # Target 2m ahead
    yaw_error_deg=0,
    pid_gains={"kp_forward": 0.5, ...}
)
# command = {"pitch_velocity": 1.0, "roll_velocity": 0, ...}
```

### WebSocket Server
Async WebSocket server for real-time communication with Android client:
```python
from visual_homing.server import WebSocketServer
from visual_homing.protocol.messages import FrameMessage

def frame_callback(frame: FrameMessage) -> dict:
    # Process frame and return control command
    return {
        "command": {"pitch_velocity": 0.5, "roll_velocity": 0, ...},
        "status": {"state": "HOMING", "keyframes_remaining": 5, ...}
    }

server = WebSocketServer(host="0.0.0.0", port=8765)
server.set_frame_callback(frame_callback)
await server.start()
```

## Coordinate Frames

```
CAMERA FRAME (Fast3R/OpenCV):     DJI BODY FRAME:
     Y (down)                          Z (down)
     │                                 │
     │                                 │
     └───── X (right)                  └───── Y (right)
    ╱                                 ╱
   ╱                                 ╱
  Z (forward)                       X (forward)
```

## Development Roadmap

- [x] Phase 1: Server Proof of Concept
  - [x] Fast3R model loading and inference
  - [x] SVD Procrustes pose estimation
  - [x] Coordinate frame validation
  
- [x] Phase 2: Communication Layer ✓
  - [x] WebSocket server (Python/asyncio)
  - [x] Binary frame encoding/decoding
  - [x] Test client simulating Android app
  - [x] Round-trip latency < 100ms verified (avg: ~1ms)
  - [x] Android client specification document
  
- [ ] Phase 3: Android Video Streaming ← **YOU ARE HERE**
- [ ] Phase 4: VirtualStick Control
- [ ] Phase 5: Keyframe Recording
- [ ] Phase 6: Homing Integration
- [ ] Phase 7: Robustness & Testing

## Next Steps: Building the Android App

See [`ANDROID_CLIENT_SPEC.md`](./ANDROID_CLIENT_SPEC.md) for the complete implementation guide.

### Quick Start for Android Development

1. **Start the mock server** (no GPU needed):
   ```bash
   python run_server.py --mock -v
   ```

2. **Build the Android app** following `ANDROID_CLIENT_SPEC.md`:
   - Implement `BinaryFrameEncoder` (Section 4.3)
   - Implement `ControlResponseParser` (Section 5.5)
   - Implement `WebSocketClientManager` (Section 3.2)
   
3. **Test on same WiFi network**:
   - Find your computer's IP: `hostname -I`
   - Connect Android app to `ws://<your-ip>:8765`

4. **Test commands** - The mock server responds to:
   - `start_recording` → State: IDLE → RECORDING
   - `stop_recording` → State: RECORDING → ARMED  
   - `start_homing` → State: ARMED → HOMING
   - `reset` → State: any → IDLE

## License

See the main repository LICENSE file.


