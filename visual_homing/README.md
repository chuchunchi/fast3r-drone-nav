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
│   └── homing_controller.py   # Main controller
├── protocol/                   # Communication protocol
│   ├── messages.py            # Message definitions
│   └── binary_codec.py        # Binary encoding/decoding
├── tests/                      # Unit tests
│   ├── test_pose_estimator.py
│   └── test_coordinate_utils.py
├── scripts/                    # Demo and utility scripts
│   ├── demo_pose_estimation.py
│   └── test_coordinate_frames.py
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
  
- [ ] Phase 2: Communication Layer
- [ ] Phase 3: Android Video Streaming
- [ ] Phase 4: VirtualStick Control
- [ ] Phase 5: Keyframe Recording
- [ ] Phase 6: Homing Integration
- [ ] Phase 7: Robustness & Testing

## License

See the main repository LICENSE file.


