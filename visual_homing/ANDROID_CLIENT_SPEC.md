# Android WebSocket Client Specification

**Version:** 1.0  
**Last Updated:** January 2026  
**Status:** Ready for Implementation

---

## 1. Overview

This document provides a complete specification for implementing the Android WebSocket client that communicates with the Visual Homing Python server. The Android app acts as a **thin client**, responsible for:

1. Streaming video frames and telemetry to the server
2. Receiving velocity commands from the server
3. Executing commands via DJI VirtualStick API

The server performs all heavy computation (Fast3R inference, pose estimation, PID control).

**Important:** This is a **single-client system**. The server accepts only one drone connection at a time. If a new client connects while one is already connected, the existing connection is terminated.

---

## 2. Communication Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Android App (Client)                          │
│                                                                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │ DJI Camera   │  │ DJI IMU/FC   │  │ VirtualStick │           │
│  │ Video Feed   │  │ Telemetry    │  │ Controller   │           │
│  └──────┬───────┘  └──────┬───────┘  └──────▲───────┘           │
│         │                  │                  │                   │
│         ▼                  ▼                  │                   │
│  ┌────────────────────────────────────────────────────┐          │
│  │            WebSocket Client Manager                  │          │
│  │  - Binary frame encoder                              │          │
│  │  - JSON response parser                              │          │
│  │  - Heartbeat/watchdog                                │          │
│  └────────────────────────────────────────────────────┘          │
│                          │                  ▲                     │
│                          │ Binary (45B+img) │ JSON                │
│                          ▼                  │                     │
└──────────────────────────┼──────────────────┼─────────────────────┘
                           │                  │
                    WebSocket Connection (ws://server:8765)
                           │                  │
                           ▼                  │
┌──────────────────────────┴──────────────────┴─────────────────────┐
│                    Python Server                                   │
│                    (Fast3R, Pose Est, PID)                         │
└───────────────────────────────────────────────────────────────────┘
```

---

## 3. WebSocket Connection

### 3.1 Connection Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Protocol | `ws://` or `wss://` | Use WSS for production |
| Default Port | `8765` | Configurable |
| Ping Interval | 20 seconds | WebSocket keep-alive |
| Ping Timeout | 10 seconds | Disconnect if pong not received |
| Max Message Size | 10 MB | For large JPEG frames |
| Reconnect Delay | 1-5 seconds | Exponential backoff |

### 3.2 Connection Lifecycle

```java
public class WebSocketClientManager {
    private static final String TAG = "WSClient";
    private static final int DEFAULT_PORT = 8765;
    private static final long RECONNECT_DELAY_MS = 2000;
    private static final long MAX_RECONNECT_DELAY_MS = 30000;
    
    private WebSocketClient webSocket;
    private String serverUrl;
    private boolean shouldReconnect = true;
    private long currentReconnectDelay = RECONNECT_DELAY_MS;
    
    public void connect(String host, int port) {
        serverUrl = "ws://" + host + ":" + port;
        attemptConnection();
    }
    
    private void attemptConnection() {
        try {
            URI uri = new URI(serverUrl);
            webSocket = new WebSocketClient(uri) {
                @Override
                public void onOpen(ServerHandshake handshake) {
                    Log.i(TAG, "Connected to server");
                    currentReconnectDelay = RECONNECT_DELAY_MS;
                    onConnectionEstablished();
                }
                
                @Override
                public void onMessage(String message) {
                    handleTextMessage(message);
                }
                
                @Override
                public void onMessage(ByteBuffer bytes) {
                    handleBinaryMessage(bytes);
                }
                
                @Override
                public void onClose(int code, String reason, boolean remote) {
                    Log.w(TAG, "Connection closed: " + reason);
                    scheduleReconnect();
                }
                
                @Override
                public void onError(Exception ex) {
                    Log.e(TAG, "WebSocket error", ex);
                }
            };
            
            webSocket.setConnectionLostTimeout(30);
            webSocket.connect();
            
        } catch (URISyntaxException e) {
            Log.e(TAG, "Invalid server URL", e);
        }
    }
    
    private void scheduleReconnect() {
        if (!shouldReconnect) return;
        
        new Handler(Looper.getMainLooper()).postDelayed(() -> {
            if (shouldReconnect) {
                Log.i(TAG, "Attempting reconnection...");
                attemptConnection();
                currentReconnectDelay = Math.min(
                    currentReconnectDelay * 2, 
                    MAX_RECONNECT_DELAY_MS
                );
            }
        }, currentReconnectDelay);
    }
    
    public void disconnect() {
        shouldReconnect = false;
        if (webSocket != null) {
            webSocket.close();
        }
    }
}
```

---

## 4. Binary Protocol (Client → Server)

### 4.1 Frame Message Format

Every frame message is a binary packet with the following structure:

```
┌────────────────────────────────────────────────────────────────┐
│ Byte Offset │ Size    │ Field            │ Type         │ Notes│
├─────────────┼─────────┼──────────────────┼──────────────┼──────┤
│ 0           │ 1       │ message_type     │ uint8        │ 0x01 │
│ 1           │ 4       │ frame_id         │ uint32 LE    │      │
│ 5           │ 8       │ timestamp_ms     │ uint64 LE    │      │
│ 13          │ 4       │ velocity_x       │ float32 LE   │ m/s  │
│ 17          │ 4       │ velocity_y       │ float32 LE   │ m/s  │
│ 21          │ 4       │ velocity_z       │ float32 LE   │ m/s  │
│ 25          │ 4       │ yaw              │ float32 LE   │ deg  │
│ 29          │ 4       │ pitch            │ float32 LE   │ deg  │
│ 33          │ 4       │ roll             │ float32 LE   │ deg  │
│ 37          │ 4       │ height           │ float32 LE   │ m    │
│ 41          │ 4       │ image_size       │ uint32 LE    │ bytes│
│ 45          │ N       │ image_data       │ bytes        │ JPEG │
└────────────────────────────────────────────────────────────────┘

Total: 45 bytes header + N bytes image data
LE = Little Endian
```

### 4.2 Message Types

```java
public class MessageType {
    public static final byte FRAME_TELEMETRY = 0x01;
    public static final byte CONTROL_COMMAND = 0x02;
    public static final byte STATE_UPDATE = 0x03;
    public static final byte HEARTBEAT = 0x04;
    public static final byte EMERGENCY_STOP = 0x05;
    public static final byte START_RECORDING = 0x10;
    public static final byte STOP_RECORDING = 0x11;
    public static final byte START_HOMING = 0x12;
    public static final byte RESET = 0x13;
}
```

### 4.3 Binary Encoder Implementation

```java
public class BinaryFrameEncoder {
    private static final int HEADER_SIZE = 45;
    
    public byte[] encodeFrame(
            int frameId,
            long timestampMs,
            float velocityX,
            float velocityY,
            float velocityZ,
            float yaw,
            float pitch,
            float roll,
            float height,
            byte[] imageData
    ) {
        ByteBuffer buffer = ByteBuffer.allocate(HEADER_SIZE + imageData.length);
        buffer.order(ByteOrder.LITTLE_ENDIAN);
        
        // Header
        buffer.put(MessageType.FRAME_TELEMETRY);       // offset 0, 1 byte
        buffer.putInt(frameId);                         // offset 1, 4 bytes
        buffer.putLong(timestampMs);                    // offset 5, 8 bytes
        buffer.putFloat(velocityX);                     // offset 13, 4 bytes
        buffer.putFloat(velocityY);                     // offset 17, 4 bytes
        buffer.putFloat(velocityZ);                     // offset 21, 4 bytes
        buffer.putFloat(yaw);                           // offset 25, 4 bytes
        buffer.putFloat(pitch);                         // offset 29, 4 bytes
        buffer.putFloat(roll);                          // offset 33, 4 bytes
        buffer.putFloat(height);                        // offset 37, 4 bytes
        buffer.putInt(imageData.length);                // offset 41, 4 bytes
        
        // Image data
        buffer.put(imageData);                          // offset 45, N bytes
        
        return buffer.array();
    }
}
```

### 4.4 Image Preparation

```java
public class ImageProcessor {
    private static final int TARGET_WIDTH = 512;
    private static final int TARGET_HEIGHT = 384;
    private static final int JPEG_QUALITY = 80;
    
    /**
     * Convert YUV frame to JPEG for transmission.
     * 
     * @param yuvData Raw YUV420 data from DJI camera
     * @param width Original frame width
     * @param height Original frame height
     * @return JPEG compressed bytes
     */
    public byte[] processFrame(byte[] yuvData, int width, int height) {
        // Convert YUV to Bitmap
        YuvImage yuvImage = new YuvImage(
            yuvData, 
            ImageFormat.NV21, 
            width, 
            height, 
            null
        );
        
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        yuvImage.compressToJpeg(
            new Rect(0, 0, width, height), 
            JPEG_QUALITY, 
            out
        );
        
        byte[] jpegBytes = out.toByteArray();
        
        // Decode to bitmap for resizing
        Bitmap bitmap = BitmapFactory.decodeByteArray(
            jpegBytes, 0, jpegBytes.length
        );
        
        // Resize to target dimensions
        Bitmap resized = Bitmap.createScaledBitmap(
            bitmap, 
            TARGET_WIDTH, 
            TARGET_HEIGHT, 
            true
        );
        bitmap.recycle();
        
        // Compress resized bitmap
        ByteArrayOutputStream resizedOut = new ByteArrayOutputStream();
        resized.compress(Bitmap.CompressFormat.JPEG, JPEG_QUALITY, resizedOut);
        resized.recycle();
        
        return resizedOut.toByteArray();
    }
}
```

---

## 5. JSON Protocol (Server → Client)

### 5.1 Control Response Format

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
        "confidence": 0.85,
        "total_keyframes": 10,
        "total_distance_m": 15.0
    },
    "server_latency_ms": 12.5
}
```

### 5.2 Command Semantics

| Field | Unit | Range | Description |
|-------|------|-------|-------------|
| `pitch_velocity` | m/s | [-5, 5] | Forward (+) / Backward (-) velocity |
| `roll_velocity` | m/s | [-5, 5] | Right (+) / Left (-) velocity |
| `vertical_velocity` | m/s | [-3, 3] | Up (+) / Down (-) velocity |
| `yaw_rate` | deg/s | [-100, 100] | Clockwise (+) / CCW (-) rotation |

### 5.3 State Values

| State | Description |
|-------|-------------|
| `IDLE` | System idle, waiting for recording |
| `RECORDING` | Recording keyframes (TEACH phase) |
| `ARMED` | Recording complete, ready for homing |
| `HOMING` | Actively navigating (REPEAT phase) |
| `COMPLETED` | Successfully returned to start |
| `ERROR` | Error state, requires intervention |
| `TIMEOUT` | Server timeout, hover commanded |

### 5.4 Other Server Messages

**State Update:**
```json
{
    "type": "state_update",
    "timestamp_ms": 1704412800000,
    "state": "RECORDING",
    "total_keyframes": 5
}
```

**Heartbeat Acknowledgment:**
```json
{
    "type": "heartbeat_ack",
    "timestamp_ms": 1704412800000,
    "state": "IDLE"
}
```

**Emergency Stop:**
```json
{
    "type": "emergency_stop",
    "timestamp_ms": 1704412800000
}
```

### 5.5 JSON Response Parser

```java
public class ControlResponseParser {
    
    public ControlResponse parse(String json) throws JSONException {
        JSONObject root = new JSONObject(json);
        
        String type = root.getString("type");
        
        if (type.equals("control")) {
            return parseControlResponse(root);
        } else if (type.equals("state_update")) {
            return parseStateUpdate(root);
        } else if (type.equals("emergency_stop")) {
            return parseEmergencyStop(root);
        } else if (type.equals("heartbeat_ack")) {
            return parseHeartbeatAck(root);
        }
        
        throw new JSONException("Unknown message type: " + type);
    }
    
    private ControlResponse parseControlResponse(JSONObject root) 
            throws JSONException {
        ControlResponse response = new ControlResponse();
        
        response.frameId = root.getInt("frame_id");
        response.timestampMs = root.getLong("timestamp_ms");
        response.serverLatencyMs = root.optDouble("server_latency_ms", 0.0);
        
        JSONObject cmd = root.getJSONObject("command");
        response.pitchVelocity = (float) cmd.getDouble("pitch_velocity");
        response.rollVelocity = (float) cmd.getDouble("roll_velocity");
        response.verticalVelocity = (float) cmd.getDouble("vertical_velocity");
        response.yawRate = (float) cmd.getDouble("yaw_rate");
        
        JSONObject status = root.getJSONObject("status");
        response.state = status.getString("state");
        response.keyframesRemaining = status.getInt("keyframes_remaining");
        response.targetDistanceM = (float) status.getDouble("target_distance_m");
        response.confidence = (float) status.getDouble("confidence");
        
        return response;
    }
}

public class ControlResponse {
    public int frameId;
    public long timestampMs;
    public double serverLatencyMs;
    
    public float pitchVelocity;
    public float rollVelocity;
    public float verticalVelocity;
    public float yawRate;
    
    public String state;
    public int keyframesRemaining;
    public float targetDistanceM;
    public float confidence;
}
```

---

## 6. Client Commands (Client → Server)

The client can send JSON text messages to trigger state transitions:

### 6.1 Start Recording
```json
{
    "type": "start_recording",
    "timestamp_ms": 1704412800000
}
```

### 6.2 Stop Recording
```json
{
    "type": "stop_recording",
    "timestamp_ms": 1704412800000
}
```

### 6.3 Start Homing
```json
{
    "type": "start_homing",
    "timestamp_ms": 1704412800000
}
```

### 6.4 Reset
```json
{
    "type": "reset",
    "timestamp_ms": 1704412800000
}
```

### 6.5 Command Sender

```java
public class CommandSender {
    private WebSocketClient webSocket;
    
    public void sendCommand(String type) {
        try {
            JSONObject cmd = new JSONObject();
            cmd.put("type", type);
            cmd.put("timestamp_ms", System.currentTimeMillis());
            webSocket.send(cmd.toString());
        } catch (JSONException e) {
            Log.e(TAG, "Failed to create command", e);
        }
    }
    
    public void startRecording() {
        sendCommand("start_recording");
    }
    
    public void stopRecording() {
        sendCommand("stop_recording");
    }
    
    public void startHoming() {
        sendCommand("start_homing");
    }
    
    public void reset() {
        sendCommand("reset");
    }
}
```

---

## 7. DJI Integration

### 7.1 Video Feed Extraction

```java
public class VideoFeedManager implements VideoFeeder.VideoDataListener {
    private ImageProcessor imageProcessor;
    private FrameSender frameSender;
    private AtomicInteger frameId = new AtomicInteger(0);
    private long lastFrameTime = 0;
    private static final long FRAME_INTERVAL_MS = 100; // 10 Hz
    
    @Override
    public void onReceive(byte[] videoBuffer, int size) {
        // Rate limiting
        long now = System.currentTimeMillis();
        if (now - lastFrameTime < FRAME_INTERVAL_MS) {
            return;
        }
        lastFrameTime = now;
        
        // Process in background thread
        executor.execute(() -> {
            try {
                // Decode H.264 to YUV (requires MediaCodec setup)
                byte[] yuvData = decodeH264ToYuv(videoBuffer, size);
                
                // Process to JPEG
                byte[] jpegData = imageProcessor.processFrame(
                    yuvData, 
                    VIDEO_WIDTH, 
                    VIDEO_HEIGHT
                );
                
                // Get telemetry
                Telemetry telemetry = getTelemetry();
                
                // Send frame
                frameSender.sendFrame(
                    frameId.incrementAndGet(),
                    now,
                    telemetry,
                    jpegData
                );
                
            } catch (Exception e) {
                Log.e(TAG, "Frame processing error", e);
            }
        });
    }
}
```

### 7.2 Telemetry Extraction

```java
public class TelemetryExtractor {
    private FlightController flightController;
    
    public Telemetry getTelemetry() {
        FlightControllerState state = flightController.getState();
        
        Telemetry telem = new Telemetry();
        
        // Velocity in body frame (m/s)
        telem.velocityX = state.getVelocityX();  // Forward
        telem.velocityY = state.getVelocityY();  // Right
        telem.velocityZ = state.getVelocityZ();  // Down
        
        // Attitude (degrees)
        Attitude attitude = state.getAttitude();
        telem.yaw = (float) attitude.yaw;
        telem.pitch = (float) attitude.pitch;
        telem.roll = (float) attitude.roll;
        
        // Height (meters)
        telem.height = state.getUltrasonicHeightInMeters();
        if (telem.height <= 0) {
            // Fallback to barometric altitude
            telem.height = (float) state.getAircraftLocation().getAltitude();
        }
        
        return telem;
    }
}

public class Telemetry {
    public float velocityX;  // Forward (m/s)
    public float velocityY;  // Right (m/s)
    public float velocityZ;  // Down (m/s)
    public float yaw;        // degrees
    public float pitch;      // degrees
    public float roll;       // degrees
    public float height;     // meters
}
```

### 7.3 VirtualStick Control

```java
public class VirtualStickController {
    private FlightController flightController;
    private ScheduledExecutorService commandExecutor;
    private volatile FlightControlData lastCommand;
    private volatile long lastCommandTime = 0;
    private static final long COMMAND_TIMEOUT_MS = 1000;
    
    public void initialize() {
        FlightController fc = flightController;
        
        // Enable Virtual Stick mode
        fc.setVirtualStickModeEnabled(true, error -> {
            if (error != null) {
                Log.e(TAG, "Failed to enable VirtualStick: " + 
                    error.getDescription());
            }
        });
        
        // Configure control modes
        fc.setRollPitchControlMode(RollPitchControlMode.VELOCITY);
        fc.setYawControlMode(YawControlMode.ANGULAR_VELOCITY);
        fc.setVerticalControlMode(VerticalControlMode.VELOCITY);
        fc.setRollPitchCoordinateSystem(FlightCoordinateSystem.BODY);
        
        // Start command sending loop (10 Hz)
        commandExecutor = Executors.newSingleThreadScheduledExecutor();
        commandExecutor.scheduleAtFixedRate(
            this::sendCurrentCommand,
            0,
            100,
            TimeUnit.MILLISECONDS
        );
    }
    
    public void updateCommand(ControlResponse response) {
        // Clamp velocities for safety
        float pitch = clamp(response.pitchVelocity, -2.0f, 2.0f);
        float roll = clamp(response.rollVelocity, -2.0f, 2.0f);
        float vertical = clamp(response.verticalVelocity, -1.0f, 1.0f);
        float yaw = clamp(response.yawRate, -30.0f, 30.0f);
        
        lastCommand = new FlightControlData(pitch, roll, yaw, vertical);
        lastCommandTime = System.currentTimeMillis();
    }
    
    private void sendCurrentCommand() {
        // Watchdog: check for command timeout
        if (System.currentTimeMillis() - lastCommandTime > COMMAND_TIMEOUT_MS) {
            Log.w(TAG, "Command timeout - sending hover");
            lastCommand = new FlightControlData(0, 0, 0, 0);
            
            // Optionally disable VirtualStick for safety
            if (System.currentTimeMillis() - lastCommandTime > 2000) {
                disableVirtualStick();
                return;
            }
        }
        
        if (lastCommand != null) {
            flightController.sendVirtualStickFlightControlData(
                lastCommand,
                error -> {
                    if (error != null) {
                        Log.e(TAG, "Control error: " + error.getDescription());
                    }
                }
            );
        }
    }
    
    private void disableVirtualStick() {
        flightController.setVirtualStickModeEnabled(false, null);
        Log.w(TAG, "VirtualStick disabled due to timeout");
    }
    
    public void emergencyStop() {
        lastCommand = new FlightControlData(0, 0, 0, 0);
        sendCurrentCommand();
        disableVirtualStick();
    }
    
    private float clamp(float value, float min, float max) {
        return Math.max(min, Math.min(max, value));
    }
}
```

---

## 8. Safety Features

### 8.1 Connection Watchdog

```java
public class ConnectionWatchdog {
    private static final long COMMAND_TIMEOUT_MS = 1000;
    private static final long ABORT_TIMEOUT_MS = 2000;
    
    private volatile long lastCommandReceivedTime = 0;
    private VirtualStickController stickController;
    private Handler handler;
    private boolean isRunning = false;
    
    public void start() {
        isRunning = true;
        handler = new Handler(Looper.getMainLooper());
        handler.post(watchdogRunnable);
    }
    
    public void stop() {
        isRunning = false;
        handler.removeCallbacks(watchdogRunnable);
    }
    
    public void onCommandReceived() {
        lastCommandReceivedTime = System.currentTimeMillis();
    }
    
    private Runnable watchdogRunnable = new Runnable() {
        @Override
        public void run() {
            if (!isRunning) return;
            
            long elapsed = System.currentTimeMillis() - lastCommandReceivedTime;
            
            if (elapsed > ABORT_TIMEOUT_MS) {
                Log.e(TAG, "WATCHDOG: Abort timeout - disabling VirtualStick");
                stickController.emergencyStop();
            } else if (elapsed > COMMAND_TIMEOUT_MS) {
                Log.w(TAG, "WATCHDOG: Command timeout - hovering");
            }
            
            handler.postDelayed(this, 100); // Check every 100ms
        }
    };
}
```

### 8.2 Emergency Stop Handler

```java
public class EmergencyStopHandler {
    private VirtualStickController stickController;
    private WebSocketClientManager wsClient;
    
    public void handleEmergencyStop() {
        Log.w(TAG, "EMERGENCY STOP triggered");
        
        // Immediately stop all motion
        stickController.emergencyStop();
        
        // Notify UI
        showEmergencyStopAlert();
    }
    
    public void handleServerEmergencyStop(JSONObject message) {
        Log.w(TAG, "Server triggered emergency stop");
        handleEmergencyStop();
    }
}
```

---

## 9. Complete Integration Example

```java
public class VisualHomingClient implements 
        WebSocketCallback, 
        VideoFeeder.VideoDataListener {
    
    private WebSocketClientManager wsClient;
    private BinaryFrameEncoder encoder;
    private ControlResponseParser parser;
    private VirtualStickController stickController;
    private TelemetryExtractor telemetryExtractor;
    private ImageProcessor imageProcessor;
    private ConnectionWatchdog watchdog;
    
    private AtomicInteger frameId = new AtomicInteger(0);
    private ExecutorService processingExecutor;
    
    public void initialize(String serverHost, int serverPort) {
        encoder = new BinaryFrameEncoder();
        parser = new ControlResponseParser();
        imageProcessor = new ImageProcessor();
        stickController = new VirtualStickController();
        telemetryExtractor = new TelemetryExtractor();
        watchdog = new ConnectionWatchdog();
        
        processingExecutor = Executors.newFixedThreadPool(2);
        
        // Initialize DJI components
        stickController.initialize();
        
        // Connect to server
        wsClient = new WebSocketClientManager();
        wsClient.setCallback(this);
        wsClient.connect(serverHost, serverPort);
        
        // Start watchdog
        watchdog.start();
    }
    
    @Override
    public void onVideoDataReceived(byte[] data, int size) {
        processingExecutor.execute(() -> {
            try {
                // Process video frame
                byte[] jpegData = imageProcessor.processFrame(data, 1920, 1080);
                
                // Get telemetry
                Telemetry telem = telemetryExtractor.getTelemetry();
                
                // Encode binary message
                byte[] packet = encoder.encodeFrame(
                    frameId.incrementAndGet(),
                    System.currentTimeMillis(),
                    telem.velocityX,
                    telem.velocityY,
                    telem.velocityZ,
                    telem.yaw,
                    telem.pitch,
                    telem.roll,
                    telem.height,
                    jpegData
                );
                
                // Send to server
                wsClient.sendBinary(packet);
                
            } catch (Exception e) {
                Log.e(TAG, "Frame processing error", e);
            }
        });
    }
    
    @Override
    public void onMessageReceived(String message) {
        try {
            ControlResponse response = parser.parse(message);
            
            // Update watchdog
            watchdog.onCommandReceived();
            
            // Execute command
            if (response.state.equals("HOMING") || 
                response.state.equals("RECORDING")) {
                stickController.updateCommand(response);
            }
            
            // Update UI
            updateStatusUI(response);
            
        } catch (JSONException e) {
            Log.e(TAG, "Failed to parse response", e);
        }
    }
    
    @Override
    public void onDisconnected() {
        Log.w(TAG, "Disconnected from server");
        stickController.emergencyStop();
    }
    
    public void startRecording() {
        wsClient.sendCommand("start_recording");
    }
    
    public void stopRecording() {
        wsClient.sendCommand("stop_recording");
    }
    
    public void startHoming() {
        wsClient.sendCommand("start_homing");
    }
    
    public void shutdown() {
        watchdog.stop();
        stickController.emergencyStop();
        wsClient.disconnect();
        processingExecutor.shutdown();
    }
}
```

---

## 10. Testing

### 10.1 Protocol Testing

Test the binary protocol using the provided Python test server:

```bash
# Start the test server
cd visual_homing
python scripts/demo_websocket.py --mode server

# The server will echo responses for any frames received
```

### 10.2 Latency Testing

The Android app should measure and log:
- Frame encoding time
- Network round-trip time
- Command execution latency

```java
public class LatencyTracker {
    private Map<Integer, Long> sendTimes = new ConcurrentHashMap<>();
    private List<Long> rttSamples = new ArrayList<>();
    
    public void onFrameSent(int frameId) {
        sendTimes.put(frameId, System.currentTimeMillis());
    }
    
    public void onResponseReceived(int frameId) {
        Long sendTime = sendTimes.remove(frameId);
        if (sendTime != null) {
            long rtt = System.currentTimeMillis() - sendTime;
            rttSamples.add(rtt);
            
            if (rttSamples.size() % 100 == 0) {
                logLatencyStats();
            }
        }
    }
    
    private void logLatencyStats() {
        if (rttSamples.isEmpty()) return;
        
        Collections.sort(rttSamples);
        int n = rttSamples.size();
        
        long min = rttSamples.get(0);
        long max = rttSamples.get(n - 1);
        long avg = rttSamples.stream().mapToLong(l -> l).sum() / n;
        long p50 = rttSamples.get(n / 2);
        long p95 = rttSamples.get((int)(n * 0.95));
        
        Log.i(TAG, String.format(
            "Latency: min=%dms, avg=%dms, max=%dms, p50=%dms, p95=%dms",
            min, avg, max, p50, p95
        ));
    }
}
```

---

## 11. Gradle Dependencies

```gradle
dependencies {
    // DJI SDK
    implementation 'com.dji:dji-sdk:4.16.4'
    compileOnly 'com.dji:dji-sdk-provided:4.16.4'
    
    // WebSocket client
    implementation 'org.java-websocket:Java-WebSocket:1.5.4'
    
    // JSON parsing
    implementation 'org.json:json:20231013'
}
```

---

## 12. Checklist

- [ ] Implement `BinaryFrameEncoder` class
- [ ] Implement `ControlResponseParser` class
- [ ] Implement `WebSocketClientManager` with reconnection
- [ ] Implement `ImageProcessor` for JPEG compression
- [ ] Implement `TelemetryExtractor` for DJI data
- [ ] Implement `VirtualStickController` with timeout
- [ ] Implement `ConnectionWatchdog` for safety
- [ ] Test binary protocol with Python server
- [ ] Measure and verify <100ms latency
- [ ] Test with DJI simulator
- [ ] Field test with real drone

---

## 13. References

- [DJI Mobile SDK Documentation](https://developer.dji.com/mobile-sdk/documentation/)
- [Java-WebSocket Library](https://github.com/TooTallNate/Java-WebSocket)
- [Visual Homing Design Spec](./design_spec.md)

