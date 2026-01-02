# AETHER Core API Reference
## Source of Truth - Last Updated: 2026-01-02

**Base URL:** `http://localhost:8891` (or `http://192.168.50.1:8891`)
**Port:** 8891 (Python Flask backend - SSOT)
**Frontend:** Port 3000 (Express proxy + React app)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Raspberry Pi (192.168.50.1)                           │
│                         Access Point Mode                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────────┐   │
│  │   Frontend   │───▶│Express Proxy │───▶│      Flask API           │   │
│  │  (port 3000) │    │  (port 3000) │    │     (port 8891)          │   │
│  │    React     │    │    Node.js   │    │   SSOT Controller        │   │
│  └──────────────┘    └──────────────┘    └────────────┬─────────────┘   │
│                                                        │                 │
│                              ┌─────────────────────────┼─────────────────┤
│                              │                         │                 │
│                              ▼                         ▼                 │
│                    ┌──────────────┐          ┌──────────────┐           │
│                    │   DMXState   │          │  NodeManager │           │
│                    │  (in-memory) │          │  (UDPJSON)   │           │
│                    └──────────────┘          └──────┬───────┘           │
│                                                      │                   │
│                                    UDP Port 6455     │                   │
│                                    (UDPJSON v2)      ▼                   │
└──────────────────────────────────────────────────────────────────────────┘
                                         │
                                         │ WiFi (SSID: AetherDMX)
                                         │
            ┌────────────────────────────┼────────────────────────────┐
            │                            │                            │
     ┌──────▼──────┐              ┌──────▼──────┐              ┌──────▼──────┐
     │  pulse-422C │              │  pulse-76E4 │              │  pulse-4690 │ ...
     │  U:2  .16   │              │  U:3  .28   │              │  U:4  .68   │
     └──────┬──────┘              └──────┬──────┘              └──────┬──────┘
            │                            │                            │
           DMX                          DMX                          DMX
         Fixtures                     Fixtures                     Fixtures
```

---

## Data Flow: DMX Output

1. **API Request** → `/api/dmx/set`, `/api/scenes/:id/apply`, `/api/chases/:id/play`
2. **ContentManager.set_channels()** → Single Source of Truth (SSOT)
3. **DMXState.set_channel()** → Updates in-memory DMX buffer
4. **NodeManager.send_udpjson()** → Sends UDPJSON v2 to ESP32 nodes (port 6455)
5. **ESP32 Node** → Receives UDP, updates local buffer, outputs DMX at 40Hz

**No OLA, no sACN** - Direct UDPJSON transport only.

---

## Core Endpoints

### Health & System

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | System health check |
| `/api/version` | GET | Firmware/software versions |
| `/api/system/stats` | GET | System statistics (CPU, memory, uptime) |
| `/api/system/update` | POST | Trigger system update |
| `/api/system/update/check` | GET | Check for available updates |
| `/api/system/autosync` | GET/POST | Auto-sync settings |

---

## DMX Control (SSOT)

All DMX operations route through the **ContentManager** (Single Source of Truth).

### Set Channels

```bash
POST /api/dmx/set
Content-Type: application/json

{
  "universe": 2,
  "channels": {
    "1": 255,
    "2": 128,
    "3": 64
  },
  "fade_ms": 500  # optional
}
```

**Response:**
```json
{
  "success": true,
  "results": [
    {"node": "Universe 2 Node", "success": true}
  ]
}
```

### Blackout

```bash
POST /api/dmx/blackout
Content-Type: application/json

{
  "universe": 2,    # optional - omit for all universes
  "fade_ms": 1000   # optional fade time
}
```

### Master Dimmer

```bash
POST /api/dmx/master
Content-Type: application/json

{
  "level": 50,      # 0-100 percent
  "capture": true   # capture current state as base
}
```

### Reset Master

```bash
POST /api/dmx/master/reset
Content-Type: application/json
{}
```

### Get Universe State

```bash
GET /api/dmx/universe/2
```

**Response:**
```json
{
  "channels": [255, 128, 64, 0, 0, ...]  // 512 values
}
```

### DMX Diagnostics

```bash
GET /api/dmx/diagnostics
```

---

## Scenes

### List Scenes

```bash
GET /api/scenes
```

**Response:**
```json
[
  {
    "scene_id": "scene_1766603545",
    "name": "All Blue",
    "universe": 1,
    "channels": {"1": 0, "2": 0, "3": 255, ...},
    "fade_ms": 500,
    "color": "#3b82f6",
    "icon": "lightbulb"
  }
]
```

### Create Scene

```bash
POST /api/scenes
Content-Type: application/json

{
  "name": "My Scene",
  "channels": {"1": 255, "2": 128, "3": 0, "4": 0},
  "fade_ms": 500,
  "color": "#ff0000"
}
```

**Response:**
```json
{
  "success": true,
  "scene_id": "scene_1767384997"
}
```

### Update Scene

```bash
PUT /api/scenes/{scene_id}
Content-Type: application/json

{
  "name": "Updated Scene Name",
  "channels": {"1": 255, "2": 128, "3": 64, "4": 0},
  "fade_ms": 1000,
  "color": "#00ff00"
}
```

**Response:**
```json
{
  "success": true,
  "scene_id": "scene_1767384997"
}
```

### Play Scene

```bash
POST /api/scenes/{scene_id}/play
Content-Type: application/json
{}
```

**Response:**
```json
{
  "success": true,
  "results": [
    {"node": "Universe 1 (Built-in)", "success": true},
    {"node": "Universe 2 Node", "success": true}
  ],
  "universes": [0, 1, 2, 4]
}
```

**Note:** Scenes are broadcast to ALL universes when played.

### Delete Scene

```bash
DELETE /api/scenes/{scene_id}
```

---

## Chases

### List Chases

```bash
GET /api/chases
```

**Response:**
```json
[
  {
    "chase_id": "chase_1766825208",
    "name": "Dallas Stars Show",
    "universe": 1,
    "bpm": 120,
    "loop": 1,
    "steps": [
      {
        "channels": {"2:1": 0, "2:2": 30, ...},  // universe:channel format
        "fade_ms": 750,
        "hold_ms": 50,
        "duration_ms": 800
      }
    ]
  }
]
```

### Create Chase

```bash
POST /api/chases
Content-Type: application/json

{
  "name": "My Chase",
  "bpm": 120,
  "loop": true,
  "color": "#ff0000",
  "steps": [
    {"channels": {"1": 255, "2": 0, "3": 0}, "fade_ms": 500, "hold_ms": 500},
    {"channels": {"1": 0, "2": 255, "3": 0}, "fade_ms": 500, "hold_ms": 500}
  ]
}
```

**Note:** Chases are universe-agnostic. Universe is selected at playback time via the modal.

**Response:**
```json
{
  "success": true,
  "chase_id": "chase_1767384997"
}
```

### Update Chase

```bash
PUT /api/chases/{chase_id}
Content-Type: application/json

{
  "name": "Updated Chase Name",
  "bpm": 140,
  "loop": true,
  "steps": [
    {"channels": {"1": 255, "2": 128, "3": 0}, "fade_ms": 300, "hold_ms": 200}
  ]
}
```

**Response:**
```json
{
  "success": true,
  "chase_id": "chase_1767384997"
}
```

### Play Chase

```bash
POST /api/chases/{chase_id}/play
Content-Type: application/json
{}
```

**Response:**
```json
{
  "success": true,
  "fade_ms": 0,
  "universes": [1, 2]
}
```

### Stop Chase

```bash
POST /api/chases/{chase_id}/stop
Content-Type: application/json
{}
```

### Delete Chase

```bash
DELETE /api/chases/{chase_id}
```

---

## Playback Control

### Get Status

```bash
GET /api/playback/status
```

**Response:**
```json
{
  "0": {"id": "scene_123", "type": "scene", "started": "2025-12-31T15:51:53"},
  "1": {"id": "chase_456", "type": "chase", "started": "2025-12-31T15:52:44"},
  "2": {"id": "chase_456", "type": "chase", "started": "2025-12-31T15:52:44"}
}
```

### Stop All Playback

```bash
POST /api/playback/stop
Content-Type: application/json
{}
```

---

## Node Management

### List Nodes

```bash
GET /api/nodes
```

**Response:**
```json
[
  {
    "node_id": "pulse-422C",
    "name": "Living Room",
    "ip_address": "192.168.50.123",
    "mac_address": "6C:F3:...",
    "universe": 2,
    "is_paired": true,
    "is_online": true,
    "firmware_version": "2.3.0"
  }
]
```

### Pair Node

```bash
POST /api/nodes/{node_id}/pair
Content-Type: application/json

{
  "universe": 2,
  "name": "Kitchen Lights"
}
```

### Configure Node

```bash
POST /api/nodes/{node_id}/configure
Content-Type: application/json

{
  "universe": 2,
  "slice_start": 1,
  "slice_end": 128,
  "name": "Kitchen"
}
```

### Unpair Node

```bash
POST /api/nodes/{node_id}/unpair
Content-Type: application/json
{}
```

### Sync to Nodes

```bash
POST /api/nodes/sync
Content-Type: application/json
{}
```

---

## Schedules

### List Schedules

```bash
GET /api/schedules
```

### Create Schedule

```bash
POST /api/schedules
Content-Type: application/json

{
  "name": "Morning Lights",
  "action_type": "scene",         # scene, chase, or blackout
  "action_id": "scene_123",       # required for scene/chase
  "trigger_type": "time",         # time, sunset, sunrise
  "trigger_time": "07:00",        # HH:MM format
  "enabled": true
}
```

### Update Schedule

```bash
PUT /api/schedules/{schedule_id}
Content-Type: application/json
{...same fields as create...}
```

### Delete Schedule

```bash
DELETE /api/schedules/{schedule_id}
```

### Trigger Manually

```bash
POST /api/schedules/{schedule_id}/trigger
Content-Type: application/json
{}
```

---

## Effects (Dynamic)

### Start Christmas Effect

```bash
POST /api/effects/christmas
Content-Type: application/json

{
  "universe": 2,
  "speed": 1.0
}
```

### Other Effects

- `POST /api/effects/twinkle`
- `POST /api/effects/smooth`
- `POST /api/effects/wave`

### Stop Effects

```bash
POST /api/effects/stop
Content-Type: application/json
{}
```

---

## Shows (Sequences)

### List Shows

```bash
GET /api/shows
```

### Play Show

```bash
POST /api/shows/{show_id}/play
Content-Type: application/json
{}
```

### Control

- `POST /api/shows/stop`
- `POST /api/shows/pause`
- `POST /api/shows/resume`
- `POST /api/shows/tempo` - Adjust playback tempo

---

## Fixtures

### List Fixtures

```bash
GET /api/fixtures
GET /api/fixtures/universe/2  # Filter by universe
```

### Create Fixture

```bash
POST /api/fixtures
Content-Type: application/json

{
  "name": "RGBW Par",
  "universe": 2,
  "start_channel": 1,
  "channel_count": 4,
  "type": "rgbw"
}
```

---

## Groups

### List Groups

```bash
GET /api/groups
```

### Create Group

```bash
POST /api/groups
Content-Type: application/json

{
  "name": "Living Room",
  "fixture_ids": ["fixture_1", "fixture_2"]
}
```

---

## Settings

### Get All Settings

```bash
GET /api/settings/all
```

### Get Category

```bash
GET /api/settings/{category}
```

### Update Category

```bash
POST /api/settings/{category}
Content-Type: application/json

{
  "setting_key": "value"
}
```

---

## Session Management

### Check for Resume

```bash
GET /api/session/resume
```

### Resume Session

```bash
POST /api/session/resume
Content-Type: application/json
{}
```

### Dismiss Resume

```bash
POST /api/session/dismiss
Content-Type: application/json
{}
```

---

## Multi-Universe Channel Format

For chases that span multiple universes, use the `universe:channel` format:

```json
{
  "channels": {
    "1:1": 255,    // Universe 1, Channel 1
    "1:2": 128,    // Universe 1, Channel 2
    "2:1": 200,    // Universe 2, Channel 1
    "4:5": 100     // Universe 4, Channel 5
  }
}
```

Simple channel numbers (e.g., `"1": 255`) apply to the scene/chase's default universe.

---

## Network Configuration

| Service | Port | Protocol | Description |
|---------|------|----------|-------------|
| Flask API | 8891 | HTTP | Main backend API (SSOT) |
| React Frontend | 3000 | HTTP | Web dashboard + Express proxy |
| **UDPJSON DMX** | **6455** | **UDP** | **DMX transport to ESP32 nodes** |
| Node Config | 8888 | UDP | Node configuration commands |
| Node Discovery | 9999 | UDP | Node registration/heartbeat |
| SSH | 22 | TCP | Remote access |
| mDNS | 5353 | UDP | .local domain discovery |

**Network Topology:**
```
┌─────────────────────────────────────────────────────────────────┐
│                      Raspberry Pi (AP Mode)                      │
│                         192.168.50.1                             │
│                                                                  │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐          │
│  │ aether-core │    │   Portal    │    │   System    │          │
│  │   :8891     │    │   :3000     │    │   :22       │          │
│  │   (API)     │    │   (Web)     │    │   (SSH)     │          │
│  └──────┬──────┘    └─────────────┘    └─────────────┘          │
│         │                                                        │
│         │ UDP :6455 (DMX commands)                               │
│         │ UDP :8888 (Config commands)                            │
│         ▼                                                        │
└─────────────────────────────────────────────────────────────────┘
          │ WiFi (AetherDMX network)
          │
    ┌─────┴─────┐
    │           │
┌───▼───┐   ┌───▼───┐
│ Node1 │   │ Node2 │     ... more nodes
│ .16   │   │ .28   │
│ U:2   │   │ U:3   │
│ 1-128 │   │ 1-128 │     (slice config)
└───────┘   └───────┘
```

---

## UDPJSON Protocol v2 (Port 6455)

The UDPJSON v2 protocol is used for all DMX communication between aether-core and ESP32 nodes.

### Message Types

| Type | Direction | Description |
|------|-----------|-------------|
| `set` | Pi → Node | Set specific channels |
| `fill` | Pi → Node | Fill channel ranges |
| `frame` | Pi → Node | Full 512-byte frame (base64) |
| `blackout` | Pi → Node | All channels to 0 |
| `panic` | Pi → Node | Immediate blackout (no fade) |
| `ping` | Pi → Node | Health check |
| `pong` | Node → Pi | Health response with telemetry |

### Set Channels (v2 compact format)

```json
{
  "v": 2,
  "type": "set",
  "u": 2,
  "seq": 12345,
  "ch": [[1, 255], [2, 128], [3, 64]],
  "fade": 500
}
```

### Fill Ranges

```json
{
  "v": 2,
  "type": "fill",
  "u": 2,
  "seq": 12346,
  "ranges": [[1, 128, 0], [129, 256, 255]],
  "fade": 1000
}
```

### Ping/Pong

```json
// Request
{"v": 2, "type": "ping", "seq": 1}

// Response
{
  "v": 2,
  "type": "pong",
  "seq": 1,
  "id": "pulse-422C",
  "u": 2,
  "ip": "192.168.50.16",
  "rssi": -45,
  "uptime": 3600,
  "heap": 180000,
  "rx": 1234,
  "rx_bad": 0,
  "dmx_fps": 40,
  "stale": false,
  "slice": [1, 128],
  "caps": ["dmx", "fade", "split", "frame", "rdm_stub", "ota"]
}
```

### Protocol Rules

1. **Sequence Numbers**: Used for duplicate detection. Packets with seq ≤ last_seq are dropped (within 256 window).
2. **Universe Filtering**: Nodes only process packets matching their configured universe (or `u: 0` for broadcast).
3. **MTU Safety**: Payloads must be < 1200 bytes. Use chunking for large channel sets.
4. **No Blackout on Silence**: Nodes hold last values indefinitely; `stale` flag indicates no recent packets.
5. **Fade Engine**: Per-channel non-blocking fades at 50Hz resolution.

---

## ESP32 Pulse Node Commands (UDP port 8888)

### Configure Node

```json
{
  "cmd": "configure",
  "universe": 2,
  "slice_start": 1,
  "slice_end": 128,
  "name": "Kitchen"
}
```

### Set Offline Mode

```json
{
  "cmd": "set_offline_mode",
  "mode": "loop"  // none, loop, chase, hold
}
```

### Store Chase for Offline Playback

```json
{
  "cmd": "store_chase",
  "loop_count": 0,
  "steps": [
    {"channels": [255, 0, 0, ...], "fade_ms": 500, "hold_ms": 1000},
    {"channels": [0, 255, 0, ...], "fade_ms": 500, "hold_ms": 1000}
  ]
}
```

### Get Offline Status

```json
{"cmd": "offline_status"}
```

### Clear Config

```json
{"cmd": "clear_config"}
```

---

## Error Responses

All endpoints return consistent error format:

```json
{
  "success": false,
  "error": "Error message here"
}
```

HTTP Status Codes:
- `200` - Success
- `400` - Bad Request (invalid JSON, missing fields)
- `404` - Not Found (scene/chase/node doesn't exist)
- `415` - Unsupported Media Type (missing Content-Type header)
- `500` - Internal Server Error

---

## Quick Reference

### Turn on lights:
```bash
curl -X POST http://localhost:8891/api/dmx/set \
  -H 'Content-Type: application/json' \
  -d '{"universe":2,"channels":{"1":255,"2":255,"3":255}}'
```

### Play a scene:
```bash
curl -X POST http://localhost:8891/api/scenes/scene_123/play \
  -H 'Content-Type: application/json' -d '{}'
```

### Blackout:
```bash
curl -X POST http://localhost:8891/api/dmx/blackout \
  -H 'Content-Type: application/json' -d '{"fade_ms":1000}'
```

### Check what's playing:
```bash
curl http://localhost:8891/api/playback/status
```
