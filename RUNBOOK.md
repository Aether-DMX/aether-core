# AETHER DMX RUNBOOK v6.0
## Source of Truth - Last Updated: 2026-01-02

---

## System Overview

AETHER is a WiFi-based DMX lighting control system with:
- **Raspberry Pi** as central controller (Access Point mode)
- **ESP32 Pulse Nodes** for wireless DMX output
- **React Frontend** for touchscreen control
- **Flask Backend** as single source of truth (SSOT)
- **UDPJSON v2 Protocol** for DMX transport

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Raspberry Pi (192.168.50.1)                   │
│                         Access Point Mode                        │
│                                                                  │
│  ┌─────────────────┐    ┌─────────────────┐                     │
│  │   aether-core   │    │  portal-backend │                     │
│  │   Flask :8891   │◄───│  Express :3000  │◄── Browser/Touch    │
│  │     (SSOT)      │    │    (Proxy)      │                     │
│  └────────┬────────┘    └─────────────────┘                     │
│           │                                                      │
│           │ UDP :6455 (UDPJSON v2)                              │
│           │ UDP :8888 (Config)                                   │
│           │ UDP :9999 (Discovery)                                │
│           ▼                                                      │
└─────────────────────────────────────────────────────────────────┘
            │
            │ WiFi (SSID: AetherDMX)
            │
    ┌───────┴───────┐───────────────┐───────────────┐
    │               │               │               │
┌───▼───┐       ┌───▼───┐       ┌───▼───┐       ┌───▼───┐
│ Node1 │       │ Node2 │       │ Node3 │       │ Node4 │
│ .16   │       │ .28   │       │ .68   │       │ .33   │
│ U:2   │       │ U:3   │       │ U:4   │       │ U:5   │
└───┬───┘       └───┬───┘       └───┬───┘       └───┬───┘
    │               │               │               │
   DMX             DMX             DMX             DMX
 Fixtures        Fixtures        Fixtures        Fixtures
```

---

## Network Configuration

| Service | Port | Protocol | Description |
|---------|------|----------|-------------|
| Flask API | 8891 | HTTP | Backend SSOT - all state lives here |
| Express Proxy | 3000 | HTTP | Frontend + API proxy to Flask |
| UDPJSON DMX | 6455 | UDP | DMX commands to ESP32 nodes |
| Node Config | 8888 | UDP | Node configuration commands |
| Node Discovery | 9999 | UDP | Heartbeat/registration |
| SSH | 22 | TCP | Remote access |

---

## Node Inventory

| Hostname | Universe | IP | MAC | Status |
|----------|----------|-----|-----|--------|
| pulse-422C | 2 | 192.168.50.16 | 4C:EB:D6:65:42:2C | Online |
| pulse-76E4 | 3 | 192.168.50.28 | 00:70:07:E6:76:E4 | Online |
| pulse-4690 | 4 | 192.168.50.68 | E0:8C:FE:5C:46:90 | Online |
| pulse-7394 | 5 | 192.168.50.33 | D4:E9:F4:E2:73:94 | Online |

**Note:** Universe 1 uses wired UART on the Pi itself.

---

## Quick Start Commands

### Check System Health
```bash
# From any machine on the network
curl http://192.168.50.1:8891/api/health

# Via SSH to Pi
ssh ramzt@aether-portal.local
curl http://localhost:8891/api/health
```

### List All Nodes
```bash
curl http://192.168.50.1:8891/api/nodes
```

### Set Channels
```bash
# Universe 2, channels 1-3 to RGB white
curl -X POST http://192.168.50.1:8891/api/dmx/set \
  -H 'Content-Type: application/json' \
  -d '{"universe":2,"channels":{"1":255,"2":255,"3":255}}'
```

### Fade Channels
```bash
# Fade universe 3 to red over 2 seconds
curl -X POST http://192.168.50.1:8891/api/dmx/fade \
  -H 'Content-Type: application/json' \
  -d '{"universe":3,"channels":{"1":255,"2":0,"3":0},"fade_ms":2000}'
```

### Blackout All
```bash
curl -X POST http://192.168.50.1:8891/api/dmx/blackout \
  -H 'Content-Type: application/json' \
  -d '{"fade_ms":1000}'
```

### Blackout Single Universe
```bash
curl -X POST http://192.168.50.1:8891/api/dmx/blackout \
  -H 'Content-Type: application/json' \
  -d '{"universe":2,"fade_ms":0}'
```

---

## API Endpoints

### DMX Control

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/dmx/set` | Set channel values instantly |
| POST | `/api/dmx/fade` | Fade channels over duration |
| POST | `/api/dmx/blackout` | Blackout (all or single universe) |
| GET | `/api/dmx/status` | Get DMX system status |
| GET | `/api/dmx/diagnostics` | Detailed diagnostics |

### Scenes

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/scenes` | List all scenes |
| POST | `/api/scenes` | Create new scene |
| GET | `/api/scenes/:id` | Get scene details |
| PUT | `/api/scenes/:id` | Update scene |
| DELETE | `/api/scenes/:id` | Delete scene |
| POST | `/api/scenes/:id/apply` | Apply scene to outputs |

### Chases

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/chases` | List all chases |
| POST | `/api/chases` | Create new chase |
| GET | `/api/chases/:id` | Get chase details |
| PUT | `/api/chases/:id` | Update chase |
| DELETE | `/api/chases/:id` | Delete chase |
| POST | `/api/chases/:id/play` | Start chase playback |
| POST | `/api/chases/:id/stop` | Stop chase |

### Effects

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/effects` | List available effects |
| POST | `/api/effects/smooth` | Start smooth color effect |
| POST | `/api/effects/wave` | Start wave effect |
| POST | `/api/effects/twinkle` | Start twinkle effect |
| POST | `/api/effects/christmas` | Start christmas effect |
| POST | `/api/effects/stop` | Stop all effects |

### Nodes

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/nodes` | List all nodes |
| POST | `/api/nodes/scan` | Trigger node discovery |
| POST | `/api/nodes/:id/configure` | Configure node settings |
| POST | `/api/nodes/:id/sync` | Sync node with backend |

### Playback

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/playback/status` | Current playback state per universe |
| POST | `/api/playback/stop` | Stop all playback |

### Groups

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/groups` | List fixture groups |
| POST | `/api/groups` | Create group |
| PUT | `/api/groups/:id` | Update group |
| DELETE | `/api/groups/:id` | Delete group |

### Fixtures

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/fixtures` | List all fixtures |
| POST | `/api/fixtures` | Create fixture |
| PUT | `/api/fixtures/:id` | Update fixture |
| DELETE | `/api/fixtures/:id` | Delete fixture |

### Shows

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/shows` | List all shows |
| POST | `/api/shows` | Create show |
| GET | `/api/shows/:id` | Get show details |
| PUT | `/api/shows/:id` | Update show |
| DELETE | `/api/shows/:id` | Delete show |
| POST | `/api/shows/:id/play` | Play show |
| POST | `/api/shows/stop` | Stop show |

### Schedules

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/schedules` | List schedules |
| POST | `/api/schedules` | Create schedule |
| PUT | `/api/schedules/:id` | Update schedule |
| DELETE | `/api/schedules/:id` | Delete schedule |

---

## UDPJSON Protocol v2

All DMX data uses UDPJSON v2 on port 6455.

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

### Set Channels Example
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

### Ping/Pong Example
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

1. **Sequence Numbers**: Duplicate detection with 256-packet window
2. **Universe Filtering**: Nodes only process their configured universe
3. **MTU Safety**: Payloads < 1200 bytes
4. **No Blackout on Silence**: Nodes hold last values indefinitely
5. **Fade Engine**: Per-channel non-blocking fades at 50Hz

---

## Troubleshooting

### Node Not Responding

1. **Check network connectivity:**
   ```bash
   ssh ramzt@aether-portal.local "ping -c 2 192.168.50.16"
   ```

2. **Send direct ping:**
   ```bash
   ssh ramzt@aether-portal.local "echo '{\"v\":2,\"type\":\"ping\",\"seq\":1}' | nc -u -w2 192.168.50.16 6455"
   ```

3. **Check node registration:**
   ```bash
   curl http://192.168.50.1:8891/api/nodes | jq '.[] | select(.universe==2)'
   ```

4. **Power cycle the node** - often fixes WiFi connection issues

### Lights Not Responding But Node Is Online

1. **Check DMX wiring** - XLR cable from node to fixture
2. **Check fixture DMX address** - should match channel numbers
3. **Check fixture power** - is it plugged in?
4. **Check universe assignment** - fixture on correct universe?

### Chase/Effect Only Works On Some Universes

1. **Check playback status:**
   ```bash
   curl http://192.168.50.1:8891/api/playback/status
   ```

2. **Check which universes the chase targets:**
   ```bash
   curl http://192.168.50.1:8891/api/chases/<id>
   ```

3. **Power supply issue** - high current effects may trip protection

### Frontend Not Loading

1. **Check Express server:**
   ```bash
   ssh ramzt@aether-portal.local "systemctl status aether-portal"
   ```

2. **Check Flask backend:**
   ```bash
   ssh ramzt@aether-portal.local "systemctl status aether-core"
   ```

3. **Clear browser cache** - Ctrl+Shift+R or hard refresh

---

## Service Management

### Start/Stop Services
```bash
# On the Pi
sudo systemctl start aether-core
sudo systemctl stop aether-core
sudo systemctl restart aether-core

sudo systemctl start aether-portal
sudo systemctl stop aether-portal
sudo systemctl restart aether-portal
```

### View Logs
```bash
# Flask backend logs
sudo journalctl -u aether-core -f

# Express frontend logs
sudo journalctl -u aether-portal -f
```

### Check Service Status
```bash
sudo systemctl status aether-core
sudo systemctl status aether-portal
```

---

## Firmware Updates

### Flash ESP32 Node (from Windows dev machine)

1. Connect node via USB
2. Run PlatformIO:
   ```bash
   cd C:/MyProjects/Aether/aether-pulse/hybrid
   pio run -e pulse_bulletproof -t upload
   ```

3. Disconnect and power via 5V
4. Node will connect to AetherDMX WiFi automatically

### OTA Updates (Future)

Nodes advertise `ota` capability but OTA is not yet implemented.

---

## File Locations

### Raspberry Pi

| Path | Description |
|------|-------------|
| `/home/ramzt/aether-core/` | Flask backend |
| `/home/ramzt/aether-portal/` | Express + React frontend |
| `/home/ramzt/Aether-DMX/settings.json` | Persistent settings |
| `/home/ramzt/Aether-DMX/scenes/` | Scene definitions |
| `/home/ramzt/Aether-DMX/chases/` | Chase definitions |

### Development (Windows)

| Path | Description |
|------|-------------|
| `C:\MyProjects\Aether\aether-core\` | Flask backend source |
| `C:\MyProjects\Aether\aether-portal-os\` | Portal frontend/backend |
| `C:\MyProjects\Aether\aether-pulse\` | ESP32 firmware |
| `C:\MyProjects\Aether\PROTOCOL.md` | Protocol specification |

---

## Hardware Specifications

### ESP32 Pulse Node

| Component | Specification |
|-----------|---------------|
| MCU | ESP32-WROOM-32 |
| WiFi | 802.11 b/g/n |
| DMX Output | RS-485 via MAX485 |
| DMX TX Pin | GPIO 17 |
| DMX Enable | GPIO 4 |
| Power | 5V DC |

### DMX Output

| Parameter | Value |
|-----------|-------|
| Protocol | DMX512-A |
| Channels | 1-512 per universe |
| Refresh Rate | 40 Hz |
| Break Time | 176 µs |
| MAB Time | 16 µs |

---

## Known Limitations

1. **Universe 1** requires wired UART (not WiFi)
2. **No RDM** - RDM is stubbed but not functional
3. **No OTA** - firmware updates require USB
4. **Single AP** - all nodes connect to Pi's hotspot
5. **UDP** - no delivery guarantee (mitigated by sequence numbers)

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 6.0 | 2026-01-02 | Bulletproof firmware, UDPJSON v2, all 4 nodes verified |
| 5.0 | 2025-12-31 | Event-driven output, pong port fix |
| 4.0 | 2025-12-30 | UDPJSON transport, removed sACN |
| 3.0 | 2025-12-15 | Flask SSOT, Express proxy |
| 2.0 | 2025-11-01 | Multi-universe support |
| 1.0 | 2025-10-01 | Initial release |

---

## Quick Reference Card

```
┌────────────────────────────────────────────────────────────────┐
│                    AETHER QUICK REFERENCE                       │
├────────────────────────────────────────────────────────────────┤
│ Portal URL:     http://192.168.50.1:3000                       │
│ API URL:        http://192.168.50.1:8891                       │
│ SSH:            ssh ramzt@aether-portal.local                  │
├────────────────────────────────────────────────────────────────┤
│ Blackout:       POST /api/dmx/blackout                         │
│ Set Channels:   POST /api/dmx/set                              │
│ Play Chase:     POST /api/chases/:id/play                      │
│ Stop All:       POST /api/playback/stop                        │
├────────────────────────────────────────────────────────────────┤
│ Node Ping:      echo '{"v":2,"type":"ping","seq":1}' |         │
│                 nc -u -w2 <node-ip> 6455                       │
├────────────────────────────────────────────────────────────────┤
│ Universes:      1=UART, 2=.16, 3=.28, 4=.68, 5=.33            │
└────────────────────────────────────────────────────────────────┘
```
