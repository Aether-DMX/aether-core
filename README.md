# AETHER Core

Python backend services for the AETHER DMX system. Single Source of Truth (SSOT) for all lighting state.

## Services

| Script | Port | Description |
|--------|------|-------------|
| `aether-core.py` | 8891 | Main service - REST API, UDP JSON v2 output, serial DMX |

## Dependencies

```bash
pip install -r requirements.txt
```

Core dependencies: `flask`, `flask-cors`, `flask-socketio`, `requests`

Optional: `pyserial` (UART gateway), `supabase` (cloud sync)

## Running

### As systemd service (recommended)
```bash
sudo systemctl start aether-core
sudo systemctl status aether-core
sudo journalctl -u aether-core -f
```

### Manually
```bash
python3 aether-core.py
```

## API Endpoints (Port 8891)

### DMX Control
```bash
# Set single channel
curl -X POST http://localhost:8891/api/dmx/set \
  -H "Content-Type: application/json" \
  -d '{"universe":1,"channel":1,"value":255}'

# Set multiple channels
curl -X POST http://localhost:8891/api/dmx/channels \
  -H "Content-Type: application/json" \
  -d '{"universe":1,"channels":{"1":255,"2":128,"3":64}}'

# Blackout
curl -X POST http://localhost:8891/api/dmx/blackout
```

### Node Management
```bash
# List discovered nodes
curl http://localhost:8891/api/nodes

# Configure node
curl -X POST http://localhost:8891/api/nodes/<node_id>/config \
  -H "Content-Type: application/json" \
  -d '{"universe":1,"name":"Stage Left"}'
```

### System
```bash
# Health check
curl http://localhost:8891/api/health

# System stats
curl http://localhost:8891/api/system/stats
```

## Architecture

```
┌──────────────────────────────────────────────────┐
│                  aether-core.py                   │
│                                                   │
│  ┌──────────┐  ┌──────────────┐  ┌───────────┐  │
│  │ Flask    │  │ UDP JSON v2  │  │ Serial    │  │
│  │ REST API │  │ Port 8888    │  │ UART DMX  │  │
│  │ :8891    │  │              │  │           │  │
│  └──────────┘  └──────────────┘  └───────────┘  │
│        │              │                │         │
└────────┼──────────────┼────────────────┼─────────┘
         │              │                │
         ▼              ▼                ▼
    Frontend       ESP32 Pulse       Wired
    (React)          Nodes           Nodes
```

### UDP JSON v2 Protocol

AETHER uses UDP JSON v2.0 as its only wireless transport (no sACN/E1.31/OLA).

**Packet format:**
```json
{"v":2,"u":1,"d":[255,128,64,0,...]}
```
- `v`: Protocol version (always 2)
- `u`: Universe number (1-8)
- `d`: Array of up to 512 DMX channel values (0-255)

## Configuration

Environment variables (with defaults):
```bash
AETHER_API_PORT=8891       # REST API port
AETHER_DISCOVERY_PORT=9999 # Node discovery port
AETHER_WIFI_PORT=8888      # UDP JSON v2 output port
```

Serial settings in `aether-core.py`:
```python
SERIAL_PORT = '/dev/serial0'  # Pi GPIO UART
SERIAL_BAUD = 115200
```

## License

MIT License
