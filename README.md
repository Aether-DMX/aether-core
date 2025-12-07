# AETHER Core

Python backend services for the AETHER DMX system.

## Services

| Script | Port | Description |
|--------|------|-------------|
| `aether-core.py` | 8891 | Main service - REST API, sACN output, serial DMX |
| `aether-discovery.py` | 9999 | Node discovery (UDP broadcast) |
| `aether-pairing-server.py` | 8888 | Node configuration |
| `aether-dmx-manager.py` | - | DMX universe management |
| `aether-command-router.py` | - | Command routing between services |

## Dependencies

```bash
pip3 install flask flask-cors pyserial sacn
```

Also requires OLA (Open Lighting Architecture):
```bash
sudo apt install ola ola-python
```

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
┌─────────────────────────────────────────────────┐
│                  aether-core.py                  │
│                                                  │
│  ┌──────────┐  ┌──────────┐  ┌──────────────┐  │
│  │ Flask    │  │ sACN     │  │ Serial       │  │
│  │ REST API │  │ Sender   │  │ (Wired DMX)  │  │
│  │ :8891    │  │ E1.31    │  │ /dev/serial0 │  │
│  └──────────┘  └──────────┘  └──────────────┘  │
│        │              │              │          │
└────────┼──────────────┼──────────────┼──────────┘
         │              │              │
         ▼              ▼              ▼
    Frontend       Wireless        Wired
    (Node.js)       Nodes          Nodes
```

## Configuration

Default settings in `aether-core.py`:
```python
SERIAL_PORT = '/dev/serial0'  # Pi GPIO UART
SERIAL_BAUD = 115200
API_PORT = 8891
DISCOVERY_PORT = 9999
```

## License

MIT License
