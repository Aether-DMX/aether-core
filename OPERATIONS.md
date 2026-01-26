# AETHER OPERATIONS GUIDE

**Version:** 5.0.0
**Last Updated:** 2026-01-26
**Audience:** Installers, Operators, Support

---

## 1. HEALTH CHECK ENDPOINTS

### System Health

```bash
# Basic health check
curl http://localhost:8891/api/health

# Response:
{
  "status": "healthy",
  "version": "5.0.0",
  "timestamp": "2026-01-26T12:00:00.000Z",
  "services": {
    "database": true,
    "discovery": true,
    "udpjson": true
  }
}
```

### Version Information

```bash
curl http://localhost:8891/api/version

# Response:
{
  "version": "5.0.0",
  "build": "production",
  "python": "3.11.x"
}
```

### Trust Status

```bash
curl http://localhost:8891/api/trust/status

# Response:
{
  "monitoring": true,
  "node_health": {
    "node-1": {"is_healthy": true, "last_heartbeat": "..."},
    "node-2": {"is_healthy": true, "last_heartbeat": "..."}
  },
  "playback_halted_due_to_failure": false,
  "active_alerts": []
}
```

---

## 2. NODE STATUS VISIBILITY

### List All Nodes

```bash
curl http://localhost:8891/api/nodes

# Response:
{
  "nodes": [
    {
      "node_id": "aether-pulse-001",
      "hostname": "aether-pulse-001",
      "ip": "192.168.1.100",
      "status": "online",
      "rssi": -45,
      "uptime": 3600,
      "universes": [1],
      "last_seen": "2026-01-26T12:00:00.000Z"
    }
  ]
}
```

### Ping All Nodes

```bash
curl -X POST http://localhost:8891/api/nodes/ping

# Response:
{
  "success": true,
  "total": 2,
  "responded": 2,
  "nodes": [
    {"node_id": "node-1", "ip": "192.168.1.100", "success": true},
    {"node_id": "node-2", "ip": "192.168.1.101", "success": true}
  ]
}
```

### Ping Specific Node

```bash
curl -X POST http://localhost:8891/api/nodes/{node_id}/ping
```

### Reset Specific Node

```bash
curl -X POST http://localhost:8891/api/nodes/{node_id}/reset
```

---

## 3. PLAYBACK SESSION INTROSPECTION

### Get Playback Status

```bash
curl http://localhost:8891/api/playback/status

# Response:
{
  "sessions": [
    {
      "id": "look_abc123",
      "type": "look",
      "name": "Warm Amber",
      "state": "playing",
      "elapsed_time": 45.2,
      "universes": [1, 2]
    }
  ],
  "engine_running": true,
  "fps": 30
}
```

### Get Unified Engine Status

```bash
curl http://localhost:8891/api/unified/status

# Response:
{
  "sessions": [...],
  "active_session_count": 1,
  "engine_fps": 30,
  "last_render_time_ms": 2.3
}
```

### Stop All Playback

```bash
curl -X POST http://localhost:8891/api/playback/stop
```

### Panic (Emergency Stop)

```bash
curl -X POST http://localhost:8891/api/dmx/panic

# Stops all playback, sends panic to all nodes, clears SSOT
```

---

## 4. RDM DISCOVERY STATUS

### Get RDM Status

```bash
curl http://localhost:8891/api/rdm/status

# Response:
{
  "enabled": true,
  "device_count": 5,
  "last_discovery": "2026-01-26T12:00:00.000Z",
  "discovery_in_progress": false,
  "devices": [...]
}
```

### Trigger Discovery

```bash
curl -X POST http://localhost:8891/api/rdm/discover

# Response:
{
  "success": true,
  "devices": [...],
  "count": 5,
  "timestamp": "2026-01-26T12:00:00.000Z"
}
```

### Get Address Suggestions

```bash
curl http://localhost:8891/api/rdm/address-suggestions

# Response:
{
  "success": true,
  "suggestions": [...],
  "conflicts": [...],
  "total_devices": 5,
  "conflicting_devices": 1
}
```

---

## 5. TROUBLESHOOTING COMMANDS

### Quick Health Check Script

```bash
#!/bin/bash
# save as: aether-health.sh

echo "=== AETHER Health Check ==="
echo ""

# API Health
echo "API Health:"
curl -s http://localhost:8891/api/health | jq .
echo ""

# Node Count
echo "Nodes Online:"
curl -s http://localhost:8891/api/nodes | jq '.nodes | length'
echo ""

# Playback Status
echo "Active Sessions:"
curl -s http://localhost:8891/api/playback/status | jq '.sessions | length'
echo ""

# Trust Status
echo "Trust Status:"
curl -s http://localhost:8891/api/trust/status | jq '{monitoring, playback_halted: .playback_halted_due_to_failure}'
```

### Service Status (systemd)

```bash
# Check service status
sudo systemctl status aether-core

# View recent logs
sudo journalctl -u aether-core -n 50

# Follow logs live
sudo journalctl -u aether-core -f
```

### Network Diagnostics

```bash
# Check if discovery port is listening
sudo ss -ulnp | grep 9999

# Check if API port is listening
sudo ss -tlnp | grep 8891

# Check node connectivity
ping 192.168.1.100  # Replace with node IP
```

---

## 6. COMMON ISSUES

### Node Not Appearing

| Symptom | Check | Fix |
|---------|-------|-----|
| Node not in list | Is it on same subnet? | Verify network config |
| Node shows offline | Ping responds? | Check node logs |
| Discovery timeout | Port 9999 open? | Check firewall |

### Playback Issues

| Symptom | Check | Fix |
|---------|-------|-----|
| Nothing playing | Session status? | Check /api/playback/status |
| Stuck on scene | Paused? | Resume or stop |
| Flickering | Node RSSI? | Improve WiFi signal |

### RDM Issues

| Symptom | Check | Fix |
|---------|-------|-----|
| No devices found | Node has RDM cap? | Check node capabilities |
| Discovery timeout | Fixtures RDM-capable? | Verify fixture specs |
| Address won't set | Conflicts? | Use address-suggestions |

---

## 7. LOG LOCATIONS

| Log | Location | Contents |
|-----|----------|----------|
| AETHER Core | journalctl -u aether-core | API, playback, trust |
| Express Proxy | journalctl -u dmx-backend | HTTP proxy logs |
| Node Logs | Serial @ 115200 | Firmware debug output |

### Log Levels

| Level | Meaning |
|-------|---------|
| INFO | Normal operations |
| WARNING | Recoverable issues |
| ERROR | Failures requiring attention |
| CRITICAL | System-affecting failures |

### Trust Event Logs

Trust events are logged with structured format:

```
TRUST CRITICAL: node_heartbeat_lost | Components: ['node-1'] | Details: {...}
TRUST WARNING: ui_state_mismatch | Components: ['fader-panel'] | Details: {...}
```

---

## 8. SAFETY ACTIONS

### Immediate Blackout

```bash
curl -X POST http://localhost:8891/api/dmx/blackout
```

### Emergency Panic

```bash
curl -X POST http://localhost:8891/api/dmx/panic
```

### Stop All Playback

```bash
curl -X POST http://localhost:8891/api/playback/stop
```

### Clear Trust Halt

If playback was halted due to node failure:

```bash
curl -X POST http://localhost:8891/api/trust/clear-halt
```

---

## 9. BACKUP AND RESTORE

### Database Location

```
/srv/aether/core/aether.db  (production)
./aether.db                  (development)
```

### Backup

```bash
# Stop service first
sudo systemctl stop aether-core

# Copy database
cp /srv/aether/core/aether.db /backup/aether-$(date +%Y%m%d).db

# Restart service
sudo systemctl start aether-core
```

### Restore

```bash
sudo systemctl stop aether-core
cp /backup/aether-20260126.db /srv/aether/core/aether.db
sudo systemctl start aether-core
```

---

## 10. CONTACT

For issues not covered here:

- GitHub Issues: https://github.com/Aether-DMX/aether-core/issues
- Logs required: Include relevant journalctl output
- Version required: Include output of /api/version
