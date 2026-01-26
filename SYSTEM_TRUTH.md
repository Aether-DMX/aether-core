# AETHER SYSTEM TRUTH

**Version:** 5.0.0
**Last Updated:** 2026-01-26
**Authority:** This document reflects code reality. If code and docs disagree, update docs or delete them.

---

## 1. CANONICAL AUTHORITY CHAIN

```
UnifiedPlaybackEngine → MergeLayer → ContentManager (SSOT) → UDPJSON → ESP32 Nodes
```

### What This Means

| Component | Role | Location |
|-----------|------|----------|
| **UnifiedPlaybackEngine** | SOLE authority for all playback timing and state | `unified_playback.py` |
| **MergeLayer** | Combines multiple sources by priority | `aether-core.py` |
| **ContentManager (SSOT)** | Single Source of Truth for DMX state | `aether-core.py` |
| **UDPJSON** | Transport protocol to ESP32 nodes | Port 6455 |
| **ESP32 Nodes** | Physical DMX output via RS-485 | `aether-pulse` firmware |

### Authority Rules

1. **UnifiedPlaybackEngine is the ONLY entity that may own a timing loop for playback**
2. Any path that bypasses `UnifiedPlaybackEngine → MergeLayer → SSOT → UDPJSON` is **ILLEGAL**
3. The ESP32 firmware is the final DMX output authority on the wire
4. Backend crash does not stop DMX output (firmware holds last values)

---

## 2. PLAYBACK DATA FLOW

### Look Playback (Static + Modifiers)

```
User Request: POST /api/looks/{id}/play
    ↓
unified_playback.play_look(look_id, look_data, universes)
    ↓
session_factory.from_look() → PlaybackSession
    ↓
unified_engine.play(session)
    ↓
unified_engine._render_loop() [30 FPS]
    ↓
ModifierRenderer.render() [if modifiers present]
    ↓
output_callback(universe, channels, fade_ms)
    ↓
MergeLayer.set_source_channels() [priority-based merge]
    ↓
ContentManager.set_channels() [SSOT update]
    ↓
UDPJSON broadcast to nodes [40 FPS refresh]
    ↓
ESP32 firmware → RS-485 → DMX fixtures
```

### Sequence Playback

```
User Request: POST /api/sequences/{id}/play
    ↓
unified_playback.play_sequence(sequence_id, sequence_data, universes)
    ↓
session_factory.from_sequence() → PlaybackSession with steps
    ↓
unified_engine.play(session)
    ↓
Step advancement based on timing/BPM
    ↓
[Same output path as Look Playback]
```

### Safety Actions (Bypass Path)

```
PANIC: /api/dmx/panic
    ↓
unified_engine.stop_all()  [stops all sessions]
    ↓
send_udpjson_panic() to ALL nodes  [immediate blackout]
    ↓
ContentManager.blackout()  [clears SSOT]
```

Safety actions bypass non-essential layers for speed.

---

## 3. TRUST RULES

### The Four Trust Rules

| Rule | Behavior | Enforcement |
|------|----------|-------------|
| **Network Loss** | Nodes HOLD last DMX value | ESP32 firmware behavior |
| **Backend Crash** | Nodes CONTINUE output | ESP32 firmware behavior |
| **UI Desync** | REALITY wins over UI | `check_ui_sync()` in `operator_trust.py` |
| **Partial Node Failure** | SYSTEM HALTS playback + ALERTS | `OperatorTrustEnforcer` |

### Trust Event Logging

All trust events emit structured logs:

```
TRUST CRITICAL: node_heartbeat_lost | Components: ['node-1'] | Details: {...}
TRUST WARNING: ui_state_mismatch | Components: ['fader-panel'] | Details: {...}
```

Silent failure is **FORBIDDEN**. Every failure mode must be visible.

### Trust API Endpoints

| Endpoint | Purpose |
|----------|---------|
| `GET /api/trust/status` | Current trust enforcement state |
| `GET /api/trust/events` | Trust event history |
| `POST /api/trust/ui-sync` | Verify UI matches reality |
| `POST /api/trust/clear-halt` | Operator acknowledges failure |

---

## 4. FAILURE BEHAVIOR

### What Happens When Things Fail

| Failure | System Response | User Impact |
|---------|-----------------|-------------|
| **Backend crashes** | Nodes hold last DMX values | Lights stay on at last state |
| **Network drops** | Nodes hold, backend logs loss | Lights stable, UI shows warning |
| **Node goes offline** | Playback halts, alert raised | Operator must acknowledge |
| **UI loses connection** | Backend continues, UI reconnects | Playback unaffected |
| **Database corrupted** | In-memory state preserved | Current show continues |

### Recovery Behavior

| Scenario | Recovery Action |
|----------|-----------------|
| Backend restart | Nodes continue, backend re-syncs state |
| Network restore | Heartbeats resume, stale flag cleared |
| Node reconnect | Re-register, sync content if paired |
| UI reconnect | Fetch current state, sync to reality |

---

## 5. WHAT THE SYSTEM DOES

### Core Capabilities

- **Playback**: Looks (static scenes), Sequences (timed steps), with modifiers
- **Control**: Faders, buttons, master dimmer, blackout, panic
- **Discovery**: Node auto-discovery via UDP broadcast (port 9999)
- **RDM**: Fixture discovery, addressing, identification (via ESP32 firmware)
- **Scheduling**: Time-based scene activation
- **Preview**: Sandbox mode for editing without output

### Supported Protocols

| Protocol | Port | Direction | Purpose |
|----------|------|-----------|---------|
| HTTP REST | 8891 | In | API requests |
| WebSocket | 8891 | Bidirectional | Real-time updates |
| UDP Discovery | 9999 | In | Node registration |
| UDPJSON v2 | 6455 | Out | DMX data to nodes |

### Data Persistence

- SQLite database for fixtures, looks, sequences, schedules
- In-memory SSOT for live DMX state
- Node state persisted per-node in database

---

## 6. WHAT THE SYSTEM DOES NOT DO

### Explicit Non-Features

| Not Supported | Reason |
|---------------|--------|
| sACN/E1.31 | Removed - UDPJSON is simpler and sufficient |
| Art-Net | Not implemented - would add complexity |
| MIDI input | Not implemented |
| OSC input | Not implemented |
| DMX input (merge) | Nodes are output-only |
| Fixture profiles | Basic RDM only, no OFL integration |
| Multi-backend sync | Single backend only |
| Cloud control | Local network only |

### Architectural Constraints

- **Single backend**: No distributed or clustered deployment
- **Local network**: No internet connectivity required or supported
- **No plugins**: All functionality is built-in
- **No scripting**: No user-defined automation logic

---

## 7. PORT ASSIGNMENTS

| Port | Protocol | Service |
|------|----------|---------|
| 8891 | HTTP/WS | AETHER Core API |
| 3001 | HTTP | Express Proxy (portal) |
| 9999 | UDP | Node Discovery |
| 6455 | UDP | UDPJSON DMX Output |

---

## 8. FILE AUTHORITY

### Canonical Files

| File | Authority |
|------|-----------|
| `unified_playback.py` | Playback engine (SOLE AUTHORITY) |
| `aether-core.py` | API routes, MergeLayer, ContentManager |
| `operator_trust.py` | Trust enforcement |
| `rdm_service.py` | RDM orchestration |
| `render_engine.py` | ModifierRenderer utility ONLY |
| `effects_engine.py` | Effect computation utility ONLY |

### Deleted Files (Phase 3)

These files no longer exist and must not be recreated:

- `playback_controller.py` - Replaced by UnifiedPlaybackEngine
- `DMXService.js` - SSOT violation
- `PersistentDMXService.js` - Authority violation
- `dmxController.js` - Dead code

---

## 9. VERSION HISTORY

| Version | Date | Changes |
|---------|------|---------|
| 5.0.0 | 2026-01-26 | Phase 4 complete: Trust, RDM v1, hardening |
| 4.x | Prior | UnifiedPlaybackEngine established |

---

## 10. DOCUMENT MAINTENANCE

This document must be updated when:

1. Authority chain changes
2. New trust rules are added
3. Failure behavior changes
4. Ports or protocols change
5. Files are added or deleted from canonical set

**If this document is stale, delete it rather than leave it wrong.**
