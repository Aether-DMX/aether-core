# AETHER PRODUCT TRUTH

**Version:** 5.0.0
**Last Updated:** 2026-01-26
**Authority:** This document describes what AETHER promises to users. No more, no less.

---

## 1. WHAT AETHER IS

AETHER is a **WiFi DMX lighting control system** for:

- Small to medium venues (bars, restaurants, event spaces)
- Permanent installations
- Operators who want simple, reliable control
- Environments where traditional DMX cabling is impractical

---

## 2. WHAT AETHER PROMISES

### Core Promises

| Promise | Guarantee |
|---------|-----------|
| **Reliable playback** | Looks and sequences play without glitches |
| **Safe failure** | Lights hold last state if backend crashes |
| **Simple operation** | Play, pause, stop, blackout work as expected |
| **Network resilience** | Brief dropouts don't cause chaos |
| **RDM support** | Discover and address fixtures remotely |

### Operational Promises

| Promise | Guarantee |
|---------|-----------|
| **Instant blackout** | Panic stops everything immediately |
| **Predictable timing** | Sequences advance at specified BPM |
| **State visibility** | Operator can always see what's playing |
| **Clean shutdown** | Stop all clears everything cleanly |

### Hardware Promises

| Promise | Guarantee |
|---------|-----------|
| **ESP32 nodes work** | UDPJSON protocol is stable |
| **Multiple universes** | Up to 5 universes supported |
| **DMX compliance** | 512 channels per universe, 0-255 values |

---

## 3. WHAT AETHER DOES NOT PROMISE

### Explicit Non-Promises

| Not Promised | Why |
|--------------|-----|
| **Sub-millisecond timing** | WiFi is not deterministic |
| **Broadcast-quality sync** | Not designed for video/music sync |
| **Zero-downtime updates** | Updates may require restart |
| **Multi-venue control** | Single-venue, single-network only |
| **Professional touring** | Not ruggedized, not road-tested |
| **Fixture library** | RDM addresses fixtures, doesn't profile them |
| **Cloud backup** | Local-only, no cloud features |
| **Mobile app** | Web UI only |

### What "WiFi DMX" Means

WiFi introduces latency and potential packet loss. AETHER mitigates this but does not eliminate it.

- **Typical latency**: 5-50ms depending on network
- **Packet loss handling**: Nodes hold last value
- **Refresh rate**: 40 FPS to nodes (25ms interval)

If you need deterministic sub-frame timing, use wired DMX.

---

## 4. SUPPORTED USE CASES

### Primary Use Cases

| Use Case | Supported | Notes |
|----------|-----------|-------|
| Bar/restaurant ambient | **YES** | Primary target |
| Event space preset recall | **YES** | Look playback |
| Architectural lighting | **YES** | Schedule-based |
| Small theater | **PARTIAL** | No cue stacking yet |
| DJ booth accent | **YES** | Effect modifiers |
| Holiday displays | **YES** | Sequences + schedules |

### Secondary Use Cases

| Use Case | Supported | Notes |
|----------|-----------|-------|
| Church services | **YES** | Scene recall, simple cues |
| Retail displays | **YES** | Schedule-driven |
| Home automation | **PARTIAL** | No external triggers |
| Art installations | **PARTIAL** | Limited interactivity |

---

## 5. NON-GOALS

These are **explicitly not goals** for AETHER:

### Product Non-Goals

| Non-Goal | Reason |
|----------|--------|
| Replace ETC/MA | Different market, different requirements |
| Compete with DMXKing | Hardware commoditization not our focus |
| Support Art-Net | UDPJSON is simpler for our use case |
| Cloud features | Adds complexity, security concerns |
| Plugin ecosystem | Maintenance burden, security risk |
| MIDI/OSC input | Would complicate architecture |

### Architectural Non-Goals

| Non-Goal | Reason |
|----------|--------|
| Distributed backend | Single-venue assumption |
| Hot failover | Complexity exceeds value |
| Real-time media sync | Not a media server |
| Fixture profiling | RDM footprint is sufficient |

---

## 6. OPERATOR EXPECTATIONS

### What Operators Should Expect

| Expectation | Reality |
|-------------|---------|
| "Press play, it plays" | **YES** |
| "Press stop, it stops" | **YES** |
| "Blackout works" | **YES** |
| "Panic kills everything" | **YES** |
| "I can see what's playing" | **YES** |
| "Updates don't break shows" | **MOSTLY** - test after updates |

### What Operators Should NOT Expect

| Expectation | Reality |
|-------------|---------|
| "WiFi is as reliable as cable" | **NO** - WiFi has inherent issues |
| "Zero learning curve" | **NO** - DMX concepts required |
| "Works without network" | **NO** - Network is required |
| "Automatic fixture setup" | **NO** - RDM helps but requires setup |

---

## 7. INSTALLER EXPECTATIONS

### What Installers Should Expect

| Expectation | Reality |
|-------------|---------|
| "Nodes auto-discover" | **YES** |
| "RDM works for addressing" | **YES** |
| "Config persists on restart" | **YES** |
| "Logs show problems" | **YES** |

### Installation Requirements

| Requirement | Specification |
|-------------|---------------|
| Network | 2.4GHz WiFi, same subnet |
| Backend | Raspberry Pi 4+ or equivalent |
| Nodes | ESP32-based with RS-485 |
| Power | 5V USB or PoE for nodes |

---

## 8. SUPPORT BOUNDARIES

### We Will Help With

- Configuration issues
- Network troubleshooting
- RDM discovery problems
- Software bugs

### We Will NOT Help With

- Custom fixture profiles
- Third-party hardware integration
- Performance tuning for edge cases
- Feature requests beyond roadmap

---

## 9. ROADMAP BOUNDARIES

### Planned (v5.x)

- Bug fixes
- Performance improvements
- Documentation updates

### Not Planned

- sACN/Art-Net support
- MIDI/OSC input
- Cloud features
- Plugin system
- Multi-backend clustering

---

## 10. DOCUMENT MAINTENANCE

This document must be updated when:

1. Product promises change
2. New use cases are validated
3. Non-goals are reconsidered
4. Support boundaries shift

**If this document doesn't match what we ship, fix it or delete it.**
