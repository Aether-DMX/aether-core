# AETHER DMX RUNBOOK - UDPJSON Transport

## System Architecture

```
Frontend (React, Port 3000)
    │
    ▼
Portal Backend (Node.js, Port 3000) ──► proxies to ──►
    │
    ▼
AETHER Core (Python Flask, Port 8891)
    │
    ├── DMX State Manager (SSOT)
    ├── 40fps Refresh Loop
    │
    ▼ UDPJSON (Port 6455)
    │
ESP32 Nodes (WiFi, Port 6455)
    │
    ▼ RS-485/DMX
    │
DMX Fixtures
```

## Transport: UDPJSON (No OLA, No sACN)

All DMX output uses direct UDP JSON commands on port **6455**.

**Protocol Messages:**

```json
// Set channels immediately
{"type":"set","universe":2,"channels":{"1":255,"2":128},"source":"frontend","ts":1730000000}

// Fade channels over duration
{"type":"fade","universe":3,"duration_ms":1200,"channels":{"1":0,"2":255},"easing":"linear","source":"frontend","ts":1730000000}

// Blackout (all channels to 0)
{"type":"blackout","universe":4,"source":"frontend","ts":1730000000}

// Health check ping
{"type":"ping","ts":1730000000}

// Pong response
{"type":"pong","node_id":"pulse-abc123","universes":[2],"slice_start":1,"slice_end":256,"slice_mode":"zero_outside","version":"2.5.0","rssi":-54,"uptime_s":12345,"dmx_tx_fps":40,"rx_udp_packets":999}
```

## Universe 1 is OFFLINE

**Universe 1 is not available** - it requires wired infrastructure that is not connected.

- All testing must use universes 2-5
- Backend rejects requests for universe 1 with 400 error
- Frontend defaults to universe 2

---

## End-to-End Tests (curl commands)

Replace `<pi-ip>` with your Pi's IP address (e.g., `192.168.50.1`).

### Test 1: Set Channel 1 to 255 on Universe 2

```bash
curl -X POST http://<pi-ip>:8891/api/dmx/set \
  -H "Content-Type: application/json" \
  -d '{"universe":2,"channels":{"1":255}}'
```

**Expected Response:**
```json
{"success":true,"universe":2,"channels_updated":1}
```

**Expected Node Log:**
```
UDPJSON: set U2, 1 channels
```

### Test 2: Set Multiple Channels on Universe 3

```bash
curl -X POST http://<pi-ip>:8891/api/dmx/set \
  -H "Content-Type: application/json" \
  -d '{"universe":3,"channels":{"1":255,"2":128,"3":64,"4":32}}'
```

### Test 3: Fade Channel 1 to 0 Over 2 Seconds on Universe 2

```bash
curl -X POST http://<pi-ip>:8891/api/dmx/fade \
  -H "Content-Type: application/json" \
  -d '{"universe":2,"duration_ms":2000,"channels":{"1":0}}'
```

**Expected Node Log:**
```
UDPJSON: fade U2, 1 channels, 2000ms
```

### Test 4: Blackout Universe 4

```bash
curl -X POST http://<pi-ip>:8891/api/dmx/blackout \
  -H "Content-Type: application/json" \
  -d '{"universe":4}'
```

**Expected Node Log:**
```
UDPJSON: blackout U4
```

### Test 5: Blackout All Universes (2-5)

```bash
curl -X POST http://<pi-ip>:8891/api/dmx/blackout \
  -H "Content-Type: application/json" \
  -d '{}'
```

### Test 6: Get DMX Status

```bash
curl http://<pi-ip>:8891/api/dmx/status
```

**Expected Response:**
```json
{
  "transport": "udpjson",
  "port": 6455,
  "online_nodes": [...],
  "universes": {
    "2": [...],
    "3": [...],
    "4": [...],
    "5": [...]
  },
  "universe_1_note": "Universe 1 is OFFLINE - use universes 2-5",
  "stats": {
    "total_sends": 12345,
    "errors": 0,
    "per_universe": {"2": 1000, "3": 1000, ...}
  }
}
```

### Test 7: Universe 1 Rejection

```bash
curl -X POST http://<pi-ip>:8891/api/dmx/set \
  -H "Content-Type: application/json" \
  -d '{"universe":1,"channels":{"1":255}}'
```

**Expected Response (400 error):**
```json
{"error":"Universe 1 is offline. Use universes 2-5.","success":false}
```

### Test 8: System Health Check

```bash
curl http://<pi-ip>:8891/api/health
```

**Expected Response:**
```json
{
  "status": "healthy",
  "version": "2.4.0",
  "services": {"database": true, "discovery": true, "udpjson": true}
}
```

### Test 9: Get Nodes List

```bash
curl http://<pi-ip>:8891/api/nodes
```

---

## Frontend Regression Checklist

After changes, verify these work in the browser:

- [ ] Dashboard loads without errors
- [ ] Console view loads and shows universes 2-5 (not universe 1)
- [ ] Faders move and fixtures respond
- [ ] Blackout button works
- [ ] Universe selector shows 2-5
- [ ] Scene playback works
- [ ] Chase playback works
- [ ] Node Management shows online nodes

---

## Debug Checklist

### Issue: Frontend works but lights don't respond

1. **Check backend is running:**
   ```bash
   curl http://<pi-ip>:8891/api/health
   ```

2. **Check nodes are online:**
   ```bash
   curl http://<pi-ip>:8891/api/dmx/status
   ```

3. **Check backend logs for UDPJSON sends:**
   ```
   📤 SSOT U2 -> 4 ch (3 non-zero), snap
   ```

4. **Check node serial output:**
   ```
   UDPJSON: set U2, 4 channels
   ```

### Issue: Backend sends but node doesn't receive

1. **Verify network connectivity:**
   ```bash
   ping <node-ip>
   ```

2. **Check node is listening on port 6455:**
   - Node should show `UDP: Config=8888, Discovery=9999, UDPJSON DMX=6455` at boot

3. **Test direct UDP send:**
   ```bash
   echo '{"type":"ping","ts":123}' | nc -u <node-ip> 6455
   ```

4. **Check node serial for pong response:**
   ```
   UDPJSON: pong sent to 192.168.50.1
   ```

### Issue: Node receives but fixtures don't change

1. **Check node universe matches:**
   ```
   UDPJSON: set U2, 4 channels  (should match fixture universe)
   ```

2. **Check slice configuration:**
   - If node slice is 1-128, channels 129-512 won't output
   - Check `slice_start` and `slice_end` in `/api/dmx/status`

3. **Check DMX output wiring:**
   - DMX TX = GPIO 17
   - DMX Enable = GPIO 4
   - Verify RS-485 transceiver is connected

4. **Check DMX output rate:**
   - Node should output at 40 fps
   - Serial log shows: `Output: TX=12345, fps=40.0 (target 40)`

### Issue: Universe mismatch

1. **Verify frontend is set to correct universe:**
   - Check browser console: `✅ Configured universes (online, excluding U1): [2, 3, 4, 5]`

2. **Verify backend routes to correct nodes:**
   - Check `/api/dmx/status` shows nodes with correct universe assignments

3. **Verify node is configured for correct universe:**
   - Check node serial output: `Config: Universe=2, Slice=1-512`
   - Or OLED display shows `U2`

---

## Expected Backend Logs

On successful DMX send:
```
📤 SSOT U2 -> 4 ch (3 non-zero), snap
📡 UDPJSON: U2 -> 4 ch (3 non-zero), fade=0ms
🔄 DMX Refresh: universes={2, 3, 4, 5}, udpjson sends=12345
```

On universe 1 rejection:
```
⚠️ Universe 1 is offline - not updating state
```

## Expected Node Logs

On UDPJSON receive:
```
UDPJSON: set U2, 4 channels
```

On fade:
```
UDPJSON: fade U2, 4 channels, 1500ms
```

On blackout:
```
UDPJSON: blackout U2
```

On ping/pong:
```
UDPJSON: pong sent to 192.168.50.1
```

Periodic status (every 10s):
```
───────────────────────────────────────────────
STATUS @ 1234s | pulse-ABC123 | Universe 2
───────────────────────────────────────────────
  Slice:    1-512 (zero_outside)
  UDP:      LIVE (last 25ms ago)
  Packets:  UDPJSON=999
  Output:   TX=49360, fps=40.0 (target 40)
  WiFi:     OK, RSSI=-54 dBm
  Preview:  ch1-4=[255,128,64,0]
───────────────────────────────────────────────
```

---

## Physical Validation Steps

1. **Power on nodes for universes 2-5**
2. **Verify nodes connect to AetherDMX WiFi**
3. **Check nodes appear in `/api/nodes` as `online`**
4. **Run set command for each universe:**
   ```bash
   for u in 2 3 4 5; do
     curl -X POST http://<pi-ip>:8891/api/dmx/set \
       -H "Content-Type: application/json" \
       -d "{\"universe\":$u,\"channels\":{\"1\":255}}"
     sleep 1
   done
   ```
5. **Verify fixtures on each universe respond**
6. **Run blackout:**
   ```bash
   curl -X POST http://<pi-ip>:8891/api/dmx/blackout -H "Content-Type: application/json" -d '{}'
   ```
7. **Verify all fixtures go dark**

---

## Constants (SSOT)

| Constant | Value | Location |
|----------|-------|----------|
| UDPJSON Port | 6455 | `aether-core.py:AETHER_UDPJSON_PORT`, `main.cpp:UDPJSON_DMX_PORT` |
| Config Port | 8888 | `aether-core.py:WIFI_COMMAND_PORT`, `main.cpp:CONFIG_PORT` |
| Discovery Port | 9999 | `aether-core.py:DISCOVERY_PORT`, `main.cpp:DISCOVERY_PORT` |
| API Port | 8891 | `aether-core.py:API_PORT` |
| DMX Output FPS | 40 | `aether-core.py:_refresh_rate`, `main.cpp:DMX_OUTPUT_FPS` |
