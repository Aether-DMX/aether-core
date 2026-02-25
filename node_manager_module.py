"""
Node Manager Module - Extracted from aether-core.py

NodeManager handles all DMX output and node communication via UDPJSON protocol.
Manages node registration, configuration, content synchronization, and playback.

This module uses the core_registry to access cross-module services like socketio,
dmx_state, content_manager, etc.
"""

import socket
import json
import sqlite3
import threading
import time
from datetime import datetime, timedelta
import logging

import core_registry as reg


# Constants - used for node communication
STALE_TIMEOUT = 60

# These constants should be accessed via registry but also defined here for reference
# The actual values come from environment or defaults
AETHER_UDPJSON_PORT = 6455  # Primary port for UDPJSON DMX commands
WIFI_COMMAND_PORT = 8888    # Default WiFi command port

# Content sync chunk settings
CHUNK_SIZE = 50
CHUNK_DELAY = 0.05


class NodeManager:
    """Node management with UDPJSON DMX output.

    DMX data is sent via UDP JSON commands to ESP32 nodes on port 6455.
    Protocol v2: {"v":2,"type":"set","u":N,"seq":M,"ch":[[ch,val],...]}
    Legacy v1:   {"type":"set","universe":N,"channels":{...},"ts":...}

    Event-driven: packets sent on value change, no continuous refresh required.
    Nodes hold last values until next update.
    """

    # Protocol version
    PROTOCOL_VERSION = 2  # v2: Compact ch/fill/frame encodings
    MAX_PAYLOAD_SIZE = 1200  # MTU-safe payload limit

    def __init__(self):
        self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        self.lock = threading.Lock()

        # Sequence number for duplicate detection
        self._seq = 0
        self._seq_lock = threading.Lock()

        # Diagnostics tracking - UDPJSON DMX output
        self._last_udpjson_send = None
        self._udpjson_send_count = 0
        self._udpjson_errors = 0
        self._udpjson_per_universe = {}  # {universe: send_count}

        # Diagnostics tracking - UDP config commands
        self._last_udp_send = None
        self._udp_send_count = 0

        # [F03] Per-node delivery tracking
        self._per_node_stats = {}  # {ip: {sent: N, errors: N, acks: N, timeouts: N, last_rtt_ms: float}}
        self._reliable_send_count = 0
        self._reliable_ack_count = 0
        self._reliable_timeout_count = 0

        # DMX refresh loop control
        self._refresh_running = False
        self._refresh_thread = None
        self._refresh_rate = 40  # Hz (frames per second)

        print(f"✅ DMX Transport: UDPJSON v{self.PROTOCOL_VERSION} (port {AETHER_UDPJSON_PORT})")

    def _next_seq(self):
        """Get next sequence number (thread-safe, wraps at 2^32)"""
        with self._seq_lock:
            self._seq = (self._seq + 1) & 0xFFFFFFFF
            return self._seq

    def _channels_to_compact(self, channels_dict):
        """Convert {channel: value} dict to compact [[ch,val],...] array.

        This is the v2 protocol format - much more compact than object keys.
        """
        return [[int(ch), int(val)] for ch, val in channels_dict.items()]

    def _estimate_payload_size(self, ch_pairs):
        """Estimate JSON payload size for channel pairs."""
        # Rough estimate: "[[1,255]," = 9 chars per pair average
        return 50 + len(ch_pairs) * 9  # 50 for header overhead

    # ─────────────────────────────────────────────────────────
    # UDPJSON DMX Protocol - Primary output method
    # ─────────────────────────────────────────────────────────
    def send_udpjson(self, ip, port, payload_dict):
        """Send a UDPJSON command to a node.

        Args:
            ip: Node IP address
            port: UDP port (typically AETHER_UDPJSON_PORT=6455)
            payload_dict: Dictionary to send as JSON

        Returns:
            True on success, False on error
        """
        try:
            json_data = json.dumps(payload_dict, separators=(',', ':'))

            # Debug: log what we're sending (format depends on v1 or v2)
            msg_type = payload_dict.get('type', 'unknown')
            version = payload_dict.get('v', 1)
            seq = payload_dict.get('seq', 0)
            universe = payload_dict.get('u') or payload_dict.get('universe', 0)

            # Count channels from either format
            ch_count = len(payload_dict.get('ch', [])) or len(payload_dict.get('channels', {}))

            # Rate-limited logging (only log every 100th packet or first few)
            if self._udpjson_send_count < 10 or self._udpjson_send_count % 100 == 0:
                print(f"📡 UDP v{version}: {ip}:{port} type={msg_type} u={universe} ch={ch_count} seq={seq} bytes={len(json_data)}", flush=True)

            self.udp_socket.sendto(json_data.encode(), (ip, port))

            # Track diagnostics
            self._udpjson_send_count += 1
            self._last_udpjson_send = time.time()

            if universe:
                self._udpjson_per_universe[universe] = self._udpjson_per_universe.get(universe, 0) + 1

            return True
        except Exception as e:
            self._udpjson_errors += 1
            print(f"❌ UDPJSON send error to {ip}:{port}: {e}")
            return False

    # ─────────────────────────────────────────────────────────
    # [F03] Reliable UDP — ACK mode for critical commands
    # ─────────────────────────────────────────────────────────
    def _track_node_stat(self, ip, field, value=1):
        """Update per-node delivery statistics."""
        if ip not in self._per_node_stats:
            self._per_node_stats[ip] = {
                'sent': 0, 'errors': 0, 'acks': 0, 'timeouts': 0,
                'last_rtt_ms': None, 'last_seen': None,
            }
        stats = self._per_node_stats[ip]
        if field == 'last_rtt_ms' or field == 'last_seen':
            stats[field] = value
        else:
            stats[field] += value

    def send_udpjson_reliable(self, ip, port, payload_dict, retries=3, timeout_ms=150):
        """Send a UDPJSON command with ACK verification and retry.

        [F03] For critical commands (panic, reset, ping) that MUST be delivered.
        Opens a temporary receive socket to listen for any response from the node,
        retries up to `retries` times with increasing backoff.

        Normal DMX frames (set/fade) should NOT use this — the 40Hz refresh loop
        provides natural retransmission every 25ms.

        Args:
            ip: Node IP address
            port: UDP port
            payload_dict: Dictionary to send as JSON
            retries: Max retry attempts (default 3)
            timeout_ms: Timeout per attempt in milliseconds (default 150ms)

        Returns:
            dict with {success: bool, attempts: int, rtt_ms: float|None, response: dict|None}
        """
        self._reliable_send_count += 1
        self._track_node_stat(ip, 'sent')

        json_data = json.dumps(payload_dict, separators=(',', ':'))
        msg_type = payload_dict.get('type', 'unknown')
        seq = payload_dict.get('seq', 0)

        for attempt in range(1, retries + 1):
            # Create a temporary socket for this request-response exchange
            recv_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            try:
                recv_sock.settimeout(timeout_ms / 1000.0)
                recv_sock.bind(('', 0))  # Bind to ephemeral port

                send_start = time.monotonic()
                self.udp_socket.sendto(json_data.encode(), (ip, port))

                # Track send diagnostics
                self._udpjson_send_count += 1
                self._last_udpjson_send = time.time()

                try:
                    data, addr = recv_sock.recvfrom(4096)
                    rtt_ms = (time.monotonic() - send_start) * 1000
                    response = json.loads(data.decode())

                    self._reliable_ack_count += 1
                    self._track_node_stat(ip, 'acks')
                    self._track_node_stat(ip, 'last_rtt_ms', rtt_ms)
                    self._track_node_stat(ip, 'last_seen', time.time())

                    if attempt > 1:
                        print(f"   ✓ {msg_type.upper()} ACK from {ip} after {attempt} attempts ({rtt_ms:.1f}ms)", flush=True)
                    else:
                        print(f"   ✓ {msg_type.upper()} ACK from {ip} ({rtt_ms:.1f}ms)", flush=True)

                    return {
                        'success': True, 'attempts': attempt,
                        'rtt_ms': round(rtt_ms, 1), 'response': response
                    }

                except socket.timeout:
                    # No response within timeout — retry with backoff
                    backoff_ms = timeout_ms * attempt  # 150ms, 300ms, 450ms
                    if attempt < retries:
                        print(f"   ⏱ {msg_type.upper()} timeout from {ip} (attempt {attempt}/{retries}, backoff {backoff_ms}ms)", flush=True)
                        time.sleep(backoff_ms / 1000.0)
                    else:
                        print(f"   ✗ {msg_type.upper()} no ACK from {ip} after {retries} attempts", flush=True)

            except Exception as e:
                self._udpjson_errors += 1
                self._track_node_stat(ip, 'errors')
                print(f"   ✗ {msg_type.upper()} send error to {ip}: {e}", flush=True)

            finally:
                recv_sock.close()

        # All retries exhausted
        self._reliable_timeout_count += 1
        self._track_node_stat(ip, 'timeouts')
        return {'success': False, 'attempts': retries, 'rtt_ms': None, 'response': None}

    def get_delivery_stats(self):
        """[F03] Get per-node delivery statistics for diagnostics."""
        return {
            'reliable_sends': self._reliable_send_count,
            'reliable_acks': self._reliable_ack_count,
            'reliable_timeouts': self._reliable_timeout_count,
            'ack_rate': round(self._reliable_ack_count / max(1, self._reliable_send_count) * 100, 1),
            'per_node': dict(self._per_node_stats),
        }

    def send_udpjson_set(self, node_ip, universe, channels_dict, source="backend", fade_ms=0):
        """Send a 'set' command to a node via UDPJSON v2 protocol.

        Uses compact [[ch,val],...] format instead of {"ch":val,...} for smaller payloads.
        Automatically splits large payloads into multiple packets to stay under MTU.

        Args:
            node_ip: Node IP address
            universe: Universe number
            channels_dict: {channel: value, ...} dictionary
            source: Source identifier (e.g., "frontend", "chase", "scene")
            fade_ms: Optional fade duration in milliseconds
        """
        ch_pairs = self._channels_to_compact(channels_dict)

        # Check if we need to split the payload
        estimated_size = self._estimate_payload_size(ch_pairs)
        if estimated_size > self.MAX_PAYLOAD_SIZE and len(ch_pairs) > 50:
            # Split into multiple packets
            return self._send_chunked(node_ip, universe, ch_pairs, fade_ms, source)

        payload = {
            "v": self.PROTOCOL_VERSION,
            "type": "set",
            "u": universe,
            "seq": self._next_seq(),
            "ch": ch_pairs
        }
        if fade_ms > 0:
            payload["fade"] = fade_ms

        return self.send_udpjson(node_ip, AETHER_UDPJSON_PORT, payload)

    def _send_chunked(self, node_ip, universe, ch_pairs, fade_ms, source):
        """Send channel updates in multiple packets to stay under MTU."""
        chunk_size = 100  # ~900 bytes per chunk
        success = True
        for i in range(0, len(ch_pairs), chunk_size):
            chunk = ch_pairs[i:i + chunk_size]
            payload = {
                "v": self.PROTOCOL_VERSION,
                "type": "set",
                "u": universe,
                "seq": self._next_seq(),
                "ch": chunk
            }
            if fade_ms > 0:
                payload["fade"] = fade_ms
            if not self.send_udpjson(node_ip, AETHER_UDPJSON_PORT, payload):
                success = False
        return success

    def send_udpjson_fade(self, node_ip, universe, channels_dict, duration_ms, easing="linear", source="backend"):
        """Send a 'set' command with fade to a node via UDPJSON v2 protocol.

        Note: v2 protocol uses 'fade' field on 'set' type, not separate 'fade' type.
        Nodes handle fading locally based on the fade duration.

        Args:
            node_ip: Node IP address
            universe: Universe number
            channels_dict: {channel: target_value, ...}
            duration_ms: Fade duration in milliseconds
            easing: Easing function (ignored in v2 - linear only for now)
            source: Source identifier
        """
        # v2 protocol: use set with fade parameter
        return self.send_udpjson_set(node_ip, universe, channels_dict, source, fade_ms=duration_ms)

    def send_udpjson_fill(self, node_ip, universe, ranges, fade_ms=0):
        """Send a 'fill' command for efficient range fills.

        Efficiently sets contiguous channel ranges to the same value.
        Ideal for blackouts, full-on, or wipes.

        Args:
            node_ip: Node IP address
            universe: Universe number
            ranges: List of [start, end, value] tuples
            fade_ms: Optional fade duration in milliseconds
        """
        payload = {
            "v": self.PROTOCOL_VERSION,
            "type": "fill",
            "u": universe,
            "seq": self._next_seq(),
            "ranges": ranges
        }
        if fade_ms > 0:
            payload["fade"] = fade_ms

        return self.send_udpjson(node_ip, AETHER_UDPJSON_PORT, payload)

    def send_udpjson_blackout(self, node_ip, universe, fade_ms=0, source="backend"):
        """Send a 'blackout' command to a node via UDPJSON v2 protocol.

        Uses efficient fill command: ranges=[[1,512,0]] instead of sending 512 zeros.
        """
        payload = {
            "v": self.PROTOCOL_VERSION,
            "type": "blackout",
            "u": universe,
            "seq": self._next_seq()
        }
        if fade_ms > 0:
            payload["fade"] = fade_ms

        return self.send_udpjson(node_ip, AETHER_UDPJSON_PORT, payload)

    def send_udpjson_panic(self, node_ip, universe):
        """Send a 'panic' command - immediate blackout with no fade.

        SAFETY ACTION: This bypasses all playback/effects and commands
        immediate zero output on the target universe.

        [F03] Uses reliable sending with ACK verification and retry.
        Falls back to fire-and-forget if ACK times out (command still sent).
        """
        if reg.audit_log:
            reg.audit_log('panic', ip=node_ip, universe=universe)  # [F16]
        print(f"🚨 PANIC: Sending to {node_ip} universe {universe}", flush=True)
        payload = {
            "v": self.PROTOCOL_VERSION,
            "type": "panic",
            "u": universe,
            "seq": self._next_seq()
        }
        # [F03] Use reliable send with 3 retries, 100ms timeout (fast for safety)
        result = self.send_udpjson_reliable(node_ip, AETHER_UDPJSON_PORT, payload,
                                            retries=3, timeout_ms=100)
        if not result['success']:
            # ACK failed but command was sent — log but don't block
            print(f"   ⚠ PANIC sent but NO ACK from {node_ip} (command may have arrived)", flush=True)
        return result.get('success') or result.get('attempts', 0) > 0  # True if at least sent

    def send_udpjson_ping(self, node_ip):
        """Send a 'ping' command to a node and wait for 'pong' response.

        SAFETY ACTION: Health check for node connectivity.

        [F03] Now actually waits for the pong response using reliable send.
        Returns the pong data (rssi, uptime, heap, etc.) on success.
        """
        print(f"🏓 PING: Sending to {node_ip}", flush=True)
        payload = {
            "v": self.PROTOCOL_VERSION,
            "type": "ping",
            "seq": self._next_seq()
        }
        # [F03] Use reliable send — actually receive the pong
        result = self.send_udpjson_reliable(node_ip, AETHER_UDPJSON_PORT, payload,
                                            retries=2, timeout_ms=500)
        return result

    def send_udpjson_reset(self, node_ip):
        """Send a 'reset' command to a node.

        SAFETY ACTION: Commands node to reset its internal state.
        This clears any stuck effects, resets DMX output, and reinitializes.

        [F03] Uses reliable sending with ACK verification and retry.
        """
        print(f"🔄 RESET: Sending to {node_ip}", flush=True)
        payload = {
            "v": self.PROTOCOL_VERSION,
            "type": "reset",
            "seq": self._next_seq()
        }
        # [F03] Use reliable send — 3 retries, 200ms timeout
        result = self.send_udpjson_reliable(node_ip, AETHER_UDPJSON_PORT, payload,
                                            retries=3, timeout_ms=200)
        return result

    def start_dmx_refresh(self):
        """Start the continuous DMX refresh loop.

        Sends DMX data at 40fps to all active nodes via UDPJSON.
        """
        if self._refresh_running:
            print("⚠️ DMX Refresh: Already running")
            return

        self._refresh_running = True
        self._refresh_thread = threading.Thread(target=self._dmx_refresh_loop, daemon=True)
        self._refresh_thread.start()
        print(f"✅ DMX Refresh: Started at {self._refresh_rate} fps (UDPJSON on port {AETHER_UDPJSON_PORT})")
        print(f"   Thread alive: {self._refresh_thread.is_alive()}")

    def stop_dmx_refresh(self):
        """Stop the DMX refresh loop"""
        self._refresh_running = False
        if self._refresh_thread:
            self._refresh_thread.join(timeout=1.0)
        print("⏹️ DMX Refresh: Stopped")

    def _refresh_node_cache(self):
        """Background refresh of node-to-universe mapping (runs off render thread).

        [F04] Optimized: single UPDATE subquery instead of row-by-row loop.
        Retry logic prevents 'database is locked' from crashing the cache refresh.
        """
        while self._refresh_running:
            try:
                conn = reg.get_db()
                c = conn.cursor()

                # [F04] Single atomic UPDATE — replaces row-by-row loop that held write lock
                # Calculates channel_ceiling = min(512, max(1, max_fixture_channel + 16))
                for attempt in range(5):
                    try:
                        c.execute("""
                            UPDATE nodes SET channel_ceiling = MIN(512, MAX(1,
                                COALESCE((
                                    SELECT MAX(f.start_channel + f.channel_count - 1) + 16
                                    FROM fixtures f WHERE f.universe = nodes.universe
                                ), 1)
                            ))
                            WHERE is_paired = 1 AND ip IS NOT NULL AND status = 'online'
                        """)
                        conn.commit()
                        break
                    except sqlite3.OperationalError as e:
                        if 'locked' in str(e) and attempt < 4:
                            time.sleep(0.2 * (attempt + 1))  # 200ms, 400ms, 600ms, 800ms backoff
                        else:
                            raise  # Give up after 5 attempts (2s total backoff)

                # Now fetch node cache with channel_ceiling (read-only, no lock contention)
                # [F20] Exclude offline/stale nodes — don't send DMX to dead IPs
                c.execute("""
                    SELECT universe, ip, channel_start, channel_end, via_seance, seance_ip, channel_ceiling
                    FROM nodes
                    WHERE is_paired = 1 AND ip IS NOT NULL AND type = 'wifi' AND status = 'online'
                """)
                new_cache = {}
                for row in c.fetchall():
                    u, ip, ch_start, ch_end, via_seance, seance_ip, ceiling = row
                    if u == 1:
                        continue
                    if u not in new_cache:
                        new_cache[u] = []
                    target_ip = seance_ip if via_seance and seance_ip else ip
                    s = ch_start or 1
                    # Use channel_ceiling if set and in auto mode, otherwise use channel_end
                    e = min(ch_end or 512, ceiling or 512)
                    new_cache[u].append((target_ip, s, e, ip, via_seance, ceiling or 512))
                conn.close()
                self._node_cache = new_cache
            except Exception as ex:
                print(f"⚠️ Node cache refresh error: {ex}")
            time.sleep(5.0)  # [F04] Reduced from 2s — less write contention

    def _dmx_refresh_loop(self):
        """Background thread that sends DMX data to nodes via UDPJSON.

        All output uses the new UDPJSON protocol on port 6455.
        Optimized for smooth fading: monotonic timing, no blocking I/O in hot path.
        """
        frame_interval = 1.0 / self._refresh_rate
        frame_count = 0
        self._node_cache = {}
        self._last_sent = {}  # {universe:ip -> {ch_str: val}} delta tracking
        print(f"🔄 DMX Refresh loop starting (interval={frame_interval:.3f}s) - UDPJSON on port {AETHER_UDPJSON_PORT}")

        # Start background node cache refresh thread
        import threading
        node_thread = threading.Thread(target=self._refresh_node_cache, daemon=True)
        node_thread.start()

        while self._refresh_running:
            try:
                loop_start = time.monotonic()

                # Get active universes from dmx_state + cached nodes
                if reg.dmx_state:
                    active_universes = set(reg.dmx_state.universes.keys())
                else:
                    active_universes = set()
                universe_to_nodes = self._node_cache
                active_universes.update(universe_to_nodes.keys())

                # Log active universes periodically (every 5 seconds)
                frame_count += 1
                if frame_count == 1 or frame_count % (self._refresh_rate * 5) == 0:
                    print(f"🔄 DMX Refresh: universes={sorted(active_universes)}, udpjson sends={self._udpjson_send_count}")

                # Send DMX data for each active universe (skip universe 1)
                for universe in active_universes:
                    if not self._refresh_running:
                        break
                    if universe == 1:
                        continue

                    # Get output values (handles fade interpolation internally)
                    if reg.dmx_state:
                        dmx_values = reg.dmx_state.get_output_values(universe)
                    else:
                        continue

                    # Send to each node in this universe via UDPJSON
                    nodes = universe_to_nodes.get(universe, [])
                    for node_data in nodes:
                        # Unpack node data (now includes ceiling)
                        if len(node_data) == 6:
                            target_ip, slice_start, slice_end, original_ip, via_seance, ceiling = node_data
                        else:
                            # Backwards compatibility
                            target_ip, slice_start, slice_end, original_ip, via_seance = node_data
                            ceiling = slice_end

                        node_key = f"{universe}:{target_ip}"
                        prev_sent = self._last_sent.get(node_key, {})

                        # Smart channel optimization: only send up to ceiling
                        # This reduces network traffic significantly when only a few fixtures are used
                        effective_end = min(slice_end, ceiling)

                        # Build channels dict: include non-zero AND channels that changed to zero
                        node_channels = {}
                        for ch in range(slice_start, effective_end + 1):
                            val = dmx_values[ch - 1]
                            ch_str = str(ch)
                            if val > 0:
                                node_channels[ch_str] = val
                            elif ch_str in prev_sent:
                                # Was non-zero last frame, now zero — must send the zero
                                node_channels[ch_str] = 0

                        # Send either when there's data, or once per second for keepalive
                        if node_channels or frame_count % self._refresh_rate == 0:
                            self.send_udpjson_set(target_ip, universe, node_channels, source="refresh")

                        # Track what we sent for next frame's delta
                        self._last_sent[node_key] = {k: v for k, v in node_channels.items() if v > 0}

                # Maintain consistent frame rate using monotonic clock
                elapsed = time.monotonic() - loop_start
                sleep_time = frame_interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

            except Exception as e:
                print(f"❌ DMX Refresh error: {e}")
                time.sleep(0.1)

    def get_all_nodes(self, include_offline=True):
        conn = reg.get_db()
        c = conn.cursor()
        if include_offline:
            c.execute('SELECT * FROM nodes ORDER BY universe, channel_start')
        else:
            c.execute('SELECT * FROM nodes WHERE status = "online" ORDER BY universe, channel_start')
        rows = c.fetchall()
        conn.close()
        return [dict(row) for row in rows]

    def get_node(self, node_id):
        conn = reg.get_db()
        c = conn.cursor()
        c.execute('SELECT * FROM nodes WHERE node_id = ?', (str(node_id),))
        row = c.fetchone()
        conn.close()
        return dict(row) if row else None

    def get_nodes_in_universe(self, universe):
        """Get all paired/builtin nodes in a universe

        All nodes receive DMX via UDPJSON on port 6455.
        """
        conn = reg.get_db()
        c = conn.cursor()
        c.execute('''SELECT * FROM nodes WHERE universe = ? AND (is_paired = 1 OR is_builtin = 1)
                     ORDER BY channel_start''', (universe,))
        rows = c.fetchall()
        conn.close()
        return [dict(row) for row in rows]

    def get_wifi_nodes_in_universe(self, universe):
        """Get only WiFi nodes in a universe (for syncing content)"""
        conn = reg.get_db()
        c = conn.cursor()
        c.execute('''SELECT * FROM nodes WHERE universe = ? AND type = "wifi"
                     AND (is_paired = 1) AND status = "online" ORDER BY channel_start''', (universe,))
        rows = c.fetchall()
        conn.close()
        return [dict(row) for row in rows]

    def register_node(self, data):
        node_id = str(data.get('node_id'))
        conn = reg.get_db()
        c = conn.cursor()
        c.execute('SELECT * FROM nodes WHERE node_id = ?', (node_id,))
        existing = c.fetchone()
        now = datetime.now().isoformat()

        was_offline = False
        # Build firmware string: prefer 'firmware' field, fall back to 'version'
        firmware_str = data.get('firmware') or data.get('version')
        transport_str = data.get('transport', '')
        if transport_str and firmware_str:
            firmware_str = f"{firmware_str} ({transport_str})"

        # Extract Seance routing info (if node is connected via Seance bridge)
        via_seance = data.get('via_seance')  # Seance node ID or SSID
        seance_ip = data.get('seance_ip')    # IP of Seance on Pi's network (192.168.50.x)
        original_ip = data.get('original_ip') or data.get('ip')  # Node's IP on Seance's AP network

        if existing:
            # Check if node was offline before updating, and current paired state
            c.execute('SELECT status, is_paired FROM nodes WHERE node_id = ?', (node_id,))
            row = c.fetchone()
            was_offline = row and row[0] == 'offline'
            pi_thinks_paired = row and row[1] == 1

            # [F10] IP pinning: warn if paired node's IP changes (DHCP re-assignment or spoof)
            existing_ip = existing['ip'] if existing else None
            incoming_ip = data.get('ip')
            if pi_thinks_paired and existing_ip and incoming_ip and existing_ip != incoming_ip:
                print(f"[F10 WARN] Paired node {node_id} IP changed: {existing_ip} → {incoming_ip} (DHCP re-assign or spoof?)")

            # Check if node reports unpaired status (intentional unpair or NVS wipe)
            node_reports_unpaired = data.get('is_paired') == False or data.get('waiting_for_config') == True

            # [F10] Block heartbeat-driven unpairing for paired nodes — only REST API can unpair
            if pi_thinks_paired and node_reports_unpaired:
                msg_type = data.get('type', 'unknown')
                if msg_type == 'heartbeat':
                    # Heartbeats cannot force-unpair — log and skip
                    print(f"[F10 BLOCK] Heartbeat from {node_id} claims unpaired — ignoring (use REST API to unpair)")
                elif msg_type == 'register':
                    # Fresh registration with is_paired=false could be legitimate NVS wipe
                    print(f"[F10 WARN] Registration from {node_id} reports unpaired — syncing Pi database (possible NVS wipe)")
                    c.execute('UPDATE nodes SET is_paired = 0 WHERE node_id = ?', (node_id,))

            # Update basic fields that always come from heartbeats/registrations
            c.execute('''UPDATE nodes SET hostname = COALESCE(?, hostname), mac = COALESCE(?, mac),
                ip = COALESCE(?, ip), uptime = COALESCE(?, uptime), rssi = COALESCE(?, rssi),
                fps = COALESCE(?, fps), firmware = COALESCE(?, firmware), status = 'online', last_seen = ?,
                via_seance = ?, seance_ip = ?
                WHERE node_id = ?''',
                (data.get('hostname'), data.get('mac'), original_ip, data.get('uptime'),
                 data.get('rssi'), data.get('fps'), firmware_str, now, via_seance, seance_ip, node_id))

            # Also update slice config if provided in heartbeat (nodes now send full config)
            # This ensures Pi always has accurate slice info even for Seance-bridged nodes
            slice_start = data.get('slice_start')
            slice_end = data.get('slice_end')
            slice_mode = data.get('slice_mode')
            universe = data.get('u') or data.get('universe')

            if slice_start is not None and slice_end is not None:
                c.execute('''UPDATE nodes SET channel_start = ?, channel_end = ?,
                    slice_mode = COALESCE(?, slice_mode), universe = COALESCE(?, universe)
                    WHERE node_id = ?''',
                    (slice_start, slice_end, slice_mode, universe, node_id))

            # Log Seance routing changes
            if via_seance:
                print(f"📡 Node {node_id} via Seance: {via_seance} @ {seance_ip}")
        else:
            # Support both legacy (startChannel/channelCount) and new (slice_start/slice_end/slice_mode) fields
            slice_start = data.get('slice_start') or data.get('startChannel', 1)
            slice_end = data.get('slice_end') or data.get('channelCount', 512)
            slice_mode = data.get('slice_mode', 'zero_outside')
            c.execute('''INSERT INTO nodes (node_id, name, hostname, mac, ip, universe, channel_start, type,
                channel_end, slice_mode, firmware, status, is_paired, first_seen, last_seen, via_seance, seance_ip)
                VALUES (?, ?, ?, ?, ?, ?, ?, 'wifi', ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                (node_id, data.get('hostname', f'Node-{node_id[-4:]}'), data.get('hostname'),
                 data.get('mac'), original_ip, data.get('universe', 1), slice_start,
                 slice_end, slice_mode, firmware_str, 'online', False, now, now, via_seance, seance_ip))
            if via_seance:
                print(f"📡 New node {node_id} via Seance: {via_seance} @ {seance_ip}")
        conn.commit()
        conn.close()
        # Re-send config to paired WiFi nodes ONLY on reconnect (was offline, now online)
        node = self.get_node(node_id)
        if node and node.get('is_paired') and node.get('type') == 'wifi' and existing and was_offline:
            print(f"🔄 Re-sending config to reconnected node {node_id}")
            self.send_config_to_node(node, {
                'name': node.get('name'),
                'universe': node.get('universe', 1),
                'channel_start': node.get('channel_start', 1),
                'channel_end': node.get('channel_end', 512),
                'slice_mode': node.get('slice_mode', 'zero_outside')
            })
        self.broadcast_status()

        # Async sync node to Supabase (non-blocking)
        if reg.get_supabase_service and node:
            try:
                supabase = reg.get_supabase_service()
                if supabase and supabase.is_enabled():
                    reg.cloud_submit(supabase.sync_node, node)
            except Exception:
                pass

        return node

    def pair_node(self, node_id, config):
        conn = reg.get_db()
        c = conn.cursor()

        # Support both legacy and new slice field names
        channel_start = config.get('channel_start') or config.get('channelStart', 1)
        channel_end = config.get('channel_end') or config.get('channelEnd', 512)
        slice_mode = config.get('slice_mode', 'zero_outside')

        c.execute('''UPDATE nodes SET name = COALESCE(?, name), universe = ?, channel_start = ?,
            channel_end = ?, slice_mode = ?, mode = COALESCE(?, 'output'), is_paired = 1 WHERE node_id = ?''',
            (config.get('name'), config.get('universe', 1), channel_start,
             channel_end, slice_mode, config.get('mode'), str(node_id)))
        conn.commit()
        conn.close()

        # Send config to node via UDP
        node = self.get_node(node_id)
        if node:
            self.send_config_to_node(node, config)
            self.sync_content_to_node(node)
            print(f"✅ Node paired: {node.get('name')} on U{config.get('universe', 1)} ch{channel_start}-{channel_end} ({slice_mode})")

        self.broadcast_status()

        # Async sync node to Supabase (non-blocking)
        if reg.get_supabase_service and node:
            try:
                supabase = reg.get_supabase_service()
                if supabase and supabase.is_enabled():
                    reg.cloud_submit(supabase.sync_node, node)
            except Exception:
                pass

        return node

    def unpair_node(self, node_id):
        # Get node info before updating DB
        node = self.get_node(node_id)

        # Send unpair command to WiFi node to clear its config
        if node and node.get('type') == 'wifi' and node.get('ip'):
            # Route through Seance if node is connected via Seance bridge
            target_ip = node.get('seance_ip') if node.get('via_seance') else node.get('ip')
            self.send_command_to_wifi(target_ip, {'cmd': 'unpair'})
            print(f"📤 Unpair sent to {node.get('name', node_id)} ({target_ip})")

        conn = reg.get_db()
        c = conn.cursor()
        c.execute('UPDATE nodes SET is_paired = 0 WHERE node_id = ? AND can_delete = 1', (str(node_id),))
        conn.commit()
        conn.close()
        self.broadcast_status()

    def delete_node(self, node_id):
        # Get node info before deleting
        node = self.get_node(node_id)

        # Send unpair command to WiFi node to clear its config
        if node and node.get('type') == 'wifi' and node.get('ip'):
            # Route through Seance if node is connected via Seance bridge
            target_ip = node.get('seance_ip') if node.get('via_seance') else node.get('ip')
            self.send_command_to_wifi(target_ip, {'cmd': 'unpair'})
            print(f"📤 Unpair sent to {node.get('name', node_id)} ({target_ip})")

        conn = reg.get_db()
        c = conn.cursor()
        c.execute('DELETE FROM nodes WHERE node_id = ? AND can_delete = 1', (str(node_id),))
        conn.commit()
        conn.close()
        self.broadcast_status()

    def check_stale_nodes(self):
        conn = reg.get_db()
        c = conn.cursor()
        cutoff = (datetime.now() - timedelta(seconds=STALE_TIMEOUT)).isoformat()
        # Find which nodes are going offline BEFORE updating
        c.execute('SELECT node_id, ip, name FROM nodes WHERE last_seen < ? AND status = "online" AND is_builtin = 0', (cutoff,))
        stale_nodes = [dict(row) for row in c.fetchall()]
        stale_node_ids = [n['node_id'] for n in stale_nodes]
        c.execute('UPDATE nodes SET status = "offline" WHERE last_seen < ? AND status = "online" AND is_builtin = 0', (cutoff,))
        if c.rowcount > 0:
            conn.commit()
            self.broadcast_status()
            # [F20] Emit specific stale alert to frontend via SocketIO
            for node in stale_nodes:
                nid = node['node_id']
                print(f"⚠️ [F20] Node {nid} ({node.get('ip','?')}) went stale/offline", flush=True)
                if reg.audit_log:
                    reg.audit_log('node_stale', node_id=nid, ip=node.get('ip'), name=node.get('name'))
            if reg.socketio:
                reg.socketio.emit('node_stale', {
                    'node_ids': stale_node_ids,
                    'count': len(stale_node_ids),
                    'timestamp': datetime.now().isoformat()
                })
            # Mark all RDM devices on stale nodes as offline in live_inventory
            if reg.rdm_manager:
                for node_id in stale_node_ids:
                    try:
                        reg.rdm_manager.mark_node_devices_offline(node_id)
                    except Exception as e:
                        print(f"⚠️ RDM inventory update failed for stale node {node_id}: {e}", flush=True)
        conn.close()

        # [F20] Periodic reconnection probing — ping offline nodes every ~90s
        # (stale_checker runs every 30s, so probe on every 3rd call)
        if not hasattr(self, '_stale_probe_counter'):
            self._stale_probe_counter = 0
        self._stale_probe_counter += 1
        if self._stale_probe_counter % 3 == 0:
            self._probe_offline_nodes()

    def _probe_offline_nodes(self):
        """[F20] Ping offline nodes to detect reconnection without heartbeat."""
        conn = reg.get_db()
        c = conn.cursor()
        c.execute('SELECT node_id, ip FROM nodes WHERE status = "offline" AND ip IS NOT NULL AND is_builtin = 0')
        offline = [dict(row) for row in c.fetchall()]
        conn.close()
        for node in offline[:5]:  # Probe max 5 per cycle to avoid blocking
            ip = node.get('ip')
            if not ip or ip == 'localhost':
                continue
            try:
                result = self.send_udpjson_reliable(ip, AETHER_UDPJSON_PORT,
                    {"v": self.PROTOCOL_VERSION, "type": "ping", "seq": self._next_seq()},
                    retries=1, timeout_ms=300)
                if result.get('success'):
                    print(f"✅ [F20] Offline node {node['node_id']} responded to probe — reconnecting", flush=True)
                    if reg.audit_log:
                        reg.audit_log('node_reconnect_probe', node_id=node['node_id'], ip=ip)
            except Exception:
                pass  # Expected — node is offline

    def broadcast_status(self):
        nodes = self.get_all_nodes()
        if reg.socketio:
            reg.socketio.emit('nodes_update', {'nodes': nodes, 'timestamp': datetime.now().isoformat()})

    # ─────────────────────────────────────────────────────────
    # Channel Translation for Universe Splitting
    # ─────────────────────────────────────────────────────────
    def translate_channels_for_node(self, node, channels):
        """Translate universe channels to channels within node's range"""
        node_start = node.get('channel_start', 1)
        node_end = node.get('channel_end', 512)
        translated = {}
        for ch_str, value in channels.items():
            ch = int(ch_str)
            if node_start <= ch <= node_end:
                # Keep original channel number - node knows its range
                translated[str(ch)] = value
        return translated

    # ─────────────────────────────────────────────────────────
    # Send Commands to Nodes - UDPJSON
    # ─────────────────────────────────────────────────────────
    def send_to_node(self, node, channels_dict, fade_ms=0):
        """Send DMX values to a node via UDPJSON

        All DMX output goes through UDPJSON to ESP32 nodes on port 6455.
        """
        universe = node.get("universe", 1)

        # Universe 1 is offline
        if universe == 1:
            print(f"⚠️ Universe 1 is offline - skipping send to node", flush=True)
            return False

        non_zero = sum(1 for v in channels_dict.values() if v > 0) if channels_dict else 0
        print(f"📡 UDPJSON: U{universe} -> {len(channels_dict) if channels_dict else 0} ch ({non_zero} non-zero), fade={fade_ms}ms", flush=True)

        return self.update_dmx_state(universe, channels_dict, fade_ms)

    def update_dmx_state(self, universe, channels_dict, fade_ms=0):
        """Update DMX state for a universe - the refresh loop handles UDPJSON output

        SSOT COMPLIANCE: This method ONLY updates dmx_state.
        The continuous refresh loop (_dmx_refresh_loop) handles:
        - Sending UDPJSON commands at consistent 40fps
        - [F07] ESP32 handles fades; SSOT returns target values via get_output_values()
        """
        # Universe 1 is offline
        if universe == 1:
            print(f"⚠️ Universe 1 is offline - not updating state", flush=True)
            return False

        try:
            non_zero = sum(1 for v in channels_dict.values() if v > 0) if channels_dict else 0

            if fade_ms > 0:
                print(f"📤 SSOT U{universe} -> {len(channels_dict)} ch ({non_zero} non-zero), fade={fade_ms}ms", flush=True)
            else:
                print(f"📤 SSOT U{universe} -> {len(channels_dict)} ch ({non_zero} non-zero), snap", flush=True)

            # Update SSOT state with fade info - refresh loop handles UDPJSON output
            if reg.dmx_state:
                reg.dmx_state.set_channels(universe, channels_dict, fade_ms=fade_ms)

            return True

        except Exception as e:
            print(f"❌ SSOT update error: {e}")
            return False

    def send_command_to_wifi(self, ip, command):
        """Send config command to WiFi node (not DMX data)"""
        try:
            json_data = json.dumps(command)
            self.udp_socket.sendto(json_data.encode(), (ip, WIFI_COMMAND_PORT))
            self._last_udp_send = datetime.now().isoformat()
            self._udp_send_count += 1
            return True
        except Exception as e:
            print(f"❌ UDP command error to {ip}: {e}")
            return False

    def send_blackout(self, node, fade_ms=1000):
        """Send blackout to a node via UDPJSON with fade"""
        universe = node.get('universe', 1)
        if universe == 1:
            print(f"⚠️ Universe 1 is offline - skipping blackout", flush=True)
            return False
        all_zeros = {str(ch): 0 for ch in range(1, 513)}
        return self.update_dmx_state(universe, all_zeros, fade_ms=fade_ms)

    def send_config_to_node(self, node, config):
        """Send configuration update to a WiFi or gateway node"""
        node_type = node.get('type')

        universe = config.get('universe', node.get('universe', 1))

        # Support both legacy and new slice field names
        channel_start = config.get('channel_start') or config.get('channelStart') or node.get('channel_start', 1)
        channel_end = config.get('channel_end') or config.get('channelEnd') or node.get('channel_end', 512)
        slice_mode = config.get('slice_mode', node.get('slice_mode', 'zero_outside'))

        if node_type == 'gateway':
            # Gateway support would be here, but for now skipped
            return False
        elif node_type == 'wifi':
            # Send config to ESP32 via UDP - include both new and legacy field names
            command = {
                'cmd': 'config',
                'name': config.get('name', node.get('name')),
                'universe': universe,
                'channel_start': channel_start,
                'channel_end': channel_end,
                'slice_mode': slice_mode,
                # Legacy field names for backward compatibility with older firmware
                'startChannel': channel_start,
                'channelCount': channel_end
            }
            # Route through Seance if node is connected via Seance bridge
            if node.get('via_seance') and node.get('seance_ip'):
                target_ip = node.get('seance_ip')
                # Add routing info for Seance to forward to correct node
                command['node_id'] = node.get('node_id')
                command['_route_to'] = node.get('ip')  # Node's IP on Seance's AP network
                print(f"📡 Config via Seance: {node.get('node_id')} -> {target_ip}:8888 (route to {node.get('ip')})", flush=True)
            else:
                target_ip = node.get('ip')
                print(f"📡 Config direct: {node.get('node_id')} -> {target_ip}:8888", flush=True)
            result = self.send_command_to_wifi(target_ip, command)
            print(f"📡 Config result: {result}", flush=True)
        else:
            return False

        return result

    # ─────────────────────────────────────────────────────────
    # Sync Content to Nodes (Scenes/Chases)
    # ─────────────────────────────────────────────────────────
    def sync_scene_to_node(self, node, scene):
        """Send a scene to a WiFi node for local storage"""
        if node.get('type') != 'wifi':
            return False

        # Route through Seance if node is connected via Seance bridge
        target_ip = node.get('seance_ip') if node.get('via_seance') else node.get('ip')

        # Filter channels for this node's range
        node_channels = self.translate_channels_for_node(node, scene.get('channels', {}))

        if not node_channels:
            print(f"  ⚠️ Scene '{scene['name']}' has no channels for {node['name']}")
            return True  # Not an error, just nothing to sync

        command = {
            'cmd': 'store_scene',
            'id': scene['scene_id'],
            'name': scene['name'],
            'channels': node_channels,
            'fade_ms': scene.get('fade_ms', 500)
        }

        # Send in chunks if needed (large scenes)
        json_data = json.dumps(command)
        if len(json_data) > 1400:  # Near MTU limit
            # Send scene metadata first
            meta_cmd = {
                'cmd': 'store_scene',
                'id': scene['scene_id'],
                'name': scene['name'],
                'channels': {},
                'fade_ms': scene.get('fade_ms', 500)
            }
            self.send_command_to_wifi(target_ip, meta_cmd)
            time.sleep(CHUNK_DELAY)

            # Then send channels in chunks
            channel_items = list(node_channels.items())
            for i in range(0, len(channel_items), CHUNK_SIZE * 2):
                chunk = dict(channel_items[i:i + CHUNK_SIZE * 2])
                chunk_cmd = {
                    'cmd': 'set_channels',
                    'channels': chunk,
                    'fade_ms': 0
                }
                self.send_command_to_wifi(target_ip, chunk_cmd)
                time.sleep(CHUNK_DELAY)
            print(f"  📤 Scene '{scene['name']}' -> {node['name']} (chunked)")
        else:
            self.send_command_to_wifi(target_ip, command)
            print(f"  📤 Scene '{scene['name']}' -> {node['name']}")

        return True

    def sync_chase_to_node(self, node, chase):
        """Send a chase to a WiFi node for local storage"""
        if node.get('type') != 'wifi':
            return False

        # Route through Seance if node is connected via Seance bridge
        target_ip = node.get('seance_ip') if node.get('via_seance') else node.get('ip')

        # Filter each step's channels for this node's range
        filtered_steps = []
        for step in chase.get('steps', []):
            step_channels = step.get('channels', {})
            node_channels = self.translate_channels_for_node(node, step_channels)
            if node_channels:
                filtered_steps.append({'channels': node_channels})

        if not filtered_steps:
            print(f"  ⚠️ Chase '{chase['name']}' has no channels for {node['name']}")
            return True

        command = {
            'cmd': 'store_chase',
            'id': chase['chase_id'],
            'name': chase['name'],
            'bpm': chase.get('bpm', 120),
            'loop': chase.get('loop', True),
            'steps': filtered_steps
        }

        # Check size and send
        json_data = json.dumps(command)
        if len(json_data) > 1400:
            # Large chase - need to send in parts
            # First clear and send metadata
            meta_cmd = {
                'cmd': 'store_chase',
                'id': chase['chase_id'],
                'name': chase['name'],
                'bpm': chase.get('bpm', 120),
                'loop': chase.get('loop', True),
                'steps': []
            }
            self.send_command_to_wifi(target_ip, meta_cmd)
            time.sleep(CHUNK_DELAY)

            # Send steps in batches
            for i in range(0, len(filtered_steps), 5):
                batch_steps = filtered_steps[i:i+5]
                batch_cmd = {
                    'cmd': 'append_chase_steps',
                    'id': chase['chase_id'],
                    'steps': batch_steps
                }
                self.send_command_to_wifi(target_ip, batch_cmd)
                time.sleep(CHUNK_DELAY)
            print(f"  📤 Chase '{chase['name']}' -> {node['name']} (chunked, {len(filtered_steps)} steps)")
        else:
            self.send_command_to_wifi(target_ip, command)
            print(f"  📤 Chase '{chase['name']}' -> {node['name']} ({len(filtered_steps)} steps)")

        return True

    def sync_content_to_node(self, node):
        """Sync all scenes and chases to a single node"""
        if node.get('type') != 'wifi':
            return

        universe = node.get('universe', 1)
        node_name = node.get('name') or node.get('node_id', 'unknown')
        print(f"🔄 Syncing content to {node_name} (U{universe})")

        # Get all scenes for this universe
        conn = reg.get_db()
        c = conn.cursor()
        c.execute('SELECT * FROM scenes WHERE universe = ?', (universe,))
        scenes = [dict(row) for row in c.fetchall()]

        c.execute('SELECT * FROM chases WHERE universe = ?', (universe,))
        chases = [dict(row) for row in c.fetchall()]
        conn.close()

        # Sync scenes
        for scene in scenes:
            scene['channels'] = json.loads(scene['channels']) if scene['channels'] else {}
            self.sync_scene_to_node(node, scene)
            time.sleep(CHUNK_DELAY)

        # Sync chases
        for chase in chases:
            chase['steps'] = json.loads(chase['steps']) if chase['steps'] else []
            self.sync_chase_to_node(node, chase)
            time.sleep(CHUNK_DELAY)

        print(f"✓ Synced {len(scenes)} scenes, {len(chases)} chases to {node_name}")

    def sync_all_content(self):
        """Sync all content to all paired WiFi nodes"""
        print("🔄 Starting full content sync to all nodes...")
        nodes = self.get_all_nodes(include_offline=False)
        for node in nodes:
            if node.get('type') == 'wifi' and node.get('is_paired'):
                self.sync_content_to_node(node)
        print("✓ Full sync complete")

    # ─────────────────────────────────────────────────────────
    # Playback Commands
    # ─────────────────────────────────────────────────────────
    def play_scene_on_nodes(self, universe, scene_id, fade_ms=None):
        """Tell all nodes in universe to play a stored scene"""
        nodes = self.get_nodes_in_universe(universe)
        results = []

        for node in nodes:
            if node.get('type') == 'wifi':
                # Route through Seance if node is connected via Seance bridge
                target_ip = node.get('seance_ip') if node.get('via_seance') else node.get('ip')
                command = {'cmd': 'play_scene', 'id': scene_id}
                if fade_ms is not None:
                    command['fade_ms'] = fade_ms
                success = self.send_command_to_wifi(target_ip, command)
                results.append({'node': node['name'], 'success': success})

        return results

    def play_chase_on_nodes(self, universe, chase_id):
        """Tell all nodes in universe to play a stored chase"""
        nodes = self.get_nodes_in_universe(universe)
        results = []

        for node in nodes:
            if node.get('type') == 'wifi':
                # Route through Seance if node is connected via Seance bridge
                target_ip = node.get('seance_ip') if node.get('via_seance') else node.get('ip')
                command = {'cmd': 'play_chase', 'id': chase_id}
                success = self.send_command_to_wifi(target_ip, command)
                results.append({'node': node['name'], 'success': success})

        return results

    def stop_playback_on_nodes(self, universe=None):
        """Tell nodes to stop playback"""
        if universe:
            nodes = self.get_nodes_in_universe(universe)
        else:
            nodes = self.get_all_nodes(include_offline=False)

        results = []
        for node in nodes:
            if node.get('type') == 'wifi':
                # Route through Seance if node is connected via Seance bridge
                target_ip = node.get('seance_ip') if node.get('via_seance') else node.get('ip')
                success = self.send_command_to_wifi(target_ip, {'cmd': 'stop'})
                results.append({'node': node['name'], 'success': success})

        return results
