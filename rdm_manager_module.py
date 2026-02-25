"""
RDM Manager Module - Extracted from aether-core.py

This module provides consolidated Remote Device Management (RDM) functionality:
- RDMDevice: Data class representing an RDM-capable device
- RDMManager: Manager for RDM discovery, device communication, and inventory

All references to global objects are resolved through core_registry for
decoupled, testable code with proper null-safety checks.

Sends RDM commands to ESP32 nodes via UDPJSON and processes responses.
Maintains authoritative live_inventory of all known RDM devices.
Emits rdm_update via SocketIO when device status changes.
"""

import socket
import json
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional

import core_registry as reg


# ============================================================
# Constants
# ============================================================

AETHER_UDPJSON_PORT = 6455

KNOWN_MANUFACTURERS = {
    0x0000: "PLASA (Development)", 0x0001: "ESTA (Standards)",
    0x414C: "Avolites", 0x4144: "ADJ", 0x4348: "Chauvet",
    0x434D: "City Theatrical", 0x454C: "ETC", 0x4D41: "Martin",
    0x5052: "PR Lighting", 0x524F: "Robe", 0x534C: "Signify (Philips)",
    0x5354: "Strong Entertainment", 0x5641: "Varilite",
}


# ============================================================
# RDMDevice
# ============================================================

@dataclass
class RDMDevice:
    """Represents an RDM-capable device discovered on the bus."""
    uid: str                           # Unique ID: "XXXX:XXXXXXXX" (manufacturer:device)
    manufacturer_id: int = 0
    device_model_id: int = 0
    dmx_address: int = 0
    dmx_footprint: int = 0
    personality_id: int = 1
    personality_count: int = 1
    software_version: int = 0
    sensor_count: int = 0
    label: str = ""                    # User-assigned label
    discovered_via: str = ""           # Node ID that discovered this device
    discovered_at: str = ""            # ISO timestamp
    last_seen: str = ""               # ISO timestamp
    manufacturer_name: str = ""        # Resolved from manufacturer ID
    model_name: str = ""              # Resolved from device info

    def to_dict(self):
        return {
            'uid': self.uid, 'manufacturer_id': self.manufacturer_id,
            'device_model_id': self.device_model_id, 'dmx_address': self.dmx_address,
            'dmx_footprint': self.dmx_footprint, 'personality_id': self.personality_id,
            'personality_count': self.personality_count, 'software_version': self.software_version,
            'sensor_count': self.sensor_count, 'label': self.label,
            'discovered_via': self.discovered_via, 'discovered_at': self.discovered_at,
            'last_seen': self.last_seen, 'manufacturer_name': self.manufacturer_name,
            'model_name': self.model_name,
        }


# ============================================================
# RDMManager
# ============================================================

class RDMManager:
    """Consolidated RDM (Remote Device Management) — Single Source of Truth.

    Sends RDM commands to ESP32 nodes via UDPJSON and processes responses.
    Maintains authoritative live_inventory of all known RDM devices.
    Emits rdm_update via SocketIO when device status changes.
    """

    RDM_TIMEOUT_MS = 5000
    DISCOVERY_TIMEOUT_MS = 30000
    UDP_PORT = 6455

    def __init__(self):
        self.discovery_tasks = {}  # node_id -> discovery status
        self.response_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.response_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.response_socket.settimeout(10.0)
        self.pending_requests = {}  # request_id -> callback

        # Authoritative live inventory
        # { "uid": { "status": "online"/"offline"/"stale", "temp": 0.0, "is_patched": bool } }
        self.live_inventory = {}

        # In-memory device cache
        self._devices: Dict[str, RDMDevice] = {}
        self._lock = threading.RLock()
        self._last_discovery = None
        self._discovery_in_progress = False
        self._last_emit_time = 0.0

        # External references (set after init)
        self._node_manager = None
        self._socketio = None
        self._playback_engine = None

        # Hydrate in-memory cache from database (devices from previous sessions)
        self._hydrate_from_db()

        print("✓ RDMManager initialized (consolidated)")

    def _hydrate_from_db(self):
        """Load existing RDM devices from database into in-memory cache on startup."""
        try:
            db_fn = reg.get_db
            if db_fn is None:
                print("  ↳ Warning: get_db not available in registry")
                return

            conn = db_fn()
            c = conn.cursor()
            c.execute('SELECT * FROM rdm_devices ORDER BY node_id, dmx_address')
            columns = [d[0] for d in c.description]
            rows = c.fetchall()
            with self._lock:
                for row in rows:
                    row_dict = dict(zip(columns, row))
                    uid = row_dict['uid']
                    self._devices[uid] = RDMDevice(
                        uid=uid,
                        manufacturer_id=row_dict.get('manufacturer_id', 0) or 0,
                        device_model_id=row_dict.get('device_model_id', 0) or 0,
                        dmx_address=row_dict.get('dmx_address', 0) or 0,
                        dmx_footprint=row_dict.get('dmx_footprint', 0) or 0,
                        personality_id=row_dict.get('personality_id', 1) or 1,
                        personality_count=row_dict.get('personality_count', 1) or 1,
                        software_version=int(row_dict.get('software_version', 0) or 0),
                        sensor_count=row_dict.get('sensor_count', 0) or 0,
                        label=row_dict.get('device_label', '') or '',
                        discovered_via=row_dict.get('node_id', '') or '',
                        discovered_at=str(row_dict.get('created_at', '')) or '',
                        last_seen=str(row_dict.get('last_seen', '')) or '',
                        manufacturer_name=KNOWN_MANUFACTURERS.get(row_dict.get('manufacturer_id', 0), 'Unknown'),
                        model_name=''
                    )
                    # Also populate live_inventory with "offline" status (will be updated by heartbeats)
                    if uid not in self.live_inventory:
                        self.live_inventory[uid] = {
                            'status': 'offline',
                            'temp': 0.0,
                            'is_patched': False
                        }
            if rows:
                print(f"  ↳ Hydrated {len(rows)} RDM devices from database")
        except Exception as e:
            print(f"  ↳ Warning: Could not hydrate RDM cache from DB: {e}")

    # ─────────────────────────────────────────────────────────
    # External Wiring
    # ─────────────────────────────────────────────────────────

    def set_node_manager(self, nm):
        """Set reference to NodeManager."""
        self._node_manager = nm

    def set_socketio(self, sio):
        """Set reference to Flask-SocketIO for real-time updates."""
        self._socketio = sio

    def set_playback_engine(self, engine):
        """Set reference to UnifiedPlaybackEngine for offset auto-cleanup."""
        self._playback_engine = engine

    # ─────────────────────────────────────────────────────────
    # Live Inventory & SocketIO
    # ─────────────────────────────────────────────────────────

    def _emit_rdm_update(self):
        """Emit live_inventory to all connected clients via SocketIO (throttled)."""
        sio = self._socketio
        if not sio:
            return
        now = time.monotonic()
        if now - self._last_emit_time < 0.5:  # Max 2 emits/sec
            return
        self._last_emit_time = now
        sio.emit('rdm_update', {
            'inventory': self.live_inventory,
            'device_count': len(self._devices),
            'timestamp': datetime.now().isoformat()
        })

    def update_inventory(self, uid, status, temp=0.0, is_patched=None):
        """Update a device's live inventory entry. Emits on status change."""
        old_entry = self.live_inventory.get(uid, {})
        old_status = old_entry.get('status', 'unknown')

        # Check is_patched from fixtures table if not provided
        if is_patched is None:
            is_patched = old_entry.get('is_patched', False)

        self.live_inventory[uid] = {
            'status': status,
            'temp': temp,
            'is_patched': is_patched,
        }

        # Auto-cleanup AI offsets when device returns to healthy
        if old_status != 'online' and status == 'online' and self._playback_engine:
            fixture_id = self._resolve_fixture_for_rdm_uid(uid)
            if fixture_id:
                self._playback_engine.clear_offsets_for_fixture(fixture_id)

        if old_status != status:
            self._emit_rdm_update()

    def mark_node_devices_offline(self, node_id):
        """Mark all devices discovered via a node as offline."""
        with self._lock:
            for uid, dev in self._devices.items():
                if dev.discovered_via == node_id:
                    self.update_inventory(uid, 'offline')

    def _resolve_fixture_for_rdm_uid(self, uid):
        """Look up fixture_id in the fixtures table for an RDM UID."""
        try:
            db_fn = reg.get_db
            if db_fn is None:
                return None

            conn = db_fn()
            c = conn.cursor()
            c.execute('SELECT fixture_id FROM fixtures WHERE rdm_uid = ?', (uid,))
            row = c.fetchone()
            return row[0] if row else None
        except Exception:
            return None

    def _send_rdm_command(self, node_ip, action, params=None):
        """Send RDM command to a node and wait for response.

        Args:
            node_ip: IP address of the ESP32 node
            action: RDM action (discover, get_info, identify, set_address, etc.)
            params: Additional parameters for the action

        Returns:
            dict with response data or error
        """
        try:
            # Build the RDM command (v:2 required for V2 protocol parser)
            payload = {"v": 2, "type": "rdm", "action": action}
            if params:
                payload.update(params)

            # Create a socket for sending and receiving
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.settimeout(15.0 if action == 'discover' else 5.0)  # Discovery takes longer

            # Send the command
            json_data = json.dumps(payload, separators=(',', ':'))
            sock.sendto(json_data.encode(), (node_ip, AETHER_UDPJSON_PORT))
            print(f"📡 RDM: {action} -> {node_ip}")

            # Wait for response
            try:
                data, addr = sock.recvfrom(4096)
                response = json.loads(data.decode())
                sock.close()
                return response
            except socket.timeout:
                sock.close()
                return {"success": False, "error": "Response timeout"}

        except Exception as e:
            print(f"❌ RDM error: {e}")
            return {"success": False, "error": str(e)}

    def discover_devices(self, node_id):
        """Start RDM discovery on a node.

        Args:
            node_id: The node to scan for RDM devices

        Returns:
            dict with discovery results
        """
        nm = self._node_manager
        if nm is None:
            return {"success": False, "error": "NodeManager not available"}

        node = nm.get_node(node_id)
        if not node or node.get('status') != 'online':
            return {"success": False, "error": "Node not found or offline"}

        self.discovery_tasks[node_id] = {"status": "scanning", "started_at": datetime.now().isoformat()}

        # Send discover command
        result = self._send_rdm_command(node['ip'], 'discover')

        # Update status
        self.discovery_tasks[node_id] = {"status": "complete", "result": result}

        # Save discovered devices to database and purge stale entries
        if result.get('success'):
            universe = node.get('universe', 1)
            found_uids = []
            for d in result.get('devices', []):
                found_uids.append(d if isinstance(d, str) else d.get('uid'))

            if result.get('devices'):
                self._save_devices(node_id, universe, result['devices'])

            # Purge stale devices: remove DB entries for this node that were NOT found
            self._purge_stale_devices(node_id, found_uids)

            # Fetch detailed info for each device and populate cache + inventory
            for device in result['devices']:
                uid = device if isinstance(device, str) else device.get('uid')
                if uid:
                    now_iso = datetime.now().isoformat()
                    # Populate in-memory cache
                    with self._lock:
                        if uid not in self._devices:
                            self._devices[uid] = RDMDevice(
                                uid=uid, discovered_via=node_id,
                                discovered_at=now_iso, last_seen=now_iso
                            )
                        else:
                            self._devices[uid].last_seen = now_iso

                    try:
                        info = self._send_rdm_command(node['ip'], 'get_info', {"uid": uid})
                        if info.get('success'):
                            self._update_device_info(uid, info)
                            # Update cache with device info
                            with self._lock:
                                if uid in self._devices:
                                    dev = self._devices[uid]
                                    dev.manufacturer_id = info.get('manufacturer_id', 0)
                                    dev.device_model_id = info.get('device_model_id', 0)
                                    dev.dmx_address = info.get('dmx_address', 0)
                                    dev.dmx_footprint = info.get('dmx_footprint', info.get('footprint', 0))
                                    dev.personality_id = info.get('personality_id', 1)
                                    dev.personality_count = info.get('personality_count', 1)
                                    dev.software_version = info.get('software_version', 0)
                                    dev.sensor_count = info.get('sensor_count', 0)
                                    mid = dev.manufacturer_id
                                    if mid in KNOWN_MANUFACTURERS:
                                        dev.manufacturer_name = KNOWN_MANUFACTURERS[mid]
                            print(f"  📋 Got info for {uid}: Ch{info.get('dmx_address', '?')}, {info.get('dmx_footprint', info.get('footprint', '?'))}ch")
                    except Exception as e:
                        print(f"  ⚠️ Failed to get info for {uid}: {e}")

                    # Check if fixture is patched
                    is_patched = self._resolve_fixture_for_rdm_uid(uid) is not None
                    self.update_inventory(uid, 'online', is_patched=is_patched)

            self._last_discovery = datetime.now()
            self._emit_rdm_update()

        return result

    def _save_devices(self, node_id, universe, devices):
        """Save discovered devices to database.

        Devices can be either:
        - List of UID strings: ["02CA:C207DFA1", ...]
        - List of dicts: [{"uid": "...", "manufacturer": ...}, ...]
        """
        db_fn = reg.get_db
        if db_fn is None:
            print("  ↳ Warning: get_db not available in registry")
            return

        conn = db_fn()
        c = conn.cursor()

        for device in devices:
            # Handle both string UIDs and dict format
            if isinstance(device, str):
                uid = device
                manufacturer_id = 0
                device_model_id = 0
            else:
                uid = device.get('uid')
                manufacturer_id = device.get('manufacturer', 0)
                device_model_id = device.get('device_id', 0)

            if not uid:
                continue

            c.execute('''INSERT OR REPLACE INTO rdm_devices
                (uid, node_id, universe, manufacturer_id, device_model_id, last_seen)
                VALUES (?, ?, ?, ?, ?, ?)''',
                (uid, node_id, universe, manufacturer_id, device_model_id,
                 datetime.now().isoformat()))
        conn.commit()

        conn.commit()
        print(f"✓ Saved {len(devices)} RDM devices to database")

    def _purge_stale_devices(self, node_id, found_uids):
        """Remove devices from DB and caches that belong to this node but were NOT found in latest discovery."""
        db_fn = reg.get_db
        if db_fn is None:
            print("  ↳ Warning: get_db not available in registry")
            return

        conn = db_fn()
        c = conn.cursor()
        c.execute('SELECT uid FROM rdm_devices WHERE node_id = ?', (node_id,))
        db_uids = [row[0] for row in c.fetchall()]

        stale_uids = [uid for uid in db_uids if uid not in found_uids]
        if not stale_uids:
            return

        for uid in stale_uids:
            c.execute('DELETE FROM rdm_devices WHERE uid = ?', (uid,))
            c.execute('DELETE FROM rdm_personalities WHERE device_uid = ?', (uid,))
            with self._lock:
                self._devices.pop(uid, None)
                self.live_inventory.pop(uid, None)

        conn.commit()
        print(f"🗑️ Purged {len(stale_uids)} stale RDM devices from {node_id}: {stale_uids}")

    def get_device_info(self, node_id, uid):
        """Get detailed info for a specific RDM device."""
        nm = self._node_manager
        if nm is None:
            return {"success": False, "error": "NodeManager not available"}

        node = nm.get_node(node_id)
        if not node or node.get('status') != 'online':
            return {"success": False, "error": "Node not found or offline"}

        result = self._send_rdm_command(node['ip'], 'get_info', {"uid": uid})

        # Update database with new info
        if result.get('success'):
            self._update_device_info(uid, result)

        return result

    def _update_device_info(self, uid, info):
        """Update device info in database."""
        db_fn = reg.get_db
        if db_fn is None:
            print("  ↳ Warning: get_db not available in registry")
            return

        conn = db_fn()
        c = conn.cursor()

        c.execute('''UPDATE rdm_devices SET
            dmx_address = ?, dmx_footprint = ?, personality_id = ?, personality_count = ?,
            software_version = ?, sensor_count = ?, last_seen = ?
            WHERE uid = ?''',
            (info.get('dmx_address', 0), info.get('dmx_footprint', info.get('footprint', 0)),
             info.get('personality_id', info.get('personality_current', 0)), info.get('personality_count', 0),
             info.get('software_version', ''), info.get('sensor_count', 0),
             datetime.now().isoformat(), uid))

        conn.commit()

    def identify_device(self, node_id, uid, state):
        """Set identify mode on a device (flashes LED)."""
        nm = self._node_manager
        if nm is None:
            return {"success": False, "error": "NodeManager not available"}

        node = nm.get_node(node_id)
        if not node or node.get('status') != 'online':
            return {"success": False, "error": "Node not found or offline"}

        return self._send_rdm_command(node['ip'], 'identify', {"uid": uid, "state": state})

    def set_dmx_address(self, node_id, uid, address):
        """Set DMX start address for a device."""
        nm = self._node_manager
        if nm is None:
            return {"success": False, "error": "NodeManager not available"}

        node = nm.get_node(node_id)
        if not node or node.get('status') != 'online':
            return {"success": False, "error": "Node not found or offline"}

        result = self._send_rdm_command(node['ip'], 'set_address', {"uid": uid, "address": address})

        # Update database
        if result.get('success'):
            db_fn = reg.get_db
            if db_fn is not None:
                try:
                    conn = db_fn()
                    c = conn.cursor()
                    c.execute('UPDATE rdm_devices SET dmx_address = ?, last_seen = ? WHERE uid = ?',
                             (address, datetime.now().isoformat(), uid))
                    conn.commit()
                except Exception:
                    pass

        return result

    def get_devices_for_node(self, node_id):
        """Get all RDM devices for a node from database."""
        db_fn = reg.get_db
        if db_fn is None:
            return []

        conn = db_fn()
        c = conn.cursor()
        c.execute('SELECT * FROM rdm_devices WHERE node_id = ? ORDER BY dmx_address', (node_id,))
        columns = [d[0] for d in c.description]
        return [dict(zip(columns, row)) for row in c.fetchall()]

    def get_all_devices(self):
        """Get all RDM devices from database, enriched with live_inventory status."""
        db_fn = reg.get_db
        if db_fn is None:
            return []

        conn = db_fn()
        c = conn.cursor()
        c.execute('SELECT * FROM rdm_devices ORDER BY node_id, dmx_address')
        columns = [d[0] for d in c.description]
        devices = [dict(zip(columns, row)) for row in c.fetchall()]
        # Merge live_inventory into each device for real-time status
        for dev in devices:
            inv = self.live_inventory.get(dev.get('uid'), {})
            dev['online'] = inv.get('status', 'offline') == 'online'
            dev['status'] = inv.get('status', 'offline')
            dev['is_patched'] = inv.get('is_patched', False)
        return devices

    def delete_device(self, uid):
        """Remove a device from the database and cache."""
        db_fn = reg.get_db
        if db_fn is not None:
            try:
                conn = db_fn()
                c = conn.cursor()
                c.execute('DELETE FROM rdm_devices WHERE uid = ?', (uid,))
                c.execute('DELETE FROM rdm_personalities WHERE device_uid = ?', (uid,))
                conn.commit()
            except Exception:
                pass

        with self._lock:
            self._devices.pop(uid, None)
        self.live_inventory.pop(uid, None)
        self._emit_rdm_update()
        return {"success": True}

    # ─────────────────────────────────────────────────────────
    # Cross-Node Discovery (absorbed from rdm_service.py)
    # ─────────────────────────────────────────────────────────

    def discover_all(self):
        """Run RDM discovery on ALL online nodes."""
        if self._discovery_in_progress:
            return {'success': False, 'error': 'Discovery already in progress', 'devices': []}

        self._discovery_in_progress = True
        print("🔍 RDM: Starting discovery on all nodes...", flush=True)

        try:
            nodes = self._get_rdm_capable_nodes()
            if not nodes:
                return {'success': False, 'error': 'No RDM-capable nodes online', 'devices': []}

            all_device_uids = []
            for node in nodes:
                node_id = node.get('node_id')
                if not node_id:
                    continue
                print(f"🔍 RDM: Discovering on {node_id}...", flush=True)
                result = self.discover_devices(node_id)
                if result.get('success'):
                    devices = result.get('devices', [])
                    for dev in devices:
                        uid = dev if isinstance(dev, str) else dev.get('uid')
                        if uid:
                            all_device_uids.append(uid)

            self._last_discovery = datetime.now()
            with self._lock:
                device_list = [self._devices[uid].to_dict() for uid in all_device_uids if uid in self._devices]

            return {
                'success': True,
                'devices': device_list,
                'count': len(all_device_uids),
                'timestamp': self._last_discovery.isoformat()
            }
        finally:
            self._discovery_in_progress = False

    def get_cached_devices(self):
        """Get list of all known RDM devices (from database)."""
        return self.get_all_devices()

    # ─────────────────────────────────────────────────────────
    # UID-Based Operations (resolve uid → node_id internally)
    # ─────────────────────────────────────────────────────────

    def _get_node_for_device(self, uid):
        """Get the node that can communicate with a device."""
        dev = self._devices.get(uid)
        if dev and dev.discovered_via and self._node_manager:
            return self._node_manager.get_node(dev.discovered_via)
        # Fallback: try any RDM-capable node
        nodes = self._get_rdm_capable_nodes()
        return nodes[0] if nodes else None

    def identify_by_uid(self, uid, state=True):
        """Identify a device by UID (resolves node automatically)."""
        node = self._get_node_for_device(uid)
        if not node:
            return {'success': False, 'error': 'No node available for device'}
        return self._send_rdm_command(node.get('ip'), 'identify', {"uid": uid, "state": state})

    def get_address_by_uid(self, uid):
        """Get DMX address for a device by UID."""
        node = self._get_node_for_device(uid)
        if not node:
            return {'success': False, 'error': 'No node available for device'}
        result = self._send_rdm_command(node.get('ip'), 'get_address', {"uid": uid})
        if result and result.get('success'):
            address = result.get('address', 0)
            with self._lock:
                if uid in self._devices:
                    self._devices[uid].dmx_address = address
            return {'success': True, 'uid': uid, 'address': address}
        return {'success': False, 'error': result.get('error', 'Unknown') if result else 'No response'}

    def set_address_by_uid(self, uid, address):
        """Set DMX address for a device by UID (with conflict check)."""
        if address < 1 or address > 512:
            return {'success': False, 'error': 'Invalid address (must be 1-512)'}
        dev = self._devices.get(uid)
        if not dev:
            return {'success': False, 'error': 'Device not found in cache'}
        # Check for conflicts
        conflicts = self._check_address_conflict(uid, address, dev.dmx_footprint)
        if conflicts:
            return {'success': False, 'error': 'Address conflict', 'conflicts': conflicts}
        node = self._get_node_for_device(uid)
        if not node:
            return {'success': False, 'error': 'No node available for device'}
        result = self._send_rdm_command(node.get('ip'), 'set_address', {"uid": uid, "address": address})
        if result and result.get('success'):
            with self._lock:
                if uid in self._devices:
                    self._devices[uid].dmx_address = address
            # Also update DB
            try:
                db_fn = reg.get_db
                if db_fn is not None:
                    conn = db_fn()
                    c = conn.cursor()
                    c.execute('UPDATE rdm_devices SET dmx_address = ?, last_seen = ? WHERE uid = ?',
                             (address, datetime.now().isoformat(), uid))
                    conn.commit()
            except Exception:
                pass
            return {'success': True, 'uid': uid, 'address': address}
        return {'success': False, 'error': result.get('error', 'Unknown') if result else 'No response'}

    def get_cached_device_info(self, uid):
        """Get info for a specific device (from database)."""
        # Try in-memory cache first
        dev = self._devices.get(uid)
        if dev:
            return {'success': True, 'device': dev.to_dict()}
        # Fallback to database
        db_fn = reg.get_db
        if db_fn is None:
            return {'success': False, 'error': 'Database not available'}

        conn = db_fn()
        c = conn.cursor()
        c.execute('SELECT * FROM rdm_devices WHERE uid = ?', (uid,))
        row = c.fetchone()
        if not row:
            return {'success': False, 'error': 'Device not found'}
        columns = [d[0] for d in c.description]
        return {'success': True, 'device': dict(zip(columns, row))}

    # ─────────────────────────────────────────────────────────
    # Address Conflict Detection (absorbed from rdm_service.py)
    # ─────────────────────────────────────────────────────────

    def _check_address_conflict(self, uid, new_address, footprint):
        """Check if setting an address would cause a conflict."""
        conflicts = []
        new_end = new_address + max(footprint, 1) - 1
        with self._lock:
            for other_uid, other in self._devices.items():
                if other_uid == uid or other.dmx_address == 0:
                    continue
                other_end = other.dmx_address + max(other.dmx_footprint, 1) - 1
                if not (new_end < other.dmx_address or new_address > other_end):
                    conflicts.append({'uid': other_uid, 'address': other.dmx_address, 'footprint': other.dmx_footprint})
        return conflicts

    def _find_all_conflicts(self):
        """Find all address conflicts among cached devices."""
        conflicts = []
        devices = list(self._devices.values())
        for i, dev1 in enumerate(devices):
            if dev1.dmx_address == 0:
                continue
            end1 = dev1.dmx_address + max(dev1.dmx_footprint, 1) - 1
            for dev2 in devices[i+1:]:
                if dev2.dmx_address == 0:
                    continue
                end2 = dev2.dmx_address + max(dev2.dmx_footprint, 1) - 1
                if not (end1 < dev2.dmx_address or dev1.dmx_address > end2):
                    conflicts.append({'device1': dev1.uid, 'device2': dev2.uid, 'overlap': True})
        return conflicts

    def suggest_addresses(self):
        """Analyze current addressing and suggest optimal assignments."""
        with self._lock:
            devices = list(self._devices.values())
        if not devices:
            return {'success': True, 'suggestions': [], 'conflicts': []}

        devices.sort(key=lambda d: d.dmx_address)
        conflicts = []
        used_ranges = []

        for dev in devices:
            if dev.dmx_address == 0:
                continue
            start = dev.dmx_address
            end = start + max(dev.dmx_footprint, 1) - 1
            for other_start, other_end, other_uid in used_ranges:
                if not (end < other_start or start > other_end):
                    conflicts.append({'device1': dev.uid, 'device2': other_uid,
                                     'range1': [start, end], 'range2': [other_start, other_end]})
            used_ranges.append((start, end, dev.uid))

        suggestions = []
        next_available = 1
        for dev in devices:
            footprint = max(dev.dmx_footprint, 1)
            has_conflict = any(c['device1'] == dev.uid or c['device2'] == dev.uid for c in conflicts)
            if has_conflict or dev.dmx_address == 0:
                while True:
                    end = next_available + footprint - 1
                    if end > 512:
                        suggestions.append({'uid': dev.uid, 'current_address': dev.dmx_address,
                                           'suggested_address': 0, 'footprint': footprint, 'reason': 'no_space'})
                        break
                    slot_free = all(
                        end < o.dmx_address or next_available > o.dmx_address + max(o.dmx_footprint, 1) - 1
                        for o in devices if o.uid != dev.uid and o.dmx_address > 0
                    )
                    if slot_free:
                        suggestions.append({'uid': dev.uid, 'current_address': dev.dmx_address,
                                           'suggested_address': next_available, 'footprint': footprint,
                                           'reason': 'conflict' if has_conflict else 'unaddressed'})
                        next_available = end + 1
                        break
                    next_available += 1
            else:
                next_available = max(next_available, dev.dmx_address + footprint)

        return {
            'success': True, 'suggestions': suggestions, 'conflicts': conflicts,
            'total_devices': len(devices),
            'conflicting_devices': len(set(c['device1'] for c in conflicts).union(c['device2'] for c in conflicts))
        }

    def auto_fix_addresses(self):
        """Automatically fix all address conflicts."""
        analysis = self.suggest_addresses()
        if not analysis.get('success'):
            return analysis
        results = []
        for suggestion in analysis.get('suggestions', []):
            uid = suggestion['uid']
            new_address = suggestion['suggested_address']
            if new_address == 0:
                results.append({'uid': uid, 'success': False, 'error': 'No space available'})
                continue
            result = self.set_address_by_uid(uid, new_address)
            results.append({'uid': uid, 'success': result.get('success', False),
                           'address': new_address if result.get('success') else None,
                           'error': result.get('error')})
        return {'success': all(r['success'] for r in results), 'results': results}

    def verify_cue_readiness(self, cue_data):
        """Verify all fixtures required for a cue are ready."""
        issues = []
        warnings = []
        required_fixtures = cue_data.get('fixtures', [])
        if not self._devices:
            warnings.append({'type': 'no_rdm_devices',
                            'message': 'No RDM devices discovered - manual verification recommended'})
        for fixture in required_fixtures:
            fixture_uid = fixture.get('rdm_uid')
            expected_address = fixture.get('dmx_address')
            expected_footprint = fixture.get('footprint')
            if not fixture_uid:
                continue
            device = self._devices.get(fixture_uid)
            if not device:
                issues.append({'type': 'device_not_found', 'uid': fixture_uid,
                              'message': f'Device {fixture_uid} not found in RDM cache'})
                continue
            if expected_address and device.dmx_address != expected_address:
                issues.append({'type': 'address_mismatch', 'uid': fixture_uid,
                              'expected': expected_address, 'actual': device.dmx_address,
                              'message': f'Wrong address: expected {expected_address}, found {device.dmx_address}'})
            if expected_footprint and device.dmx_footprint != expected_footprint:
                warnings.append({'type': 'footprint_mismatch', 'uid': fixture_uid,
                                'expected': expected_footprint, 'actual': device.dmx_footprint,
                                'message': f'Footprint mismatch: expected {expected_footprint}, device reports {device.dmx_footprint}'})
        conflicts = self._find_all_conflicts()
        if conflicts:
            issues.append({'type': 'address_conflicts', 'conflicts': conflicts,
                          'message': f'Found {len(conflicts)} address conflicts'})
        ready = len(issues) == 0
        return {'ready': ready, 'issues': issues, 'warnings': warnings,
                'can_proceed': ready, 'recommendation': 'OK to proceed' if ready else 'Review issues before proceeding'}

    # ─────────────────────────────────────────────────────────
    # Internal Helpers
    # ─────────────────────────────────────────────────────────

    def _get_rdm_capable_nodes(self):
        """Get list of online nodes that support RDM."""
        if not self._node_manager:
            return []
        all_nodes = self._node_manager.get_all_nodes(include_offline=False)
        return [n for n in all_nodes if 'rdm' in n.get('caps', [])]

    def get_status(self):
        """Get consolidated RDM status."""
        all_devices = self.get_all_devices()
        with self._lock:
            return {
                'enabled': True,
                'device_count': len(all_devices),
                'last_discovery': self._last_discovery.isoformat() if self._last_discovery else None,
                'discovery_in_progress': self._discovery_in_progress,
                'live_inventory': self.live_inventory,
                'devices': all_devices
            }
