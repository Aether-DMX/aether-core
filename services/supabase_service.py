"""
AETHER Supabase Service
=======================
Cloud persistence, sync, and coordination layer.

IMPORTANT: Supabase is NOT the playback authority.
- Flask (aether-core) remains SSOT for live state
- Supabase exists to: persist data, sync data, authenticate users
- If Supabase goes offline, AETHER continues to function locally

Feature Gating:
- Set ENABLE_SUPABASE=false to disable all cloud functionality
- When disabled, all methods return gracefully without errors

Usage:
    from services.supabase_service import get_supabase_service

    supabase = get_supabase_service()
    if supabase.is_enabled():
        await supabase.sync_node(node_data)
"""

import os
import uuid
import json
import asyncio
import threading
from datetime import datetime, timezone
from typing import Optional, Dict, List, Any, Callable
from pathlib import Path
from functools import wraps

# Supabase import with graceful fallback
try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False
    Client = None

# ============================================================
# Configuration
# ============================================================

# Feature flag - set to "false" to disable all Supabase functionality
ENABLE_SUPABASE = os.getenv("ENABLE_SUPABASE", "true").lower() == "true"

# Supabase credentials (backend only - NEVER expose to frontend)
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")

# Installation ID - unique identifier for this AETHER installation
# Generated once, persisted locally, never regenerated automatically
HOME_DIR = os.path.expanduser("~")
SETTINGS_FILE = os.path.join(HOME_DIR, "aether-settings.json")

# Sync configuration
SYNC_RETRY_DELAY = 5  # seconds between retry attempts
MAX_SYNC_RETRIES = 3
SYNC_BATCH_SIZE = 50  # max records per batch sync

# Deterministic UUID namespace for local->cloud ID mapping
# All AETHER installations produce the same UUID for the same local ID
AETHER_UUID_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, 'https://aetherdmx.com/ids')
ID_MAP_FILE = os.path.join(HOME_DIR, "aether-id-map.json")

# Table name -> primary key column mapping (for ON CONFLICT and UUID conversion)
TABLE_PK_MAP = {
    "devices": "device_id",
    "scene_templates": "template_id",
    "fixture_library": "fixture_id",
    "conversations": "conversation_id",
    "messages": "message_id",
    "usage_events": "event_id",
    "schedules": "schedule_id",
    "groups": "group_id",
}


# ============================================================
# Installation ID Management
# ============================================================

def _load_settings() -> Dict[str, Any]:
    """Load settings from local file"""
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️ Failed to load settings: {e}")
    return {}


def _save_settings(settings: Dict[str, Any]):
    """Save settings to local file"""
    try:
        with open(SETTINGS_FILE, 'w') as f:
            json.dump(settings, f, indent=2)
    except Exception as e:
        print(f"⚠️ Failed to save settings: {e}")


def get_installation_id() -> str:
    """
    Get or generate the installation ID.

    - Generated once as UUID v4
    - Persisted locally in settings.json
    - Never regenerated automatically
    - Changing requires explicit operator action
    """
    settings = _load_settings()

    if "installation_id" not in settings:
        # Generate new installation ID
        settings["installation_id"] = str(uuid.uuid4())
        settings["installation_id_created_at"] = datetime.now(timezone.utc).isoformat()
        _save_settings(settings)
        print(f"🆔 Generated new installation ID: {settings['installation_id']}")

    return settings["installation_id"]


# ============================================================
# Pending Sync Queue
# ============================================================

class PendingSyncQueue:
    """
    Queue for operations that failed to sync to Supabase.
    Persisted to disk for recovery after restart.
    """

    def __init__(self, queue_file: str = None):
        self.queue_file = queue_file or os.path.join(HOME_DIR, "aether-pending-sync.json")
        self.lock = threading.Lock()
        self._queue: List[Dict[str, Any]] = []
        self._load()

    def _load(self):
        """Load pending operations from disk"""
        if os.path.exists(self.queue_file):
            try:
                with open(self.queue_file, 'r') as f:
                    self._queue = json.load(f)
                if self._queue:
                    print(f"📋 Loaded {len(self._queue)} pending sync operations")
            except Exception as e:
                print(f"⚠️ Failed to load pending sync queue: {e}")
                self._queue = []

    def _save(self):
        """Save pending operations to disk"""
        try:
            with open(self.queue_file, 'w') as f:
                json.dump(self._queue, f, indent=2)
        except Exception as e:
            print(f"⚠️ Failed to save pending sync queue: {e}")

    def add(self, operation: str, table: str, data: Dict[str, Any], record_id: str = None):
        """Add a pending operation"""
        with self.lock:
            self._queue.append({
                "id": str(uuid.uuid4()),
                "operation": operation,  # "upsert", "delete"
                "table": table,
                "record_id": record_id,
                "data": data,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "retry_count": 0
            })
            self._save()

    def get_pending(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get pending operations"""
        with self.lock:
            return self._queue[:limit]

    def mark_complete(self, operation_ids: List[str]):
        """Mark operations as complete (remove from queue)"""
        with self.lock:
            self._queue = [op for op in self._queue if op["id"] not in operation_ids]
            self._save()

    def mark_failed(self, operation_id: str):
        """Increment retry count for failed operation"""
        with self.lock:
            for op in self._queue:
                if op["id"] == operation_id:
                    op["retry_count"] = op.get("retry_count", 0) + 1
                    op["last_error_at"] = datetime.now(timezone.utc).isoformat()
                    break
            self._save()

    def clear(self):
        """Clear all pending operations"""
        with self.lock:
            self._queue = []
            self._save()

    def count(self) -> int:
        """Get count of pending operations"""
        with self.lock:
            return len(self._queue)


# ============================================================
# Supabase Service
# ============================================================

class SupabaseService:
    """
    Supabase integration service for AETHER.

    Responsibilities:
    - CRUD for Looks, Sequences, Modifiers, Nodes
    - User/installation management
    - Audit logging
    - Sync timestamps

    This service:
    - Converts Supabase rows ↔ internal data models
    - Never touches RenderEngine directly
    - Never blocks playback
    - All operations are async and fail-soft
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        """Singleton pattern for service instance"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._initialized = True
        self._client: Optional[Client] = None
        self._enabled = False
        self._connected = False
        self._installation_id: Optional[str] = None
        self._pending_queue = PendingSyncQueue()
        self._last_sync_at: Optional[datetime] = None
        self._sync_in_progress = False
        self._callbacks: Dict[str, List[Callable]] = {}
        self._id_map: Dict[str, str] = {}
        self._id_map_lock = threading.RLock()
        self._id_map_dirty = False

        # Initialize if enabled
        self._initialize()

    def _initialize(self):
        """Initialize Supabase client if enabled and configured"""
        if not ENABLE_SUPABASE:
            print("☁️ Supabase integration DISABLED (ENABLE_SUPABASE=false)")
            return

        if not SUPABASE_AVAILABLE:
            print("⚠️ Supabase package not installed. Run: pip install supabase")
            return

        if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
            print("⚠️ Supabase credentials not configured (SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)")
            return

        try:
            self._client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
            self._installation_id = get_installation_id()
            self._enabled = True
            self._connected = True
            print(f"☁️ Supabase connected - Installation: {self._installation_id[:8]}...")

            # Load local->cloud UUID mapping and clear stale pending ops
            self._load_id_map()
            self._clear_stale_pending()
        except Exception as e:
            print(f"⚠️ Supabase connection failed: {e}")
            self._enabled = True  # Still enabled, just not connected
            self._connected = False

    # ---- Status Methods ----

    def is_enabled(self) -> bool:
        """Check if Supabase integration is enabled"""
        return self._enabled

    def is_connected(self) -> bool:
        """Check if currently connected to Supabase"""
        return self._connected

    def get_installation_id(self) -> Optional[str]:
        """Get this installation's ID"""
        return self._installation_id

    def get_pending_count(self) -> int:
        """Get count of pending sync operations"""
        return self._pending_queue.count()

    def get_status(self) -> Dict[str, Any]:
        """Get current sync status"""
        return {
            "enabled": self._enabled,
            "connected": self._connected,
            "installation_id": self._installation_id,
            "pending_operations": self._pending_queue.count(),
            "last_sync_at": self._last_sync_at.isoformat() if self._last_sync_at else None,
            "sync_in_progress": self._sync_in_progress
        }

    # ---- ID Mapping (local string IDs -> Supabase UUIDs) ----

    def _load_id_map(self) -> None:
        """Load persisted local-ID-to-cloud-UUID map from disk."""
        with self._id_map_lock:
            if os.path.exists(ID_MAP_FILE):
                try:
                    with open(ID_MAP_FILE, 'r') as f:
                        self._id_map = json.load(f)
                    print(f"🆔 Loaded {len(self._id_map)} ID mappings from cache")
                except Exception as e:
                    print(f"⚠️ Failed to load ID map: {e}")
                    self._id_map = {}

            # Seed from cloud if cache is empty or on first run
            if not self._id_map and self._connected:
                self._populate_id_map_from_cloud()

    def _save_id_map(self) -> None:
        """Persist ID map to disk."""
        with self._id_map_lock:
            try:
                with open(ID_MAP_FILE, 'w') as f:
                    json.dump(self._id_map, f, indent=2)
                self._id_map_dirty = False
            except Exception as e:
                print(f"⚠️ Failed to save ID map: {e}")

    def _populate_id_map_from_cloud(self) -> None:
        """
        Fetch existing records from Supabase to build the reverse map.

        Handles the "lost UUID derivation" problem: the original UUID generation
        method is unrecoverable, so we query Supabase for all existing records
        and map their known local IDs back to their cloud UUIDs.
        """
        if not self._client:
            return

        count = 0
        try:
            # Fetch devices — map by hostname (matches node_id pattern e.g. "pulse-4690")
            result = self._client.table("devices").select("device_id, name, hostname").execute()
            for row in (result.data or []):
                cloud_uuid = row.get("device_id")
                # hostname is the best match for local node_id
                local_id = row.get("hostname")
                if local_id and cloud_uuid:
                    self._id_map[local_id] = cloud_uuid
                    count += 1

            # Fetch scene_templates — map by name (best available reverse key)
            result = self._client.table("scene_templates").select("template_id, name, mood").execute()
            for row in (result.data or []):
                cloud_uuid = row.get("template_id")
                name = row.get("name")
                if name and cloud_uuid:
                    self._id_map[name] = cloud_uuid
                    count += 1

            # Fetch fixture_library — map by model name
            result = self._client.table("fixture_library").select("fixture_id, model").execute()
            for row in (result.data or []):
                cloud_uuid = row.get("fixture_id")
                model = row.get("model")
                if model and cloud_uuid:
                    self._id_map[model] = cloud_uuid
                    count += 1

            if count > 0:
                print(f"🆔 Populated {count} ID mappings from Supabase")
                self._save_id_map()

        except Exception as e:
            print(f"⚠️ Failed to populate ID map from cloud: {e}")

    def _to_cloud_uuid(self, local_id: str) -> str:
        """
        Convert a local string ID to a valid UUID for Supabase.

        Strategy:
        1. If already a valid UUID, return as-is
        2. Check in-memory cache (populated from disk + cloud on startup)
        3. Generate deterministic UUID v5 from AETHER namespace
        4. Cache the new mapping
        """
        if not local_id:
            return str(uuid.uuid4())

        # Already a valid UUID? Return as-is
        try:
            uuid.UUID(local_id)
            return local_id
        except (ValueError, AttributeError):
            pass

        # Check cache
        if local_id in self._id_map:
            return self._id_map[local_id]

        # Generate deterministic UUID v5
        cloud_uuid = str(uuid.uuid5(AETHER_UUID_NAMESPACE, local_id))
        self._id_map[local_id] = cloud_uuid
        self._id_map_dirty = True

        return cloud_uuid

    def _clear_stale_pending(self) -> int:
        """
        Remove pending operations that have exceeded max retries or contain
        stale data (raw string IDs that predate the UUID fix).
        Called once at startup.
        """
        count = self._pending_queue.count()
        if count > 0:
            # Clear all existing pending ops — they contain raw string IDs
            # in their data payloads and cannot be retried successfully
            self._pending_queue.clear()
            print(f"🧹 Cleared {count} stale pending sync operations")
        return count

    # ---- Core Sync Methods ----

    async def _execute_with_retry(
        self,
        operation: Callable,
        table: str,
        data: Dict[str, Any] = None,
        record_id: str = None,
        op_type: str = "upsert"
    ) -> Optional[Dict[str, Any]]:
        """
        Execute a Supabase operation with retry and fallback.

        - If Supabase fails, queue operation for later retry
        - Never raises exceptions to caller
        - Returns result or None
        """
        if not self._enabled or not self._client:
            return None

        try:
            result = await asyncio.get_event_loop().run_in_executor(None, operation)
            return result
        except Exception as e:
            print(f"⚠️ Supabase {op_type} failed for {table}: {e}")
            self._connected = False

            # Queue for retry if data provided
            if data:
                self._pending_queue.add(op_type, table, data, record_id)

            return None

    def _sync_execute(
        self,
        operation: Callable,
        table: str,
        data: Dict[str, Any] = None,
        record_id: str = None,
        op_type: str = "upsert"
    ) -> Optional[Dict[str, Any]]:
        """Synchronous wrapper for execute with retry"""
        if not self._enabled or not self._client:
            return None

        try:
            result = operation()
            self._connected = True
            if self._id_map_dirty:
                self._save_id_map()
            return result
        except Exception as e:
            print(f"⚠️ Supabase {op_type} failed for {table}: {e}")
            self._connected = False

            if data:
                self._pending_queue.add(op_type, table, data, record_id)

            return None

    # ---- Node/Device Sync ----

    def sync_node(self, node_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Sync a node to Supabase devices table.

        Maps SQLite nodes → Supabase devices
        """
        if not self._enabled or not self._client:
            return None

        device_record = {
            "device_id": self._to_cloud_uuid(node_data.get("node_id")),
            "name": node_data.get("name", "Unknown Node"),
            "hostname": node_data.get("hostname"),
            "mac_address": node_data.get("mac"),
            "ip_address": node_data.get("ip"),
            "version": node_data.get("firmware"),
            "last_seen_at": node_data.get("last_seen") or datetime.now(timezone.utc).isoformat(),
            "is_online": node_data.get("status") == "online",
            "settings": json.dumps({
                "universe": node_data.get("universe", 1),
                "channel_start": node_data.get("channel_start", 1),
                "channel_end": node_data.get("channel_end", 512),
                "slice_mode": node_data.get("slice_mode", "zero_outside"),
                "mode": node_data.get("mode", "output"),
                "type": node_data.get("type", "wifi")
            })
        }

        def upsert():
            return self._client.table("devices").upsert(
                device_record,
                on_conflict="device_id"
            ).execute()

        return self._sync_execute(upsert, "devices", device_record, node_data.get("node_id"))

    # ---- Look Sync ----

    def sync_look(self, look_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Sync a look to Supabase.

        Uses scene_templates table with type='look'
        """
        if not self._enabled or not self._client:
            return None

        template_record = {
            "template_id": self._to_cloud_uuid(look_data.get("look_id")),
            "name": look_data.get("name"),
            "description": look_data.get("description", ""),
            "mood": "look",  # Use mood field to identify type
            "template_data": json.dumps({
                "type": "look",
                "channels": look_data.get("channels"),
                "modifiers": look_data.get("modifiers", []),
                "fade_ms": look_data.get("fade_ms", 0),
                "color": look_data.get("color"),
                "icon": look_data.get("icon"),
                "installation_id": self._installation_id
            }),
            "is_official": False
        }

        def upsert():
            return self._client.table("scene_templates").upsert(
                template_record,
                on_conflict="template_id"
            ).execute()

        return self._sync_execute(upsert, "scene_templates", template_record, look_data.get("look_id"))

    # ---- Sequence Sync ----

    def sync_sequence(self, sequence_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Sync a sequence to Supabase.

        Uses scene_templates table with type='sequence'
        """
        if not self._enabled or not self._client:
            return None

        template_record = {
            "template_id": self._to_cloud_uuid(sequence_data.get("sequence_id")),
            "name": sequence_data.get("name"),
            "description": sequence_data.get("description", ""),
            "mood": "sequence",  # Use mood field to identify type
            "template_data": json.dumps({
                "type": "sequence",
                "steps": sequence_data.get("steps"),
                "bpm": sequence_data.get("bpm", 120),
                "loop": sequence_data.get("loop", True),
                "color": sequence_data.get("color"),
                "installation_id": self._installation_id
            }),
            "is_official": False
        }

        def upsert():
            return self._client.table("scene_templates").upsert(
                template_record,
                on_conflict="template_id"
            ).execute()

        return self._sync_execute(upsert, "scene_templates", template_record, sequence_data.get("sequence_id"))

    # ---- Scene Sync ----

    def sync_scene(self, scene_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Sync a scene to Supabase.

        Uses scene_templates table with type='scene'
        """
        if not self._enabled or not self._client:
            return None

        template_record = {
            "template_id": self._to_cloud_uuid(scene_data.get("scene_id")),
            "name": scene_data.get("name"),
            "description": scene_data.get("description", ""),
            "mood": "scene",
            "template_data": json.dumps({
                "type": "scene",
                "channels": scene_data.get("channels"),
                "universe": scene_data.get("universe", 1),
                "fade_ms": scene_data.get("fade_ms", 500),
                "curve": scene_data.get("curve", "linear"),
                "color": scene_data.get("color"),
                "icon": scene_data.get("icon"),
                "installation_id": self._installation_id
            }),
            "is_official": False
        }

        def upsert():
            return self._client.table("scene_templates").upsert(
                template_record,
                on_conflict="template_id"
            ).execute()

        return self._sync_execute(upsert, "scene_templates", template_record, scene_data.get("scene_id"))

    # ---- Chase Sync ----

    def sync_chase(self, chase_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Sync a chase to Supabase.

        Uses scene_templates table with type='chase'
        """
        if not self._enabled or not self._client:
            return None

        template_record = {
            "template_id": self._to_cloud_uuid(chase_data.get("chase_id")),
            "name": chase_data.get("name"),
            "description": chase_data.get("description", ""),
            "mood": "chase",
            "template_data": json.dumps({
                "type": "chase",
                "steps": chase_data.get("steps"),
                "bpm": chase_data.get("bpm", 120),
                "loop": chase_data.get("loop", True),
                "fade_ms": chase_data.get("fade_ms", 0),
                "universe": chase_data.get("universe", 1),
                "color": chase_data.get("color"),
                "installation_id": self._installation_id
            }),
            "is_official": False
        }

        def upsert():
            return self._client.table("scene_templates").upsert(
                template_record,
                on_conflict="template_id"
            ).execute()

        return self._sync_execute(upsert, "scene_templates", template_record, chase_data.get("chase_id"))

    # ---- Show Sync ----

    def sync_show(self, show_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Sync a show to Supabase.

        Uses scene_templates table with mood='show'
        """
        if not self._enabled or not self._client:
            return None

        timeline = show_data.get("timeline")
        if isinstance(timeline, str):
            timeline = json.loads(timeline)

        template_record = {
            "template_id": self._to_cloud_uuid(show_data.get("show_id")),
            "name": show_data.get("name"),
            "description": show_data.get("description", ""),
            "mood": "show",
            "template_data": json.dumps({
                "type": "show",
                "timeline": timeline,
                "duration_ms": show_data.get("duration_ms", 0),
                "installation_id": self._installation_id
            }),
            "is_official": False
        }

        def upsert():
            return self._client.table("scene_templates").upsert(
                template_record,
                on_conflict="template_id"
            ).execute()

        return self._sync_execute(upsert, "scene_templates", template_record, show_data.get("show_id"))

    # ---- Schedule Sync ----

    def sync_schedule(self, schedule_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Sync a schedule to Supabase schedules table.
        """
        if not self._enabled or not self._client:
            return None

        schedule_record = {
            "schedule_id": self._to_cloud_uuid(schedule_data.get("schedule_id")),
            "installation_id": self._installation_id,
            "name": schedule_data.get("name"),
            "cron": schedule_data.get("cron"),
            "action_type": schedule_data.get("action_type"),
            "action_id": schedule_data.get("action_id"),
            "action_params": json.dumps(schedule_data.get("action_params", {})),
            "enabled": schedule_data.get("enabled", True),
            "last_run": schedule_data.get("last_run"),
            "next_run": schedule_data.get("next_run"),
        }

        def upsert():
            return self._client.table("schedules").upsert(
                schedule_record,
                on_conflict="schedule_id"
            ).execute()

        return self._sync_execute(upsert, "schedules", schedule_record, schedule_data.get("schedule_id"))

    # ---- Group Sync ----

    def sync_group(self, group_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Sync a group to Supabase groups table.
        """
        if not self._enabled or not self._client:
            return None

        channels = group_data.get("channels", [])
        if isinstance(channels, str):
            channels = json.loads(channels)

        group_record = {
            "group_id": self._to_cloud_uuid(group_data.get("group_id")),
            "installation_id": self._installation_id,
            "name": group_data.get("name"),
            "universe": group_data.get("universe", 1),
            "channels": json.dumps(channels),
            "color": group_data.get("color", "#8b5cf6"),
        }

        def upsert():
            return self._client.table("groups").upsert(
                group_record,
                on_conflict="group_id"
            ).execute()

        return self._sync_execute(upsert, "groups", group_record, group_data.get("group_id"))

    # ---- Cloud Delete Helpers ----

    def delete_from_cloud(self, table: str, pk_column: str, local_id: str) -> Optional[Dict[str, Any]]:
        """Delete a record from Supabase by its local ID."""
        if not self._enabled or not self._client:
            return None

        cloud_uuid = self._to_cloud_uuid(local_id)

        def delete_op():
            return self._client.table(table).delete().eq(pk_column, cloud_uuid).execute()

        return self._sync_execute(delete_op, table, record_id=local_id, op_type="delete")

    def delete_show(self, show_id: str) -> Optional[Dict[str, Any]]:
        """Delete a show from Supabase."""
        return self.delete_from_cloud("scene_templates", "template_id", show_id)

    def delete_schedule(self, schedule_id: str) -> Optional[Dict[str, Any]]:
        """Delete a schedule from Supabase."""
        return self.delete_from_cloud("schedules", "schedule_id", schedule_id)

    def delete_group(self, group_id: str) -> Optional[Dict[str, Any]]:
        """Delete a group from Supabase."""
        return self.delete_from_cloud("groups", "group_id", group_id)

    # ---- Fixture Sync ----

    def sync_fixture(self, fixture_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Sync a fixture to Supabase fixture_library.
        """
        if not self._enabled or not self._client:
            return None

        fixture_record = {
            "fixture_id": self._to_cloud_uuid(fixture_data.get("fixture_id")),
            "manufacturer": fixture_data.get("manufacturer", "Unknown"),
            "model": fixture_data.get("model") or fixture_data.get("name", "Unknown"),
            "category": fixture_data.get("type", "generic"),
            "modes": json.dumps({
                "default": {
                    "name": "Default",
                    "channel_count": fixture_data.get("channel_count", 1),
                    "channel_map": fixture_data.get("channel_map"),
                    "start_channel": fixture_data.get("start_channel", 1),
                    "universe": fixture_data.get("universe", 1)
                }
            }),
            "verified": False
        }

        def upsert():
            return self._client.table("fixture_library").upsert(
                fixture_record,
                on_conflict="fixture_id"
            ).execute()

        return self._sync_execute(upsert, "fixture_library", fixture_record, fixture_data.get("fixture_id"))

    # ---- Audit Logging ----

    def log_event(self, event_type: str, event_data: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
        """
        Log an audit event to Supabase.

        Use for: playback events, user actions, system events
        NOT for: live DMX frames, real-time data
        """
        if not self._enabled or not self._client:
            return None

        event_record = {
            "event_type": event_type,
            "device_id": self._installation_id,
            "event_data": json.dumps(event_data or {}),
            "created_at": datetime.now(timezone.utc).isoformat()
        }

        def insert():
            return self._client.table("usage_events").insert(event_record).execute()

        # Don't queue audit events - they're not critical
        try:
            return insert()
        except Exception as e:
            print(f"⚠️ Audit log failed: {e}")
            return None

    # ---- AI Conversation Logging ----

    def log_conversation(self, session_id: str, messages: List[Dict], metadata: Dict = None) -> Optional[Dict]:
        """
        Upsert a conversation and its messages to Supabase.

        Args:
            session_id: AI session ID from Node.js
            messages: Array of {role, content, tokens_used?, tool_calls?}
            metadata: Optional {model, title}
        """
        if not self._enabled or not self._client:
            return None

        meta = metadata or {}
        conversation_id = self._to_cloud_uuid(f"conv-{session_id}")

        # Use first user message as title if none provided
        title = meta.get("title")
        if not title:
            for msg in messages:
                if msg.get("role") == "user" and msg.get("content"):
                    content = msg["content"]
                    title = (content[:80] + "...") if len(content) > 80 else content
                    break
        title = title or f"Session {session_id}"

        conversation_record = {
            "conversation_id": conversation_id,
            "device_id": self._installation_id,
            "title": title,
            "context": json.dumps({
                "model": meta.get("model", "unknown"),
                "session_id": session_id,
                "installation_id": self._installation_id,
            }),
            "message_count": len(messages),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

        def upsert():
            return self._client.table("conversations").upsert(
                conversation_record,
                on_conflict="conversation_id"
            ).execute()

        result = self._sync_execute(upsert, "conversations", conversation_record, conversation_id)

        # Insert individual messages (non-critical, don't queue on failure)
        if result and messages:
            for msg in messages:
                self.log_message(
                    conversation_id=conversation_id,
                    role=msg.get("role", "user"),
                    content=msg.get("content", ""),
                    tokens_used=msg.get("tokens_used"),
                    tool_calls=msg.get("tool_calls"),
                )

        return result

    def log_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        tokens_used: int = None,
        tool_calls: Any = None,
    ) -> Optional[Dict]:
        """
        Insert a single message into Supabase.

        Non-critical: failures are silently dropped (not queued).
        """
        if not self._enabled or not self._client:
            return None

        # Handle complex content objects (Claude API returns arrays)
        if isinstance(content, (list, dict)):
            content = json.dumps(content)

        message_record = {
            "message_id": str(uuid.uuid4()),
            "conversation_id": conversation_id,
            "role": role,
            "content": content[:10000] if content else "",
            "tokens_used": tokens_used,
            "tool_calls": json.dumps(tool_calls) if tool_calls else None,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        def insert():
            return self._client.table("messages").insert(message_record).execute()

        try:
            return insert()
        except Exception as e:
            print(f"⚠️ Message log failed: {e}")
            return None

    # ---- AI Learning Pipeline ----

    def log_learning(self, learning_data: Dict[str, Any]) -> Optional[Dict]:
        """
        Log an AI learning event to Supabase ai_learnings table.

        Args:
            learning_data: {
                category: str,        # Tool category (e.g., "scene", "chase")
                pattern_key: str,     # Action pattern (e.g., "scene.create")
                pattern_data: dict,   # Full context: user_input, tool_input, result
                success_rate: float,  # 1.0 or 0.0
            }
        """
        if not self._enabled or not self._client:
            return None

        record = {
            "learning_id": str(uuid.uuid4()),
            "category": learning_data.get("category", "unknown"),
            "pattern_key": learning_data.get("pattern_key", "unknown"),
            "pattern_data": json.dumps(learning_data.get("pattern_data", {})),
            "success_rate": learning_data.get("success_rate"),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        def insert():
            return self._client.table("ai_learnings").insert(record).execute()

        try:
            return insert()
        except Exception as e:
            print(f"⚠️ Learning log failed: {e}")
            return None

    def log_feedback(self, feedback_data: Dict[str, Any]) -> Optional[Dict]:
        """
        Log user feedback (thumbs up/down) on an AI response.

        Args:
            feedback_data: {
                message_id: str,     # UUID of the message being rated
                rating: int,         # 1 (thumbs up) or -1 (thumbs down)
                feedback_text: str,  # Optional comment
            }
        """
        if not self._enabled or not self._client:
            return None

        record = {
            "feedback_id": str(uuid.uuid4()),
            "message_id": feedback_data.get("message_id"),
            "rating": feedback_data.get("rating"),
            "feedback_text": feedback_data.get("feedback_text", ""),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        def insert():
            return self._client.table("ai_feedback").insert(record).execute()

        try:
            return insert()
        except Exception as e:
            print(f"⚠️ Feedback log failed: {e}")
            return None

    def get_learned_patterns(self, category: str = None, limit: int = 20) -> List[Dict]:
        """Fetch recent AI learnings, optionally filtered by category."""
        if not self._enabled or not self._client:
            return []

        try:
            query = self._client.table("ai_learnings").select("*").order(
                "created_at", desc=True
            ).limit(limit)

            if category:
                query = query.eq("category", category)

            result = query.execute()
            return result.data or []
        except Exception as e:
            print(f"⚠️ Failed to fetch learnings: {e}")
            return []

    # ---- Bulk Sync ----

    def initial_sync(
        self,
        nodes: List[Dict] = None,
        looks: List[Dict] = None,
        sequences: List[Dict] = None,
        scenes: List[Dict] = None,
        chases: List[Dict] = None,
        fixtures: List[Dict] = None,
        shows: List[Dict] = None,
        schedules: List[Dict] = None,
        groups: List[Dict] = None
    ) -> Dict[str, int]:
        """
        Perform initial one-way sync from SQLite to Supabase.

        Called on Flask startup when Supabase is reachable.
        Returns count of synced records per type.
        """
        if not self._enabled or not self._client:
            return {"error": "Supabase not enabled or connected"}

        if self._sync_in_progress:
            return {"error": "Sync already in progress"}

        self._sync_in_progress = True
        results = {
            "nodes": 0,
            "looks": 0,
            "sequences": 0,
            "scenes": 0,
            "chases": 0,
            "fixtures": 0,
            "shows": 0,
            "schedules": 0,
            "groups": 0,
            "errors": 0
        }

        try:
            print("☁️ Starting initial sync to Supabase...")

            # Sync nodes
            if nodes:
                for node in nodes:
                    if self.sync_node(node):
                        results["nodes"] += 1
                    else:
                        results["errors"] += 1

            # Sync looks
            if looks:
                for look in looks:
                    if self.sync_look(look):
                        results["looks"] += 1
                    else:
                        results["errors"] += 1

            # Sync sequences
            if sequences:
                for seq in sequences:
                    if self.sync_sequence(seq):
                        results["sequences"] += 1
                    else:
                        results["errors"] += 1

            # Sync scenes
            if scenes:
                for scene in scenes:
                    if self.sync_scene(scene):
                        results["scenes"] += 1
                    else:
                        results["errors"] += 1

            # Sync chases
            if chases:
                for chase in chases:
                    if self.sync_chase(chase):
                        results["chases"] += 1
                    else:
                        results["errors"] += 1

            # Sync fixtures
            if fixtures:
                for fixture in fixtures:
                    if self.sync_fixture(fixture):
                        results["fixtures"] += 1
                    else:
                        results["errors"] += 1

            # Sync shows
            if shows:
                for show in shows:
                    if self.sync_show(show):
                        results["shows"] += 1
                    else:
                        results["errors"] += 1

            # Sync schedules
            if schedules:
                for schedule in schedules:
                    if self.sync_schedule(schedule):
                        results["schedules"] += 1
                    else:
                        results["errors"] += 1

            # Sync groups
            if groups:
                for group in groups:
                    if self.sync_group(group):
                        results["groups"] += 1
                    else:
                        results["errors"] += 1

            self._last_sync_at = datetime.now(timezone.utc)

            total = sum(v for k, v in results.items() if k != "errors")
            print(f"☁️ Initial sync complete: {total} records synced, {results['errors']} errors")

            # Log sync event
            self.log_event("initial_sync", results)

        except Exception as e:
            print(f"⚠️ Initial sync failed: {e}")
            results["error"] = str(e)
        finally:
            self._sync_in_progress = False

        return results

    # ---- Retry Pending Operations ----

    def retry_pending(self) -> Dict[str, int]:
        """
        Retry pending sync operations.

        Called periodically or when connection is restored.
        """
        if not self._enabled or not self._client:
            return {"error": "Supabase not enabled"}

        if self._sync_in_progress:
            return {"skipped": "Sync in progress"}

        pending = self._pending_queue.get_pending(SYNC_BATCH_SIZE)
        if not pending:
            return {"pending": 0}

        self._sync_in_progress = True
        completed = []
        failed = 0

        try:
            for op in pending:
                try:
                    table = op["table"]
                    conflict_col = TABLE_PK_MAP.get(table, "id")

                    if op["operation"] == "upsert":
                        data = op["data"]
                        # Convert PK field to UUID if it's a raw string ID
                        if conflict_col in data and data[conflict_col]:
                            data[conflict_col] = self._to_cloud_uuid(data[conflict_col])

                        result = self._client.table(table).upsert(
                            data,
                            on_conflict=conflict_col
                        ).execute()
                        if result:
                            completed.append(op["id"])
                    elif op["operation"] == "delete":
                        record_id = op.get("record_id", "")
                        if record_id:
                            record_id = self._to_cloud_uuid(record_id)
                        result = self._client.table(table).delete().eq(
                            conflict_col, record_id
                        ).execute()
                        if result:
                            completed.append(op["id"])
                except Exception as e:
                    print(f"⚠️ Retry failed for {op['table']}: {e}")
                    self._pending_queue.mark_failed(op["id"])
                    failed += 1

            if completed:
                self._pending_queue.mark_complete(completed)
                self._connected = True

        finally:
            self._sync_in_progress = False

        return {
            "completed": len(completed),
            "failed": failed,
            "remaining": self._pending_queue.count()
        }

    # ---- Fetch from Cloud ----

    def fetch_looks(self) -> List[Dict[str, Any]]:
        """Fetch looks from Supabase for this installation"""
        if not self._enabled or not self._client:
            return []

        try:
            result = self._client.table("scene_templates").select("*").eq(
                "mood", "look"
            ).execute()

            looks = []
            for row in result.data or []:
                template_data = json.loads(row.get("template_data", "{}"))
                if template_data.get("installation_id") == self._installation_id:
                    looks.append({
                        "look_id": row.get("template_id"),
                        "name": row.get("name"),
                        "description": row.get("description"),
                        "channels": template_data.get("channels"),
                        "modifiers": template_data.get("modifiers", []),
                        "fade_ms": template_data.get("fade_ms", 0),
                        "color": template_data.get("color"),
                        "icon": template_data.get("icon")
                    })

            return looks
        except Exception as e:
            print(f"⚠️ Failed to fetch looks: {e}")
            return []

    def fetch_sequences(self) -> List[Dict[str, Any]]:
        """Fetch sequences from Supabase for this installation"""
        if not self._enabled or not self._client:
            return []

        try:
            result = self._client.table("scene_templates").select("*").eq(
                "mood", "sequence"
            ).execute()

            sequences = []
            for row in result.data or []:
                template_data = json.loads(row.get("template_data", "{}"))
                if template_data.get("installation_id") == self._installation_id:
                    sequences.append({
                        "sequence_id": row.get("template_id"),
                        "name": row.get("name"),
                        "description": row.get("description"),
                        "steps": template_data.get("steps"),
                        "bpm": template_data.get("bpm", 120),
                        "loop": template_data.get("loop", True),
                        "color": template_data.get("color")
                    })

            return sequences
        except Exception as e:
            print(f"⚠️ Failed to fetch sequences: {e}")
            return []


# ============================================================
# Module-level Singleton Access
# ============================================================

_service_instance: Optional[SupabaseService] = None

def get_supabase_service() -> SupabaseService:
    """Get the singleton Supabase service instance"""
    global _service_instance
    if _service_instance is None:
        _service_instance = SupabaseService()
    return _service_instance


# ============================================================
# Convenience Decorators
# ============================================================

def sync_to_cloud(entity_type: str):
    """
    Decorator to automatically sync entity changes to Supabase.

    Usage:
        @sync_to_cloud("look")
        def create_look(look_data):
            # ... create in SQLite ...
            return look_data
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)

            # Async sync to cloud (non-blocking)
            if result:
                service = get_supabase_service()
                if service.is_enabled():
                    try:
                        if entity_type == "look":
                            service.sync_look(result)
                        elif entity_type == "sequence":
                            service.sync_sequence(result)
                        elif entity_type == "scene":
                            service.sync_scene(result)
                        elif entity_type == "chase":
                            service.sync_chase(result)
                        elif entity_type == "node":
                            service.sync_node(result)
                        elif entity_type == "fixture":
                            service.sync_fixture(result)
                        elif entity_type == "show":
                            service.sync_show(result)
                        elif entity_type == "schedule":
                            service.sync_schedule(result)
                        elif entity_type == "group":
                            service.sync_group(result)
                    except Exception as e:
                        print(f"⚠️ Cloud sync failed for {entity_type}: {e}")

            return result
        return wrapper
    return decorator


# ============================================================
# CLI for Testing
# ============================================================

if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("AETHER Supabase Service Test")
    print("=" * 60)

    service = get_supabase_service()
    status = service.get_status()

    print(f"\nStatus:")
    for key, value in status.items():
        print(f"  {key}: {value}")

    if service.is_enabled() and service.is_connected():
        print("\n✅ Supabase service is ready")

        # Test audit log
        result = service.log_event("test_event", {"message": "Service test"})
        if result:
            print("✅ Audit log test passed")
        else:
            print("⚠️ Audit log test failed")
    else:
        print("\n⚠️ Supabase service not connected")
        print("Check environment variables:")
        print("  ENABLE_SUPABASE=true")
        print("  SUPABASE_URL=<your-url>")
        print("  SUPABASE_SERVICE_ROLE_KEY=<your-key>")
