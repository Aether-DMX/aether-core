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
            "device_id": node_data.get("node_id"),
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
            "template_id": look_data.get("look_id"),
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
            "template_id": sequence_data.get("sequence_id"),
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
            "template_id": scene_data.get("scene_id"),
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
            "template_id": chase_data.get("chase_id"),
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

    # ---- Fixture Sync ----

    def sync_fixture(self, fixture_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Sync a fixture to Supabase fixture_library.
        """
        if not self._enabled or not self._client:
            return None

        fixture_record = {
            "fixture_id": fixture_data.get("fixture_id"),
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

    # ---- Bulk Sync ----

    def initial_sync(
        self,
        nodes: List[Dict] = None,
        looks: List[Dict] = None,
        sequences: List[Dict] = None,
        scenes: List[Dict] = None,
        chases: List[Dict] = None,
        fixtures: List[Dict] = None
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
                    if op["operation"] == "upsert":
                        result = self._client.table(op["table"]).upsert(
                            op["data"],
                            on_conflict=op.get("record_id", "id")
                        ).execute()
                        if result:
                            completed.append(op["id"])
                    elif op["operation"] == "delete":
                        result = self._client.table(op["table"]).delete().eq(
                            "id", op["record_id"]
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
