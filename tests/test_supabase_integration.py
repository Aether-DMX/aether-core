"""
AETHER Supabase Integration Tests
=================================
Tests for cloud sync functionality.

Run with: pytest tests/test_supabase_integration.py -v
"""

import os
import sys
import json
import uuid
import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timezone

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================
# Test Fixtures
# ============================================================

@pytest.fixture
def mock_supabase_client():
    """Create a mock Supabase client"""
    client = MagicMock()

    # Mock table operations
    table_mock = MagicMock()
    table_mock.upsert.return_value.execute.return_value = MagicMock(data=[{"id": "test"}])
    table_mock.insert.return_value.execute.return_value = MagicMock(data=[{"id": "test"}])
    table_mock.select.return_value.eq.return_value.execute.return_value = MagicMock(data=[])

    client.table.return_value = table_mock
    return client


@pytest.fixture
def sample_node():
    """Sample node data for testing"""
    return {
        "node_id": f"node_{uuid.uuid4().hex[:8]}",
        "name": "Test Node",
        "hostname": "test-node",
        "mac": "AA:BB:CC:DD:EE:FF",
        "ip": "192.168.1.100",
        "universe": 1,
        "channel_start": 1,
        "channel_end": 4,
        "slice_mode": "zero_outside",
        "mode": "output",
        "type": "wifi",
        "firmware": "1.0.0",
        "status": "online",
        "last_seen": datetime.now(timezone.utc).isoformat()
    }


@pytest.fixture
def sample_look():
    """Sample look data for testing"""
    return {
        "look_id": f"look_{uuid.uuid4().hex[:8]}",
        "name": "Test Look",
        "channels": {"1:1": 255, "1:2": 128, "1:3": 64, "1:4": 32},
        "modifiers": [],
        "fade_ms": 500,
        "color": "blue",
        "icon": "lightbulb",
        "description": "A test look"
    }


@pytest.fixture
def sample_sequence():
    """Sample sequence data for testing"""
    return {
        "sequence_id": f"sequence_{uuid.uuid4().hex[:8]}",
        "name": "Test Sequence",
        "steps": [
            {"channels": {"1:1": 255}, "fade_ms": 100, "hold_ms": 500},
            {"channels": {"1:1": 0}, "fade_ms": 100, "hold_ms": 500}
        ],
        "bpm": 120,
        "loop": True,
        "color": "green",
        "description": "A test sequence"
    }


@pytest.fixture
def sample_scene():
    """Sample scene data for testing"""
    return {
        "scene_id": f"scene_{uuid.uuid4().hex[:8]}",
        "name": "Test Scene",
        "channels": {"1": 255, "2": 128},
        "universe": 1,
        "fade_ms": 500,
        "curve": "linear",
        "color": "#3b82f6",
        "icon": "lightbulb",
        "description": "A test scene"
    }


@pytest.fixture
def sample_chase():
    """Sample chase data for testing"""
    return {
        "chase_id": f"chase_{uuid.uuid4().hex[:8]}",
        "name": "Test Chase",
        "steps": [
            {"channels": {"1": 255}, "fade_ms": 50, "hold_ms": 100},
            {"channels": {"2": 255}, "fade_ms": 50, "hold_ms": 100}
        ],
        "bpm": 120,
        "loop": True,
        "universe": 1,
        "fade_ms": 0,
        "color": "#10b981"
    }


# ============================================================
# Installation ID Tests
# ============================================================

class TestInstallationId:
    """Tests for installation ID management"""

    def test_installation_id_generation(self, tmp_path):
        """Test that installation ID is generated correctly"""
        # Mock the settings file path
        settings_file = tmp_path / "aether-settings.json"

        with patch('services.supabase_service.SETTINGS_FILE', str(settings_file)):
            from services.supabase_service import get_installation_id, _load_settings

            # First call should generate new ID
            install_id = get_installation_id()

            assert install_id is not None
            assert len(install_id) == 36  # UUID format
            assert '-' in install_id

            # Verify it's persisted
            settings = _load_settings()
            assert settings.get("installation_id") == install_id

    def test_installation_id_persistence(self, tmp_path):
        """Test that installation ID persists across calls"""
        settings_file = tmp_path / "aether-settings.json"

        with patch('services.supabase_service.SETTINGS_FILE', str(settings_file)):
            from services.supabase_service import get_installation_id

            # Get ID twice
            id1 = get_installation_id()
            id2 = get_installation_id()

            # Should be the same
            assert id1 == id2


# ============================================================
# Pending Sync Queue Tests
# ============================================================

class TestPendingSyncQueue:
    """Tests for the pending sync queue"""

    def test_queue_add_and_get(self, tmp_path):
        """Test adding and retrieving from queue"""
        queue_file = tmp_path / "pending-sync.json"

        from services.supabase_service import PendingSyncQueue

        queue = PendingSyncQueue(str(queue_file))

        # Add an operation
        queue.add("upsert", "devices", {"id": "test123"}, "test123")

        # Get pending operations
        pending = queue.get_pending()

        assert len(pending) == 1
        assert pending[0]["operation"] == "upsert"
        assert pending[0]["table"] == "devices"
        assert pending[0]["data"]["id"] == "test123"

    def test_queue_mark_complete(self, tmp_path):
        """Test marking operations as complete"""
        queue_file = tmp_path / "pending-sync.json"

        from services.supabase_service import PendingSyncQueue

        queue = PendingSyncQueue(str(queue_file))

        # Add operations
        queue.add("upsert", "devices", {"id": "test1"}, "test1")
        queue.add("upsert", "devices", {"id": "test2"}, "test2")

        # Get pending and mark first as complete
        pending = queue.get_pending()
        queue.mark_complete([pending[0]["id"]])

        # Should only have one left
        remaining = queue.get_pending()
        assert len(remaining) == 1
        assert remaining[0]["data"]["id"] == "test2"

    def test_queue_persistence(self, tmp_path):
        """Test queue persists to disk"""
        queue_file = tmp_path / "pending-sync.json"

        from services.supabase_service import PendingSyncQueue

        # Create queue and add item
        queue1 = PendingSyncQueue(str(queue_file))
        queue1.add("upsert", "devices", {"id": "persist_test"}, "persist_test")

        # Create new queue instance (simulates restart)
        queue2 = PendingSyncQueue(str(queue_file))
        pending = queue2.get_pending()

        assert len(pending) == 1
        assert pending[0]["data"]["id"] == "persist_test"


# ============================================================
# Supabase Service Tests
# ============================================================

class TestSupabaseService:
    """Tests for the Supabase service"""

    def test_service_disabled_when_env_false(self):
        """Test service is disabled when ENABLE_SUPABASE=false"""
        with patch.dict(os.environ, {"ENABLE_SUPABASE": "false"}):
            # Need to reload the module to pick up env change
            import importlib
            import services.supabase_service as supa_module

            # Create fresh instance
            with patch.object(supa_module.SupabaseService, '_instance', None):
                service = supa_module.SupabaseService()
                assert service.is_enabled() == False

    def test_service_returns_none_when_disabled(self, sample_node):
        """Test sync methods return None when disabled"""
        with patch.dict(os.environ, {"ENABLE_SUPABASE": "false"}):
            import importlib
            import services.supabase_service as supa_module

            with patch.object(supa_module.SupabaseService, '_instance', None):
                service = supa_module.SupabaseService()

                result = service.sync_node(sample_node)
                assert result is None

    def test_get_status(self, tmp_path):
        """Test get_status returns correct structure"""
        with patch.dict(os.environ, {"ENABLE_SUPABASE": "false"}):
            import services.supabase_service as supa_module

            with patch.object(supa_module.SupabaseService, '_instance', None):
                service = supa_module.SupabaseService()
                status = service.get_status()

                assert "enabled" in status
                assert "connected" in status
                assert "installation_id" in status
                assert "pending_operations" in status
                assert "last_sync_at" in status
                assert "sync_in_progress" in status


# ============================================================
# Sync Method Tests (with mocked client)
# ============================================================

class TestSyncMethods:
    """Tests for individual sync methods"""

    def test_sync_node_creates_correct_record(self, mock_supabase_client, sample_node):
        """Test sync_node creates correct device record"""
        import services.supabase_service as supa_module

        with patch.object(supa_module.SupabaseService, '_instance', None):
            service = supa_module.SupabaseService()
            service._client = mock_supabase_client
            service._enabled = True
            service._connected = True
            service._installation_id = "test-install-id"

            service.sync_node(sample_node)

            # Verify table was called with devices
            mock_supabase_client.table.assert_called_with("devices")

    def test_sync_look_creates_correct_record(self, mock_supabase_client, sample_look):
        """Test sync_look creates correct scene_template record"""
        import services.supabase_service as supa_module

        with patch.object(supa_module.SupabaseService, '_instance', None):
            service = supa_module.SupabaseService()
            service._client = mock_supabase_client
            service._enabled = True
            service._connected = True
            service._installation_id = "test-install-id"

            service.sync_look(sample_look)

            # Verify table was called with scene_templates
            mock_supabase_client.table.assert_called_with("scene_templates")

    def test_sync_sequence_creates_correct_record(self, mock_supabase_client, sample_sequence):
        """Test sync_sequence creates correct scene_template record"""
        import services.supabase_service as supa_module

        with patch.object(supa_module.SupabaseService, '_instance', None):
            service = supa_module.SupabaseService()
            service._client = mock_supabase_client
            service._enabled = True
            service._connected = True
            service._installation_id = "test-install-id"

            service.sync_sequence(sample_sequence)

            mock_supabase_client.table.assert_called_with("scene_templates")

    def test_sync_scene_creates_correct_record(self, mock_supabase_client, sample_scene):
        """Test sync_scene creates correct scene_template record"""
        import services.supabase_service as supa_module

        with patch.object(supa_module.SupabaseService, '_instance', None):
            service = supa_module.SupabaseService()
            service._client = mock_supabase_client
            service._enabled = True
            service._connected = True
            service._installation_id = "test-install-id"

            service.sync_scene(sample_scene)

            mock_supabase_client.table.assert_called_with("scene_templates")

    def test_sync_chase_creates_correct_record(self, mock_supabase_client, sample_chase):
        """Test sync_chase creates correct scene_template record"""
        import services.supabase_service as supa_module

        with patch.object(supa_module.SupabaseService, '_instance', None):
            service = supa_module.SupabaseService()
            service._client = mock_supabase_client
            service._enabled = True
            service._connected = True
            service._installation_id = "test-install-id"

            service.sync_chase(sample_chase)

            mock_supabase_client.table.assert_called_with("scene_templates")


# ============================================================
# Error Handling Tests
# ============================================================

class TestErrorHandling:
    """Tests for error handling and fail-soft behavior"""

    def test_sync_failure_queues_operation(self, sample_node, tmp_path):
        """Test that failed sync operations are queued for retry"""
        import services.supabase_service as supa_module

        queue_file = tmp_path / "pending-sync.json"

        with patch.object(supa_module.SupabaseService, '_instance', None):
            service = supa_module.SupabaseService()
            service._enabled = True
            service._connected = True
            service._installation_id = "test-install-id"
            service._pending_queue = supa_module.PendingSyncQueue(str(queue_file))

            # Create a client that raises an exception
            failing_client = MagicMock()
            failing_client.table.return_value.upsert.return_value.execute.side_effect = Exception("Network error")
            service._client = failing_client

            # Sync should not raise, but should queue
            result = service.sync_node(sample_node)

            assert result is None
            assert service._pending_queue.count() == 1

    def test_service_continues_when_supabase_unavailable(self):
        """Test that service gracefully handles unavailable Supabase"""
        with patch.dict(os.environ, {
            "ENABLE_SUPABASE": "true",
            "SUPABASE_URL": "",
            "SUPABASE_SERVICE_ROLE_KEY": ""
        }):
            import services.supabase_service as supa_module

            with patch.object(supa_module.SupabaseService, '_instance', None):
                # Should not raise even with missing credentials
                service = supa_module.SupabaseService()

                # Service should be disabled due to missing credentials
                assert service.is_connected() == False


# ============================================================
# Initial Sync Tests
# ============================================================

class TestInitialSync:
    """Tests for initial sync functionality"""

    def test_initial_sync_processes_all_types(self, mock_supabase_client):
        """Test initial_sync processes all entity types"""
        import services.supabase_service as supa_module

        with patch.object(supa_module.SupabaseService, '_instance', None):
            service = supa_module.SupabaseService()
            service._client = mock_supabase_client
            service._enabled = True
            service._connected = True
            service._installation_id = "test-install-id"

            result = service.initial_sync(
                nodes=[{"node_id": "n1", "name": "Node 1"}],
                looks=[{"look_id": "l1", "name": "Look 1", "channels": {}}],
                sequences=[{"sequence_id": "s1", "name": "Seq 1", "steps": []}],
                scenes=[{"scene_id": "sc1", "name": "Scene 1", "channels": {}}],
                chases=[{"chase_id": "c1", "name": "Chase 1", "steps": []}],
                fixtures=[{"fixture_id": "f1", "name": "Fixture 1"}]
            )

            assert result["nodes"] == 1
            assert result["looks"] == 1
            assert result["sequences"] == 1
            assert result["scenes"] == 1
            assert result["chases"] == 1
            assert result["fixtures"] == 1

    def test_initial_sync_returns_error_counts(self, mock_supabase_client):
        """Test initial_sync tracks errors"""
        import services.supabase_service as supa_module

        with patch.object(supa_module.SupabaseService, '_instance', None):
            service = supa_module.SupabaseService()
            service._enabled = True
            service._connected = True
            service._installation_id = "test-install-id"

            # Create client that fails on some operations
            failing_client = MagicMock()
            call_count = [0]

            def fail_every_other(*args, **kwargs):
                call_count[0] += 1
                if call_count[0] % 2 == 0:
                    raise Exception("Simulated failure")
                return MagicMock(data=[{"id": "test"}])

            failing_client.table.return_value.upsert.return_value.execute.side_effect = fail_every_other
            service._client = failing_client

            result = service.initial_sync(
                nodes=[
                    {"node_id": "n1", "name": "Node 1"},
                    {"node_id": "n2", "name": "Node 2"}
                ]
            )

            # Should have 1 success and 1 error
            assert result["nodes"] == 1
            assert result["errors"] == 1


# ============================================================
# Feature Gate Tests
# ============================================================

class TestFeatureGating:
    """Tests for feature gating behavior"""

    def test_disabled_service_returns_gracefully(self):
        """Test all methods return gracefully when disabled"""
        with patch.dict(os.environ, {"ENABLE_SUPABASE": "false"}):
            import services.supabase_service as supa_module

            with patch.object(supa_module.SupabaseService, '_instance', None):
                service = supa_module.SupabaseService()

                # All sync methods should return None
                assert service.sync_node({}) is None
                assert service.sync_look({}) is None
                assert service.sync_sequence({}) is None
                assert service.sync_scene({}) is None
                assert service.sync_chase({}) is None
                assert service.sync_fixture({}) is None
                assert service.log_event("test") is None

                # Fetch methods should return empty lists
                assert service.fetch_looks() == []
                assert service.fetch_sequences() == []

    def test_rollback_by_disabling(self):
        """Test that setting ENABLE_SUPABASE=false effectively disables all cloud features"""
        # This simulates the rollback scenario
        with patch.dict(os.environ, {"ENABLE_SUPABASE": "false"}):
            import services.supabase_service as supa_module

            with patch.object(supa_module.SupabaseService, '_instance', None):
                service = supa_module.SupabaseService()

                status = service.get_status()

                assert status["enabled"] == False
                assert status["connected"] == False


# ============================================================
# Run tests if executed directly
# ============================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
