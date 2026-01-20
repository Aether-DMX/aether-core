"""
Unit Tests for RDM Discovery Engine

Tests for:
- DiscoverySession state management
- DiscoveryEngine discovery workflow
- Conflict detection
- Enrichment logic
- Progress callbacks
- Timeout handling
- Error handling
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, MagicMock
from datetime import datetime

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from core.rdm.discovery import (
    DiscoverySession,
    DiscoveryEngine,
    RdmDiscovery,
)
from core.rdm.types import (
    RdmUid,
    DiscoveredDevice,
    DiscoveredFixture,
    DiscoveryState,
    DiscoveryStatus,
    RdmDeviceInfo,
    RdmPersonality,
)
from core.rdm.transport import RdmTimeoutError, RdmTransportError


class TestDiscoveredFixture:
    """Tests for DiscoveredFixture dataclass."""

    def test_channel_range(self):
        """Test channel range calculation."""
        fixture = DiscoveredFixture(
            uid="02CA:12345678",
            universe=1,
            start_address=10,
            channel_count=8
        )
        r = fixture.channel_range()
        assert r.start == 10
        assert r.stop == 18
        assert list(r) == [10, 11, 12, 13, 14, 15, 16, 17]

    def test_channel_range_single_channel(self):
        """Test channel range for single channel fixture."""
        fixture = DiscoveredFixture(
            uid="02CA:12345678",
            universe=1,
            start_address=1,
            channel_count=1
        )
        r = fixture.channel_range()
        assert list(r) == [1]

    def test_has_conflicts_empty(self):
        """Test has_conflicts with no conflicts."""
        fixture = DiscoveredFixture(
            uid="02CA:12345678",
            universe=1,
            start_address=1,
            channel_count=8
        )
        assert fixture.has_conflicts() is False

    def test_has_conflicts_with_conflicts(self):
        """Test has_conflicts with conflicts."""
        fixture = DiscoveredFixture(
            uid="02CA:12345678",
            universe=1,
            start_address=1,
            channel_count=8,
            conflicts=["Overlaps with fixture X"]
        )
        assert fixture.has_conflicts() is True

    def test_to_dict(self):
        """Test serialization to dict."""
        fixture = DiscoveredFixture(
            uid="02CA:12345678",
            universe=1,
            start_address=25,
            channel_count=16,
            personality_index=2,
            personality_label="16-Channel",
            manufacturer="Test Mfg",
            model="Test Model",
            device_id=1234,
            serial_number="ABC123",
            software_version="1.0.0",
            capabilities=["color_wheel", "gobo"],
            fixture_type="moving_head",
            conflicts=["Overlap warning"]
        )
        d = fixture.to_dict()

        assert d["uid"] == "02CA:12345678"
        assert d["universe"] == 1
        assert d["start_address"] == 25
        assert d["channel_count"] == 16
        assert d["personality_label"] == "16-Channel"
        assert d["manufacturer"] == "Test Mfg"
        assert d["fixture_type"] == "moving_head"
        assert d["channel_range"] == [25, 40]
        assert d["has_conflicts"] is True

    def test_from_device_info(self):
        """Test creating DiscoveredFixture from RdmDeviceInfo."""
        uid = RdmUid(0x02CA, 0x12345678)
        personalities = [
            RdmPersonality(id=1, name="8-Channel", footprint=8),
            RdmPersonality(id=2, name="16-Channel", footprint=16),
        ]
        info = RdmDeviceInfo(
            uid=uid,
            manufacturer_id=714,
            device_model_id=1234,
            manufacturer_label="Test Mfg",
            device_model="Test Model",
            device_label="My Light",
            dmx_address=25,
            dmx_footprint=16,
            current_personality=2,
            personalities=personalities,
            software_version="1.2.3"
        )

        fixture = DiscoveredFixture.from_device_info(info, universe=1)

        assert fixture.uid == "02CA:12345678"
        assert fixture.universe == 1
        assert fixture.start_address == 25
        assert fixture.channel_count == 16
        assert fixture.personality_index == 2
        assert fixture.personality_label == "16-Channel"
        assert fixture.manufacturer == "Test Mfg"
        assert fixture.model == "Test Model"


class TestDiscoverySession:
    """Tests for discovery session state management."""

    def test_session_starts_idle(self):
        """Test initial session state."""
        session = DiscoverySession("node1", "192.168.1.100", 1)

        assert session.status.state == DiscoveryState.IDLE
        assert session.status.devices_found == 0
        assert session.status.devices_queried == 0
        assert session.devices == {}

    def test_start_sets_broadcasting_state(self):
        """Test state transition on start."""
        session = DiscoverySession("node1", "192.168.1.100", 1)
        session.start()

        assert session.status.state == DiscoveryState.BROADCASTING
        assert session.status.started_at is not None
        assert session.status.error is None

    def test_set_uids_transitions_to_querying(self):
        """Test state transition on UIDs received."""
        session = DiscoverySession("node1", "192.168.1.100", 1)
        session.start()

        uids = [
            RdmUid(0x02CA, 0x11111111),
            RdmUid(0x02CA, 0x22222222),
        ]
        session.set_uids(uids)

        assert session.status.state == DiscoveryState.QUERYING
        assert session.status.devices_found == 2

    def test_add_device_increments_count(self):
        """Test device counting."""
        session = DiscoverySession("node1", "192.168.1.100", 1)
        session.start()

        uid = RdmUid(0x02CA, 0x12345678)
        device = DiscoveredDevice(
            uid=uid,
            node_id="node1",
            universe=1,
            manufacturer_id=714,
            device_model_id=1234
        )
        session.add_device(device)

        assert session.status.devices_queried == 1
        assert "02CA:12345678" in session.devices

    def test_complete_sets_final_state(self):
        """Test completion state."""
        session = DiscoverySession("node1", "192.168.1.100", 1)
        session.start()
        session.complete()

        assert session.status.state == DiscoveryState.COMPLETE
        assert session.status.completed_at is not None

    def test_fail_captures_error(self):
        """Test error handling."""
        session = DiscoverySession("node1", "192.168.1.100", 1)
        session.start()
        session.fail("Connection refused")

        assert session.status.state == DiscoveryState.ERROR
        assert session.status.error == "Connection refused"
        assert session.status.completed_at is not None

    def test_is_active_broadcasting(self):
        """Test is_active during broadcasting."""
        session = DiscoverySession("node1", "192.168.1.100", 1)
        session.start()

        assert session.is_active() is True

    def test_is_active_querying(self):
        """Test is_active during querying."""
        session = DiscoverySession("node1", "192.168.1.100", 1)
        session.start()
        session.set_uids([RdmUid(0x02CA, 0x12345678)])

        assert session.is_active() is True

    def test_is_active_complete(self):
        """Test is_active after completion."""
        session = DiscoverySession("node1", "192.168.1.100", 1)
        session.start()
        session.complete()

        assert session.is_active() is False

    def test_is_active_error(self):
        """Test is_active after error."""
        session = DiscoverySession("node1", "192.168.1.100", 1)
        session.start()
        session.fail("Error")

        assert session.is_active() is False


class TestDiscoveryEngine:
    """Tests for DiscoveryEngine."""

    @pytest.fixture
    def mock_transport(self):
        """Create mock transport."""
        transport = Mock()
        transport.discover = AsyncMock(return_value=[])
        transport.get_device_info = AsyncMock()
        return transport

    @pytest.fixture
    def engine(self, mock_transport):
        """Create discovery engine with mock transport."""
        return DiscoveryEngine(mock_transport)

    def test_init(self, mock_transport):
        """Test engine initialization."""
        engine = DiscoveryEngine(mock_transport)

        assert engine.transport == mock_transport
        assert engine.fixture_db is None
        assert engine._state == DiscoveryState.IDLE

    def test_init_with_fixture_db(self, mock_transport):
        """Test engine initialization with fixture database."""
        fixture_db = Mock()
        engine = DiscoveryEngine(mock_transport, fixture_db)

        assert engine.fixture_db == fixture_db

    @pytest.mark.asyncio
    async def test_discover_empty_universe(self, engine, mock_transport):
        """Test discovery with no fixtures."""
        mock_transport.discover.return_value = []

        fixtures = await engine.discover_universes("192.168.1.100", [1])

        assert fixtures == {}
        mock_transport.discover.assert_called_once_with("192.168.1.100", 1)

    @pytest.mark.asyncio
    async def test_discover_single_fixture(self, engine, mock_transport):
        """Test discovery with single fixture."""
        uid = RdmUid(0x02CA, 0x12345678)
        mock_transport.discover.return_value = [uid]
        mock_transport.get_device_info.return_value = RdmDeviceInfo(
            uid=uid,
            manufacturer_id=714,
            device_model_id=1234,
            manufacturer_label="Test Mfg",
            device_model="Test Model",
            dmx_address=1,
            dmx_footprint=8
        )

        fixtures = await engine.discover_universes("192.168.1.100", [1])

        assert len(fixtures) == 1
        assert "02CA:12345678" in fixtures
        fixture = fixtures["02CA:12345678"]
        assert fixture.manufacturer == "Test Mfg"
        assert fixture.model == "Test Model"
        assert fixture.start_address == 1
        assert fixture.channel_count == 8

    @pytest.mark.asyncio
    async def test_discover_multiple_fixtures(self, engine, mock_transport):
        """Test discovery with multiple fixtures."""
        uid1 = RdmUid(0x02CA, 0x11111111)
        uid2 = RdmUid(0x02CA, 0x22222222)
        uid3 = RdmUid(0x02CA, 0x33333333)

        mock_transport.discover.return_value = [uid1, uid2, uid3]

        def make_info(uid, address):
            return RdmDeviceInfo(
                uid=uid,
                manufacturer_id=714,
                device_model_id=1234,
                manufacturer_label="Test Mfg",
                device_model="LED Par",
                dmx_address=address,
                dmx_footprint=8
            )

        mock_transport.get_device_info.side_effect = [
            make_info(uid1, 1),
            make_info(uid2, 9),
            make_info(uid3, 17),
        ]

        fixtures = await engine.discover_universes("192.168.1.100", [1])

        assert len(fixtures) == 3

    @pytest.mark.asyncio
    async def test_discover_multiple_universes(self, engine, mock_transport):
        """Test discovery across multiple universes."""
        uid1 = RdmUid(0x02CA, 0x11111111)
        uid2 = RdmUid(0x02CA, 0x22222222)

        # Universe 1 has uid1, universe 2 has uid2
        mock_transport.discover.side_effect = [
            [uid1],  # Universe 1
            [uid2],  # Universe 2
        ]

        mock_transport.get_device_info.side_effect = [
            RdmDeviceInfo(
                uid=uid1,
                manufacturer_id=714,
                device_model_id=1234,
                manufacturer_label="Mfg",
                device_model="Model",
                dmx_address=1,
                dmx_footprint=8
            ),
            RdmDeviceInfo(
                uid=uid2,
                manufacturer_id=714,
                device_model_id=1234,
                manufacturer_label="Mfg",
                device_model="Model",
                dmx_address=1,
                dmx_footprint=8
            ),
        ]

        fixtures = await engine.discover_universes("192.168.1.100", [1, 2])

        assert len(fixtures) == 2
        assert fixtures["02CA:11111111"].universe == 1
        assert fixtures["02CA:22222222"].universe == 2

    @pytest.mark.asyncio
    async def test_discover_timeout_per_fixture(self, engine, mock_transport):
        """Test timeout handling for individual fixtures."""
        uid1 = RdmUid(0x02CA, 0x11111111)
        uid2 = RdmUid(0x02CA, 0x22222222)

        mock_transport.discover.return_value = [uid1, uid2]

        # First fixture times out, second succeeds
        mock_transport.get_device_info.side_effect = [
            RdmTimeoutError("Timeout"),
            RdmDeviceInfo(
                uid=uid2,
                manufacturer_id=714,
                device_model_id=1234,
                manufacturer_label="Mfg",
                device_model="Model",
                dmx_address=1,
                dmx_footprint=8
            ),
        ]

        fixtures = await engine.discover_universes("192.168.1.100", [1])

        # Should skip timed out fixture but continue
        assert len(fixtures) == 1
        assert "02CA:22222222" in fixtures

    @pytest.mark.asyncio
    async def test_discover_transport_error(self, engine, mock_transport):
        """Test transport error handling."""
        mock_transport.discover.side_effect = RdmTransportError("Connection failed")

        # Should not raise, returns empty
        fixtures = await engine.discover_universes("192.168.1.100", [1])
        assert fixtures == {}

    @pytest.mark.asyncio
    async def test_progress_callbacks(self, engine, mock_transport):
        """Test progress callback emission."""
        mock_transport.discover.return_value = []

        progress_updates = []

        def on_progress(percent, msg):
            progress_updates.append((percent, msg))

        engine.on_progress(on_progress)

        await engine.discover_universes("192.168.1.100", [1])

        # Should have at least start (0%) and end (100%)
        assert any(p[0] == 0 for p in progress_updates)
        assert any(p[0] == 100 for p in progress_updates)

    def test_off_progress(self, engine):
        """Test unregistering progress callback."""
        def callback(pct, msg):
            pass

        engine.on_progress(callback)
        assert callback in engine._progress_callbacks

        engine.off_progress(callback)
        assert callback not in engine._progress_callbacks


class TestConflictDetection:
    """Tests for address conflict detection."""

    @pytest.fixture
    def mock_transport(self):
        """Create mock transport."""
        transport = Mock()
        transport.discover = AsyncMock(return_value=[])
        transport.get_device_info = AsyncMock()
        return transport

    @pytest.fixture
    def engine(self, mock_transport):
        """Create discovery engine."""
        return DiscoveryEngine(mock_transport)

    def test_no_conflicts(self, engine):
        """Test fixtures with no overlap."""
        fixtures = {
            "uid1": DiscoveredFixture(
                uid="uid1", universe=1, start_address=1, channel_count=8
            ),
            "uid2": DiscoveredFixture(
                uid="uid2", universe=1, start_address=9, channel_count=8
            ),
        }

        engine._detect_conflicts(fixtures)

        assert fixtures["uid1"].conflicts == []
        assert fixtures["uid2"].conflicts == []

    def test_overlap_detected(self, engine):
        """Test overlapping fixtures are detected."""
        fixtures = {
            "uid1": DiscoveredFixture(
                uid="uid1", universe=1, start_address=1, channel_count=10,
                manufacturer="Mfg1", model="Model1"
            ),
            "uid2": DiscoveredFixture(
                uid="uid2", universe=1, start_address=5, channel_count=10,
                manufacturer="Mfg2", model="Model2"
            ),
        }

        engine._detect_conflicts(fixtures)

        # Both should have conflicts
        assert len(fixtures["uid1"].conflicts) == 1
        assert len(fixtures["uid2"].conflicts) == 1
        assert "uid2" in fixtures["uid1"].conflicts[0]
        assert "uid1" in fixtures["uid2"].conflicts[0]

    def test_different_universes_no_conflict(self, engine):
        """Test fixtures on different universes don't conflict."""
        fixtures = {
            "uid1": DiscoveredFixture(
                uid="uid1", universe=1, start_address=1, channel_count=100
            ),
            "uid2": DiscoveredFixture(
                uid="uid2", universe=2, start_address=1, channel_count=100
            ),
        }

        engine._detect_conflicts(fixtures)

        assert fixtures["uid1"].conflicts == []
        assert fixtures["uid2"].conflicts == []

    def test_exact_overlap(self, engine):
        """Test exact same address range."""
        fixtures = {
            "uid1": DiscoveredFixture(
                uid="uid1", universe=1, start_address=1, channel_count=8,
                manufacturer="A", model="A"
            ),
            "uid2": DiscoveredFixture(
                uid="uid2", universe=1, start_address=1, channel_count=8,
                manufacturer="B", model="B"
            ),
        }

        engine._detect_conflicts(fixtures)

        assert len(fixtures["uid1"].conflicts) == 1
        assert len(fixtures["uid2"].conflicts) == 1

    def test_partial_overlap(self, engine):
        """Test partial channel overlap."""
        fixtures = {
            "uid1": DiscoveredFixture(
                uid="uid1", universe=1, start_address=1, channel_count=8,
                manufacturer="A", model="A"
            ),
            "uid2": DiscoveredFixture(
                uid="uid2", universe=1, start_address=6, channel_count=8,
                manufacturer="B", model="B"
            ),
        }

        engine._detect_conflicts(fixtures)

        # uid1: 1-8, uid2: 6-13, overlap: 6-8
        assert len(fixtures["uid1"].conflicts) == 1
        assert len(fixtures["uid2"].conflicts) == 1
        # Check overlap range mentioned in conflict
        assert "6-8" in fixtures["uid1"].conflicts[0]

    def test_multiple_conflicts(self, engine):
        """Test fixture with multiple conflicts."""
        fixtures = {
            "uid1": DiscoveredFixture(
                uid="uid1", universe=1, start_address=1, channel_count=512,
                manufacturer="A", model="A"
            ),
            "uid2": DiscoveredFixture(
                uid="uid2", universe=1, start_address=10, channel_count=8,
                manufacturer="B", model="B"
            ),
            "uid3": DiscoveredFixture(
                uid="uid3", universe=1, start_address=100, channel_count=8,
                manufacturer="C", model="C"
            ),
        }

        engine._detect_conflicts(fixtures)

        # uid1 overlaps with both uid2 and uid3
        assert len(fixtures["uid1"].conflicts) == 2
        assert len(fixtures["uid2"].conflicts) == 1
        assert len(fixtures["uid3"].conflicts) == 1


class TestEnrichment:
    """Tests for fixture enrichment."""

    @pytest.fixture
    def mock_transport(self):
        """Create mock transport."""
        return Mock()

    def test_guess_fixture_type_moving_head(self, mock_transport):
        """Test guessing moving head type."""
        engine = DiscoveryEngine(mock_transport)

        assert engine._guess_fixture_type("Moving Head Spot") == "moving_head"
        assert engine._guess_fixture_type("MH-500") == "moving_head"

    def test_guess_fixture_type_wash(self, mock_transport):
        """Test guessing wash type."""
        engine = DiscoveryEngine(mock_transport)

        assert engine._guess_fixture_type("LED Wash 600") == "wash"
        assert engine._guess_fixture_type("Studio Flood") == "wash"

    def test_guess_fixture_type_par(self, mock_transport):
        """Test guessing par type."""
        engine = DiscoveryEngine(mock_transport)

        assert engine._guess_fixture_type("LED Par 64") == "par"
        assert engine._guess_fixture_type("RGB Par Can") == "par"

    def test_guess_fixture_type_strobe(self, mock_transport):
        """Test guessing strobe type."""
        engine = DiscoveryEngine(mock_transport)

        assert engine._guess_fixture_type("Atomic 3000") == "strobe"
        assert engine._guess_fixture_type("LED Strobe") == "strobe"

    def test_guess_fixture_type_unknown(self, mock_transport):
        """Test unknown fixture type."""
        engine = DiscoveryEngine(mock_transport)

        assert engine._guess_fixture_type("Mystery Box XYZ") is None

    def test_enrich_without_db(self, mock_transport):
        """Test enrichment without fixture database."""
        engine = DiscoveryEngine(mock_transport)
        fixture = DiscoveredFixture(
            uid="uid1", universe=1, start_address=1, channel_count=8,
            model="LED Par 64"
        )

        engine._enrich_fixture(fixture)

        assert fixture.fixture_type == "par"

    def test_enrich_with_db_match(self, mock_transport):
        """Test enrichment with database match."""
        fixture_db = Mock()
        fixture_db.get_profile_by_rdm_ids = Mock(return_value={
            'fixture_type': 'moving_head',
            'capabilities': ['pan', 'tilt', 'color_wheel', 'gobo']
        })

        engine = DiscoveryEngine(mock_transport, fixture_db)
        fixture = DiscoveredFixture(
            uid="02CA:12345678", universe=1, start_address=1, channel_count=16,
            device_id=1234
        )

        engine._enrich_fixture(fixture)

        assert fixture.fixture_type == "moving_head"
        assert "pan" in fixture.capabilities
        assert "color_wheel" in fixture.capabilities

    def test_enrich_with_db_no_match(self, mock_transport):
        """Test enrichment with no database match."""
        fixture_db = Mock()
        fixture_db.get_profile_by_rdm_ids = Mock(return_value=None)
        fixture_db.get_profile_by_name = Mock(return_value=None)

        engine = DiscoveryEngine(mock_transport, fixture_db)
        fixture = DiscoveredFixture(
            uid="02CA:12345678", universe=1, start_address=1, channel_count=16,
            device_id=1234,
            model="LED Wash 600"
        )

        engine._enrich_fixture(fixture)

        # Falls back to guessing
        assert fixture.fixture_type == "wash"


class TestRdmDiscoveryLegacy:
    """Tests for legacy RdmDiscovery class."""

    @pytest.fixture
    def mock_transport(self):
        """Create mock transport."""
        transport = Mock()
        transport.discover = AsyncMock(return_value=[])
        transport.get_device_info = AsyncMock()
        return transport

    @pytest.fixture
    def discovery(self, mock_transport):
        """Create RdmDiscovery instance."""
        return RdmDiscovery(mock_transport)

    @pytest.mark.asyncio
    async def test_discover_creates_session(self, discovery, mock_transport):
        """Test session creation on discover."""
        await discovery.discover_node("node1", "192.168.1.100", 1)

        assert "node1" in discovery.sessions
        assert discovery.sessions["node1"].status.state == DiscoveryState.COMPLETE

    @pytest.mark.asyncio
    async def test_discover_prevents_concurrent(self, discovery, mock_transport):
        """Test concurrent discovery prevention."""
        # Start a discovery that won't complete immediately
        async def slow_discover(*args):
            await asyncio.sleep(10)
            return []

        mock_transport.discover = slow_discover

        # Start first discovery
        task = asyncio.create_task(
            discovery.discover_node("node1", "192.168.1.100", 1)
        )
        await asyncio.sleep(0.01)  # Let it start

        # Try to start second
        with pytest.raises(RuntimeError, match="already in progress"):
            await discovery.discover_node("node1", "192.168.1.100", 1)

        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_discover_caches_devices(self, discovery, mock_transport):
        """Test device caching."""
        uid = RdmUid(0x02CA, 0x12345678)
        mock_transport.discover.return_value = [uid]
        mock_transport.get_device_info.return_value = RdmDeviceInfo(
            uid=uid,
            manufacturer_id=714,
            device_model_id=1234,
            dmx_address=1,
            dmx_footprint=8
        )

        await discovery.discover_node("node1", "192.168.1.100", 1)

        assert "02CA:12345678" in discovery.device_cache

    def test_get_cached_devices_all(self, discovery):
        """Test getting all cached devices."""
        uid = RdmUid(0x02CA, 0x12345678)
        device = DiscoveredDevice(
            uid=uid, node_id="node1", universe=1,
            manufacturer_id=714, device_model_id=1234
        )
        discovery.device_cache[str(uid)] = device

        devices = discovery.get_cached_devices()

        assert len(devices) == 1

    def test_get_cached_devices_by_node(self, discovery):
        """Test filtering cached devices by node."""
        uid1 = RdmUid(0x02CA, 0x11111111)
        uid2 = RdmUid(0x02CA, 0x22222222)
        device1 = DiscoveredDevice(
            uid=uid1, node_id="node1", universe=1,
            manufacturer_id=714, device_model_id=1234
        )
        device2 = DiscoveredDevice(
            uid=uid2, node_id="node2", universe=1,
            manufacturer_id=714, device_model_id=1234
        )
        discovery.device_cache[str(uid1)] = device1
        discovery.device_cache[str(uid2)] = device2

        devices = discovery.get_cached_devices(node_id="node1")

        assert len(devices) == 1
        assert str(devices[0].uid) == "02CA:11111111"

    def test_get_cached_devices_by_universe(self, discovery):
        """Test filtering cached devices by universe."""
        uid1 = RdmUid(0x02CA, 0x11111111)
        uid2 = RdmUid(0x02CA, 0x22222222)
        device1 = DiscoveredDevice(
            uid=uid1, node_id="node1", universe=1,
            manufacturer_id=714, device_model_id=1234
        )
        device2 = DiscoveredDevice(
            uid=uid2, node_id="node1", universe=2,
            manufacturer_id=714, device_model_id=1234
        )
        discovery.device_cache[str(uid1)] = device1
        discovery.device_cache[str(uid2)] = device2

        devices = discovery.get_cached_devices(universe=2)

        assert len(devices) == 1
        assert devices[0].universe == 2

    def test_clear_cache_all(self, discovery):
        """Test clearing all cached devices."""
        uid = RdmUid(0x02CA, 0x12345678)
        device = DiscoveredDevice(
            uid=uid, node_id="node1", universe=1,
            manufacturer_id=714, device_model_id=1234
        )
        discovery.device_cache[str(uid)] = device

        discovery.clear_cache()

        assert len(discovery.device_cache) == 0

    def test_clear_cache_by_node(self, discovery):
        """Test clearing cache for specific node."""
        uid1 = RdmUid(0x02CA, 0x11111111)
        uid2 = RdmUid(0x02CA, 0x22222222)
        device1 = DiscoveredDevice(
            uid=uid1, node_id="node1", universe=1,
            manufacturer_id=714, device_model_id=1234
        )
        device2 = DiscoveredDevice(
            uid=uid2, node_id="node2", universe=1,
            manufacturer_id=714, device_model_id=1234
        )
        discovery.device_cache[str(uid1)] = device1
        discovery.device_cache[str(uid2)] = device2

        discovery.clear_cache(node_id="node1")

        assert len(discovery.device_cache) == 1
        assert str(uid2) in discovery.device_cache

    def test_event_callbacks(self, discovery):
        """Test event callback registration."""
        events = []

        def callback(data):
            events.append(data)

        discovery.on("test_event", callback)
        discovery._emit("test_event", {"test": "data"})

        assert len(events) == 1
        assert events[0] == {"test": "data"}

    def test_event_callback_removal(self, discovery):
        """Test event callback removal."""
        events = []

        def callback(data):
            events.append(data)

        discovery.on("test_event", callback)
        discovery.off("test_event", callback)
        discovery._emit("test_event", {"test": "data"})

        assert len(events) == 0


class TestDiscoveryStateEnum:
    """Tests for DiscoveryState enum."""

    def test_all_states_exist(self):
        """Test all required states exist."""
        assert DiscoveryState.IDLE.value == "idle"
        assert DiscoveryState.BROADCASTING.value == "broadcasting"
        assert DiscoveryState.QUERYING.value == "querying"
        assert DiscoveryState.ENRICHING.value == "enriching"
        assert DiscoveryState.CONFLICT_CHECK.value == "conflict_check"
        assert DiscoveryState.COMPLETE.value == "complete"
        assert DiscoveryState.ERROR.value == "error"

    def test_discovering_alias(self):
        """Test DISCOVERING is alias for BROADCASTING."""
        assert DiscoveryState.DISCOVERING.value == "broadcasting"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
