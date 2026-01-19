"""
Fixture Library Module for Aether DMX

Provides:
1. Fixture profile definitions with channel mappings
2. Open Fixture Library (OFL) integration for 15,000+ fixture profiles
3. RDM-to-fixture linking for auto-configuration
4. Intelligent channel mapping for scenes/chases

Architecture:
- FixtureProfile: Defines a fixture type (channels, capabilities)
- FixtureInstance: A placed fixture (profile + universe + address)
- FixtureLibrary: Manages profiles and instances
"""

import json
import os
import sqlite3
import urllib.request
import urllib.error
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from datetime import datetime
import threading

# ============================================================
# Data Models
# ============================================================

@dataclass
class ChannelCapability:
    """Defines what a channel controls"""
    name: str  # e.g., "Red", "Pan", "Dimmer"
    type: str  # dimmer, color, position, control, speed, gobo, prism, etc.
    default: int = 0
    min_value: int = 0
    max_value: int = 255
    fine_channel: Optional[int] = None  # For 16-bit channels (e.g., Pan Fine)

@dataclass
class FixtureMode:
    """A fixture personality/mode - different channel configurations"""
    mode_id: str
    name: str  # e.g., "8-channel", "16-channel Extended"
    channel_count: int
    channels: List[ChannelCapability] = field(default_factory=list)

@dataclass
class FixtureProfile:
    """
    A fixture profile defines the capabilities of a fixture model.
    This is the "template" - what channels do what.
    """
    profile_id: str  # Unique ID (e.g., "chauvet-slimpar-pro-h-usb")
    manufacturer: str
    model: str
    category: str  # par, moving_head, wash, beam, strobe, dimmer, etc.
    modes: List[FixtureMode] = field(default_factory=list)

    # Physical info (optional)
    weight_kg: float = 0
    power_watts: int = 0

    # RDM matching
    rdm_manufacturer_id: Optional[int] = None
    rdm_device_model_id: Optional[int] = None

    # Source
    source: str = "manual"  # manual, ofl, rdm
    ofl_key: Optional[str] = None  # Open Fixture Library key

    def get_mode(self, mode_id: str) -> Optional[FixtureMode]:
        for mode in self.modes:
            if mode.mode_id == mode_id:
                return mode
        return self.modes[0] if self.modes else None

    def get_channel_by_type(self, mode_id: str, channel_type: str) -> List[int]:
        """Get channel offsets for a given type (e.g., all 'color' channels)"""
        mode = self.get_mode(mode_id)
        if not mode:
            return []
        return [i for i, ch in enumerate(mode.channels) if ch.type == channel_type]

    def get_rgb_channels(self, mode_id: str) -> Dict[str, int]:
        """Get RGB channel offsets for color mixing"""
        mode = self.get_mode(mode_id)
        if not mode:
            return {}
        result = {}
        for i, ch in enumerate(mode.channels):
            name_lower = ch.name.lower()
            if 'red' in name_lower and 'r' not in result:
                result['r'] = i
            elif 'green' in name_lower and 'g' not in result:
                result['g'] = i
            elif 'blue' in name_lower and 'b' not in result:
                result['b'] = i
            elif 'white' in name_lower and 'w' not in result:
                result['w'] = i
            elif 'amber' in name_lower and 'a' not in result:
                result['a'] = i
        return result

@dataclass
class FixtureInstance:
    """
    A placed fixture - links a profile to a physical location on DMX.
    This is the actual fixture in your rig.
    """
    fixture_id: str
    name: str
    profile_id: str
    mode_id: str
    universe: int
    start_channel: int

    # Optional grouping
    group: Optional[str] = None  # e.g., "Stage Left", "Truss 1"
    position: Optional[str] = None  # e.g., "1", "A", location identifier

    # RDM link
    rdm_uid: Optional[str] = None

    # Metadata
    notes: str = ""
    color: str = "#8b5cf6"  # UI color
    created_at: str = ""
    updated_at: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        self.updated_at = datetime.now().isoformat()


# ============================================================
# Built-in Profiles (common generic types)
# ============================================================

BUILTIN_PROFILES = [
    FixtureProfile(
        profile_id="generic-dimmer",
        manufacturer="Generic",
        model="Dimmer",
        category="dimmer",
        modes=[
            FixtureMode("1ch", "1-Channel", 1, [
                ChannelCapability("Dimmer", "dimmer", 0)
            ])
        ]
    ),
    FixtureProfile(
        profile_id="generic-rgb",
        manufacturer="Generic",
        model="RGB Par",
        category="par",
        modes=[
            FixtureMode("3ch", "3-Channel RGB", 3, [
                ChannelCapability("Red", "color", 0),
                ChannelCapability("Green", "color", 0),
                ChannelCapability("Blue", "color", 0),
            ]),
            FixtureMode("4ch", "4-Channel RGBD", 4, [
                ChannelCapability("Red", "color", 0),
                ChannelCapability("Green", "color", 0),
                ChannelCapability("Blue", "color", 0),
                ChannelCapability("Dimmer", "dimmer", 255),
            ])
        ]
    ),
    FixtureProfile(
        profile_id="generic-rgbw",
        manufacturer="Generic",
        model="RGBW Par",
        category="par",
        modes=[
            FixtureMode("4ch", "4-Channel RGBW", 4, [
                ChannelCapability("Red", "color", 0),
                ChannelCapability("Green", "color", 0),
                ChannelCapability("Blue", "color", 0),
                ChannelCapability("White", "color", 0),
            ]),
            FixtureMode("5ch", "5-Channel RGBWD", 5, [
                ChannelCapability("Red", "color", 0),
                ChannelCapability("Green", "color", 0),
                ChannelCapability("Blue", "color", 0),
                ChannelCapability("White", "color", 0),
                ChannelCapability("Dimmer", "dimmer", 255),
            ])
        ]
    ),
    FixtureProfile(
        profile_id="generic-rgbwa",
        manufacturer="Generic",
        model="RGBWA Par",
        category="par",
        modes=[
            FixtureMode("5ch", "5-Channel RGBWA", 5, [
                ChannelCapability("Red", "color", 0),
                ChannelCapability("Green", "color", 0),
                ChannelCapability("Blue", "color", 0),
                ChannelCapability("White", "color", 0),
                ChannelCapability("Amber", "color", 0),
            ]),
            FixtureMode("6ch", "6-Channel RGBWAD", 6, [
                ChannelCapability("Red", "color", 0),
                ChannelCapability("Green", "color", 0),
                ChannelCapability("Blue", "color", 0),
                ChannelCapability("White", "color", 0),
                ChannelCapability("Amber", "color", 0),
                ChannelCapability("Dimmer", "dimmer", 255),
            ])
        ]
    ),
    FixtureProfile(
        profile_id="generic-moving-head",
        manufacturer="Generic",
        model="Moving Head",
        category="moving_head",
        modes=[
            FixtureMode("16ch", "16-Channel", 16, [
                ChannelCapability("Pan", "position", 128),
                ChannelCapability("Pan Fine", "position", 0),
                ChannelCapability("Tilt", "position", 128),
                ChannelCapability("Tilt Fine", "position", 0),
                ChannelCapability("Pan/Tilt Speed", "speed", 0),
                ChannelCapability("Dimmer", "dimmer", 0),
                ChannelCapability("Strobe", "control", 0),
                ChannelCapability("Red", "color", 0),
                ChannelCapability("Green", "color", 0),
                ChannelCapability("Blue", "color", 0),
                ChannelCapability("White", "color", 0),
                ChannelCapability("Color Wheel", "control", 0),
                ChannelCapability("Gobo", "gobo", 0),
                ChannelCapability("Gobo Rotation", "speed", 0),
                ChannelCapability("Prism", "prism", 0),
                ChannelCapability("Focus", "control", 128),
            ])
        ]
    ),
    FixtureProfile(
        profile_id="generic-wash",
        manufacturer="Generic",
        model="LED Wash",
        category="wash",
        modes=[
            FixtureMode("7ch", "7-Channel", 7, [
                ChannelCapability("Red", "color", 0),
                ChannelCapability("Green", "color", 0),
                ChannelCapability("Blue", "color", 0),
                ChannelCapability("White", "color", 0),
                ChannelCapability("Dimmer", "dimmer", 255),
                ChannelCapability("Pan", "position", 128),
                ChannelCapability("Tilt", "position", 128),
            ])
        ]
    ),
    FixtureProfile(
        profile_id="generic-strobe",
        manufacturer="Generic",
        model="Strobe",
        category="strobe",
        modes=[
            FixtureMode("2ch", "2-Channel", 2, [
                ChannelCapability("Dimmer", "dimmer", 0),
                ChannelCapability("Strobe Speed", "speed", 0),
            ])
        ]
    ),
]


# ============================================================
# Open Fixture Library Integration
# ============================================================

class OFLClient:
    """
    Client for Open Fixture Library (https://open-fixture-library.org)
    Provides access to 15,000+ fixture profiles.
    """

    BASE_URL = "https://open-fixture-library.org/api/v1"
    CACHE_DIR = os.path.expanduser("~/.aether/ofl_cache")

    def __init__(self):
        os.makedirs(self.CACHE_DIR, exist_ok=True)
        self._manufacturers_cache = None
        self._cache_lock = threading.Lock()

    def _fetch_json(self, endpoint: str, timeout: int = 10) -> Optional[Dict]:
        """Fetch JSON from OFL API"""
        url = f"{self.BASE_URL}/{endpoint}"
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Aether-DMX/1.0'})
            with urllib.request.urlopen(req, timeout=timeout) as response:
                return json.loads(response.read().decode())
        except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError) as e:
            print(f"OFL fetch error for {endpoint}: {e}")
            return None

    def _get_cache_path(self, key: str) -> str:
        """Get cache file path for a key"""
        safe_key = key.replace("/", "_").replace("\\", "_")
        return os.path.join(self.CACHE_DIR, f"{safe_key}.json")

    def _read_cache(self, key: str) -> Optional[Dict]:
        """Read from cache if exists and not expired (7 days)"""
        cache_path = self._get_cache_path(key)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r') as f:
                    data = json.load(f)
                    # Check expiry (7 days)
                    if data.get('_cached_at'):
                        cached_time = datetime.fromisoformat(data['_cached_at'])
                        if (datetime.now() - cached_time).days < 7:
                            return data.get('data')
            except (OSError, json.JSONDecodeError, ValueError, KeyError):
                pass  # Cache corrupted or unreadable, will refetch
        return None

    def _write_cache(self, key: str, data: Dict):
        """Write to cache"""
        cache_path = self._get_cache_path(key)
        try:
            with open(cache_path, 'w') as f:
                json.dump({'data': data, '_cached_at': datetime.now().isoformat()}, f)
        except OSError:
            pass  # Cache write failed, non-critical

    def get_manufacturers(self) -> List[Dict]:
        """Get list of all manufacturers"""
        with self._cache_lock:
            if self._manufacturers_cache:
                return self._manufacturers_cache

            cached = self._read_cache("manufacturers")
            if cached:
                self._manufacturers_cache = cached
                return cached

            data = self._fetch_json("manufacturers")
            if data:
                result = [{"key": k, "name": v.get("name", k)} for k, v in data.items()]
                self._manufacturers_cache = result
                self._write_cache("manufacturers", result)
                return result
            return []

    def search_fixtures(self, query: str, manufacturer: str = None) -> List[Dict]:
        """Search for fixtures by name"""
        cache_key = f"search_{query}_{manufacturer or 'all'}"
        cached = self._read_cache(cache_key)
        if cached:
            return cached

        endpoint = f"fixtures/search/{urllib.parse.quote(query)}"
        if manufacturer:
            endpoint += f"?manufacturer={urllib.parse.quote(manufacturer)}"

        data = self._fetch_json(endpoint)
        if data and isinstance(data, list):
            self._write_cache(cache_key, data)
            return data
        return []

    def get_fixture(self, manufacturer: str, fixture: str) -> Optional[Dict]:
        """Get full fixture definition from OFL"""
        cache_key = f"fixture_{manufacturer}_{fixture}"
        cached = self._read_cache(cache_key)
        if cached:
            return cached

        data = self._fetch_json(f"{manufacturer}/{fixture}")
        if data:
            self._write_cache(cache_key, data)
            return data
        return None

    def ofl_to_profile(self, ofl_data: Dict, manufacturer_key: str, fixture_key: str) -> Optional[FixtureProfile]:
        """Convert OFL fixture data to our FixtureProfile format"""
        if not ofl_data:
            return None

        try:
            modes = []
            for mode_data in ofl_data.get('modes', []):
                channels = []
                for ch_key in mode_data.get('channels', []):
                    ch_data = ofl_data.get('availableChannels', {}).get(ch_key, {})
                    ch_type = self._classify_ofl_channel(ch_key, ch_data)
                    channels.append(ChannelCapability(
                        name=ch_key,
                        type=ch_type,
                        default=ch_data.get('defaultValue', 0)
                    ))

                modes.append(FixtureMode(
                    mode_id=mode_data.get('shortName', mode_data.get('name', 'default')),
                    name=mode_data.get('name', 'Default'),
                    channel_count=len(channels),
                    channels=channels
                ))

            # Determine category
            categories = ofl_data.get('categories', [])
            category = 'generic'
            if 'Moving Head' in categories:
                category = 'moving_head'
            elif 'Color Changer' in categories or 'LED' in categories:
                category = 'par'
            elif 'Strobe' in categories:
                category = 'strobe'
            elif 'Dimmer' in categories:
                category = 'dimmer'
            elif 'Wash' in categories or 'Batten' in categories:
                category = 'wash'

            return FixtureProfile(
                profile_id=f"ofl-{manufacturer_key}-{fixture_key}",
                manufacturer=ofl_data.get('manufacturer', {}).get('name', manufacturer_key),
                model=ofl_data.get('name', fixture_key),
                category=category,
                modes=modes,
                source="ofl",
                ofl_key=f"{manufacturer_key}/{fixture_key}",
                rdm_manufacturer_id=ofl_data.get('rdm', {}).get('manufacturerId'),
                rdm_device_model_id=ofl_data.get('rdm', {}).get('deviceModelId'),
            )
        except Exception as e:
            print(f"Error converting OFL fixture: {e}")
            return None

    def _classify_ofl_channel(self, name: str, data: Dict) -> str:
        """Classify OFL channel type to our type system"""
        name_lower = name.lower()

        # Check OFL type first
        ofl_type = data.get('type', '').lower()
        if ofl_type in ('intensity', 'dimmer'):
            return 'dimmer'
        if ofl_type == 'pan':
            return 'position'
        if ofl_type == 'tilt':
            return 'position'
        if ofl_type == 'color':
            return 'color'

        # Fall back to name matching
        if any(c in name_lower for c in ('red', 'green', 'blue', 'white', 'amber', 'cyan', 'magenta', 'yellow', 'uv')):
            return 'color'
        if 'dimmer' in name_lower or 'intensity' in name_lower or 'master' in name_lower:
            return 'dimmer'
        if 'pan' in name_lower or 'tilt' in name_lower:
            return 'position'
        if 'speed' in name_lower:
            return 'speed'
        if 'gobo' in name_lower:
            return 'gobo'
        if 'prism' in name_lower:
            return 'prism'
        if 'strobe' in name_lower or 'shutter' in name_lower:
            return 'control'

        return 'control'


# ============================================================
# Fixture Library Manager
# ============================================================

class FixtureLibrary:
    """
    Manages fixture profiles and instances.
    Integrates with database for persistence and OFL for imports.
    """

    def __init__(self, database_path: str):
        self.database_path = database_path
        self.ofl = OFLClient()
        self._profiles_cache: Dict[str, FixtureProfile] = {}
        self._init_database()
        self._load_builtin_profiles()

    def _get_db(self):
        conn = sqlite3.connect(self.database_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_database(self):
        """Initialize fixture library tables"""
        conn = self._get_db()
        c = conn.cursor()

        # Fixture profiles table (the templates)
        c.execute('''CREATE TABLE IF NOT EXISTS fixture_profiles (
            profile_id TEXT PRIMARY KEY,
            manufacturer TEXT,
            model TEXT,
            category TEXT,
            modes TEXT,
            rdm_manufacturer_id INTEGER,
            rdm_device_model_id INTEGER,
            source TEXT DEFAULT 'manual',
            ofl_key TEXT,
            weight_kg REAL DEFAULT 0,
            power_watts INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )''')

        # RDM manufacturer lookup table
        c.execute('''CREATE TABLE IF NOT EXISTS rdm_manufacturers (
            manufacturer_id INTEGER PRIMARY KEY,
            name TEXT
        )''')

        # Seed some known RDM manufacturers
        known_manufacturers = [
            (0x0618, "Chauvet"),
            (0x00A0, "American DJ"),
            (0x454C, "Elation"),
            (0x4D50, "Martin Professional"),
            (0x0170, "Vari-Lite"),
            (0x4845, "High End Systems"),
            (0x524F, "Robe"),
            (0x0200, "Clay Paky"),
            (0x0104, "LumenRadio"),
        ]
        for mid, name in known_manufacturers:
            c.execute('INSERT OR IGNORE INTO rdm_manufacturers (manufacturer_id, name) VALUES (?, ?)', (mid, name))

        conn.commit()
        conn.close()

    def _load_builtin_profiles(self):
        """Load built-in generic profiles into cache"""
        for profile in BUILTIN_PROFILES:
            self._profiles_cache[profile.profile_id] = profile

    # ─────────────────────────────────────────────────────────
    # Profile Management
    # ─────────────────────────────────────────────────────────

    def get_profile(self, profile_id: str) -> Optional[FixtureProfile]:
        """Get a fixture profile by ID"""
        # Check cache first
        if profile_id in self._profiles_cache:
            return self._profiles_cache[profile_id]

        # Load from database
        conn = self._get_db()
        c = conn.cursor()
        c.execute('SELECT * FROM fixture_profiles WHERE profile_id = ?', (profile_id,))
        row = c.fetchone()
        conn.close()

        if row:
            profile = self._row_to_profile(row)
            self._profiles_cache[profile_id] = profile
            return profile
        return None

    def get_all_profiles(self, category: str = None) -> List[FixtureProfile]:
        """Get all fixture profiles, optionally filtered by category"""
        # Start with built-in profiles
        profiles = list(BUILTIN_PROFILES)

        # Add database profiles
        conn = self._get_db()
        c = conn.cursor()
        if category:
            c.execute('SELECT * FROM fixture_profiles WHERE category = ? ORDER BY manufacturer, model', (category,))
        else:
            c.execute('SELECT * FROM fixture_profiles ORDER BY manufacturer, model')

        for row in c.fetchall():
            profile = self._row_to_profile(row)
            # Don't duplicate built-ins
            if not any(p.profile_id == profile.profile_id for p in profiles):
                profiles.append(profile)

        conn.close()
        return profiles

    def save_profile(self, profile: FixtureProfile) -> bool:
        """Save a fixture profile to database"""
        conn = self._get_db()
        c = conn.cursor()

        modes_json = json.dumps([{
            'mode_id': m.mode_id,
            'name': m.name,
            'channel_count': m.channel_count,
            'channels': [asdict(ch) for ch in m.channels]
        } for m in profile.modes])

        c.execute('''INSERT OR REPLACE INTO fixture_profiles
            (profile_id, manufacturer, model, category, modes, rdm_manufacturer_id,
             rdm_device_model_id, source, ofl_key, weight_kg, power_watts, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
            (profile.profile_id, profile.manufacturer, profile.model, profile.category,
             modes_json, profile.rdm_manufacturer_id, profile.rdm_device_model_id,
             profile.source, profile.ofl_key, profile.weight_kg, profile.power_watts,
             datetime.now().isoformat()))

        conn.commit()
        conn.close()

        # Update cache
        self._profiles_cache[profile.profile_id] = profile
        return True

    def _row_to_profile(self, row) -> FixtureProfile:
        """Convert database row to FixtureProfile"""
        modes_data = json.loads(row['modes']) if row['modes'] else []
        modes = []
        for m in modes_data:
            channels = [ChannelCapability(**ch) for ch in m.get('channels', [])]
            modes.append(FixtureMode(
                mode_id=m['mode_id'],
                name=m['name'],
                channel_count=m['channel_count'],
                channels=channels
            ))

        return FixtureProfile(
            profile_id=row['profile_id'],
            manufacturer=row['manufacturer'],
            model=row['model'],
            category=row['category'],
            modes=modes,
            rdm_manufacturer_id=row['rdm_manufacturer_id'],
            rdm_device_model_id=row['rdm_device_model_id'],
            source=row['source'],
            ofl_key=row['ofl_key'],
            weight_kg=row['weight_kg'] or 0,
            power_watts=row['power_watts'] or 0
        )

    # ─────────────────────────────────────────────────────────
    # OFL Integration
    # ─────────────────────────────────────────────────────────

    def search_ofl(self, query: str) -> List[Dict]:
        """Search Open Fixture Library"""
        return self.ofl.search_fixtures(query)

    def import_from_ofl(self, manufacturer: str, fixture: str) -> Optional[FixtureProfile]:
        """Import a fixture from Open Fixture Library"""
        ofl_data = self.ofl.get_fixture(manufacturer, fixture)
        if not ofl_data:
            return None

        profile = self.ofl.ofl_to_profile(ofl_data, manufacturer, fixture)
        if profile:
            self.save_profile(profile)
        return profile

    def get_ofl_manufacturers(self) -> List[Dict]:
        """Get list of manufacturers from OFL"""
        return self.ofl.get_manufacturers()

    # ─────────────────────────────────────────────────────────
    # RDM Integration
    # ─────────────────────────────────────────────────────────

    def find_profile_by_rdm(self, manufacturer_id: int, device_model_id: int) -> Optional[FixtureProfile]:
        """Find a profile matching RDM IDs"""
        # Check database first
        conn = self._get_db()
        c = conn.cursor()
        c.execute('''SELECT * FROM fixture_profiles
            WHERE rdm_manufacturer_id = ? AND rdm_device_model_id = ?''',
            (manufacturer_id, device_model_id))
        row = c.fetchone()
        conn.close()

        if row:
            return self._row_to_profile(row)

        # Could search OFL here by RDM IDs (future enhancement)
        return None

    def get_rdm_manufacturer_name(self, manufacturer_id: int) -> str:
        """Get manufacturer name from RDM ID"""
        conn = self._get_db()
        c = conn.cursor()
        c.execute('SELECT name FROM rdm_manufacturers WHERE manufacturer_id = ?', (manufacturer_id,))
        row = c.fetchone()
        conn.close()
        return row['name'] if row else f"Unknown ({hex(manufacturer_id)})"

    def create_fixture_from_rdm(self, rdm_device: Dict, profile: FixtureProfile = None) -> FixtureInstance:
        """
        Create a fixture instance from RDM device data.
        If no profile provided, tries to auto-match or creates generic.
        """
        uid = rdm_device.get('uid')
        manufacturer_id = rdm_device.get('manufacturer_id', 0)
        device_model_id = rdm_device.get('device_model_id', 0)
        dmx_address = rdm_device.get('dmx_address', 1)
        footprint = rdm_device.get('dmx_footprint', 4)
        universe = rdm_device.get('universe', 1)
        label = rdm_device.get('device_label', '')

        # Try to find matching profile
        if not profile:
            profile = self.find_profile_by_rdm(manufacturer_id, device_model_id)

        # If still no profile, create a generic one based on footprint
        if not profile:
            profile = self._create_generic_profile_for_footprint(footprint, manufacturer_id)

        # Find appropriate mode based on footprint
        mode_id = profile.modes[0].mode_id if profile.modes else 'default'
        for mode in profile.modes:
            if mode.channel_count == footprint:
                mode_id = mode.mode_id
                break

        manufacturer_name = self.get_rdm_manufacturer_name(manufacturer_id)

        return FixtureInstance(
            fixture_id=f"rdm_{uid}",
            name=label or f"{manufacturer_name} @ {dmx_address}",
            profile_id=profile.profile_id,
            mode_id=mode_id,
            universe=universe,
            start_channel=dmx_address,
            rdm_uid=uid
        )

    def _create_generic_profile_for_footprint(self, footprint: int, manufacturer_id: int = 0) -> FixtureProfile:
        """Create a generic profile for unknown fixtures based on channel count"""
        # Try to match common footprints
        if footprint == 1:
            return self._profiles_cache.get('generic-dimmer', BUILTIN_PROFILES[0])
        elif footprint == 3:
            return self._profiles_cache.get('generic-rgb', BUILTIN_PROFILES[1])
        elif footprint == 4:
            return self._profiles_cache.get('generic-rgbw', BUILTIN_PROFILES[2])
        elif footprint == 5:
            return self._profiles_cache.get('generic-rgbwa', BUILTIN_PROFILES[3])

        # Create custom generic profile
        channels = []
        for i in range(footprint):
            channels.append(ChannelCapability(f"Channel {i+1}", "control", 0))

        return FixtureProfile(
            profile_id=f"generic-{footprint}ch",
            manufacturer="Generic",
            model=f"{footprint}-Channel Fixture",
            category="generic",
            modes=[FixtureMode("default", f"{footprint}-Channel", footprint, channels)]
        )


# ============================================================
# Channel Mapper - Translates fixture-level commands to DMX
# ============================================================

class ChannelMapper:
    """
    Translates high-level fixture commands to raw DMX channel values.
    This is the key component for intelligent fixture mapping.
    """

    def __init__(self, library: FixtureLibrary):
        self.library = library

    def get_channels_for_fixture(self, fixture: FixtureInstance, values: Dict[str, int]) -> Dict[str, int]:
        """
        Convert fixture-level values to DMX channels.

        Args:
            fixture: The fixture instance
            values: Dict of capability type -> value (e.g., {"r": 255, "g": 0, "b": 128, "dimmer": 200})

        Returns:
            Dict of channel number (string) -> value
        """
        profile = self.library.get_profile(fixture.profile_id)
        if not profile:
            return {}

        mode = profile.get_mode(fixture.mode_id)
        if not mode:
            return {}

        result = {}
        for i, channel in enumerate(mode.channels):
            dmx_channel = fixture.start_channel + i
            channel_key = str(dmx_channel)

            # Match by type
            value = None
            name_lower = channel.name.lower()

            # Color channels
            if 'red' in name_lower and 'r' in values:
                value = values['r']
            elif 'green' in name_lower and 'g' in values:
                value = values['g']
            elif 'blue' in name_lower and 'b' in values:
                value = values['b']
            elif 'white' in name_lower and 'w' in values:
                value = values['w']
            elif 'amber' in name_lower and 'a' in values:
                value = values['a']
            # Dimmer
            elif channel.type == 'dimmer' and 'dimmer' in values:
                value = values['dimmer']
            elif channel.type == 'dimmer' and 'intensity' in values:
                value = values['intensity']
            # Position
            elif 'pan' in name_lower and 'fine' not in name_lower and 'pan' in values:
                value = values['pan']
            elif 'tilt' in name_lower and 'fine' not in name_lower and 'tilt' in values:
                value = values['tilt']
            # Direct channel override
            elif f"ch{i+1}" in values:
                value = values[f"ch{i+1}"]

            if value is not None:
                result[channel_key] = max(0, min(255, int(value)))

        return result

    def apply_color_to_fixtures(self, fixtures: List[FixtureInstance], r: int, g: int, b: int,
                                 w: int = 0, dimmer: int = 255) -> Dict[str, int]:
        """
        Apply an RGB(W) color to multiple fixtures.
        Returns combined channel dict for all fixtures.
        """
        all_channels = {}
        values = {'r': r, 'g': g, 'b': b, 'w': w, 'dimmer': dimmer}

        for fixture in fixtures:
            channels = self.get_channels_for_fixture(fixture, values)
            all_channels.update(channels)

        return all_channels

    def replicate_pattern_to_fixtures(self, fixtures: List[FixtureInstance],
                                       pattern: Dict[str, int]) -> Dict[str, int]:
        """
        Replicate a channel pattern across multiple fixtures.
        Pattern is relative offsets (0-based).
        """
        all_channels = {}

        for fixture in fixtures:
            profile = self.library.get_profile(fixture.profile_id)
            if not profile:
                continue
            mode = profile.get_mode(fixture.mode_id)
            if not mode:
                continue

            for offset_str, value in pattern.items():
                offset = int(offset_str)
                if offset < mode.channel_count:
                    dmx_channel = fixture.start_channel + offset
                    all_channels[str(dmx_channel)] = value

        return all_channels


# ============================================================
# Singleton instance (will be initialized by aether-core.py)
# ============================================================

_fixture_library: Optional[FixtureLibrary] = None
_channel_mapper: Optional[ChannelMapper] = None

def init_fixture_library(database_path: str):
    """Initialize the fixture library singleton"""
    global _fixture_library, _channel_mapper
    _fixture_library = FixtureLibrary(database_path)
    _channel_mapper = ChannelMapper(_fixture_library)
    print("✓ Fixture Library initialized")
    return _fixture_library

def get_fixture_library() -> Optional[FixtureLibrary]:
    return _fixture_library

def get_channel_mapper() -> Optional[ChannelMapper]:
    return _channel_mapper
