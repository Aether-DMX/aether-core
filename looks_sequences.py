"""
Looks & Sequences Module - Unified Scene/Chase Architecture with Effect Modifiers

This module implements the new canonical data model:
- Look: Base channels + optional modifiers (replaces Scene)
- Sequence: Ordered list of Looks with timing (replaces Chase)

Modifiers are schema-driven effects that animate base channel values.
All modifiers are composable and applied at playback time.

Version: 1.1.0 - Enhanced modifier system with registry and presets
"""

import json
import sqlite3
import time
import threading
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Union, Tuple
from enum import Enum

# Import the ModifierRegistry as single source of truth
from modifier_registry import (
    modifier_registry,
    validate_modifier,
    normalize_modifier,
    get_modifier_schemas,
    get_modifier_presets,
    ModifierType,
)


# ============================================================
# Schema Version - For migrations
# ============================================================
SCHEMA_VERSION = 2  # Bumped for enhanced modifier system




# ============================================================
# Canonical Data Models
# ============================================================

@dataclass
class Modifier:
    """
    A single effect modifier applied to a Look.

    Enhanced with:
    - id: Unique identifier for this modifier instance
    - preset_id: Optional reference to preset used to create this modifier
    """
    id: str  # Unique modifier instance ID
    type: str  # Must be a registered modifier type
    params: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    preset_id: Optional[str] = None  # Which preset this was created from

    def to_dict(self) -> dict:
        result = {
            "id": self.id,
            "type": self.type,
            "params": self.params,
            "enabled": self.enabled,
        }
        if self.preset_id:
            result["preset_id"] = self.preset_id
        return result

    @classmethod
    def from_dict(cls, data: dict) -> "Modifier":
        # Normalize through registry to ensure all params have values
        normalized = normalize_modifier(data)
        return cls(
            id=normalized.get("id", modifier_registry.generate_id()),
            type=normalized.get("type", ""),
            params=normalized.get("params", {}),
            enabled=normalized.get("enabled", True),
            preset_id=normalized.get("preset_id"),
        )

    @classmethod
    def from_preset(cls, modifier_type: str, preset_id: str) -> Optional["Modifier"]:
        """Create a new modifier from a preset"""
        preset_data = modifier_registry.create_from_preset(modifier_type, preset_id)
        if not preset_data:
            return None
        return cls.from_dict(preset_data)

    def validate(self) -> Tuple[bool, Optional[str]]:
        """Validate this modifier against its schema"""
        return validate_modifier(self.to_dict())

    def apply_preset(self, preset_id: str) -> bool:
        """Apply a preset to this modifier, updating params"""
        preset = modifier_registry.get_preset(self.type, preset_id)
        if not preset:
            return False
        self.params.update(preset["params"])
        self.preset_id = preset_id
        return True


@dataclass
class Look:
    """
    A Look is the base unit: channel values + optional modifiers.
    Replaces Scene in the legacy model.
    """
    look_id: str
    name: str
    channels: Dict[str, int]  # {"1": 255, "2": 128} or {"1:1": 255} for universe:channel
    modifiers: List[Modifier] = field(default_factory=list)
    fade_ms: int = 0
    color: str = "blue"  # UI tag color
    icon: str = "lightbulb"
    description: str = ""
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

    # Migration tracking
    migrated_from: Optional[str] = None  # Original scene_id if migrated

    def to_dict(self) -> dict:
        return {
            "look_id": self.look_id,
            "name": self.name,
            "channels": self.channels,
            "modifiers": [m.to_dict() for m in self.modifiers],
            "fade_ms": self.fade_ms,
            "color": self.color,
            "icon": self.icon,
            "description": self.description,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "migrated_from": self.migrated_from,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Look":
        modifiers = [Modifier.from_dict(m) for m in data.get("modifiers", [])]
        return cls(
            look_id=data.get("look_id", ""),
            name=data.get("name", ""),
            channels=data.get("channels", {}),
            modifiers=modifiers,
            fade_ms=data.get("fade_ms", 0),
            color=data.get("color", "blue"),
            icon=data.get("icon", "lightbulb"),
            description=data.get("description", ""),
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
            migrated_from=data.get("migrated_from"),
        )

    @classmethod
    def from_scene(cls, scene: dict) -> "Look":
        """Migrate a legacy Scene to a Look"""
        scene_id = scene.get("scene_id", "")
        return cls(
            look_id=f"look_{int(time.time() * 1000)}",
            name=scene.get("name", "Untitled"),
            channels=scene.get("channels", {}),
            modifiers=[],  # Scenes have no modifiers
            fade_ms=scene.get("fade_ms", 0),
            color=scene.get("color", "blue"),
            icon=scene.get("icon", "lightbulb"),
            description=scene.get("description", ""),
            created_at=scene.get("created_at"),
            updated_at=scene.get("updated_at"),
            migrated_from=scene_id,
        )


@dataclass
class SequenceStep:
    """A single step in a Sequence, containing a Look reference or inline Look data"""
    step_id: str
    name: str
    # Either reference an existing Look or embed channel data inline
    look_id: Optional[str] = None  # Reference to existing Look
    channels: Optional[Dict[str, int]] = None  # Inline channels (if no look_id)
    modifiers: List[Modifier] = field(default_factory=list)  # Step-specific modifiers
    fade_ms: int = 0  # Fade INTO this step
    hold_ms: int = 500  # Hold AFTER fade

    def to_dict(self) -> dict:
        return {
            "step_id": self.step_id,
            "name": self.name,
            "look_id": self.look_id,
            "channels": self.channels,
            "modifiers": [m.to_dict() for m in self.modifiers],
            "fade_ms": self.fade_ms,
            "hold_ms": self.hold_ms,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SequenceStep":
        modifiers = [Modifier.from_dict(m) for m in data.get("modifiers", [])]
        return cls(
            step_id=data.get("step_id", f"step_{int(time.time() * 1000)}"),
            name=data.get("name", "Step"),
            look_id=data.get("look_id"),
            channels=data.get("channels"),
            modifiers=modifiers,
            fade_ms=data.get("fade_ms", 0),
            hold_ms=data.get("hold_ms", 500),
        )

    @classmethod
    def from_chase_step(cls, step: dict, index: int) -> "SequenceStep":
        """Migrate a legacy chase step to a SequenceStep"""
        return cls(
            step_id=f"step_{int(time.time() * 1000)}_{index}",
            name=step.get("name", f"Step {index + 1}"),
            look_id=step.get("scene_id"),  # Chase steps can reference scenes
            channels=step.get("channels"),
            modifiers=[],
            fade_ms=step.get("fade_ms", 0),
            hold_ms=step.get("hold_ms", 500),
        )


@dataclass
class Sequence:
    """
    A Sequence is an ordered list of steps with timing.
    Replaces Chase in the legacy model.
    """
    sequence_id: str
    name: str
    steps: List[SequenceStep] = field(default_factory=list)
    bpm: int = 120  # Beats per minute (affects default step timing)
    loop: bool = True
    color: str = "green"  # UI tag color
    description: str = ""
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

    # Migration tracking
    migrated_from: Optional[str] = None  # Original chase_id if migrated

    def to_dict(self) -> dict:
        return {
            "sequence_id": self.sequence_id,
            "name": self.name,
            "steps": [s.to_dict() for s in self.steps],
            "bpm": self.bpm,
            "loop": self.loop,
            "color": self.color,
            "description": self.description,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "migrated_from": self.migrated_from,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Sequence":
        steps = [SequenceStep.from_dict(s) for s in data.get("steps", [])]
        return cls(
            sequence_id=data.get("sequence_id", ""),
            name=data.get("name", ""),
            steps=steps,
            bpm=data.get("bpm", 120),
            loop=data.get("loop", True),
            color=data.get("color", "green"),
            description=data.get("description", ""),
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
            migrated_from=data.get("migrated_from"),
        )

    @classmethod
    def from_chase(cls, chase: dict) -> "Sequence":
        """Migrate a legacy Chase to a Sequence"""
        chase_id = chase.get("chase_id", "")
        legacy_steps = chase.get("steps", [])
        if isinstance(legacy_steps, str):
            legacy_steps = json.loads(legacy_steps)

        steps = [SequenceStep.from_chase_step(s, i) for i, s in enumerate(legacy_steps)]

        return cls(
            sequence_id=f"sequence_{int(time.time() * 1000)}",
            name=chase.get("name", "Untitled"),
            steps=steps,
            bpm=chase.get("bpm", 120),
            loop=chase.get("loop", True),
            color=chase.get("color", "green"),
            description=chase.get("description", ""),
            created_at=chase.get("created_at"),
            updated_at=chase.get("updated_at"),
            migrated_from=chase_id,
        )


# ============================================================
# Database Schema
# ============================================================

def init_looks_sequences_tables(db_path: str):
    """Initialize the looks and sequences tables"""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    # Looks table
    c.execute('''CREATE TABLE IF NOT EXISTS looks (
        look_id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        channels TEXT NOT NULL,
        modifiers TEXT DEFAULT '[]',
        fade_ms INTEGER DEFAULT 0,
        color TEXT DEFAULT 'blue',
        icon TEXT DEFAULT 'lightbulb',
        description TEXT DEFAULT '',
        migrated_from TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )''')

    # Sequences table
    c.execute('''CREATE TABLE IF NOT EXISTS sequences (
        sequence_id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        steps TEXT NOT NULL,
        bpm INTEGER DEFAULT 120,
        loop BOOLEAN DEFAULT 1,
        color TEXT DEFAULT 'green',
        description TEXT DEFAULT '',
        migrated_from TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )''')

    # Schema version tracking
    c.execute('''CREATE TABLE IF NOT EXISTS schema_versions (
        module TEXT PRIMARY KEY,
        version INTEGER NOT NULL,
        migrated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )''')

    # Artifact version history for looks and sequences
    c.execute('''CREATE TABLE IF NOT EXISTS artifact_versions (
        version_id TEXT PRIMARY KEY,
        artifact_id TEXT NOT NULL,
        artifact_type TEXT NOT NULL,
        version_number INTEGER NOT NULL,
        data_json TEXT NOT NULL,
        author TEXT DEFAULT 'user',
        message TEXT DEFAULT '',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )''')

    # Record schema version
    c.execute('''INSERT OR REPLACE INTO schema_versions (module, version, migrated_at)
                 VALUES ('looks_sequences', ?, CURRENT_TIMESTAMP)''', (SCHEMA_VERSION,))

    # Create indices for common queries
    c.execute('CREATE INDEX IF NOT EXISTS idx_looks_name ON looks(name)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_looks_migrated_from ON looks(migrated_from)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_sequences_name ON sequences(name)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_sequences_migrated_from ON sequences(migrated_from)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_artifact_versions_artifact ON artifact_versions(artifact_id, artifact_type)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_artifact_versions_created ON artifact_versions(created_at DESC)')

    conn.commit()
    conn.close()
    print("✅ Looks & Sequences tables initialized")


# ============================================================
# CRUD Operations
# ============================================================

class LooksSequencesManager:
    """
    Manager for Look and Sequence CRUD operations.
    Thread-safe with connection-per-operation pattern.
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.lock = threading.Lock()
        init_looks_sequences_tables(db_path)

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    # ---- Looks CRUD ----

    def create_look(self, look: Look) -> Look:
        """Create a new Look"""
        with self.lock:
            conn = self._get_conn()
            c = conn.cursor()

            now = datetime.now().isoformat()
            look.created_at = now
            look.updated_at = now

            if not look.look_id:
                look.look_id = f"look_{int(time.time() * 1000)}"

            c.execute('''INSERT INTO looks
                        (look_id, name, channels, modifiers, fade_ms, color, icon, description, migrated_from, created_at, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                     (look.look_id, look.name, json.dumps(look.channels),
                      json.dumps([m.to_dict() for m in look.modifiers]),
                      look.fade_ms, look.color, look.icon, look.description,
                      look.migrated_from, look.created_at, look.updated_at))

            conn.commit()
            conn.close()
            print(f"✅ Created look: {look.name} ({look.look_id})")
            return look

    def get_look(self, look_id: str) -> Optional[Look]:
        """Get a Look by ID"""
        conn = self._get_conn()
        c = conn.cursor()
        c.execute('SELECT * FROM looks WHERE look_id = ?', (look_id,))
        row = c.fetchone()
        conn.close()

        if not row:
            return None

        return self._row_to_look(row)

    def get_all_looks(self) -> List[Look]:
        """Get all Looks"""
        conn = self._get_conn()
        c = conn.cursor()
        c.execute('SELECT * FROM looks ORDER BY name')
        rows = c.fetchall()
        conn.close()

        return [self._row_to_look(row) for row in rows]

    def update_look(self, look_id: str, updates: dict, save_version: bool = True) -> Optional[Look]:
        """Update a Look"""
        with self.lock:
            conn = self._get_conn()
            c = conn.cursor()

            # Get existing
            c.execute('SELECT * FROM looks WHERE look_id = ?', (look_id,))
            row = c.fetchone()
            if not row:
                conn.close()
                return None

            existing = self._row_to_look(row)

            # Save version before modifying (outside lock to avoid deadlock)
            if save_version:
                conn.close()
                self._save_version(look_id, 'look', existing.to_dict(), 'Auto-save before update')
                self.cleanup_old_versions(look_id, 'look')
                conn = self._get_conn()
                c = conn.cursor()

            # Apply updates
            if "name" in updates:
                existing.name = updates["name"]
            if "channels" in updates:
                existing.channels = updates["channels"]
            if "modifiers" in updates:
                existing.modifiers = [Modifier.from_dict(m) for m in updates["modifiers"]]
            if "fade_ms" in updates:
                existing.fade_ms = updates["fade_ms"]
            if "color" in updates:
                existing.color = updates["color"]
            if "icon" in updates:
                existing.icon = updates["icon"]
            if "description" in updates:
                existing.description = updates["description"]

            existing.updated_at = datetime.now().isoformat()

            c.execute('''UPDATE looks SET
                        name=?, channels=?, modifiers=?, fade_ms=?, color=?, icon=?, description=?, updated_at=?
                        WHERE look_id=?''',
                     (existing.name, json.dumps(existing.channels),
                      json.dumps([m.to_dict() for m in existing.modifiers]),
                      existing.fade_ms, existing.color, existing.icon, existing.description,
                      existing.updated_at, look_id))

            conn.commit()
            conn.close()
            print(f"✅ Updated look: {existing.name} ({look_id})")
            return existing

    def delete_look(self, look_id: str) -> bool:
        """Delete a Look"""
        with self.lock:
            conn = self._get_conn()
            c = conn.cursor()
            c.execute('DELETE FROM looks WHERE look_id = ?', (look_id,))
            deleted = c.rowcount > 0
            conn.commit()
            conn.close()
            if deleted:
                print(f"🗑️ Deleted look: {look_id}")
            return deleted

    def _row_to_look(self, row: sqlite3.Row) -> Look:
        """Convert a database row to a Look object"""
        channels = json.loads(row["channels"]) if row["channels"] else {}
        modifiers_data = json.loads(row["modifiers"]) if row["modifiers"] else []
        modifiers = [Modifier.from_dict(m) for m in modifiers_data]

        return Look(
            look_id=row["look_id"],
            name=row["name"],
            channels=channels,
            modifiers=modifiers,
            fade_ms=row["fade_ms"] or 0,
            color=row["color"] or "blue",
            icon=row["icon"] or "lightbulb",
            description=row["description"] or "",
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            migrated_from=row["migrated_from"],
        )

    # ---- Sequences CRUD ----

    def create_sequence(self, sequence: Sequence) -> Sequence:
        """Create a new Sequence"""
        with self.lock:
            conn = self._get_conn()
            c = conn.cursor()

            now = datetime.now().isoformat()
            sequence.created_at = now
            sequence.updated_at = now

            if not sequence.sequence_id:
                sequence.sequence_id = f"sequence_{int(time.time() * 1000)}"

            c.execute('''INSERT INTO sequences
                        (sequence_id, name, steps, bpm, loop, color, description, migrated_from, created_at, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                     (sequence.sequence_id, sequence.name,
                      json.dumps([s.to_dict() for s in sequence.steps]),
                      sequence.bpm, sequence.loop, sequence.color, sequence.description,
                      sequence.migrated_from, sequence.created_at, sequence.updated_at))

            conn.commit()
            conn.close()
            print(f"✅ Created sequence: {sequence.name} ({sequence.sequence_id})")
            return sequence

    def get_sequence(self, sequence_id: str) -> Optional[Sequence]:
        """Get a Sequence by ID"""
        conn = self._get_conn()
        c = conn.cursor()
        c.execute('SELECT * FROM sequences WHERE sequence_id = ?', (sequence_id,))
        row = c.fetchone()
        conn.close()

        if not row:
            return None

        return self._row_to_sequence(row)

    def get_all_sequences(self) -> List[Sequence]:
        """Get all Sequences"""
        conn = self._get_conn()
        c = conn.cursor()
        c.execute('SELECT * FROM sequences ORDER BY name')
        rows = c.fetchall()
        conn.close()

        return [self._row_to_sequence(row) for row in rows]

    def update_sequence(self, sequence_id: str, updates: dict, save_version: bool = True) -> Optional[Sequence]:
        """Update a Sequence"""
        with self.lock:
            conn = self._get_conn()
            c = conn.cursor()

            # Get existing
            c.execute('SELECT * FROM sequences WHERE sequence_id = ?', (sequence_id,))
            row = c.fetchone()
            if not row:
                conn.close()
                return None

            existing = self._row_to_sequence(row)

            # Save version before modifying (outside lock to avoid deadlock)
            if save_version:
                conn.close()
                self._save_version(sequence_id, 'sequence', existing.to_dict(), 'Auto-save before update')
                self.cleanup_old_versions(sequence_id, 'sequence')
                conn = self._get_conn()
                c = conn.cursor()

            # Apply updates
            if "name" in updates:
                existing.name = updates["name"]
            if "steps" in updates:
                existing.steps = [SequenceStep.from_dict(s) for s in updates["steps"]]
            if "bpm" in updates:
                existing.bpm = updates["bpm"]
            if "loop" in updates:
                existing.loop = updates["loop"]
            if "color" in updates:
                existing.color = updates["color"]
            if "description" in updates:
                existing.description = updates["description"]

            existing.updated_at = datetime.now().isoformat()

            c.execute('''UPDATE sequences SET
                        name=?, steps=?, bpm=?, loop=?, color=?, description=?, updated_at=?
                        WHERE sequence_id=?''',
                     (existing.name, json.dumps([s.to_dict() for s in existing.steps]),
                      existing.bpm, existing.loop, existing.color, existing.description,
                      existing.updated_at, sequence_id))

            conn.commit()
            conn.close()
            print(f"✅ Updated sequence: {existing.name} ({sequence_id})")
            return existing

    def delete_sequence(self, sequence_id: str) -> bool:
        """Delete a Sequence"""
        with self.lock:
            conn = self._get_conn()
            c = conn.cursor()
            c.execute('DELETE FROM sequences WHERE sequence_id = ?', (sequence_id,))
            deleted = c.rowcount > 0
            conn.commit()
            conn.close()
            if deleted:
                print(f"🗑️ Deleted sequence: {sequence_id}")
            return deleted

    def _row_to_sequence(self, row: sqlite3.Row) -> Sequence:
        """Convert a database row to a Sequence object"""
        steps_data = json.loads(row["steps"]) if row["steps"] else []
        steps = [SequenceStep.from_dict(s) for s in steps_data]

        return Sequence(
            sequence_id=row["sequence_id"],
            name=row["name"],
            steps=steps,
            bpm=row["bpm"] or 120,
            loop=bool(row["loop"]),
            color=row["color"] or "green",
            description=row["description"] or "",
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            migrated_from=row["migrated_from"],
        )

    # ---- Version History ----

    def _save_version(self, artifact_id: str, artifact_type: str, data: dict, message: str = "") -> str:
        """Save a version snapshot of an artifact before modification"""
        conn = self._get_conn()
        c = conn.cursor()

        # Get next version number for this artifact
        c.execute('''SELECT COALESCE(MAX(version_number), 0) + 1
                    FROM artifact_versions
                    WHERE artifact_id = ? AND artifact_type = ?''',
                  (artifact_id, artifact_type))
        version_number = c.fetchone()[0]

        version_id = f"ver_{artifact_id}_{version_number}_{int(time.time() * 1000)}"
        now = datetime.now().isoformat()

        c.execute('''INSERT INTO artifact_versions
                    (version_id, artifact_id, artifact_type, version_number, data_json, author, message, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
                  (version_id, artifact_id, artifact_type, version_number,
                   json.dumps(data), 'user', message, now))

        conn.commit()
        conn.close()
        print(f"📜 Saved version {version_number} of {artifact_type} {artifact_id}")
        return version_id

    def get_versions(self, artifact_id: str, artifact_type: str) -> List[dict]:
        """Get all versions of an artifact"""
        conn = self._get_conn()
        c = conn.cursor()
        c.execute('''SELECT version_id, artifact_id, artifact_type, version_number,
                           data_json, author, message, created_at
                    FROM artifact_versions
                    WHERE artifact_id = ? AND artifact_type = ?
                    ORDER BY version_number DESC''',
                  (artifact_id, artifact_type))
        rows = c.fetchall()
        conn.close()

        return [{
            'version_id': row['version_id'],
            'artifact_id': row['artifact_id'],
            'artifact_type': row['artifact_type'],
            'version_number': row['version_number'],
            'data': json.loads(row['data_json']),
            'author': row['author'],
            'message': row['message'],
            'created_at': row['created_at']
        } for row in rows]

    def get_version(self, version_id: str) -> Optional[dict]:
        """Get a specific version by ID"""
        conn = self._get_conn()
        c = conn.cursor()
        c.execute('''SELECT version_id, artifact_id, artifact_type, version_number,
                           data_json, author, message, created_at
                    FROM artifact_versions
                    WHERE version_id = ?''',
                  (version_id,))
        row = c.fetchone()
        conn.close()

        if not row:
            return None

        return {
            'version_id': row['version_id'],
            'artifact_id': row['artifact_id'],
            'artifact_type': row['artifact_type'],
            'version_number': row['version_number'],
            'data': json.loads(row['data_json']),
            'author': row['author'],
            'message': row['message'],
            'created_at': row['created_at']
        }

    def revert_to_version(self, version_id: str) -> Optional[dict]:
        """Revert an artifact to a specific version"""
        version = self.get_version(version_id)
        if not version:
            return None

        artifact_type = version['artifact_type']
        artifact_id = version['artifact_id']
        data = version['data']

        if artifact_type == 'look':
            # Save current state before reverting
            current = self.get_look(artifact_id)
            if current:
                self._save_version(artifact_id, 'look', current.to_dict(),
                                   f"Before revert to v{version['version_number']}")
            # Apply version data
            result = self.update_look(artifact_id, data)
            return result.to_dict() if result else None

        elif artifact_type == 'sequence':
            # Save current state before reverting
            current = self.get_sequence(artifact_id)
            if current:
                self._save_version(artifact_id, 'sequence', current.to_dict(),
                                   f"Before revert to v{version['version_number']}")
            # Apply version data
            result = self.update_sequence(artifact_id, data)
            return result.to_dict() if result else None

        return None

    def cleanup_old_versions(self, artifact_id: str, artifact_type: str, keep_count: int = 20):
        """Keep only the most recent N versions of an artifact"""
        with self.lock:
            conn = self._get_conn()
            c = conn.cursor()

            # Get version IDs to delete (all except the most recent keep_count)
            c.execute('''SELECT version_id FROM artifact_versions
                        WHERE artifact_id = ? AND artifact_type = ?
                        ORDER BY version_number DESC
                        LIMIT -1 OFFSET ?''',
                      (artifact_id, artifact_type, keep_count))
            to_delete = [row['version_id'] for row in c.fetchall()]

            if to_delete:
                placeholders = ','.join('?' * len(to_delete))
                c.execute(f'DELETE FROM artifact_versions WHERE version_id IN ({placeholders})',
                          to_delete)
                conn.commit()
                print(f"🧹 Cleaned up {len(to_delete)} old versions of {artifact_type} {artifact_id}")

            conn.close()


# ============================================================
# Migration Functions
# ============================================================

def migrate_scenes_to_looks(db_path: str, manager: LooksSequencesManager) -> dict:
    """
    Migrate all legacy scenes to looks.
    Returns migration report.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    # Check if scenes table exists
    c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='scenes'")
    if not c.fetchone():
        conn.close()
        return {"status": "skipped", "reason": "No scenes table found", "migrated": 0}

    # Get all scenes
    c.execute('SELECT * FROM scenes')
    scenes = c.fetchall()
    conn.close()

    migrated = 0
    errors = []

    for scene in scenes:
        try:
            scene_dict = dict(scene)
            # Parse channels JSON
            if scene_dict.get("channels"):
                scene_dict["channels"] = json.loads(scene_dict["channels"])
            else:
                scene_dict["channels"] = {}

            # Check if already migrated
            existing = manager.get_all_looks()
            already_migrated = any(l.migrated_from == scene_dict.get("scene_id") for l in existing)

            if already_migrated:
                continue

            # Create Look from Scene
            look = Look.from_scene(scene_dict)
            manager.create_look(look)
            migrated += 1

        except Exception as e:
            errors.append({"scene_id": scene_dict.get("scene_id"), "error": str(e)})

    return {
        "status": "completed",
        "migrated": migrated,
        "total_scenes": len(scenes),
        "errors": errors
    }


def migrate_chases_to_sequences(db_path: str, manager: LooksSequencesManager) -> dict:
    """
    Migrate all legacy chases to sequences.
    Returns migration report.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    # Check if chases table exists
    c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='chases'")
    if not c.fetchone():
        conn.close()
        return {"status": "skipped", "reason": "No chases table found", "migrated": 0}

    # Get all chases
    c.execute('SELECT * FROM chases')
    chases = c.fetchall()
    conn.close()

    migrated = 0
    errors = []

    for chase in chases:
        try:
            chase_dict = dict(chase)

            # Check if already migrated
            existing = manager.get_all_sequences()
            already_migrated = any(s.migrated_from == chase_dict.get("chase_id") for s in existing)

            if already_migrated:
                continue

            # Create Sequence from Chase
            sequence = Sequence.from_chase(chase_dict)
            manager.create_sequence(sequence)
            migrated += 1

        except Exception as e:
            errors.append({"chase_id": chase_dict.get("chase_id"), "error": str(e)})

    return {
        "status": "completed",
        "migrated": migrated,
        "total_chases": len(chases),
        "errors": errors
    }


def run_full_migration(db_path: str) -> dict:
    """
    Run complete migration from legacy scenes/chases to looks/sequences.
    Safe to run multiple times - skips already migrated items.
    """
    print("🔄 Starting Looks & Sequences migration...")

    manager = LooksSequencesManager(db_path)

    scenes_report = migrate_scenes_to_looks(db_path, manager)
    print(f"   Scenes → Looks: {scenes_report['migrated']} migrated")

    chases_report = migrate_chases_to_sequences(db_path, manager)
    print(f"   Chases → Sequences: {chases_report['migrated']} migrated")

    print("✅ Migration complete")

    return {
        "scenes_to_looks": scenes_report,
        "chases_to_sequences": chases_report
    }


# ============================================================
# API Endpoint Helpers
# ============================================================



def validate_look_data(data: dict) -> tuple[bool, Optional[str]]:
    """Validate Look creation/update data"""
    if not data.get("name"):
        return False, "Look name is required"

    if not data.get("channels"):
        return False, "Look must have at least one channel"

    # Validate modifiers if present
    for mod in data.get("modifiers", []):
        valid, error = validate_modifier(mod)
        if not valid:
            return False, error

    return True, None


def validate_sequence_data(data: dict) -> tuple[bool, Optional[str]]:
    """Validate Sequence creation/update data"""
    if not data.get("name"):
        return False, "Sequence name is required"

    steps = data.get("steps", [])
    if not steps:
        return False, "Sequence must have at least one step"

    for i, step in enumerate(steps):
        # Each step must have either look_id or channels
        if not step.get("look_id") and not step.get("channels"):
            return False, f"Step {i+1} must have either look_id or channels"

        # Validate step modifiers if present
        for mod in step.get("modifiers", []):
            valid, error = validate_modifier(mod)
            if not valid:
                return False, f"Step {i+1}: {error}"

    return True, None
