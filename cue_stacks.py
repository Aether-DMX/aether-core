"""
Cue Stacks Module - Manual Theatrical Cueing with Go/Back Controls

This module implements a theatrical cueing system:
- CueStack: An ordered list of cues for manual triggering
- Cue: A single cue with a number, name, look reference, and timing

Unlike Sequences (BPM-based auto-playback), Cue Stacks are manually triggered
with Go and Back buttons, similar to theatrical lighting consoles.

Version: 1.0.0 - Initial implementation
"""

import json
import sqlite3
import time
import threading
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum


# ============================================================
# Schema Version - For migrations
# ============================================================
SCHEMA_VERSION = 1


# ============================================================
# Canonical Data Models
# ============================================================

@dataclass
class Cue:
    """
    A single cue in a CueStack.

    Cue numbers are strings to support theatrical conventions like:
    - "1", "2", "3" (simple numbering)
    - "2.5" (point cues inserted between)
    - "10A", "10B" (lettered variants)
    """
    cue_id: str
    cue_number: str  # "1", "2.5", "10A" - theatrical cue number
    name: str
    look_id: Optional[str] = None  # Reference to Look for channel data
    channels: Optional[Dict[str, int]] = None  # Inline channels if no look_id
    fade_time_ms: int = 1000  # Fade INTO this cue
    wait_time_ms: int = 0  # 0 = manual trigger, >0 = auto-follow after this delay
    follow_time_ms: int = 0  # Deprecated - use wait_time_ms
    notes: str = ""  # Operator notes

    def to_dict(self) -> dict:
        return {
            "cue_id": self.cue_id,
            "cue_number": self.cue_number,
            "name": self.name,
            "look_id": self.look_id,
            "channels": self.channels,
            "fade_time_ms": self.fade_time_ms,
            "wait_time_ms": self.wait_time_ms,
            "follow_time_ms": self.follow_time_ms,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Cue":
        return cls(
            cue_id=data.get("cue_id", f"cue_{int(time.time() * 1000)}"),
            cue_number=str(data.get("cue_number", "1")),
            name=data.get("name", "Cue"),
            look_id=data.get("look_id"),
            channels=data.get("channels"),
            fade_time_ms=data.get("fade_time_ms", 1000),
            wait_time_ms=data.get("wait_time_ms", 0),
            follow_time_ms=data.get("follow_time_ms", 0),
            notes=data.get("notes", ""),
        )

    @staticmethod
    def sort_key(cue_number: str) -> Tuple:
        """
        Generate a sort key for cue numbers that handles:
        - Simple numbers: "1" < "2" < "10"
        - Point cues: "2" < "2.5" < "3"
        - Lettered cues: "10" < "10A" < "10B" < "11"
        """
        import re
        parts = re.split(r'(\d+\.?\d*)', cue_number)
        result = []
        for part in parts:
            if not part:
                continue
            try:
                # Try to parse as number
                result.append((0, float(part)))
            except ValueError:
                # It's a letter suffix
                result.append((1, part.upper()))
        return tuple(result) if result else (0, 0)


@dataclass
class CueStack:
    """
    A CueStack is an ordered list of cues for manual theatrical cueing.
    """
    stack_id: str
    name: str
    cues: List[Cue] = field(default_factory=list)
    color: str = "purple"  # UI tag color
    description: str = ""
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "stack_id": self.stack_id,
            "name": self.name,
            "cues": [c.to_dict() for c in self.cues],
            "color": self.color,
            "description": self.description,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CueStack":
        cues = [Cue.from_dict(c) for c in data.get("cues", [])]
        return cls(
            stack_id=data.get("stack_id", ""),
            name=data.get("name", ""),
            cues=cues,
            color=data.get("color", "purple"),
            description=data.get("description", ""),
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
        )

    def sort_cues(self):
        """Sort cues by cue number"""
        self.cues.sort(key=lambda c: Cue.sort_key(c.cue_number))

    def get_cue_by_number(self, cue_number: str) -> Optional[Cue]:
        """Find a cue by its number"""
        for cue in self.cues:
            if cue.cue_number == cue_number:
                return cue
        return None

    def get_cue_index(self, cue_number: str) -> int:
        """Get the index of a cue by number, -1 if not found"""
        for i, cue in enumerate(self.cues):
            if cue.cue_number == cue_number:
                return i
        return -1


# ============================================================
# Playback State
# ============================================================

@dataclass
class CueStackPlaybackState:
    """
    Runtime state for a playing cue stack.
    Not persisted - only exists during playback.
    """
    stack_id: str
    current_index: int = -1  # -1 = not started, 0+ = current cue index
    is_playing: bool = False
    last_go_time: Optional[float] = None
    fade_progress: float = 0.0  # 0.0 to 1.0

    def to_dict(self) -> dict:
        return {
            "stack_id": self.stack_id,
            "current_index": self.current_index,
            "is_playing": self.is_playing,
            "last_go_time": self.last_go_time,
            "fade_progress": self.fade_progress,
        }


# ============================================================
# Database Schema
# ============================================================

def init_cue_stacks_tables(db_path: str):
    """Initialize the cue stacks table"""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    # Cue Stacks table (cues stored as JSON array)
    c.execute('''CREATE TABLE IF NOT EXISTS cue_stacks (
        stack_id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        cues TEXT NOT NULL DEFAULT '[]',
        color TEXT DEFAULT 'purple',
        description TEXT DEFAULT '',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )''')

    # Schema version tracking
    c.execute('''CREATE TABLE IF NOT EXISTS schema_versions (
        module TEXT PRIMARY KEY,
        version INTEGER NOT NULL,
        migrated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )''')

    # Record schema version
    c.execute('''INSERT OR REPLACE INTO schema_versions (module, version, migrated_at)
                 VALUES ('cue_stacks', ?, CURRENT_TIMESTAMP)''', (SCHEMA_VERSION,))

    # Create indices
    c.execute('CREATE INDEX IF NOT EXISTS idx_cue_stacks_name ON cue_stacks(name)')

    conn.commit()
    conn.close()
    print("✅ Cue Stacks table initialized")


# ============================================================
# CRUD Operations
# ============================================================

class CueStacksManager:
    """
    Manager for CueStack CRUD operations.
    Thread-safe with connection-per-operation pattern.
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.lock = threading.Lock()
        init_cue_stacks_tables(db_path)

        # Runtime playback state (not persisted)
        self._playback_states: Dict[str, CueStackPlaybackState] = {}

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    # ---- CueStack CRUD ----

    def create_cue_stack(self, stack: CueStack) -> CueStack:
        """Create a new CueStack"""
        with self.lock:
            conn = self._get_conn()
            c = conn.cursor()

            now = datetime.now().isoformat()
            stack.created_at = now
            stack.updated_at = now

            if not stack.stack_id:
                stack.stack_id = f"stack_{int(time.time() * 1000)}"

            # Sort cues before saving
            stack.sort_cues()

            c.execute('''INSERT INTO cue_stacks
                        (stack_id, name, cues, color, description, created_at, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?)''',
                     (stack.stack_id, stack.name,
                      json.dumps([c.to_dict() for c in stack.cues]),
                      stack.color, stack.description,
                      stack.created_at, stack.updated_at))

            conn.commit()
            conn.close()
            print(f"✅ Created cue stack: {stack.name} ({stack.stack_id})")
            return stack

    def get_cue_stack(self, stack_id: str) -> Optional[CueStack]:
        """Get a CueStack by ID"""
        conn = self._get_conn()
        c = conn.cursor()
        c.execute('SELECT * FROM cue_stacks WHERE stack_id = ?', (stack_id,))
        row = c.fetchone()
        conn.close()

        if not row:
            return None

        return self._row_to_cue_stack(row)

    def get_all_cue_stacks(self) -> List[CueStack]:
        """Get all CueStacks"""
        conn = self._get_conn()
        c = conn.cursor()
        c.execute('SELECT * FROM cue_stacks ORDER BY name')
        rows = c.fetchall()
        conn.close()

        return [self._row_to_cue_stack(row) for row in rows]

    def update_cue_stack(self, stack_id: str, updates: dict) -> Optional[CueStack]:
        """Update a CueStack"""
        with self.lock:
            conn = self._get_conn()
            c = conn.cursor()

            # Get existing
            c.execute('SELECT * FROM cue_stacks WHERE stack_id = ?', (stack_id,))
            row = c.fetchone()
            if not row:
                conn.close()
                return None

            existing = self._row_to_cue_stack(row)

            # Apply updates
            if "name" in updates:
                existing.name = updates["name"]
            if "cues" in updates:
                existing.cues = [Cue.from_dict(c) for c in updates["cues"]]
                existing.sort_cues()
            if "color" in updates:
                existing.color = updates["color"]
            if "description" in updates:
                existing.description = updates["description"]

            existing.updated_at = datetime.now().isoformat()

            c.execute('''UPDATE cue_stacks SET
                        name=?, cues=?, color=?, description=?, updated_at=?
                        WHERE stack_id=?''',
                     (existing.name, json.dumps([c.to_dict() for c in existing.cues]),
                      existing.color, existing.description,
                      existing.updated_at, stack_id))

            conn.commit()
            conn.close()
            print(f"✅ Updated cue stack: {existing.name} ({stack_id})")
            return existing

    def delete_cue_stack(self, stack_id: str) -> bool:
        """Delete a CueStack"""
        with self.lock:
            conn = self._get_conn()
            c = conn.cursor()
            c.execute('DELETE FROM cue_stacks WHERE stack_id = ?', (stack_id,))
            deleted = c.rowcount > 0
            conn.commit()
            conn.close()
            if deleted:
                # Clean up playback state
                if stack_id in self._playback_states:
                    del self._playback_states[stack_id]
                print(f"🗑️ Deleted cue stack: {stack_id}")
            return deleted

    def _row_to_cue_stack(self, row: sqlite3.Row) -> CueStack:
        """Convert a database row to a CueStack object"""
        cues_data = json.loads(row["cues"]) if row["cues"] else []
        cues = [Cue.from_dict(c) for c in cues_data]

        return CueStack(
            stack_id=row["stack_id"],
            name=row["name"],
            cues=cues,
            color=row["color"] or "purple",
            description=row["description"] or "",
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    # ---- Playback Control ----

    def get_playback_state(self, stack_id: str) -> Optional[CueStackPlaybackState]:
        """Get current playback state for a stack"""
        return self._playback_states.get(stack_id)

    def go(self, stack_id: str, look_resolver=None) -> dict:
        """
        Execute the next cue (Go button).
        Returns the cue data to play.

        Args:
            stack_id: The cue stack ID
            look_resolver: Optional callable(look_id) -> channels dict
        """
        stack = self.get_cue_stack(stack_id)
        if not stack or not stack.cues:
            return {"success": False, "error": "Stack not found or empty"}

        # Get or create playback state
        if stack_id not in self._playback_states:
            self._playback_states[stack_id] = CueStackPlaybackState(stack_id=stack_id)

        state = self._playback_states[stack_id]

        # Move to next cue
        state.current_index += 1

        # Check bounds
        if state.current_index >= len(stack.cues):
            # At end of stack
            state.current_index = len(stack.cues) - 1
            return {
                "success": True,
                "at_end": True,
                "cue": stack.cues[state.current_index].to_dict(),
                "message": "Already at last cue"
            }

        state.is_playing = True
        state.last_go_time = time.time()
        state.fade_progress = 0.0

        current_cue = stack.cues[state.current_index]

        # Resolve channels
        channels = current_cue.channels
        if not channels and current_cue.look_id and look_resolver:
            channels = look_resolver(current_cue.look_id)

        # Get next cue info (for preview)
        next_cue = None
        if state.current_index + 1 < len(stack.cues):
            next_cue = stack.cues[state.current_index + 1].to_dict()

        # Get previous cue info
        prev_cue = None
        if state.current_index > 0:
            prev_cue = stack.cues[state.current_index - 1].to_dict()

        return {
            "success": True,
            "cue": current_cue.to_dict(),
            "channels": channels,
            "fade_time_ms": current_cue.fade_time_ms,
            "current_index": state.current_index,
            "total_cues": len(stack.cues),
            "next_cue": next_cue,
            "prev_cue": prev_cue,
            "auto_follow_ms": current_cue.wait_time_ms if current_cue.wait_time_ms > 0 else None,
        }

    def back(self, stack_id: str, look_resolver=None) -> dict:
        """
        Go back to previous cue (Back button).
        Returns the cue data to play.
        """
        stack = self.get_cue_stack(stack_id)
        if not stack or not stack.cues:
            return {"success": False, "error": "Stack not found or empty"}

        # Get or create playback state
        if stack_id not in self._playback_states:
            self._playback_states[stack_id] = CueStackPlaybackState(stack_id=stack_id)

        state = self._playback_states[stack_id]

        # Move to previous cue
        state.current_index -= 1

        # Check bounds
        if state.current_index < 0:
            state.current_index = 0
            return {
                "success": True,
                "at_start": True,
                "cue": stack.cues[0].to_dict(),
                "message": "Already at first cue"
            }

        state.is_playing = True
        state.last_go_time = time.time()
        state.fade_progress = 0.0

        current_cue = stack.cues[state.current_index]

        # Resolve channels
        channels = current_cue.channels
        if not channels and current_cue.look_id and look_resolver:
            channels = look_resolver(current_cue.look_id)

        # Get next/prev cue info
        next_cue = None
        if state.current_index + 1 < len(stack.cues):
            next_cue = stack.cues[state.current_index + 1].to_dict()

        prev_cue = None
        if state.current_index > 0:
            prev_cue = stack.cues[state.current_index - 1].to_dict()

        return {
            "success": True,
            "cue": current_cue.to_dict(),
            "channels": channels,
            "fade_time_ms": current_cue.fade_time_ms,
            "current_index": state.current_index,
            "total_cues": len(stack.cues),
            "next_cue": next_cue,
            "prev_cue": prev_cue,
        }

    def goto(self, stack_id: str, cue_number: str, look_resolver=None) -> dict:
        """
        Jump to a specific cue by number.
        """
        stack = self.get_cue_stack(stack_id)
        if not stack or not stack.cues:
            return {"success": False, "error": "Stack not found or empty"}

        # Find the cue
        index = stack.get_cue_index(cue_number)
        if index < 0:
            return {"success": False, "error": f"Cue {cue_number} not found"}

        # Get or create playback state
        if stack_id not in self._playback_states:
            self._playback_states[stack_id] = CueStackPlaybackState(stack_id=stack_id)

        state = self._playback_states[stack_id]
        state.current_index = index
        state.is_playing = True
        state.last_go_time = time.time()
        state.fade_progress = 0.0

        current_cue = stack.cues[index]

        # Resolve channels
        channels = current_cue.channels
        if not channels and current_cue.look_id and look_resolver:
            channels = look_resolver(current_cue.look_id)

        # Get next/prev cue info
        next_cue = None
        if index + 1 < len(stack.cues):
            next_cue = stack.cues[index + 1].to_dict()

        prev_cue = None
        if index > 0:
            prev_cue = stack.cues[index - 1].to_dict()

        return {
            "success": True,
            "cue": current_cue.to_dict(),
            "channels": channels,
            "fade_time_ms": current_cue.fade_time_ms,
            "current_index": index,
            "total_cues": len(stack.cues),
            "next_cue": next_cue,
            "prev_cue": prev_cue,
        }

    def stop(self, stack_id: str) -> dict:
        """Stop playback and reset state"""
        if stack_id in self._playback_states:
            del self._playback_states[stack_id]
        return {"success": True, "message": "Playback stopped"}

    def get_status(self, stack_id: str) -> dict:
        """Get current playback status"""
        stack = self.get_cue_stack(stack_id)
        if not stack:
            return {"success": False, "error": "Stack not found"}

        state = self._playback_states.get(stack_id)

        if not state or state.current_index < 0:
            return {
                "success": True,
                "stack_id": stack_id,
                "stack_name": stack.name,
                "is_playing": False,
                "current_cue": None,
                "current_index": -1,
                "total_cues": len(stack.cues),
            }

        current_cue = stack.cues[state.current_index] if state.current_index < len(stack.cues) else None
        next_cue = stack.cues[state.current_index + 1] if state.current_index + 1 < len(stack.cues) else None
        prev_cue = stack.cues[state.current_index - 1] if state.current_index > 0 else None

        return {
            "success": True,
            "stack_id": stack_id,
            "stack_name": stack.name,
            "is_playing": state.is_playing,
            "current_cue": current_cue.to_dict() if current_cue else None,
            "current_index": state.current_index,
            "total_cues": len(stack.cues),
            "next_cue": next_cue.to_dict() if next_cue else None,
            "prev_cue": prev_cue.to_dict() if prev_cue else None,
        }


# ============================================================
# API Endpoint Helpers
# ============================================================

def validate_cue_stack_data(data: dict) -> Tuple[bool, Optional[str]]:
    """Validate CueStack creation/update data"""
    if not data.get("name"):
        return False, "Cue stack name is required"

    cues = data.get("cues", [])
    cue_numbers = set()

    for i, cue in enumerate(cues):
        cue_num = cue.get("cue_number")
        if not cue_num:
            return False, f"Cue {i+1} must have a cue_number"

        if cue_num in cue_numbers:
            return False, f"Duplicate cue number: {cue_num}"
        cue_numbers.add(cue_num)

        # Each cue must have either look_id or channels
        if not cue.get("look_id") and not cue.get("channels"):
            return False, f"Cue {cue_num} must have either look_id or channels"

    return True, None


def validate_cue_data(data: dict) -> Tuple[bool, Optional[str]]:
    """Validate single cue data"""
    if not data.get("cue_number"):
        return False, "Cue number is required"

    if not data.get("look_id") and not data.get("channels"):
        return False, "Cue must have either look_id or channels"

    return True, None
