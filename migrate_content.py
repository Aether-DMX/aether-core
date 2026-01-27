#!/usr/bin/env python3
"""
Migration script: Move looks and sequences to scenes/chases (SQLite version)

- Static looks (no modifiers or empty modifiers) → Scenes
- Dynamic looks (with enabled modifiers) → Chases
- All sequences → Chases
"""

import sqlite3
import json
import os
from datetime import datetime

DB_PATH = os.path.expanduser('~/aether-core.db')

def migrate():
    print("=" * 50)
    print("AETHER Content Migration (SQLite)")
    print("=" * 50)

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Get counts
    cursor.execute("SELECT COUNT(*) FROM looks")
    look_count = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM sequences")
    seq_count = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM scenes")
    scene_count = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM chases")
    chase_count = cursor.fetchone()[0]

    print(f"\nCurrent counts:")
    print(f"  Looks: {look_count}")
    print(f"  Sequences: {seq_count}")
    print(f"  Scenes: {scene_count}")
    print(f"  Chases: {chase_count}")

    # Get existing names to avoid duplicates
    cursor.execute("SELECT LOWER(name) FROM scenes")
    existing_scene_names = {row[0] for row in cursor.fetchall()}
    cursor.execute("SELECT LOWER(name) FROM chases")
    existing_chase_names = {row[0] for row in cursor.fetchall()}

    migrated_to_scenes = []
    migrated_to_chases = []

    # --- Process Looks ---
    print("\n--- Processing Looks ---")
    cursor.execute("SELECT * FROM looks")
    looks = cursor.fetchall()

    for look in looks:
        name = look['name']
        modifiers_json = look['modifiers'] or '[]'
        modifiers = json.loads(modifiers_json)
        has_active_modifiers = any(m.get('enabled', True) for m in modifiers) and len(modifiers) > 0
        channels = json.loads(look['channels'] or '{}')

        if has_active_modifiers:
            # Dynamic look → Chase
            if name.lower() not in existing_chase_names:
                chase_data = make_chase_from_look(look, modifiers, channels)
                insert_chase(cursor, chase_data)
                migrated_to_chases.append(name)
                existing_chase_names.add(name.lower())
                print(f"  → Chase: {name} ({modifiers[0].get('type', 'unknown')} modifier)")
            else:
                print(f"  ✓ Skip: {name} (exists in chases)")
        else:
            # Static look → Scene
            if name.lower() not in existing_scene_names:
                scene_data = make_scene_from_look(look, channels)
                insert_scene(cursor, scene_data)
                migrated_to_scenes.append(name)
                existing_scene_names.add(name.lower())
                print(f"  → Scene: {name} (static)")
            else:
                print(f"  ✓ Skip: {name} (exists in scenes)")

    # --- Process Sequences ---
    print("\n--- Processing Sequences ---")
    cursor.execute("SELECT * FROM sequences")
    sequences = cursor.fetchall()

    for seq in sequences:
        name = seq['name']
        if name.lower() not in existing_chase_names:
            chase_data = make_chase_from_sequence(seq)
            insert_chase(cursor, chase_data)
            migrated_to_chases.append(name)
            existing_chase_names.add(name.lower())
            steps = json.loads(seq['steps'] or '[]')
            print(f"  → Chase: {name} ({len(steps)} steps)")
        else:
            print(f"  ✓ Skip: {name} (exists in chases)")

    # Clear looks and sequences
    print("\n--- Clearing migrated content ---")
    cursor.execute("DELETE FROM looks")
    cursor.execute("DELETE FROM sequences")
    print(f"  Cleared looks table ({look_count} rows)")
    print(f"  Cleared sequences table ({seq_count} rows)")

    conn.commit()
    conn.close()

    print("\n" + "=" * 50)
    print("Migration Complete!")
    print(f"  Migrated to Scenes: {len(migrated_to_scenes)}")
    print(f"  Migrated to Chases: {len(migrated_to_chases)}")
    print("=" * 50)

def make_scene_from_look(look, channels):
    """Convert static look to scene data"""
    return {
        'scene_id': f"scene_from_{look['look_id']}",
        'name': look['name'],
        'description': look['description'] or '',
        'channels': json.dumps(channels),
        'color': look['color'] or '#3b82f6',
        'icon': look['icon'] or 'lightbulb',
        'fade_ms': look['fade_ms'] or 500,
        'curve': 'linear',
        'universe': 1,
        'play_count': 0,
        'synced_to_nodes': 0,
        'created_at': look['created_at'],
        'updated_at': datetime.now().isoformat(),
    }

def make_chase_from_look(look, modifiers, channels):
    """Convert dynamic look to chase data"""
    mod_type = modifiers[0].get('type', 'pulse') if modifiers else 'pulse'
    params = modifiers[0].get('params', {}) if modifiers else {}

    steps = generate_chase_steps(mod_type, params, channels)

    # Estimate BPM from modifier params
    speed = params.get('speed', 1.0)
    rate = params.get('rate', 5)
    if mod_type == 'strobe':
        bpm = int(rate * 60)
    elif mod_type in ['pulse', 'rainbow', 'wave']:
        bpm = int(60 / speed) if speed > 0 else 60
    else:
        bpm = 120

    return {
        'chase_id': f"chase_from_{look['look_id']}",
        'name': look['name'],
        'description': f"Migrated from look ({mod_type} effect)",
        'steps': json.dumps(steps),
        'color': look['color'] or '#a855f7',
        'bpm': min(300, max(20, bpm)),
        'loop': 1,
        'fade_ms': 200,
        'distribution_mode': 'all',
        'created_at': look['created_at'],
        'updated_at': datetime.now().isoformat(),
    }

def make_chase_from_sequence(seq):
    """Convert sequence to chase data"""
    steps = json.loads(seq['steps'] or '[]')
    chase_steps = []
    for step in steps:
        chase_steps.append({
            'channels': step.get('channels', {}),
            'fade_ms': step.get('fade_ms', 500),
            'hold_ms': step.get('hold_ms', 500),
            'name': step.get('name', 'Step'),
        })

    return {
        'chase_id': f"chase_from_{seq['sequence_id']}",
        'name': seq['name'],
        'description': seq['description'] or '',
        'steps': json.dumps(chase_steps),
        'color': seq['color'] or '#a855f7',
        'bpm': seq['bpm'] or 120,
        'loop': 1 if seq['loop'] else 0,
        'fade_ms': 200,
        'distribution_mode': 'all',
        'created_at': seq['created_at'],
        'updated_at': datetime.now().isoformat(),
    }

def generate_chase_steps(mod_type, params, channels):
    """Generate chase steps based on modifier type"""
    if mod_type == 'strobe':
        rate = params.get('rate', 5)
        hold_ms = max(25, int(1000 / (rate * 2)))
        return [
            {'channels': channels, 'fade_ms': 0, 'hold_ms': hold_ms, 'name': 'On'},
            {'channels': {k: 0 for k in channels}, 'fade_ms': 0, 'hold_ms': hold_ms, 'name': 'Off'},
        ]
    elif mod_type == 'pulse':
        speed = params.get('speed', 0.5)
        fade_ms = int(speed * 1000)
        min_b = params.get('min_brightness', 20) / 100
        return [
            {'channels': {k: int(float(v) * min_b) for k, v in channels.items()}, 'fade_ms': fade_ms, 'hold_ms': 0, 'name': 'Dim'},
            {'channels': channels, 'fade_ms': fade_ms, 'hold_ms': 0, 'name': 'Bright'},
        ]
    elif mod_type == 'rainbow':
        speed = params.get('speed', 0.3)
        fade_ms = int(speed * 1000)
        return [
            {'channels': {'1': 255, '2': 0, '3': 0}, 'fade_ms': fade_ms, 'hold_ms': 0, 'name': 'Red'},
            {'channels': {'1': 255, '2': 128, '3': 0}, 'fade_ms': fade_ms, 'hold_ms': 0, 'name': 'Orange'},
            {'channels': {'1': 255, '2': 255, '3': 0}, 'fade_ms': fade_ms, 'hold_ms': 0, 'name': 'Yellow'},
            {'channels': {'1': 0, '2': 255, '3': 0}, 'fade_ms': fade_ms, 'hold_ms': 0, 'name': 'Green'},
            {'channels': {'1': 0, '2': 255, '3': 255}, 'fade_ms': fade_ms, 'hold_ms': 0, 'name': 'Cyan'},
            {'channels': {'1': 0, '2': 0, '3': 255}, 'fade_ms': fade_ms, 'hold_ms': 0, 'name': 'Blue'},
            {'channels': {'1': 128, '2': 0, '3': 255}, 'fade_ms': fade_ms, 'hold_ms': 0, 'name': 'Purple'},
        ]
    elif mod_type == 'wave':
        speed = params.get('speed', 1.5)
        fade_ms = int(500 / speed) if speed > 0 else 500
        min_b = params.get('min_brightness', 0) / 100
        return [
            {'channels': {k: int(float(v) * min_b) for k, v in channels.items()}, 'fade_ms': fade_ms, 'hold_ms': 0, 'name': 'Low'},
            {'channels': channels, 'fade_ms': fade_ms, 'hold_ms': 0, 'name': 'High'},
        ]
    elif mod_type == 'flicker':
        speed = params.get('speed', 5)
        hold_ms = max(30, int(200 / speed))
        min_b = params.get('min_brightness', 20) / 100
        return [
            {'channels': channels, 'fade_ms': 50, 'hold_ms': hold_ms, 'name': 'Bright'},
            {'channels': {k: int(float(v) * min_b) for k, v in channels.items()}, 'fade_ms': 30, 'hold_ms': int(hold_ms * 0.5), 'name': 'Dim'},
            {'channels': channels, 'fade_ms': 40, 'hold_ms': int(hold_ms * 0.7), 'name': 'Bright 2'},
        ]
    elif mod_type == 'twinkle':
        fade_time = params.get('fade_time', 300)
        hold_time = params.get('hold_time', 100)
        min_b = params.get('min_brightness', 20) / 100
        return [
            {'channels': {k: int(float(v) * min_b) for k, v in channels.items()}, 'fade_ms': fade_time, 'hold_ms': hold_time, 'name': 'Dim'},
            {'channels': channels, 'fade_ms': fade_time, 'hold_ms': hold_time, 'name': 'Bright'},
        ]
    else:
        return [{'channels': channels, 'fade_ms': 500, 'hold_ms': 500, 'name': 'Step 1'}]

def insert_scene(cursor, data):
    """Insert scene into database"""
    cursor.execute("""
        INSERT INTO scenes (scene_id, name, description, channels, color, icon, fade_ms, curve, universe, play_count, synced_to_nodes, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        data['scene_id'], data['name'], data['description'], data['channels'],
        data['color'], data['icon'], data['fade_ms'], data['curve'],
        data['universe'], data['play_count'], data['synced_to_nodes'],
        data['created_at'], data['updated_at']
    ))

def insert_chase(cursor, data):
    """Insert chase into database"""
    cursor.execute("""
        INSERT INTO chases (chase_id, name, description, steps, color, bpm, loop, fade_ms, distribution_mode, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        data['chase_id'], data['name'], data['description'], data['steps'],
        data['color'], data['bpm'], data['loop'], data['fade_ms'],
        data['distribution_mode'], data['created_at'], data['updated_at']
    ))

if __name__ == '__main__':
    migrate()
