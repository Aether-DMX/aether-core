#!/usr/bin/env python3
"""
Migration script: Move looks and sequences to scenes/chases

- Static looks (no modifiers) → Scenes
- Dynamic looks (with modifiers) → Chases
- All sequences → Chases
"""

import json
import os
from datetime import datetime

DATA_DIR = '/srv/aether/core/data'

def load_json(filename):
    path = os.path.join(DATA_DIR, filename)
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return []

def save_json(filename, data):
    path = os.path.join(DATA_DIR, filename)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    print(f"  Saved {filename} ({len(data)} items)")

def migrate():
    print("=" * 50)
    print("AETHER Content Migration")
    print("=" * 50)

    # Load current data
    looks = load_json('looks.json')
    sequences = load_json('sequences.json')
    scenes = load_json('scenes.json')
    chases = load_json('chases.json')

    print(f"\nCurrent counts:")
    print(f"  Looks: {len(looks)}")
    print(f"  Sequences: {len(sequences)}")
    print(f"  Scenes: {len(scenes)}")
    print(f"  Chases: {len(chases)}")

    # Track what we migrate
    migrated_to_scenes = []
    migrated_to_chases = []

    # Get existing scene/chase names to avoid duplicates
    existing_scene_names = {s.get('name', '').lower() for s in scenes}
    existing_chase_names = {c.get('name', '').lower() for c in chases}

    print("\n--- Processing Looks ---")
    for look in looks:
        name = look.get('name', 'Unnamed')
        modifiers = look.get('modifiers', [])
        has_active_modifiers = any(m.get('enabled', True) for m in modifiers)

        if has_active_modifiers and modifiers:
            # Dynamic look → Chase
            if name.lower() not in existing_chase_names:
                chase = convert_look_to_chase(look)
                chases.append(chase)
                migrated_to_chases.append(name)
                existing_chase_names.add(name.lower())
                print(f"  → Chase: {name} (has {len(modifiers)} modifier(s))")
            else:
                print(f"  ✓ Skip: {name} (already exists in chases)")
        else:
            # Static look → Scene
            if name.lower() not in existing_scene_names:
                scene = convert_look_to_scene(look)
                scenes.append(scene)
                migrated_to_scenes.append(name)
                existing_scene_names.add(name.lower())
                print(f"  → Scene: {name} (static)")
            else:
                print(f"  ✓ Skip: {name} (already exists in scenes)")

    print("\n--- Processing Sequences ---")
    for seq in sequences:
        name = seq.get('name', 'Unnamed')
        if name.lower() not in existing_chase_names:
            chase = convert_sequence_to_chase(seq)
            chases.append(chase)
            migrated_to_chases.append(name)
            existing_chase_names.add(name.lower())
            print(f"  → Chase: {name} ({len(seq.get('steps', []))} steps)")
        else:
            print(f"  ✓ Skip: {name} (already exists in chases)")

    # Save updated data
    print("\n--- Saving ---")
    save_json('scenes.json', scenes)
    save_json('chases.json', chases)

    # Clear looks and sequences (backup first)
    if looks:
        save_json('looks_backup.json', looks)
        save_json('looks.json', [])
        print("  Cleared looks.json (backup saved)")

    if sequences:
        save_json('sequences_backup.json', sequences)
        save_json('sequences.json', [])
        print("  Cleared sequences.json (backup saved)")

    print("\n" + "=" * 50)
    print("Migration Complete!")
    print(f"  Migrated to Scenes: {len(migrated_to_scenes)}")
    print(f"  Migrated to Chases: {len(migrated_to_chases)}")
    print("=" * 50)

def convert_look_to_scene(look):
    """Convert a static look to a scene"""
    return {
        'scene_id': f"scene_{look.get('look_id', '').replace('look_', '')}",
        'name': look.get('name', 'Unnamed'),
        'description': look.get('description', ''),
        'channels': look.get('channels', {}),
        'color': look.get('color', '#3b82f6'),
        'icon': look.get('icon', 'lightbulb'),
        'fade_ms': look.get('fade_ms', 500),
        'curve': 'linear',
        'universe': 1,
        'play_count': 0,
        'created_at': look.get('created_at', datetime.now().isoformat()),
        'updated_at': datetime.now().isoformat(),
        'migrated_from': look.get('look_id'),
    }

def convert_look_to_chase(look):
    """Convert a dynamic look (with modifiers) to a chase"""
    modifiers = look.get('modifiers', [])
    channels = look.get('channels', {})

    # Generate chase steps based on the modifier type
    steps = []
    mod_type = modifiers[0].get('type', 'pulse') if modifiers else 'pulse'
    params = modifiers[0].get('params', {}) if modifiers else {}

    if mod_type == 'strobe':
        # Strobe: on/off steps
        rate = params.get('rate', 5)
        hold_ms = max(25, int(1000 / (rate * 2)))
        steps = [
            {'channels': channels, 'fade_ms': 0, 'hold_ms': hold_ms, 'name': 'On'},
            {'channels': {k: 0 for k in channels}, 'fade_ms': 0, 'hold_ms': hold_ms, 'name': 'Off'},
        ]
    elif mod_type == 'pulse':
        # Pulse: fade up/down
        speed = params.get('speed', 0.5)
        fade_ms = int(speed * 1000)
        min_b = params.get('min_brightness', 20) / 100
        max_b = params.get('max_brightness', 100) / 100
        dim_channels = {k: int(v * min_b) for k, v in channels.items()}
        bright_channels = {k: int(v * max_b) for k, v in channels.items()}
        steps = [
            {'channels': dim_channels, 'fade_ms': fade_ms, 'hold_ms': 0, 'name': 'Dim'},
            {'channels': bright_channels, 'fade_ms': fade_ms, 'hold_ms': 0, 'name': 'Bright'},
        ]
    elif mod_type == 'rainbow':
        # Rainbow: cycle through colors
        speed = params.get('speed', 0.3)
        fade_ms = int(speed * 1000)
        steps = [
            {'channels': {'1': 255, '2': 0, '3': 0}, 'fade_ms': fade_ms, 'hold_ms': 0, 'name': 'Red'},
            {'channels': {'1': 255, '2': 128, '3': 0}, 'fade_ms': fade_ms, 'hold_ms': 0, 'name': 'Orange'},
            {'channels': {'1': 255, '2': 255, '3': 0}, 'fade_ms': fade_ms, 'hold_ms': 0, 'name': 'Yellow'},
            {'channels': {'1': 0, '2': 255, '3': 0}, 'fade_ms': fade_ms, 'hold_ms': 0, 'name': 'Green'},
            {'channels': {'1': 0, '2': 255, '3': 255}, 'fade_ms': fade_ms, 'hold_ms': 0, 'name': 'Cyan'},
            {'channels': {'1': 0, '2': 0, '3': 255}, 'fade_ms': fade_ms, 'hold_ms': 0, 'name': 'Blue'},
            {'channels': {'1': 128, '2': 0, '3': 255}, 'fade_ms': fade_ms, 'hold_ms': 0, 'name': 'Purple'},
            {'channels': {'1': 255, '2': 0, '3': 128}, 'fade_ms': fade_ms, 'hold_ms': 0, 'name': 'Pink'},
        ]
    elif mod_type == 'wave':
        # Wave: similar to pulse but with wave-like params
        speed = params.get('speed', 1.5)
        fade_ms = int(500 / speed)
        min_b = params.get('min_brightness', 0) / 100
        steps = [
            {'channels': {k: int(v * min_b) for k, v in channels.items()}, 'fade_ms': fade_ms, 'hold_ms': 0, 'name': 'Low'},
            {'channels': channels, 'fade_ms': fade_ms, 'hold_ms': 0, 'name': 'High'},
        ]
    elif mod_type == 'twinkle':
        # Twinkle: random-ish on/off
        fade_time = params.get('fade_time', 300)
        hold_time = params.get('hold_time', 100)
        min_b = params.get('min_brightness', 20) / 100
        steps = [
            {'channels': {k: int(v * min_b) for k, v in channels.items()}, 'fade_ms': fade_time, 'hold_ms': hold_time, 'name': 'Dim'},
            {'channels': channels, 'fade_ms': fade_time, 'hold_ms': hold_time, 'name': 'Bright'},
        ]
    elif mod_type == 'flicker':
        # Flicker: irregular on/off
        speed = params.get('speed', 5)
        hold_ms = max(30, int(200 / speed))
        min_b = params.get('min_brightness', 20) / 100
        steps = [
            {'channels': channels, 'fade_ms': 50, 'hold_ms': hold_ms, 'name': 'On'},
            {'channels': {k: int(v * min_b) for k, v in channels.items()}, 'fade_ms': 30, 'hold_ms': int(hold_ms * 0.5), 'name': 'Dim'},
            {'channels': channels, 'fade_ms': 40, 'hold_ms': int(hold_ms * 0.7), 'name': 'On 2'},
            {'channels': {k: int(v * (min_b + 0.3)) for k, v in channels.items()}, 'fade_ms': 60, 'hold_ms': hold_ms, 'name': 'Mid'},
        ]
    else:
        # Default: just the base channels as single step
        steps = [{'channels': channels, 'fade_ms': 500, 'hold_ms': 500, 'name': 'Step 1'}]

    return {
        'chase_id': f"chase_{look.get('look_id', '').replace('look_', '')}",
        'name': look.get('name', 'Unnamed'),
        'description': f"Migrated from look with {mod_type} modifier",
        'steps': steps,
        'color': look.get('color', '#a855f7'),
        'bpm': 120,
        'loop': 1,
        'fade_ms': look.get('fade_ms', 200),
        'distribution_mode': 'all',
        'created_at': look.get('created_at', datetime.now().isoformat()),
        'updated_at': datetime.now().isoformat(),
        'migrated_from': look.get('look_id'),
    }

def convert_sequence_to_chase(seq):
    """Convert a sequence to a chase"""
    steps = []
    for step in seq.get('steps', []):
        steps.append({
            'channels': step.get('channels', {}),
            'fade_ms': step.get('fade_ms', 500),
            'hold_ms': step.get('hold_ms', 500),
            'name': step.get('name', 'Step'),
        })

    return {
        'chase_id': f"chase_{seq.get('sequence_id', '').replace('sequence_', '')}",
        'name': seq.get('name', 'Unnamed'),
        'description': seq.get('description', ''),
        'steps': steps,
        'color': seq.get('color', '#a855f7'),
        'bpm': seq.get('bpm', 120),
        'loop': 1 if seq.get('loop', True) else 0,
        'fade_ms': 200,
        'distribution_mode': 'all',
        'created_at': seq.get('created_at', datetime.now().isoformat()),
        'updated_at': datetime.now().isoformat(),
        'migrated_from': seq.get('sequence_id'),
    }

if __name__ == '__main__':
    migrate()
