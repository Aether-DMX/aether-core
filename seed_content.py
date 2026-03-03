"""
AETHER Core — Factory Seed Content
Stock scenes, chases, and a demo show that ship with every unit.

All content uses RGBW format: ch1=R, ch2=G, ch3=B, ch4=W
All content targets universe 1.
IDs prefixed with stock_scene_, stock_chase_, stock_show_ for identification.
"""

import json
from datetime import datetime


# ═══════════════════════════════════════════════════════════════════
# STOCK SCENES (19 total)
# ═══════════════════════════════════════════════════════════════════

STOCK_SCENES = [
    # ── Primary Colors ──
    {
        "scene_id": "stock_scene_red", "name": "Red",
        "description": "Pure red",
        "channels": {"1": 255, "2": 0, "3": 0, "4": 0},
        "fade_ms": 500, "color": "#ef4444", "icon": "lightbulb",
    },
    {
        "scene_id": "stock_scene_green", "name": "Green",
        "description": "Pure green",
        "channels": {"1": 0, "2": 255, "3": 0, "4": 0},
        "fade_ms": 500, "color": "#22c55e", "icon": "lightbulb",
    },
    {
        "scene_id": "stock_scene_blue", "name": "Blue",
        "description": "Pure blue",
        "channels": {"1": 0, "2": 0, "3": 255, "4": 0},
        "fade_ms": 500, "color": "#3b82f6", "icon": "lightbulb",
    },
    {
        "scene_id": "stock_scene_warm_white", "name": "Warm White",
        "description": "Warm white with soft amber tint",
        "channels": {"1": 40, "2": 20, "3": 5, "4": 255},
        "fade_ms": 500, "color": "#fef3c7", "icon": "lightbulb",
    },
    {
        "scene_id": "stock_scene_cool_white", "name": "Cool White",
        "description": "Cool white with subtle blue tint",
        "channels": {"1": 10, "2": 15, "3": 40, "4": 255},
        "fade_ms": 500, "color": "#e0f2fe", "icon": "lightbulb",
    },

    # ── Off-Colors ──
    {
        "scene_id": "stock_scene_amber", "name": "Amber",
        "description": "Warm amber",
        "channels": {"1": 255, "2": 130, "3": 0, "4": 0},
        "fade_ms": 500, "color": "#f59e0b", "icon": "lightbulb",
    },
    {
        "scene_id": "stock_scene_magenta", "name": "Magenta",
        "description": "Vibrant magenta",
        "channels": {"1": 255, "2": 0, "3": 200, "4": 0},
        "fade_ms": 500, "color": "#d946ef", "icon": "lightbulb",
    },
    {
        "scene_id": "stock_scene_cyan", "name": "Cyan",
        "description": "Bright cyan",
        "channels": {"1": 0, "2": 255, "3": 255, "4": 0},
        "fade_ms": 500, "color": "#06b6d4", "icon": "lightbulb",
    },
    {
        "scene_id": "stock_scene_lavender", "name": "Lavender",
        "description": "Soft lavender",
        "channels": {"1": 150, "2": 80, "3": 255, "4": 50},
        "fade_ms": 500, "color": "#a78bfa", "icon": "lightbulb",
    },
    {
        "scene_id": "stock_scene_coral", "name": "Coral",
        "description": "Warm coral",
        "channels": {"1": 255, "2": 100, "3": 80, "4": 0},
        "fade_ms": 500, "color": "#fb7185", "icon": "lightbulb",
    },
    {
        "scene_id": "stock_scene_teal", "name": "Teal",
        "description": "Deep teal",
        "channels": {"1": 0, "2": 180, "3": 180, "4": 0},
        "fade_ms": 500, "color": "#14b8a6", "icon": "lightbulb",
    },
    {
        "scene_id": "stock_scene_deep_purple", "name": "Deep Purple",
        "description": "Rich deep purple",
        "channels": {"1": 100, "2": 0, "3": 255, "4": 0},
        "fade_ms": 500, "color": "#7c3aed", "icon": "lightbulb",
    },
    {
        "scene_id": "stock_scene_sunrise_orange", "name": "Sunrise Orange",
        "description": "Warm sunrise orange",
        "channels": {"1": 255, "2": 80, "3": 10, "4": 30},
        "fade_ms": 500, "color": "#fb923c", "icon": "lightbulb",
    },
    {
        "scene_id": "stock_scene_mint", "name": "Mint",
        "description": "Fresh mint green",
        "channels": {"1": 50, "2": 255, "3": 150, "4": 30},
        "fade_ms": 500, "color": "#34d399", "icon": "lightbulb",
    },

    # ── Mood / Dim ──
    {
        "scene_id": "stock_scene_candlelight", "name": "Candlelight",
        "description": "Warm flickering candle feel",
        "channels": {"1": 180, "2": 80, "3": 10, "4": 40},
        "fade_ms": 1000, "color": "#fbbf24", "icon": "lightbulb",
    },
    {
        "scene_id": "stock_scene_moonlight", "name": "Moonlight",
        "description": "Cool subtle moonlight",
        "channels": {"1": 20, "2": 30, "3": 80, "4": 60},
        "fade_ms": 1500, "color": "#93c5fd", "icon": "lightbulb",
    },
    {
        "scene_id": "stock_scene_golden_hour", "name": "Golden Hour",
        "description": "Warm golden sunset glow",
        "channels": {"1": 255, "2": 160, "3": 40, "4": 80},
        "fade_ms": 2000, "color": "#fcd34d", "icon": "lightbulb",
    },

    # ── Utility ──
    {
        "scene_id": "stock_scene_all_max", "name": "All Channels Max",
        "description": "Every channel at full intensity",
        "channels": {"1": 255, "2": 255, "3": 255, "4": 255},
        "fade_ms": 500, "color": "#ffffff", "icon": "lightbulb",
    },
    {
        "scene_id": "stock_scene_blackout", "name": "Blackout",
        "description": "All channels off",
        "channels": {"1": 0, "2": 0, "3": 0, "4": 0},
        "fade_ms": 500, "color": "#1e1e1e", "icon": "lightbulb",
    },
]


# ═══════════════════════════════════════════════════════════════════
# STOCK CHASES (8 total)
# ═══════════════════════════════════════════════════════════════════

STOCK_CHASES = [
    # ── Rainbow Cycle ──
    {
        "chase_id": "stock_chase_rainbow",
        "name": "Rainbow Cycle",
        "description": "Smooth ROYGBIV fade cycle",
        "bpm": 30, "loop": True, "color": "#ef4444", "fade_ms": 1500,
        "steps": [
            {"channels": {"1": 255, "2": 0, "3": 0, "4": 0}, "fade_ms": 1500, "hold_ms": 500},
            {"channels": {"1": 255, "2": 130, "3": 0, "4": 0}, "fade_ms": 1500, "hold_ms": 500},
            {"channels": {"1": 255, "2": 255, "3": 0, "4": 0}, "fade_ms": 1500, "hold_ms": 500},
            {"channels": {"1": 0, "2": 255, "3": 0, "4": 0}, "fade_ms": 1500, "hold_ms": 500},
            {"channels": {"1": 0, "2": 150, "3": 255, "4": 0}, "fade_ms": 1500, "hold_ms": 500},
            {"channels": {"1": 0, "2": 0, "3": 255, "4": 0}, "fade_ms": 1500, "hold_ms": 500},
            {"channels": {"1": 75, "2": 0, "3": 130, "4": 0}, "fade_ms": 1500, "hold_ms": 500},
            {"channels": {"1": 148, "2": 0, "3": 211, "4": 0}, "fade_ms": 1500, "hold_ms": 500},
        ],
    },

    # ── Color Pulse (Breathe) ──
    {
        "chase_id": "stock_chase_pulse",
        "name": "Color Pulse",
        "description": "Blue breathing pulse effect",
        "bpm": 40, "loop": True, "color": "#3b82f6", "fade_ms": 800,
        "steps": [
            {"channels": {"1": 0, "2": 0, "3": 255, "4": 0}, "fade_ms": 800, "hold_ms": 200},
            {"channels": {"1": 0, "2": 0, "3": 40, "4": 0}, "fade_ms": 800, "hold_ms": 200},
            {"channels": {"1": 0, "2": 0, "3": 255, "4": 0}, "fade_ms": 800, "hold_ms": 200},
            {"channels": {"1": 0, "2": 0, "3": 40, "4": 0}, "fade_ms": 800, "hold_ms": 600},
        ],
    },

    # ── Police Lights ──
    {
        "chase_id": "stock_chase_police",
        "name": "Police Lights",
        "description": "Red and blue alternating flash",
        "bpm": 120, "loop": True, "color": "#ef4444", "fade_ms": 0,
        "steps": [
            {"channels": {"1": 255, "2": 0, "3": 0, "4": 0}, "fade_ms": 0, "hold_ms": 150},
            {"channels": {"1": 0, "2": 0, "3": 0, "4": 0}, "fade_ms": 0, "hold_ms": 80},
            {"channels": {"1": 255, "2": 0, "3": 0, "4": 0}, "fade_ms": 0, "hold_ms": 150},
            {"channels": {"1": 0, "2": 0, "3": 0, "4": 0}, "fade_ms": 0, "hold_ms": 300},
            {"channels": {"1": 0, "2": 0, "3": 255, "4": 0}, "fade_ms": 0, "hold_ms": 150},
            {"channels": {"1": 0, "2": 0, "3": 0, "4": 0}, "fade_ms": 0, "hold_ms": 80},
            {"channels": {"1": 0, "2": 0, "3": 255, "4": 0}, "fade_ms": 0, "hold_ms": 150},
            {"channels": {"1": 0, "2": 0, "3": 0, "4": 0}, "fade_ms": 0, "hold_ms": 300},
        ],
    },

    # ── Fire Flicker ──
    {
        "chase_id": "stock_chase_fire",
        "name": "Fire Flicker",
        "description": "Warm randomized flickering fire effect",
        "bpm": 60, "loop": True, "color": "#f97316", "fade_ms": 100,
        "steps": [
            {"channels": {"1": 255, "2": 80, "3": 0, "4": 20}, "fade_ms": 100, "hold_ms": 120},
            {"channels": {"1": 200, "2": 50, "3": 5, "4": 10}, "fade_ms": 80, "hold_ms": 100},
            {"channels": {"1": 255, "2": 100, "3": 0, "4": 30}, "fade_ms": 120, "hold_ms": 80},
            {"channels": {"1": 180, "2": 40, "3": 0, "4": 5}, "fade_ms": 60, "hold_ms": 150},
            {"channels": {"1": 255, "2": 70, "3": 10, "4": 25}, "fade_ms": 100, "hold_ms": 100},
            {"channels": {"1": 220, "2": 90, "3": 0, "4": 15}, "fade_ms": 90, "hold_ms": 110},
            {"channels": {"1": 240, "2": 60, "3": 5, "4": 35}, "fade_ms": 110, "hold_ms": 90},
            {"channels": {"1": 190, "2": 45, "3": 0, "4": 8}, "fade_ms": 70, "hold_ms": 130},
        ],
    },

    # ── Ocean Wave ──
    {
        "chase_id": "stock_chase_ocean",
        "name": "Ocean Wave",
        "description": "Blues and teals flowing like water",
        "bpm": 25, "loop": True, "color": "#0ea5e9", "fade_ms": 2000,
        "steps": [
            {"channels": {"1": 0, "2": 40, "3": 200, "4": 10}, "fade_ms": 2000, "hold_ms": 800},
            {"channels": {"1": 0, "2": 100, "3": 255, "4": 20}, "fade_ms": 2000, "hold_ms": 600},
            {"channels": {"1": 0, "2": 180, "3": 180, "4": 5}, "fade_ms": 2000, "hold_ms": 800},
            {"channels": {"1": 0, "2": 60, "3": 220, "4": 15}, "fade_ms": 2000, "hold_ms": 600},
            {"channels": {"1": 10, "2": 120, "3": 200, "4": 0}, "fade_ms": 2000, "hold_ms": 700},
        ],
    },

    # ── Party Strobe ──
    {
        "chase_id": "stock_chase_strobe",
        "name": "Party Strobe",
        "description": "Fast multi-color strobe effect",
        "bpm": 180, "loop": True, "color": "#f43f5e", "fade_ms": 0,
        "steps": [
            {"channels": {"1": 255, "2": 0, "3": 0, "4": 0}, "fade_ms": 0, "hold_ms": 80},
            {"channels": {"1": 0, "2": 0, "3": 0, "4": 0}, "fade_ms": 0, "hold_ms": 80},
            {"channels": {"1": 0, "2": 255, "3": 0, "4": 0}, "fade_ms": 0, "hold_ms": 80},
            {"channels": {"1": 0, "2": 0, "3": 0, "4": 0}, "fade_ms": 0, "hold_ms": 80},
            {"channels": {"1": 0, "2": 0, "3": 255, "4": 0}, "fade_ms": 0, "hold_ms": 80},
            {"channels": {"1": 0, "2": 0, "3": 0, "4": 0}, "fade_ms": 0, "hold_ms": 80},
            {"channels": {"1": 255, "2": 0, "3": 255, "4": 0}, "fade_ms": 0, "hold_ms": 80},
            {"channels": {"1": 0, "2": 0, "3": 0, "4": 0}, "fade_ms": 0, "hold_ms": 80},
            {"channels": {"1": 255, "2": 255, "3": 255, "4": 255}, "fade_ms": 0, "hold_ms": 80},
            {"channels": {"1": 0, "2": 0, "3": 0, "4": 0}, "fade_ms": 0, "hold_ms": 80},
        ],
    },

    # ── Warm Cool Sweep ──
    {
        "chase_id": "stock_chase_warm_cool",
        "name": "Warm Cool Sweep",
        "description": "Warm white to cool white and back",
        "bpm": 20, "loop": True, "color": "#fef3c7", "fade_ms": 3000,
        "steps": [
            {"channels": {"1": 40, "2": 20, "3": 5, "4": 255}, "fade_ms": 3000, "hold_ms": 1000},
            {"channels": {"1": 10, "2": 15, "3": 40, "4": 255}, "fade_ms": 3000, "hold_ms": 1000},
        ],
    },

    # ── Sunset Fade ──
    {
        "chase_id": "stock_chase_sunset",
        "name": "Sunset Fade",
        "description": "Warm colors slow transition like a sunset",
        "bpm": 15, "loop": True, "color": "#f97316", "fade_ms": 4000,
        "steps": [
            {"channels": {"1": 255, "2": 200, "3": 50, "4": 80}, "fade_ms": 4000, "hold_ms": 2000},
            {"channels": {"1": 255, "2": 120, "3": 20, "4": 40}, "fade_ms": 4000, "hold_ms": 2000},
            {"channels": {"1": 255, "2": 60, "3": 40, "4": 10}, "fade_ms": 4000, "hold_ms": 2000},
            {"channels": {"1": 200, "2": 30, "3": 80, "4": 5}, "fade_ms": 4000, "hold_ms": 2000},
            {"channels": {"1": 100, "2": 20, "3": 120, "4": 0}, "fade_ms": 4000, "hold_ms": 2000},
        ],
    },
]


# ═══════════════════════════════════════════════════════════════════
# STOCK DEMO SHOW
# ═══════════════════════════════════════════════════════════════════

STOCK_SHOW = {
    "show_id": "stock_show_demo",
    "name": "AETHER Demo Show",
    "description": "A 47-second automated showcase of scenes, chases, and transitions",
    "duration_ms": 47000,
    "timeline": [
        # 0s — Fade up warm white
        {"time_ms": 0, "action_type": "scene", "action_id": "stock_scene_warm_white", "fade_ms": 1500},
        # 4s — Transition to blue
        {"time_ms": 4000, "action_type": "scene", "action_id": "stock_scene_blue", "fade_ms": 1500},
        # 8s — Deep purple
        {"time_ms": 8000, "action_type": "scene", "action_id": "stock_scene_deep_purple", "fade_ms": 1500},
        # 12s — Rainbow chase
        {"time_ms": 12000, "action_type": "chase", "action_id": "stock_chase_rainbow"},
        # 20s — Quick blackout
        {"time_ms": 20000, "action_type": "blackout", "fade_ms": 500},
        # 21s — Fire flicker
        {"time_ms": 21000, "action_type": "chase", "action_id": "stock_chase_fire"},
        # 28s — Ocean wave
        {"time_ms": 28000, "action_type": "chase", "action_id": "stock_chase_ocean"},
        # 36s — Golden hour scene
        {"time_ms": 36000, "action_type": "scene", "action_id": "stock_scene_golden_hour", "fade_ms": 2000},
        # 42s — Coral accent
        {"time_ms": 42000, "action_type": "scene", "action_id": "stock_scene_coral", "fade_ms": 1500},
        # 45s — Fade to black
        {"time_ms": 45000, "action_type": "blackout", "fade_ms": 2000},
    ],
}


# ═══════════════════════════════════════════════════════════════════
# SEED FUNCTION
# ═══════════════════════════════════════════════════════════════════

def seed_database(get_db):
    """Seed the database with factory default scenes, chases, and a demo show.

    Args:
        get_db: Function that returns a thread-local SQLite connection.

    Returns:
        dict with counts: {'scenes': N, 'chases': N, 'shows': N}
    """
    conn = get_db()
    c = conn.cursor()
    now = datetime.now().isoformat()

    # ── Scenes ──
    scene_count = 0
    for s in STOCK_SCENES:
        c.execute('''INSERT OR REPLACE INTO scenes
            (scene_id, name, description, universe, channels, fade_ms, curve, color, icon,
             is_favorite, play_count, synced_to_nodes, distribution_mode, created_at, updated_at)
            VALUES (?, ?, ?, 1, ?, ?, 'linear', ?, ?, 0, 0, 0, 'unified', ?, ?)''',
            (s['scene_id'], s['name'], s.get('description', ''),
             json.dumps(s['channels']), s.get('fade_ms', 500),
             s.get('color', '#3b82f6'), s.get('icon', 'lightbulb'),
             now, now))
        scene_count += 1

    # ── Chases ──
    chase_count = 0
    for ch in STOCK_CHASES:
        c.execute('''INSERT OR REPLACE INTO chases
            (chase_id, name, description, universe, bpm, loop, steps, color, fade_ms,
             synced_to_nodes, distribution_mode, created_at, updated_at)
            VALUES (?, ?, ?, 1, ?, ?, ?, ?, ?, 0, 'unified', ?, ?)''',
            (ch['chase_id'], ch['name'], ch.get('description', ''),
             ch.get('bpm', 120), 1 if ch.get('loop', True) else 0,
             json.dumps(ch['steps']), ch.get('color', '#10b981'),
             ch.get('fade_ms', 0), now, now))
        chase_count += 1

    # ── Demo Show ──
    show = STOCK_SHOW
    c.execute('''INSERT OR REPLACE INTO shows
        (show_id, name, description, timeline, duration_ms, distributed, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, 0, ?, ?)''',
        (show['show_id'], show['name'], show['description'],
         json.dumps(show['timeline']), show['duration_ms'], now, now))

    conn.commit()
    conn.close()

    print(f"🌱 Seeded: {scene_count} scenes, {chase_count} chases, 1 show")
    return {'scenes': scene_count, 'chases': chase_count, 'shows': 1}
