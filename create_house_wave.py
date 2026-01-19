#!/usr/bin/env python3
import json
import os
import requests

# API endpoint - configure via environment variable or default to localhost
API_BASE = os.environ.get("AETHER_CORE_URL", "http://localhost:8891")

chase = {
    "name": "House Wave v3",
    "description": "Smooth continuous wave - longer fades",
    "bpm": 60,
    "color": "#FF6600",
    "loop": True,
    "steps": []
}

# Universe order: 2 -> 4 -> 5 -> 6 -> 7
universes = [2, 4, 5, 6, 7]

# Colors (RGBW)
purple = {"1": 180, "2": 0, "3": 255, "4": 0}
orange = {"1": 255, "2": 100, "3": 0, "4": 50}
warm_dim = {"1": 80, "2": 30, "3": 0, "4": 20}

def make_step(active_idx):
    # Longer fade (950ms), shorter hold (50ms) = same 1000ms total
    step = {"fade_ms": 950, "hold_ms": 50, "channels": {}}

    for i, u in enumerate(universes):
        if i == active_idx:
            color = orange
        elif i == (active_idx + 1) % len(universes):
            color = purple
        else:
            color = warm_dim

        for ch, val in color.items():
            step["channels"][f"{u}:{ch}"] = val

    return step

for i in range(len(universes)):
    chase["steps"].append(make_step(i))

r = requests.post(f"{API_BASE}/api/chases", json=chase)
result = r.json()
print(json.dumps(result, indent=2))

if result.get("success"):
    chase_id = result.get("chase_id")
    print(f"\nPlaying chase {chase_id}...")
    play = requests.post(f"{API_BASE}/api/chases/{chase_id}/play",
                        json={"universes": [2, 4, 5, 6, 7]})
    print(json.dumps(play.json(), indent=2))
