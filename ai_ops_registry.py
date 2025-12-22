#!/usr/bin/env python3
"""AETHER AI Safe Operations Registry"""

SAFE_OPS = {
    "get_status": {"confirm": False, "risk": "none", "params": []},
    "list_nodes": {"confirm": False, "risk": "none", "params": []},
    "list_scenes": {"confirm": False, "risk": "none", "params": []},
    "list_chases": {"confirm": False, "risk": "none", "params": []},
    "rescan_nodes": {"confirm": False, "risk": "low", "params": []},
    "play_scene": {"confirm": False, "risk": "low", "modal": True, "params": ["scene_id"]},
    "play_chase": {"confirm": False, "risk": "low", "modal": True, "params": ["chase_id"]},
    "stop_playback": {"confirm": False, "risk": "low", "params": []},
    "blackout": {"confirm": True, "risk": "medium", "params": []},
    "restart_node": {"confirm": True, "risk": "medium", "params": ["node_id"]},
    "restart_olad": {"confirm": True, "risk": "high", "params": []},
}

def get_op(name): return SAFE_OPS.get(name)
def is_allowed(name): return name in SAFE_OPS
def needs_confirm(name): return SAFE_OPS.get(name, {}).get("confirm", True)
def get_risk(name): return SAFE_OPS.get(name, {}).get("risk", "unknown")
def needs_modal(name): return SAFE_OPS.get(name, {}).get("modal", False)
def list_ops(): return SAFE_OPS
