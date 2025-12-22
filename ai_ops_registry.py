#!/usr/bin/env python3
"""
AETHER AI Safe Operations Registry
Only operations in this registry can be executed by AI.
"""

SAFE_OPS = {
    # === No confirmation needed ===
    'get_status': {
        'confirm': False, 'risk': 'none',
        'params': [],
        'desc': 'Get current playback status'
    },
    'list_nodes': {
        'confirm': False, 'risk': 'none',
        'params': [],
        'desc': 'List all nodes'
    },
    'list_scenes': {
        'confirm': False, 'risk': 'none',
        'params': [],
        'desc': 'List all scenes'
    },
    'list_chases': {
        'confirm': False, 'risk': 'none',
        'params': [],
        'desc': 'List all chases'
    },
    'get_playback': {
        'confirm': False, 'risk': 'none',
        'params': ['universe'],
        'desc': 'Get playback for universe'
    },
    'rescan_nodes': {
        'confirm': False, 'risk': 'low',
        'params': [],
        'desc': 'Trigger node discovery'
    },
    
    # === Playback (uses app modal) ===
    'play_scene': {
        'confirm': False, 'risk': 'low', 'app_modal': True,
        'params': ['scene_id', 'fade_ms'],
        'desc': 'Play a scene'
    },
    'play_chase': {
        'confirm': False, 'risk': 'low', 'app_modal': True,
        'params': ['chase_id'],
        'desc': 'Play a chase'
    },
    'stop_playback': {
        'confirm': False, 'risk': 'low',
        'params': ['universe'],
        'desc': 'Stop playback'
    },
    'blackout': {
        'confirm': True, 'risk': 'medium',
        'params': ['fade_ms'],
        'desc': 'Blackout all outputs'
    },
    
    # === High risk - explicit confirmation ===
    'restart_olad': {
        'confirm': True, 'risk': 'high',
        'params': [],
        'desc': 'Restart OLA daemon'
    },
    'restart_aether_core': {
        'confirm': True, 'risk': 'high',
        'params': [],
        'desc': 'Restart Aether Core service'
    },
    'restart_node': {
        'confirm': True, 'risk': 'medium',
        'params': ['node_id'],
        'desc': 'Restart a Pulse node'
    },
}

def get_op(op_name):
    return SAFE_OPS.get(op_name)

def is_allowed(op_name):
    return op_name in SAFE_OPS

def needs_confirm(op_name):
    op = SAFE_OPS.get(op_name)
    return op['confirm'] if op else True

def needs_modal(op_name):
    op = SAFE_OPS.get(op_name)
    return op.get('app_modal', False) if op else False

def list_ops():
    return list(SAFE_OPS.keys())
