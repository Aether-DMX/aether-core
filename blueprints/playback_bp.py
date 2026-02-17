"""
AETHER Core — Playback, Render & Playback-Manager Blueprint
Routes: /api/playback/*, /api/render/*, /api/playback-manager/*
Dependencies: unified_get_status, unified_pause, unified_resume, stop_all_playback, render_engine, playback_manager
"""

from flask import Blueprint, jsonify, request

playback_bp = Blueprint('playback', __name__)

# Dependencies injected at registration time
_unified_get_status = None
_unified_pause = None
_unified_resume = None
_stop_all_playback = None
_render_engine = None
_playback_manager = None


def init_app(unified_get_status_fn, unified_pause_fn, unified_resume_fn, stop_all_playback_fn, render_engine, playback_manager):
    """Initialize blueprint with required dependencies."""
    global _unified_get_status, _unified_pause, _unified_resume, _stop_all_playback, _render_engine, _playback_manager
    _unified_get_status = unified_get_status_fn
    _unified_pause = unified_pause_fn
    _unified_resume = unified_resume_fn
    _stop_all_playback = stop_all_playback_fn
    _render_engine = render_engine
    _playback_manager = playback_manager


# ─────────────────────────────────────────────────────────
# Legacy Playback API (maintains backward compatibility)
# ─────────────────────────────────────────────────────────

@playback_bp.route('/api/playback/status', methods=['GET'])
def get_playback_status():
    """Get playback status from UnifiedPlaybackEngine (canonical authority)."""
    status = _unified_get_status()
    # Add playing flag for backward compatibility
    status['playing'] = bool(status.get('sessions'))
    return jsonify(status)


@playback_bp.route('/api/playback/stop', methods=['POST'])
def api_stop_all_playback():
    """Stop all playback (Look, Sequence, Chase, Effect, Show).

    SSOT: This endpoint uses the unified stop_all_playback function to ensure
    consistent behavior across all stop controls (UI buttons, hotkeys, etc.)

    Optional body params:
        blackout: bool - If true, also send blackout command
        fade_ms: int - Fade time for blackout (default 1000)
        universe: int - Specific universe to stop (default all)
    """
    data = request.get_json(silent=True) or {}
    blackout = data.get('blackout', False)
    fade_ms = data.get('fade_ms', 1000)
    universe = data.get('universe')

    # Use unified SSOT stop function
    result = _stop_all_playback(blackout=blackout, fade_ms=fade_ms, universe=universe)

    # Also stop render engine (not in unified function as it's a separate system)
    _render_engine.stop_rendering()

    return jsonify(result)


@playback_bp.route('/api/playback/pause', methods=['POST'])
def pause_playback():
    """Pause current playback via UnifiedPlaybackEngine (canonical authority)."""
    return jsonify(_unified_pause())


@playback_bp.route('/api/playback/resume', methods=['POST'])
def resume_playback():
    """Resume paused playback via UnifiedPlaybackEngine (canonical authority)."""
    return jsonify(_unified_resume())

    return jsonify({'success': False, 'error': 'Nothing paused to resume'})


# ─────────────────────────────────────────────────────────
# Render Engine Routes
# ─────────────────────────────────────────────────────────

@playback_bp.route('/api/render/status', methods=['GET'])
def get_render_status():
    """Get current render engine status"""
    return jsonify(_render_engine.get_status())

@playback_bp.route('/api/render/stop', methods=['POST'])
def stop_render():
    """Stop all rendering"""
    _render_engine.stop_rendering()
    return jsonify({'success': True, 'stopped': True})

# ─────────────────────────────────────────────────────────
# Legacy Playback Manager Routes (for backward compatibility)
# ─────────────────────────────────────────────────────────
@playback_bp.route('/api/playback-manager/status', methods=['GET'])
def playback_manager_status():
    """Legacy: Get playback manager status (use /api/playback/status for unified controller)"""
    return jsonify(_playback_manager.get_status())

@playback_bp.route('/api/playback-manager/stop', methods=['POST'])
def stop_playback_manager():
    """Stop all playback via unified SSOT function.

    SSOT: Redirects to unified stop_all_playback for consistent behavior.
    Use /api/playback/stop for full control with blackout option.
    """
    data = request.get_json() or {}
    universe = data.get('universe')
    # Use unified SSOT stop function
    return jsonify(_stop_all_playback(blackout=False, universe=universe))
