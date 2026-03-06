"""
AETHER Core — Settings & Screen Context Blueprint
Routes: /api/settings/*, /api/screen-context
Dependencies: app_settings dict, save_settings func, socketio, cloud_submit, get_supabase_service
"""

from flask import Blueprint, jsonify, request
from datetime import datetime

settings_bp = Blueprint('settings', __name__)

# Dependencies injected at registration time
_app_settings = None
_save_settings = None
_socketio = None
_cloud_submit = None
_get_supabase_service = None


def init_app(app_settings, save_settings_func, socketio, cloud_submit=None, get_supabase_service=None):
    """Initialize blueprint with required dependencies."""
    global _app_settings, _save_settings, _socketio, _cloud_submit, _get_supabase_service
    _app_settings = app_settings
    _save_settings = save_settings_func
    _socketio = socketio
    _cloud_submit = cloud_submit
    _get_supabase_service = get_supabase_service


def _sync_settings_to_cloud():
    """Fire-and-forget settings sync to Supabase (only if cloudBackup premium feature enabled)"""
    if not _cloud_submit or not _get_supabase_service:
        return
    # Gate behind premium feature flag
    if _app_settings and not _app_settings.get('features', {}).get('cloudBackup', False):
        return
    settings_snapshot = dict(_app_settings) if _app_settings else {}
    def _do_sync():
        try:
            svc = _get_supabase_service()
            if svc and svc.is_enabled() and svc.is_connected():
                svc.sync_settings(settings_snapshot)
        except Exception as e:
            print(f"⚠️ Settings cloud sync failed: {e}")
    _cloud_submit(_do_sync)


@settings_bp.route('/api/settings/all', methods=['GET'])
def get_all_settings():
    return jsonify(_app_settings)

@settings_bp.route('/api/settings/<category>', methods=['GET'])
def get_settings_category(category):
    return jsonify(_app_settings.get(category, {}))

@settings_bp.route('/api/settings/<category>', methods=['POST', 'PUT'])
def update_settings_category(category):
    data = request.get_json()
    if category in _app_settings:
        _app_settings[category].update(data)
        _save_settings(_app_settings)
        _socketio.emit('settings_update', {'category': category, 'data': _app_settings[category]})
        _sync_settings_to_cloud()
        return jsonify({'success': True, category: _app_settings[category]})
    return jsonify({'error': 'Category not found'}), 404

@settings_bp.route('/api/screen-context', methods=['POST'])
def screen_context():
    data = request.get_json()
    _socketio.emit('screen:context', {'page': data.get('page', 'Unknown'),
                                      'action': data.get('action'),
                                      'timestamp': datetime.now().isoformat()})
    return jsonify({'success': True})


# Setup Complete Routes (for browser onboarding persistence)
@settings_bp.route('/api/settings/setup-complete', methods=['GET'])
def get_setup_complete():
    """Get setup completion status - shared across all browsers"""
    setup = _app_settings.get('setup', {'complete': False})
    return jsonify(setup)

@settings_bp.route('/api/settings/setup-complete', methods=['POST'])
def set_setup_complete():
    """Mark setup as complete - persists on server for all browsers"""
    data = request.get_json() or {}
    if 'setup' not in _app_settings:
        _app_settings['setup'] = {'complete': False, 'mode': None, 'userProfile': {}}
    _app_settings['setup']['complete'] = data.get('complete', True)
    if 'mode' in data:
        _app_settings['setup']['mode'] = data['mode']
    if 'userProfile' in data:
        _app_settings['setup']['userProfile'].update(data['userProfile'])
    _save_settings(_app_settings)
    _socketio.emit('settings_update', {'category': 'setup', 'data': _app_settings['setup']})
    _sync_settings_to_cloud()
    return jsonify({'success': True, 'setup': _app_settings['setup']})

@settings_bp.route('/api/settings/setup-reset', methods=['POST'])
def reset_setup():
    """Reset setup (for debugging/testing)"""
    _app_settings['setup'] = {'complete': False, 'mode': None, 'userProfile': {}}
    _save_settings(_app_settings)
    return jsonify({'success': True, 'setup': _app_settings['setup']})
