"""
AETHER Core — System Blueprint
Routes: /api/system/*
Dependencies: app_settings, save_settings, version constants, get_or_create_device_id
"""

import os
import time
import platform
import subprocess
import threading
from datetime import datetime
from flask import Blueprint, jsonify, request

system_bp = Blueprint('system', __name__)

# Dependencies injected at registration time
_app_settings = None
_save_settings = None
_AETHER_VERSION = None
_AETHER_COMMIT = None
_AETHER_FILE_PATH = None
_AETHER_START_TIME = None
_get_or_create_device_id = None
_app = None  # Flask app reference for autosync state

# Factory reset dependencies (optional, injected via keyword args)
_get_db = None
_node_manager = None
_unified_engine = None
_dmx_state = None
_show_engine = None
_chase_engine = None
_effects_engine = None
_content_manager = None
_socketio = None


def init_app(app_settings, save_settings_fn, aether_version, aether_commit,
             aether_file_path, aether_start_time, get_or_create_device_id_fn, app,
             get_db=None, node_manager=None, unified_engine=None, dmx_state=None,
             show_engine=None, chase_engine=None, effects_engine=None,
             content_manager=None, socketio=None):
    """Initialize blueprint with required dependencies."""
    global _app_settings, _save_settings, _AETHER_VERSION, _AETHER_COMMIT
    global _AETHER_FILE_PATH, _AETHER_START_TIME, _get_or_create_device_id, _app
    global _get_db, _node_manager, _unified_engine, _dmx_state
    global _show_engine, _chase_engine, _effects_engine, _content_manager, _socketio
    _app_settings = app_settings
    _save_settings = save_settings_fn
    _AETHER_VERSION = aether_version
    _AETHER_COMMIT = aether_commit
    _AETHER_FILE_PATH = aether_file_path
    _AETHER_START_TIME = aether_start_time
    _get_or_create_device_id = get_or_create_device_id_fn
    _app = app
    _get_db = get_db
    _node_manager = node_manager
    _unified_engine = unified_engine
    _dmx_state = dmx_state
    _show_engine = show_engine
    _chase_engine = chase_engine
    _effects_engine = effects_engine
    _content_manager = content_manager
    _socketio = socketio


@system_bp.route('/api/system/info', methods=['GET'])
def system_info():
    """System information for frontend mode detection"""
    device_id = _get_or_create_device_id()
    is_pi = os.path.exists('/sys/firmware/devicetree/base/model')
    # Check for display: env vars first, then X11 socket, then loginctl session
    has_display = bool(
        os.environ.get('DISPLAY') or
        os.environ.get('WAYLAND_DISPLAY') or
        os.path.exists('/tmp/.X11-unix/X0') or
        os.path.exists('/run/user/1000/wayland-0')
    )

    # Determine mode from settings
    mode_setting = _app_settings.get('system', {}).get('uiMode', 'auto')
    if mode_setting == 'auto':
        mode = 'kiosk' if (is_pi and has_display) else 'desktop'
    else:
        mode = mode_setting

    return jsonify({
        'deviceId': device_id,
        'hostname': platform.node(),
        'platform': 'pi5' if is_pi else platform.system().lower(),
        'version': _AETHER_VERSION,
        'commit': _AETHER_COMMIT,
        'mode': mode,
        'capabilities': {
            'maxUniverses': 16,
            'rdmSupported': True,
            'transport': 'UDP JSON',
        }
    })


@system_bp.route('/api/system/mode', methods=['POST'])
def set_system_mode():
    """Set UI mode preference"""
    data = request.get_json() or {}
    mode = data.get('mode', 'auto')
    if mode not in ['auto', 'kiosk', 'desktop', 'mobile']:
        return jsonify({'error': 'Invalid mode'}), 400
    if 'system' not in _app_settings:
        _app_settings['system'] = {}
    _app_settings['system']['uiMode'] = mode
    _save_settings(_app_settings)
    return jsonify({'success': True, 'mode': mode})


@system_bp.route('/api/system/stats', methods=['GET'])
def system_stats():
    """Get system statistics (CPU, memory, temperature)"""
    stats = {
        'cpu_percent': None,
        'memory_used': None,
        'memory_total': None,
        'cpu_temp': None,
        'disk_used': None,
        'disk_total': None,
        'uptime': None
    }

    try:
        with open('/proc/loadavg', 'r') as f:
            load = f.read().split()
            stats['cpu_percent'] = float(load[0]) * 25
    except (OSError, ValueError, IndexError):
        pass

    try:
        with open('/proc/meminfo', 'r') as f:
            meminfo = {}
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    meminfo[parts[0].rstrip(':')] = int(parts[1]) * 1024
            stats['memory_total'] = meminfo.get('MemTotal', 0)
            stats['memory_used'] = stats['memory_total'] - meminfo.get('MemAvailable', 0)
    except (OSError, ValueError, KeyError):
        pass

    try:
        with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
            stats['cpu_temp'] = int(f.read().strip()) / 1000.0
    except (OSError, ValueError):
        pass

    try:
        statvfs = os.statvfs('/')
        stats['disk_total'] = statvfs.f_blocks * statvfs.f_frsize
        stats['disk_used'] = (statvfs.f_blocks - statvfs.f_bfree) * statvfs.f_frsize
    except (OSError, AttributeError):
        pass

    try:
        with open('/proc/uptime', 'r') as f:
            stats['uptime'] = float(f.read().split()[0])
    except (OSError, ValueError, IndexError):
        pass

    return jsonify(stats)

@system_bp.route('/api/system/update', methods=['POST'])
def system_update():
    """Pull latest code from git and deploy to runtime location"""
    results = {'steps': [], 'success': False}

    git_dir = os.path.dirname(_AETHER_FILE_PATH)
    if '/aether-core-git/' not in _AETHER_FILE_PATH and '/aether-core/' not in _AETHER_FILE_PATH:
        git_dir = os.path.expanduser('~/aether-core-git/aether-core')

    runtime_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'aether-core.py')

    try:
        fetch_result = subprocess.run(
            ['git', 'fetch', 'origin'],
            capture_output=True, text=True, timeout=30,
            cwd=git_dir
        )
        results['steps'].append({
            'step': 'git_fetch',
            'success': fetch_result.returncode == 0,
            'output': fetch_result.stdout + fetch_result.stderr
        })

        status_result = subprocess.run(
            ['git', 'status', '-uno'],
            capture_output=True, text=True, timeout=10,
            cwd=git_dir
        )
        behind = 'behind' in status_result.stdout
        results['update_available'] = behind

        if not behind:
            results['message'] = 'Already up to date'
            results['success'] = True
            return jsonify(results)

        pull_result = subprocess.run(
            ['git', 'pull', 'origin', 'main'],
            capture_output=True, text=True, timeout=60,
            cwd=git_dir
        )
        results['steps'].append({
            'step': 'git_pull',
            'success': pull_result.returncode == 0,
            'output': pull_result.stdout + pull_result.stderr
        })

        if pull_result.returncode != 0:
            results['message'] = 'Git pull failed'
            return jsonify(results), 500

        commit_result = subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'],
            capture_output=True, text=True, timeout=5,
            cwd=git_dir
        )
        new_commit = commit_result.stdout.strip()
        results['new_commit'] = new_commit
        results['old_commit'] = _AETHER_COMMIT

        source_file = os.path.join(git_dir, 'aether-core.py')
        if os.path.exists(source_file) and runtime_file != source_file:
            import shutil
            shutil.copy2(source_file, runtime_file)

            timestamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
            sed_cmd = f'sed -i "s/AETHER_COMMIT = get_git_commit()/AETHER_COMMIT = \\"{new_commit}\\"  # Baked at deploy: {timestamp}/" {runtime_file}'
            subprocess.run(['bash', '-c', sed_cmd], capture_output=True)

            results['steps'].append({
                'step': 'deploy_to_runtime',
                'success': True,
                'output': f'Copied to {runtime_file} with embedded commit {new_commit}'
            })

        results['message'] = 'Update deployed. Restarting service...'
        results['success'] = True

        def restart_service():
            time.sleep(1)
            os.system('sudo systemctl restart aether-core')

        restart_thread = threading.Thread(target=restart_service, daemon=True)
        restart_thread.start()

        return jsonify(results)

    except Exception as e:
        results['error'] = str(e)
        return jsonify(results), 500

@system_bp.route('/api/system/update/check', methods=['GET'])
def system_update_check():
    """Check if updates are available without applying them"""
    git_dir = os.path.dirname(_AETHER_FILE_PATH)
    try:
        fetch_result = subprocess.run(
            ['git', 'fetch', 'origin'],
            capture_output=True, text=True, timeout=30,
            cwd=git_dir
        )

        # Detect connectivity failure — git fetch returns non-zero or stderr contains "fatal"
        fetch_failed = fetch_result.returncode != 0 or 'fatal' in (fetch_result.stderr or '').lower()
        if fetch_failed:
            return jsonify({
                'current_commit': _AETHER_COMMIT,
                'update_available': False,
                'commits_behind': 0,
                'latest_commit': None,
                'offline': True,
                'error': 'Cannot reach remote — check internet connection'
            })

        result = subprocess.run(
            ['git', 'rev-list', 'HEAD..origin/main', '--count'],
            capture_output=True, text=True, timeout=10,
            cwd=git_dir
        )

        commits_behind = int(result.stdout.strip()) if result.returncode == 0 else 0

        log_result = subprocess.run(
            ['git', 'log', 'origin/main', '-1', '--format=%h %s'],
            capture_output=True, text=True, timeout=10,
            cwd=git_dir
        )

        return jsonify({
            'current_commit': _AETHER_COMMIT,
            'update_available': commits_behind > 0,
            'commits_behind': commits_behind,
            'latest_commit': log_result.stdout.strip() if log_result.returncode == 0 else None,
            'offline': False
        })
    except subprocess.TimeoutExpired:
        return jsonify({
            'current_commit': _AETHER_COMMIT,
            'update_available': False,
            'offline': True,
            'error': 'Git fetch timed out — no internet?'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@system_bp.route('/api/system/autosync', methods=['GET'])
def get_autosync_status():
    """Get auto-sync status (persisted in settings)"""
    # Load from persistent settings first, fall back to runtime state
    autosync_settings = _app_settings.get('autosync', {})
    return jsonify({
        'enabled': autosync_settings.get('enabled', getattr(_app, '_autosync_enabled', False)),
        'interval_minutes': autosync_settings.get('interval_minutes', getattr(_app, '_autosync_interval', 30)),
        'last_check': getattr(_app, '_autosync_last_check', None),
        'last_update': getattr(_app, '_autosync_last_update', None)
    })

@system_bp.route('/api/system/autosync', methods=['POST'])
def set_autosync():
    """Enable/disable auto-sync (persisted to settings)"""
    data = request.get_json() or {}
    enabled = data.get('enabled', False)
    interval = max(5, min(1440, data.get('interval_minutes', 30)))

    _app._autosync_enabled = enabled
    _app._autosync_interval = interval

    # Persist to settings file so it survives restarts
    _app_settings['autosync'] = {'enabled': enabled, 'interval_minutes': interval}
    _save_settings(_app_settings)

    if enabled:
        _start_autosync_thread()
        print(f"✓ Auto-sync enabled: checking every {interval} minutes (persisted)")
    else:
        print("✓ Auto-sync disabled (persisted)")

    return jsonify({
        'success': True,
        'enabled': enabled,
        'interval_minutes': interval
    })


def _start_autosync_thread():
    """Start background thread for auto-sync"""
    def autosync_worker():
        git_dir = os.path.dirname(_AETHER_FILE_PATH)
        if '/aether-core-git/' not in _AETHER_FILE_PATH and '/aether-core/' not in _AETHER_FILE_PATH:
            git_dir = os.path.expanduser('~/aether-core-git/aether-core')
        runtime_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'aether-core.py')

        while getattr(_app, '_autosync_enabled', False):
            try:
                interval = getattr(_app, '_autosync_interval', 30) * 60
                time.sleep(interval)

                if not getattr(_app, '_autosync_enabled', False):
                    break

                _app._autosync_last_check = datetime.now().isoformat()

                subprocess.run(['git', 'fetch', 'origin'],
                    capture_output=True, timeout=30,
                    cwd=git_dir)

                result = subprocess.run(
                    ['git', 'rev-list', 'HEAD..origin/main', '--count'],
                    capture_output=True, text=True, timeout=10,
                    cwd=git_dir)

                commits_behind = int(result.stdout.strip()) if result.returncode == 0 else 0

                if commits_behind > 0:
                    print(f"🔄 Auto-sync: {commits_behind} updates available, pulling...")

                    pull_result = subprocess.run(
                        ['git', 'pull', 'origin', 'main'],
                        capture_output=True, text=True, timeout=60,
                        cwd=git_dir)

                    if pull_result.returncode == 0:
                        source_file = os.path.join(git_dir, 'aether-core.py')
                        if os.path.exists(source_file) and runtime_file != source_file:
                            import shutil
                            shutil.copy2(source_file, runtime_file)

                            commit_result = subprocess.run(
                                ['git', 'rev-parse', '--short', 'HEAD'],
                                capture_output=True, text=True, timeout=5,
                                cwd=git_dir)
                            new_commit = commit_result.stdout.strip()
                            timestamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
                            sed_cmd = f'sed -i "s/AETHER_COMMIT = get_git_commit()/AETHER_COMMIT = \\"{new_commit}\\"  # Baked at deploy: {timestamp}/" {runtime_file}'
                            subprocess.run(['bash', '-c', sed_cmd], capture_output=True)

                            print(f"✓ Auto-sync: deployed {new_commit} to {runtime_file}")

                        _app._autosync_last_update = datetime.now().isoformat()
                        print("✓ Auto-sync: restarting service...")
                        time.sleep(1)
                        os.system('sudo systemctl restart aether-core')
                    else:
                        print(f"❌ Auto-sync pull failed: {pull_result.stderr}")

            except Exception as e:
                print(f"❌ Auto-sync error: {e}")

    if not getattr(_app, '_autosync_thread', None) or not _app._autosync_thread.is_alive():
        _app._autosync_thread = threading.Thread(target=autosync_worker, daemon=True)
        _app._autosync_thread.start()


# ═══════════════════════════════════════════════════════════════════
# FACTORY RESET
# ═══════════════════════════════════════════════════════════════════

@system_bp.route('/api/system/factory-reset', methods=['POST'])
def factory_reset():
    """Factory reset: stop playback, unpair nodes, wipe DB, reset settings, re-seed stock content."""
    import time as _time
    from copy import deepcopy

    results = {'steps': [], 'success': False}

    if not _get_db:
        return jsonify({'error': 'Factory reset not available — dependencies not initialized'}), 503

    try:
        # ── Step 1: Stop all playback ──────────────────────────────────
        try:
            if _unified_engine:
                _unified_engine.stop_all()
            if _show_engine:
                _show_engine.stop()
            if _chase_engine:
                _chase_engine.stop_all()
            if _effects_engine:
                _effects_engine.stop_effect()
            results['steps'].append({'step': 'stop_playback', 'success': True})
        except Exception as e:
            results['steps'].append({'step': 'stop_playback', 'success': False, 'error': str(e)})

        # ── Step 2: Unpair all WiFi nodes ──────────────────────────────
        unpaired_count = 0
        try:
            if _node_manager:
                all_nodes = _node_manager.get_all_nodes(include_offline=True)
                for node in all_nodes:
                    if node.get('type') == 'wifi' and node.get('ip') and not node.get('is_builtin'):
                        target_ip = node.get('seance_ip') if node.get('via_seance') else node.get('ip')
                        if target_ip and target_ip != 'localhost':
                            try:
                                _node_manager.send_command_to_wifi(target_ip, {'cmd': 'unpair'})
                                unpaired_count += 1
                            except Exception:
                                pass  # Node may be offline
                            _time.sleep(0.05)
            results['steps'].append({'step': 'unpair_nodes', 'success': True, 'count': unpaired_count})
        except Exception as e:
            results['steps'].append({'step': 'unpair_nodes', 'success': False, 'error': str(e)})

        # ── Step 3: Wipe database tables ───────────────────────────────
        tables_to_wipe = [
            'scenes', 'chases', 'shows', 'fixtures', 'groups',
            'schedules', 'timers', 'rdm_devices', 'rdm_personalities',
            'node_groups', 'looks', 'sequences', 'cue_stacks',
            'ai_preferences', 'ai_outcomes', 'ai_audit_log',
        ]
        tables_wiped = []
        try:
            conn = _get_db()
            c = conn.cursor()

            # Delete all non-builtin nodes
            c.execute('DELETE FROM nodes WHERE is_builtin = 0')
            tables_wiped.append('nodes')

            # Wipe all content tables
            for table in tables_to_wipe:
                try:
                    c.execute(f'DELETE FROM {table}')
                    tables_wiped.append(table)
                except Exception:
                    pass  # Table may not exist yet (lazy-created)

            conn.commit()
            conn.close()
            results['steps'].append({'step': 'wipe_database', 'success': True, 'tables': tables_wiped})
        except Exception as e:
            results['steps'].append({'step': 'wipe_database', 'success': False, 'error': str(e)})

        # ── Step 4: Reset settings to defaults ─────────────────────────
        try:
            default_settings = {
                "theme": {"mode": "dark", "accentColor": "#3b82f6", "fontSize": "medium"},
                "background": {"type": "gradient", "gradient": "purple-blue", "bubbles": True,
                               "bubbleCount": 15, "bubbleSpeed": 1.0},
                "ai": {"enabled": True, "model": "claude-3-sonnet", "contextLength": 4096,
                       "temperature": 0.7},
                "dmx": {"defaultFadeMs": 500, "refreshRate": 40, "maxUniverse": 64},
                "security": {"pinEnabled": False, "sessionTimeout": 3600},
                "setup": {"complete": False, "mode": None, "userProfile": {}},
            }
            _app_settings.clear()
            _app_settings.update(deepcopy(default_settings))
            _save_settings(_app_settings)
            results['steps'].append({'step': 'reset_settings', 'success': True})
        except Exception as e:
            results['steps'].append({'step': 'reset_settings', 'success': False, 'error': str(e)})

        # ── Step 5: Reset DMX state ────────────────────────────────────
        try:
            if _dmx_state:
                with _dmx_state.lock:
                    _dmx_state.universes.clear()
                    _dmx_state.targets.clear()
                    _dmx_state.fade_info.clear()
                    _dmx_state.master_level = 100
                    _dmx_state.master_base.clear()
                    _dmx_state.last_session = None
                    _dmx_state.last_sessions = []
                _dmx_state.save_state_now()
            results['steps'].append({'step': 'reset_dmx_state', 'success': True})
        except Exception as e:
            results['steps'].append({'step': 'reset_dmx_state', 'success': False, 'error': str(e)})

        # ── Step 6: Re-seed with stock content ─────────────────────────
        try:
            from seed_content import seed_database
            seed_result = seed_database(_get_db)
            results['steps'].append({'step': 'seed_content', 'success': True, 'seeded': seed_result})
        except Exception as e:
            results['steps'].append({'step': 'seed_content', 'success': False, 'error': str(e)})

        # ── Step 7: Notify frontend ────────────────────────────────────
        try:
            if _socketio:
                _socketio.emit('factory_reset', {'status': 'complete'})
                _socketio.emit('scenes_update', {})
                _socketio.emit('chases_update', {})
                _socketio.emit('nodes_update', {})
                _socketio.emit('dmx_state', {'universe': 1, 'channels': [0] * 512})
            results['steps'].append({'step': 'notify_frontend', 'success': True})
        except Exception as e:
            results['steps'].append({'step': 'notify_frontend', 'success': False, 'error': str(e)})

        results['success'] = True
        results['message'] = 'Factory reset complete. System restored to defaults with stock content.'
        print(f"🏭 FACTORY RESET COMPLETE: {len(tables_wiped)} tables wiped, {unpaired_count} nodes unpaired")
        return jsonify(results)

    except Exception as e:
        results['error'] = str(e)
        return jsonify(results), 500
