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


def init_app(app_settings, save_settings_fn, aether_version, aether_commit,
             aether_file_path, aether_start_time, get_or_create_device_id_fn, app):
    """Initialize blueprint with required dependencies."""
    global _app_settings, _save_settings, _AETHER_VERSION, _AETHER_COMMIT
    global _AETHER_FILE_PATH, _AETHER_START_TIME, _get_or_create_device_id, _app
    _app_settings = app_settings
    _save_settings = save_settings_fn
    _AETHER_VERSION = aether_version
    _AETHER_COMMIT = aether_commit
    _AETHER_FILE_PATH = aether_file_path
    _AETHER_START_TIME = aether_start_time
    _get_or_create_device_id = get_or_create_device_id_fn
    _app = app


@system_bp.route('/api/system/info', methods=['GET'])
def system_info():
    """System information for frontend mode detection"""
    device_id = _get_or_create_device_id()
    is_pi = os.path.exists('/sys/firmware/devicetree/base/model')
    has_display = bool(os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY'))

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

    runtime_file = '/home/ramzt/aether-core.py'

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
    try:
        subprocess.run(
            ['git', 'fetch', 'origin'],
            capture_output=True, text=True, timeout=30,
            cwd=os.path.dirname(_AETHER_FILE_PATH)
        )

        result = subprocess.run(
            ['git', 'rev-list', 'HEAD..origin/main', '--count'],
            capture_output=True, text=True, timeout=10,
            cwd=os.path.dirname(_AETHER_FILE_PATH)
        )

        commits_behind = int(result.stdout.strip()) if result.returncode == 0 else 0

        log_result = subprocess.run(
            ['git', 'log', 'origin/main', '-1', '--format=%h %s'],
            capture_output=True, text=True, timeout=10,
            cwd=os.path.dirname(_AETHER_FILE_PATH)
        )

        return jsonify({
            'current_commit': _AETHER_COMMIT,
            'update_available': commits_behind > 0,
            'commits_behind': commits_behind,
            'latest_commit': log_result.stdout.strip() if log_result.returncode == 0 else None
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@system_bp.route('/api/system/autosync', methods=['GET'])
def get_autosync_status():
    """Get auto-sync status"""
    return jsonify({
        'enabled': getattr(_app, '_autosync_enabled', False),
        'interval_minutes': getattr(_app, '_autosync_interval', 30),
        'last_check': getattr(_app, '_autosync_last_check', None),
        'last_update': getattr(_app, '_autosync_last_update', None)
    })

@system_bp.route('/api/system/autosync', methods=['POST'])
def set_autosync():
    """Enable/disable auto-sync"""
    data = request.get_json() or {}
    enabled = data.get('enabled', False)
    interval = data.get('interval_minutes', 30)

    _app._autosync_enabled = enabled
    _app._autosync_interval = max(5, min(1440, interval))

    if enabled:
        _start_autosync_thread()
        print(f"✓ Auto-sync enabled: checking every {_app._autosync_interval} minutes")
    else:
        print("✓ Auto-sync disabled")

    return jsonify({
        'success': True,
        'enabled': _app._autosync_enabled,
        'interval_minutes': _app._autosync_interval
    })


def _start_autosync_thread():
    """Start background thread for auto-sync"""
    def autosync_worker():
        git_dir = os.path.dirname(_AETHER_FILE_PATH)
        if '/aether-core-git/' not in _AETHER_FILE_PATH and '/aether-core/' not in _AETHER_FILE_PATH:
            git_dir = os.path.expanduser('~/aether-core-git/aether-core')
        runtime_file = '/home/ramzt/aether-core.py'

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
