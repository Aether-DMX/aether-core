"""
AETHER Playback State Manager - Tracks current playback state across universes

Extracted from aether-core.py for modularity.
Uses core_registry for cross-module references.
"""

import threading
from datetime import datetime
import core_registry as reg


class PlaybackManager:
    """Tracks current playback state across all universes"""
    def __init__(self):
        self.lock = threading.Lock()
        self.current = {}  # {universe: {'type': 'scene'|'chase', 'id': '...', 'started': datetime}}

    def set_playing(self, universe, content_type, content_id):
        with self.lock:
            self.current[universe] = {
                'type': content_type,
                'id': content_id,
                'started': datetime.now().isoformat()
            }
        if reg.socketio:
            reg.socketio.emit('playback_update', {'universe': universe, 'playback': self.current.get(universe)})

    def stop(self, universe=None):
        with self.lock:
            if universe:
                self.current.pop(universe, None)
            else:
                self.current.clear()
        if reg.socketio:
            reg.socketio.emit('playback_update', {'universe': universe, 'playback': None})

    def get_status(self, universe=None):
        with self.lock:
            status = self.current.copy()
            # Include running effects
            if reg.effects_engine and reg.effects_engine.running:
                for effect_id in reg.effects_engine.running.keys():
                    # Extract effect type from id (e.g., "strobe_1234" -> "strobe")
                    effect_type = effect_id.split('_')[0] if '_' in effect_id else effect_id
                    # Add to status as type 'effect'
                    status['effect'] = {
                        'type': 'effect',
                        'id': effect_id,
                        'name': effect_type.capitalize(),
                        'started': None
                    }
            if universe:
                return status.get(universe)
            return status
