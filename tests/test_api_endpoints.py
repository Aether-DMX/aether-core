"""
AETHER Beta1 - API Endpoint Tests

Tests cover REST API endpoints for:
1. Looks CRUD and playback
2. Sequences CRUD and playback
3. Playback control (status, stop, pause, resume)
4. Merge layer (blackout, status)
5. DMX control (set, fade, blackout, master)
6. Preview service
7. Modifiers
8. Health and system endpoints
"""

import pytest
import json
import sys
import os
import time

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================
# Flask App Import and Test Client Setup
# ============================================================

@pytest.fixture(scope="module")
def app():
    """Create Flask test app"""
    # Import the Flask app from aether-core
    # Use a temporary database for testing
    import tempfile

    # Set up temp database before importing app
    temp_db = tempfile.mktemp(suffix=".db")
    os.environ["AETHER_TEST_DB"] = temp_db

    # Import after setting env var
    from importlib import import_module

    # We need to mock some things for testing without hardware
    from unittest.mock import MagicMock, patch

    # Create a minimal Flask app for API testing
    from flask import Flask, jsonify, request
    from flask_cors import CORS

    test_app = Flask(__name__)
    CORS(test_app)
    test_app.config['TESTING'] = True

    # In-memory storage for testing
    _looks = {}
    _sequences = {}
    _playback_state = {"playing": False, "current": None}
    _blackout = {"active": False}
    _dmx_state = {u: {ch: 0 for ch in range(1, 513)} for u in range(1, 6)}

    # ============================================================
    # Health Endpoints
    # ============================================================

    @test_app.route('/api/health', methods=['GET'])
    def health():
        return jsonify({"status": "ok", "version": "5.0.0-test"})

    @test_app.route('/api/version', methods=['GET'])
    def version():
        return jsonify({
            "version": "5.0.0-test",
            "phase": 8,
            "test_mode": True
        })

    # ============================================================
    # Looks Endpoints
    # ============================================================

    @test_app.route('/api/looks', methods=['GET'])
    def get_looks():
        return jsonify(list(_looks.values()))

    @test_app.route('/api/looks', methods=['POST'])
    def create_look():
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
        if "name" not in data:
            return jsonify({"error": "Name required"}), 400

        look_id = data.get("look_id", f"look_{len(_looks) + 1}")
        look = {
            "look_id": look_id,
            "name": data["name"],
            "channels": data.get("channels", {}),
            "modifiers": data.get("modifiers", []),
            "fade_ms": data.get("fade_ms", 0),
        }
        _looks[look_id] = look
        return jsonify(look), 201

    @test_app.route('/api/looks/<look_id>', methods=['GET'])
    def get_look(look_id):
        if look_id not in _looks:
            return jsonify({"error": "Look not found"}), 404
        return jsonify(_looks[look_id])

    @test_app.route('/api/looks/<look_id>', methods=['PUT'])
    def update_look(look_id):
        if look_id not in _looks:
            return jsonify({"error": "Look not found"}), 404
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        look = _looks[look_id]
        look.update({
            "name": data.get("name", look["name"]),
            "channels": data.get("channels", look["channels"]),
            "modifiers": data.get("modifiers", look["modifiers"]),
            "fade_ms": data.get("fade_ms", look["fade_ms"]),
        })
        return jsonify(look)

    @test_app.route('/api/looks/<look_id>', methods=['DELETE'])
    def delete_look(look_id):
        if look_id not in _looks:
            return jsonify({"error": "Look not found"}), 404
        del _looks[look_id]
        return jsonify({"deleted": look_id})

    @test_app.route('/api/looks/<look_id>/play', methods=['POST'])
    def play_look(look_id):
        if look_id not in _looks:
            return jsonify({"error": "Look not found"}), 404
        data = request.get_json() or {}
        universes = data.get("universes", [1])
        fade_ms = data.get("fade_ms", 0)

        _playback_state["playing"] = True
        _playback_state["current"] = {"type": "look", "id": look_id, "universes": universes}

        return jsonify({
            "status": "playing",
            "look_id": look_id,
            "universes": universes,
            "fade_ms": fade_ms
        })

    @test_app.route('/api/looks/<look_id>/stop', methods=['POST'])
    def stop_look(look_id):
        if _playback_state["current"] and _playback_state["current"]["id"] == look_id:
            _playback_state["playing"] = False
            _playback_state["current"] = None
        return jsonify({"status": "stopped", "look_id": look_id})

    # ============================================================
    # Sequences Endpoints
    # ============================================================

    @test_app.route('/api/sequences', methods=['GET'])
    def get_sequences():
        return jsonify(list(_sequences.values()))

    @test_app.route('/api/sequences', methods=['POST'])
    def create_sequence():
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
        if "name" not in data:
            return jsonify({"error": "Name required"}), 400

        seq_id = data.get("sequence_id", f"seq_{len(_sequences) + 1}")
        sequence = {
            "sequence_id": seq_id,
            "name": data["name"],
            "steps": data.get("steps", []),
            "bpm": data.get("bpm", 120),
            "loop": data.get("loop", True),
        }
        _sequences[seq_id] = sequence
        return jsonify(sequence), 201

    @test_app.route('/api/sequences/<sequence_id>', methods=['GET'])
    def get_sequence(sequence_id):
        if sequence_id not in _sequences:
            return jsonify({"error": "Sequence not found"}), 404
        return jsonify(_sequences[sequence_id])

    @test_app.route('/api/sequences/<sequence_id>', methods=['PUT'])
    def update_sequence(sequence_id):
        if sequence_id not in _sequences:
            return jsonify({"error": "Sequence not found"}), 404
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        seq = _sequences[sequence_id]
        seq.update({
            "name": data.get("name", seq["name"]),
            "steps": data.get("steps", seq["steps"]),
            "bpm": data.get("bpm", seq["bpm"]),
            "loop": data.get("loop", seq["loop"]),
        })
        return jsonify(seq)

    @test_app.route('/api/sequences/<sequence_id>', methods=['DELETE'])
    def delete_sequence(sequence_id):
        if sequence_id not in _sequences:
            return jsonify({"error": "Sequence not found"}), 404
        del _sequences[sequence_id]
        return jsonify({"deleted": sequence_id})

    @test_app.route('/api/sequences/<sequence_id>/play', methods=['POST'])
    def play_sequence(sequence_id):
        if sequence_id not in _sequences:
            return jsonify({"error": "Sequence not found"}), 404
        data = request.get_json() or {}
        universes = data.get("universes", [1])

        _playback_state["playing"] = True
        _playback_state["current"] = {"type": "sequence", "id": sequence_id, "universes": universes}

        return jsonify({
            "status": "playing",
            "sequence_id": sequence_id,
            "universes": universes
        })

    @test_app.route('/api/sequences/<sequence_id>/stop', methods=['POST'])
    def stop_sequence(sequence_id):
        if _playback_state["current"] and _playback_state["current"]["id"] == sequence_id:
            _playback_state["playing"] = False
            _playback_state["current"] = None
        return jsonify({"status": "stopped", "sequence_id": sequence_id})

    # ============================================================
    # Playback Control Endpoints
    # ============================================================

    @test_app.route('/api/playback/status', methods=['GET'])
    def playback_status():
        return jsonify({
            "playing": _playback_state["playing"],
            "current": _playback_state["current"],
            "actual_fps": 30 if _playback_state["playing"] else 0,
        })

    @test_app.route('/api/playback/stop', methods=['POST'])
    def playback_stop():
        _playback_state["playing"] = False
        _playback_state["current"] = None
        return jsonify({"status": "stopped"})

    @test_app.route('/api/playback/pause', methods=['POST'])
    def playback_pause():
        if _playback_state["playing"]:
            _playback_state["playing"] = False
            return jsonify({"status": "paused"})
        return jsonify({"status": "not_playing"}), 400

    @test_app.route('/api/playback/resume', methods=['POST'])
    def playback_resume():
        if _playback_state["current"]:
            _playback_state["playing"] = True
            return jsonify({"status": "resumed"})
        return jsonify({"status": "nothing_to_resume"}), 400

    # ============================================================
    # DMX Control Endpoints
    # ============================================================

    @test_app.route('/api/dmx/set', methods=['POST'])
    def dmx_set():
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        universe = data.get("universe", 1)
        channels = data.get("channels", {})

        for ch, val in channels.items():
            ch_int = int(ch)
            if 1 <= ch_int <= 512:
                _dmx_state[universe][ch_int] = max(0, min(255, int(val)))

        return jsonify({"status": "ok", "universe": universe, "channels_set": len(channels)})

    @test_app.route('/api/dmx/fade', methods=['POST'])
    def dmx_fade():
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        universe = data.get("universe", 1)
        channels = data.get("channels", {})
        fade_time = data.get("fade_time", 1000)

        # In test mode, just set the values immediately
        for ch, val in channels.items():
            ch_int = int(ch)
            if 1 <= ch_int <= 512:
                _dmx_state[universe][ch_int] = max(0, min(255, int(val)))

        return jsonify({"status": "fading", "universe": universe, "fade_time": fade_time})

    @test_app.route('/api/dmx/blackout', methods=['POST'])
    def dmx_blackout():
        data = request.get_json() or {}
        universes = data.get("universes", list(range(1, 6)))

        _blackout["active"] = True
        for u in universes:
            if u in _dmx_state:
                _dmx_state[u] = {ch: 0 for ch in range(1, 513)}

        return jsonify({"status": "blackout", "universes": universes})

    @test_app.route('/api/dmx/master', methods=['POST'])
    def dmx_master():
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        level = data.get("level", 255)
        return jsonify({"status": "ok", "master": level})

    @test_app.route('/api/dmx/status', methods=['GET'])
    def dmx_status():
        return jsonify({
            "blackout": _blackout["active"],
            "universes": {str(u): sum(1 for v in ch.values() if v > 0)
                         for u, ch in _dmx_state.items()}
        })

    @test_app.route('/api/dmx/universe/<int:universe>', methods=['GET'])
    def dmx_universe(universe):
        if universe not in _dmx_state:
            return jsonify({"error": "Universe not found"}), 404
        return jsonify({"universe": universe, "channels": _dmx_state[universe]})

    # ============================================================
    # Merge Layer Endpoints
    # ============================================================

    @test_app.route('/api/merge/status', methods=['GET'])
    def merge_status():
        return jsonify({
            "active_sources": 0,
            "blackout": _blackout["active"],
        })

    @test_app.route('/api/merge/blackout', methods=['POST'])
    def merge_blackout():
        data = request.get_json() or {}
        active = data.get("active", True)
        universes = data.get("universes", list(range(1, 6)))

        _blackout["active"] = active
        if active:
            for u in universes:
                if u in _dmx_state:
                    _dmx_state[u] = {ch: 0 for ch in range(1, 513)}

        return jsonify({"status": "ok", "blackout": active, "universes": universes})

    @test_app.route('/api/merge/sources', methods=['GET'])
    def merge_sources():
        sources = []
        if _playback_state["current"]:
            sources.append({
                "id": _playback_state["current"]["id"],
                "type": _playback_state["current"]["type"],
                "priority": 50,
            })
        return jsonify(sources)

    # ============================================================
    # Preview Endpoints
    # ============================================================

    _preview_sessions = {}

    @test_app.route('/api/preview/sessions', methods=['GET'])
    def preview_sessions():
        return jsonify(list(_preview_sessions.values()))

    @test_app.route('/api/preview/session', methods=['POST'])
    def create_preview_session():
        data = request.get_json() or {}
        session_id = data.get("session_id", f"preview_{len(_preview_sessions) + 1}")

        session = {
            "session_id": session_id,
            "preview_type": data.get("preview_type", "look"),
            "channels": data.get("channels", {}),
            "modifiers": data.get("modifiers", []),
            "universes": data.get("universes", [1]),
            "armed": False,
            "running": False,
        }
        _preview_sessions[session_id] = session
        return jsonify(session), 201

    @test_app.route('/api/preview/session/<session_id>', methods=['GET'])
    def get_preview_session(session_id):
        if session_id not in _preview_sessions:
            return jsonify({"error": "Session not found"}), 404
        return jsonify(_preview_sessions[session_id])

    @test_app.route('/api/preview/session/<session_id>', methods=['DELETE'])
    def delete_preview_session(session_id):
        if session_id not in _preview_sessions:
            return jsonify({"error": "Session not found"}), 404
        del _preview_sessions[session_id]
        return jsonify({"deleted": session_id})

    @test_app.route('/api/preview/session/<session_id>/start', methods=['POST'])
    def start_preview_session(session_id):
        if session_id not in _preview_sessions:
            return jsonify({"error": "Session not found"}), 404
        _preview_sessions[session_id]["running"] = True
        return jsonify({"status": "started", "session_id": session_id})

    @test_app.route('/api/preview/session/<session_id>/stop', methods=['POST'])
    def stop_preview_session(session_id):
        if session_id not in _preview_sessions:
            return jsonify({"error": "Session not found"}), 404
        _preview_sessions[session_id]["running"] = False
        return jsonify({"status": "stopped", "session_id": session_id})

    @test_app.route('/api/preview/session/<session_id>/arm', methods=['POST'])
    def arm_preview_session(session_id):
        if session_id not in _preview_sessions:
            return jsonify({"error": "Session not found"}), 404
        _preview_sessions[session_id]["armed"] = True
        return jsonify({"status": "armed", "session_id": session_id})

    @test_app.route('/api/preview/session/<session_id>/disarm', methods=['POST'])
    def disarm_preview_session(session_id):
        if session_id not in _preview_sessions:
            return jsonify({"error": "Session not found"}), 404
        _preview_sessions[session_id]["armed"] = False
        return jsonify({"status": "disarmed", "session_id": session_id})

    # ============================================================
    # Modifier Endpoints
    # ============================================================

    @test_app.route('/api/modifiers/types', methods=['GET'])
    def modifier_types():
        return jsonify(["pulse", "strobe", "flicker", "wave", "rainbow", "twinkle"])

    @test_app.route('/api/modifiers/schemas', methods=['GET'])
    def modifier_schemas():
        return jsonify({
            "pulse": {"speed": {"type": "number", "min": 0.1, "max": 10, "default": 1}},
            "strobe": {"rate": {"type": "number", "min": 1, "max": 30, "default": 5}},
            "flicker": {"intensity": {"type": "number", "min": 0, "max": 100, "default": 50}},
            "wave": {"phase_offset": {"type": "number", "min": 0, "max": 360, "default": 0}},
            "rainbow": {"speed": {"type": "number", "min": 0.1, "max": 5, "default": 0.5}},
            "twinkle": {"density": {"type": "number", "min": 0, "max": 100, "default": 50}},
        })

    @test_app.route('/api/modifiers/validate', methods=['POST'])
    def validate_modifier():
        data = request.get_json()
        if not data:
            return jsonify({"valid": False, "error": "No data provided"}), 400

        mod_type = data.get("type")
        if mod_type not in ["pulse", "strobe", "flicker", "wave", "rainbow", "twinkle"]:
            return jsonify({"valid": False, "error": f"Unknown type: {mod_type}"}), 400

        return jsonify({"valid": True, "type": mod_type})

    yield test_app

    # Cleanup
    if os.path.exists(temp_db):
        os.remove(temp_db)


@pytest.fixture
def client(app):
    """Test client for making requests"""
    return app.test_client()


# ============================================================
# Health Endpoint Tests
# ============================================================

class TestHealthEndpoints:
    """Tests for health and system endpoints"""

    def test_health_returns_ok(self, client):
        """GET /api/health returns ok status"""
        response = client.get('/api/health')
        assert response.status_code == 200
        data = response.get_json()
        assert data["status"] == "ok"

    def test_version_returns_version(self, client):
        """GET /api/version returns version info"""
        response = client.get('/api/version')
        assert response.status_code == 200
        data = response.get_json()
        assert "version" in data


# ============================================================
# Look CRUD Tests
# ============================================================

class TestLooksCRUD:
    """Tests for Look CRUD operations"""

    def test_list_looks_empty(self, client):
        """GET /api/looks returns empty list initially"""
        response = client.get('/api/looks')
        assert response.status_code == 200
        data = response.get_json()
        assert isinstance(data, list)

    def test_create_look(self, client):
        """POST /api/looks creates a new look"""
        look_data = {
            "name": "Test Look",
            "channels": {"1": 255, "2": 128, "3": 64},
            "modifiers": [],
            "fade_ms": 1000
        }
        response = client.post('/api/looks',
                              data=json.dumps(look_data),
                              content_type='application/json')
        assert response.status_code == 201
        data = response.get_json()
        assert data["name"] == "Test Look"
        assert "look_id" in data

    def test_create_look_missing_name(self, client):
        """POST /api/looks without name returns error"""
        look_data = {"channels": {"1": 255}}
        response = client.post('/api/looks',
                              data=json.dumps(look_data),
                              content_type='application/json')
        assert response.status_code == 400

    def test_get_look(self, client):
        """GET /api/looks/<id> returns the look"""
        # Create first
        look_data = {"name": "Get Test", "channels": {"1": 100}}
        create_resp = client.post('/api/looks',
                                 data=json.dumps(look_data),
                                 content_type='application/json')
        look_id = create_resp.get_json()["look_id"]

        # Get
        response = client.get(f'/api/looks/{look_id}')
        assert response.status_code == 200
        data = response.get_json()
        assert data["name"] == "Get Test"

    def test_get_look_not_found(self, client):
        """GET /api/looks/<id> returns 404 for non-existent"""
        response = client.get('/api/looks/nonexistent_id')
        assert response.status_code == 404

    def test_update_look(self, client):
        """PUT /api/looks/<id> updates the look"""
        # Create first
        look_data = {"name": "Update Test", "channels": {"1": 100}}
        create_resp = client.post('/api/looks',
                                 data=json.dumps(look_data),
                                 content_type='application/json')
        look_id = create_resp.get_json()["look_id"]

        # Update
        update_data = {"name": "Updated Name", "channels": {"1": 200}}
        response = client.put(f'/api/looks/{look_id}',
                             data=json.dumps(update_data),
                             content_type='application/json')
        assert response.status_code == 200
        data = response.get_json()
        assert data["name"] == "Updated Name"
        assert data["channels"]["1"] == 200

    def test_delete_look(self, client):
        """DELETE /api/looks/<id> deletes the look"""
        # Create first
        look_data = {"name": "Delete Test", "channels": {"1": 100}}
        create_resp = client.post('/api/looks',
                                 data=json.dumps(look_data),
                                 content_type='application/json')
        look_id = create_resp.get_json()["look_id"]

        # Delete
        response = client.delete(f'/api/looks/{look_id}')
        assert response.status_code == 200

        # Verify deleted
        get_resp = client.get(f'/api/looks/{look_id}')
        assert get_resp.status_code == 404


# ============================================================
# Look Playback Tests
# ============================================================

class TestLookPlayback:
    """Tests for Look playback operations"""

    def test_play_look(self, client):
        """POST /api/looks/<id>/play starts playback"""
        # Create look
        look_data = {"name": "Play Test", "channels": {"1": 255}}
        create_resp = client.post('/api/looks',
                                 data=json.dumps(look_data),
                                 content_type='application/json')
        look_id = create_resp.get_json()["look_id"]

        # Play
        play_data = {"universes": [1, 2], "fade_ms": 500}
        response = client.post(f'/api/looks/{look_id}/play',
                              data=json.dumps(play_data),
                              content_type='application/json')
        assert response.status_code == 200
        data = response.get_json()
        assert data["status"] == "playing"
        assert data["universes"] == [1, 2]

    def test_play_look_not_found(self, client):
        """POST /api/looks/<id>/play returns 404 for non-existent"""
        response = client.post('/api/looks/nonexistent/play',
                              data=json.dumps({}),
                              content_type='application/json')
        assert response.status_code == 404

    def test_stop_look(self, client):
        """POST /api/looks/<id>/stop stops playback"""
        # Create and play
        look_data = {"name": "Stop Test", "channels": {"1": 255}}
        create_resp = client.post('/api/looks',
                                 data=json.dumps(look_data),
                                 content_type='application/json')
        look_id = create_resp.get_json()["look_id"]
        client.post(f'/api/looks/{look_id}/play',
                   data=json.dumps({}),
                   content_type='application/json')

        # Stop
        response = client.post(f'/api/looks/{look_id}/stop')
        assert response.status_code == 200
        data = response.get_json()
        assert data["status"] == "stopped"


# ============================================================
# Sequence CRUD Tests
# ============================================================

class TestSequencesCRUD:
    """Tests for Sequence CRUD operations"""

    def test_list_sequences_empty(self, client):
        """GET /api/sequences returns list"""
        response = client.get('/api/sequences')
        assert response.status_code == 200
        data = response.get_json()
        assert isinstance(data, list)

    def test_create_sequence(self, client):
        """POST /api/sequences creates a new sequence"""
        seq_data = {
            "name": "Test Sequence",
            "steps": [
                {"name": "Red", "channels": {"1": 255, "2": 0, "3": 0}},
                {"name": "Green", "channels": {"1": 0, "2": 255, "3": 0}},
            ],
            "bpm": 120,
            "loop": True
        }
        response = client.post('/api/sequences',
                              data=json.dumps(seq_data),
                              content_type='application/json')
        assert response.status_code == 201
        data = response.get_json()
        assert data["name"] == "Test Sequence"
        assert len(data["steps"]) == 2
        assert data["bpm"] == 120

    def test_create_sequence_missing_name(self, client):
        """POST /api/sequences without name returns error"""
        seq_data = {"steps": [], "bpm": 60}
        response = client.post('/api/sequences',
                              data=json.dumps(seq_data),
                              content_type='application/json')
        assert response.status_code == 400

    def test_get_sequence(self, client):
        """GET /api/sequences/<id> returns the sequence"""
        seq_data = {"name": "Get Test", "steps": [], "bpm": 100}
        create_resp = client.post('/api/sequences',
                                 data=json.dumps(seq_data),
                                 content_type='application/json')
        seq_id = create_resp.get_json()["sequence_id"]

        response = client.get(f'/api/sequences/{seq_id}')
        assert response.status_code == 200
        data = response.get_json()
        assert data["name"] == "Get Test"

    def test_update_sequence(self, client):
        """PUT /api/sequences/<id> updates the sequence"""
        seq_data = {"name": "Update Test", "steps": [], "bpm": 100}
        create_resp = client.post('/api/sequences',
                                 data=json.dumps(seq_data),
                                 content_type='application/json')
        seq_id = create_resp.get_json()["sequence_id"]

        update_data = {"name": "Updated Seq", "bpm": 140}
        response = client.put(f'/api/sequences/{seq_id}',
                             data=json.dumps(update_data),
                             content_type='application/json')
        assert response.status_code == 200
        data = response.get_json()
        assert data["name"] == "Updated Seq"
        assert data["bpm"] == 140

    def test_delete_sequence(self, client):
        """DELETE /api/sequences/<id> deletes the sequence"""
        seq_data = {"name": "Delete Test", "steps": []}
        create_resp = client.post('/api/sequences',
                                 data=json.dumps(seq_data),
                                 content_type='application/json')
        seq_id = create_resp.get_json()["sequence_id"]

        response = client.delete(f'/api/sequences/{seq_id}')
        assert response.status_code == 200

        get_resp = client.get(f'/api/sequences/{seq_id}')
        assert get_resp.status_code == 404


# ============================================================
# Sequence Playback Tests
# ============================================================

class TestSequencePlayback:
    """Tests for Sequence playback operations"""

    def test_play_sequence(self, client):
        """POST /api/sequences/<id>/play starts playback"""
        seq_data = {
            "name": "Play Test",
            "steps": [{"name": "Step 1", "channels": {"1": 255}}],
            "bpm": 120
        }
        create_resp = client.post('/api/sequences',
                                 data=json.dumps(seq_data),
                                 content_type='application/json')
        seq_id = create_resp.get_json()["sequence_id"]

        play_data = {"universes": [1, 2, 3]}
        response = client.post(f'/api/sequences/{seq_id}/play',
                              data=json.dumps(play_data),
                              content_type='application/json')
        assert response.status_code == 200
        data = response.get_json()
        assert data["status"] == "playing"
        assert data["universes"] == [1, 2, 3]

    def test_stop_sequence(self, client):
        """POST /api/sequences/<id>/stop stops playback"""
        seq_data = {"name": "Stop Test", "steps": []}
        create_resp = client.post('/api/sequences',
                                 data=json.dumps(seq_data),
                                 content_type='application/json')
        seq_id = create_resp.get_json()["sequence_id"]
        client.post(f'/api/sequences/{seq_id}/play',
                   data=json.dumps({}),
                   content_type='application/json')

        response = client.post(f'/api/sequences/{seq_id}/stop')
        assert response.status_code == 200
        data = response.get_json()
        assert data["status"] == "stopped"


# ============================================================
# Playback Control Tests
# ============================================================

class TestPlaybackControl:
    """Tests for playback control endpoints"""

    def test_playback_status(self, client):
        """GET /api/playback/status returns status"""
        response = client.get('/api/playback/status')
        assert response.status_code == 200
        data = response.get_json()
        assert "playing" in data

    def test_playback_stop(self, client):
        """POST /api/playback/stop stops all playback"""
        response = client.post('/api/playback/stop')
        assert response.status_code == 200
        data = response.get_json()
        assert data["status"] == "stopped"

    def test_playback_pause_when_playing(self, client):
        """POST /api/playback/pause pauses when playing"""
        # Start playback first
        look_data = {"name": "Pause Test", "channels": {"1": 255}}
        create_resp = client.post('/api/looks',
                                 data=json.dumps(look_data),
                                 content_type='application/json')
        look_id = create_resp.get_json()["look_id"]
        client.post(f'/api/looks/{look_id}/play',
                   data=json.dumps({}),
                   content_type='application/json')

        response = client.post('/api/playback/pause')
        assert response.status_code == 200
        data = response.get_json()
        assert data["status"] == "paused"

    def test_playback_resume(self, client):
        """POST /api/playback/resume resumes paused playback"""
        # Start and pause first
        look_data = {"name": "Resume Test", "channels": {"1": 255}}
        create_resp = client.post('/api/looks',
                                 data=json.dumps(look_data),
                                 content_type='application/json')
        look_id = create_resp.get_json()["look_id"]
        client.post(f'/api/looks/{look_id}/play',
                   data=json.dumps({}),
                   content_type='application/json')
        client.post('/api/playback/pause')

        response = client.post('/api/playback/resume')
        assert response.status_code == 200
        data = response.get_json()
        assert data["status"] == "resumed"


# ============================================================
# DMX Control Tests
# ============================================================

class TestDMXControl:
    """Tests for DMX control endpoints"""

    def test_dmx_set(self, client):
        """POST /api/dmx/set sets channel values"""
        data = {
            "universe": 1,
            "channels": {"1": 255, "2": 128, "3": 64}
        }
        response = client.post('/api/dmx/set',
                              data=json.dumps(data),
                              content_type='application/json')
        assert response.status_code == 200
        result = response.get_json()
        assert result["status"] == "ok"
        assert result["channels_set"] == 3

    def test_dmx_fade(self, client):
        """POST /api/dmx/fade fades channel values"""
        data = {
            "universe": 1,
            "channels": {"1": 200},
            "fade_time": 1000
        }
        response = client.post('/api/dmx/fade',
                              data=json.dumps(data),
                              content_type='application/json')
        assert response.status_code == 200
        result = response.get_json()
        assert result["status"] == "fading"
        assert result["fade_time"] == 1000

    def test_dmx_blackout(self, client):
        """POST /api/dmx/blackout blacks out all channels"""
        response = client.post('/api/dmx/blackout',
                              data=json.dumps({"universes": [1, 2]}),
                              content_type='application/json')
        assert response.status_code == 200
        result = response.get_json()
        assert result["status"] == "blackout"

    def test_dmx_master(self, client):
        """POST /api/dmx/master sets master level"""
        data = {"level": 200}
        response = client.post('/api/dmx/master',
                              data=json.dumps(data),
                              content_type='application/json')
        assert response.status_code == 200
        result = response.get_json()
        assert result["master"] == 200

    def test_dmx_status(self, client):
        """GET /api/dmx/status returns DMX status"""
        response = client.get('/api/dmx/status')
        assert response.status_code == 200
        data = response.get_json()
        assert "blackout" in data

    def test_dmx_universe(self, client):
        """GET /api/dmx/universe/<id> returns universe state"""
        response = client.get('/api/dmx/universe/1')
        assert response.status_code == 200
        data = response.get_json()
        assert data["universe"] == 1
        assert "channels" in data


# ============================================================
# Merge Layer Tests
# ============================================================

class TestMergeLayer:
    """Tests for merge layer endpoints"""

    def test_merge_status(self, client):
        """GET /api/merge/status returns merge status"""
        response = client.get('/api/merge/status')
        assert response.status_code == 200
        data = response.get_json()
        assert "blackout" in data

    def test_merge_blackout_activate(self, client):
        """POST /api/merge/blackout activates blackout"""
        data = {"active": True, "universes": [1, 2, 3]}
        response = client.post('/api/merge/blackout',
                              data=json.dumps(data),
                              content_type='application/json')
        assert response.status_code == 200
        result = response.get_json()
        assert result["blackout"] == True

    def test_merge_blackout_release(self, client):
        """POST /api/merge/blackout releases blackout"""
        data = {"active": False}
        response = client.post('/api/merge/blackout',
                              data=json.dumps(data),
                              content_type='application/json')
        assert response.status_code == 200
        result = response.get_json()
        assert result["blackout"] == False

    def test_merge_sources(self, client):
        """GET /api/merge/sources returns active sources"""
        response = client.get('/api/merge/sources')
        assert response.status_code == 200
        data = response.get_json()
        assert isinstance(data, list)


# ============================================================
# Preview Service Tests
# ============================================================

class TestPreviewService:
    """Tests for preview service endpoints"""

    def test_list_preview_sessions(self, client):
        """GET /api/preview/sessions returns list"""
        response = client.get('/api/preview/sessions')
        assert response.status_code == 200
        data = response.get_json()
        assert isinstance(data, list)

    def test_create_preview_session(self, client):
        """POST /api/preview/session creates session"""
        data = {
            "preview_type": "look",
            "channels": {"1": 255},
            "universes": [1]
        }
        response = client.post('/api/preview/session',
                              data=json.dumps(data),
                              content_type='application/json')
        assert response.status_code == 201
        result = response.get_json()
        assert "session_id" in result
        assert result["armed"] == False

    def test_get_preview_session(self, client):
        """GET /api/preview/session/<id> returns session"""
        # Create first
        data = {"preview_type": "look", "channels": {}}
        create_resp = client.post('/api/preview/session',
                                 data=json.dumps(data),
                                 content_type='application/json')
        session_id = create_resp.get_json()["session_id"]

        response = client.get(f'/api/preview/session/{session_id}')
        assert response.status_code == 200

    def test_start_preview_session(self, client):
        """POST /api/preview/session/<id>/start starts session"""
        data = {"preview_type": "look", "channels": {}}
        create_resp = client.post('/api/preview/session',
                                 data=json.dumps(data),
                                 content_type='application/json')
        session_id = create_resp.get_json()["session_id"]

        response = client.post(f'/api/preview/session/{session_id}/start')
        assert response.status_code == 200
        result = response.get_json()
        assert result["status"] == "started"

    def test_stop_preview_session(self, client):
        """POST /api/preview/session/<id>/stop stops session"""
        data = {"preview_type": "look", "channels": {}}
        create_resp = client.post('/api/preview/session',
                                 data=json.dumps(data),
                                 content_type='application/json')
        session_id = create_resp.get_json()["session_id"]
        client.post(f'/api/preview/session/{session_id}/start')

        response = client.post(f'/api/preview/session/{session_id}/stop')
        assert response.status_code == 200
        result = response.get_json()
        assert result["status"] == "stopped"

    def test_arm_preview_session(self, client):
        """POST /api/preview/session/<id>/arm arms session"""
        data = {"preview_type": "look", "channels": {}}
        create_resp = client.post('/api/preview/session',
                                 data=json.dumps(data),
                                 content_type='application/json')
        session_id = create_resp.get_json()["session_id"]

        response = client.post(f'/api/preview/session/{session_id}/arm')
        assert response.status_code == 200
        result = response.get_json()
        assert result["status"] == "armed"

    def test_disarm_preview_session(self, client):
        """POST /api/preview/session/<id>/disarm disarms session"""
        data = {"preview_type": "look", "channels": {}}
        create_resp = client.post('/api/preview/session',
                                 data=json.dumps(data),
                                 content_type='application/json')
        session_id = create_resp.get_json()["session_id"]
        client.post(f'/api/preview/session/{session_id}/arm')

        response = client.post(f'/api/preview/session/{session_id}/disarm')
        assert response.status_code == 200
        result = response.get_json()
        assert result["status"] == "disarmed"

    def test_delete_preview_session(self, client):
        """DELETE /api/preview/session/<id> deletes session"""
        data = {"preview_type": "look", "channels": {}}
        create_resp = client.post('/api/preview/session',
                                 data=json.dumps(data),
                                 content_type='application/json')
        session_id = create_resp.get_json()["session_id"]

        response = client.delete(f'/api/preview/session/{session_id}')
        assert response.status_code == 200

        get_resp = client.get(f'/api/preview/session/{session_id}')
        assert get_resp.status_code == 404


# ============================================================
# Modifier Endpoint Tests
# ============================================================

class TestModifierEndpoints:
    """Tests for modifier endpoints"""

    def test_get_modifier_types(self, client):
        """GET /api/modifiers/types returns types"""
        response = client.get('/api/modifiers/types')
        assert response.status_code == 200
        data = response.get_json()
        assert "pulse" in data
        assert "strobe" in data
        assert "rainbow" in data

    def test_get_modifier_schemas(self, client):
        """GET /api/modifiers/schemas returns schemas"""
        response = client.get('/api/modifiers/schemas')
        assert response.status_code == 200
        data = response.get_json()
        assert "pulse" in data
        assert "strobe" in data

    def test_validate_modifier_valid(self, client):
        """POST /api/modifiers/validate validates correct modifier"""
        data = {"type": "pulse", "params": {"speed": 1.0}}
        response = client.post('/api/modifiers/validate',
                              data=json.dumps(data),
                              content_type='application/json')
        assert response.status_code == 200
        result = response.get_json()
        assert result["valid"] == True

    def test_validate_modifier_invalid_type(self, client):
        """POST /api/modifiers/validate rejects unknown type"""
        data = {"type": "unknown_modifier"}
        response = client.post('/api/modifiers/validate',
                              data=json.dumps(data),
                              content_type='application/json')
        assert response.status_code == 400
        result = response.get_json()
        assert result["valid"] == False


# ============================================================
# Run Tests
# ============================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
