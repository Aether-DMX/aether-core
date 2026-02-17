"""
AETHER Core — Scenes Blueprint (Legacy)
Routes: /api/scenes/*
Dependencies: content_manager
"""

from flask import Blueprint, jsonify, request

scenes_bp = Blueprint('scenes', __name__)

_content_manager = None


def init_app(content_manager):
    """Initialize blueprint with required dependencies."""
    global _content_manager
    _content_manager = content_manager


@scenes_bp.route('/api/scenes', methods=['GET'])
def get_scenes():
    return jsonify(_content_manager.get_scenes())

@scenes_bp.route('/api/scenes', methods=['POST'])
def create_scene():
    return jsonify(_content_manager.create_scene(request.get_json()))

@scenes_bp.route('/api/scenes/<scene_id>', methods=['GET'])
def get_scene(scene_id):
    scene = _content_manager.get_scene(scene_id)
    return jsonify(scene) if scene else (jsonify({'error': 'Scene not found'}), 404)

@scenes_bp.route('/api/scenes/<scene_id>', methods=['PUT'])
def update_scene(scene_id):
    """Update an existing scene"""
    data = request.get_json() or {}
    data['scene_id'] = scene_id  # Ensure scene_id is set for the update
    return jsonify(_content_manager.create_scene(data))

@scenes_bp.route('/api/scenes/<scene_id>', methods=['DELETE'])
def delete_scene(scene_id):
    return jsonify(_content_manager.delete_scene(scene_id))

@scenes_bp.route('/api/scenes/<scene_id>/play', methods=['POST'])
def play_scene(scene_id):
    data = request.get_json() or {}
    print(f"Scene play request: scene_id={scene_id}, payload={data}", flush=True)
    return jsonify(_content_manager.play_scene(
        scene_id,
        fade_ms=data.get('fade_ms'),
        use_local=data.get('use_local', True),
        target_channels=data.get('target_channels'),
        universe=data.get('universe'),
        universes=data.get('universes')
    ))
