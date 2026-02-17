"""
AETHER Core — Migration Utilities Blueprint
Routes: /api/migrate/*
Dependencies: looks_sequences module, DATABASE constant
"""

from flask import Blueprint, jsonify
from looks_sequences import run_full_migration

migrate_bp = Blueprint('migrate', __name__)

# DATABASE path injected at registration time via init_app()
_database_path = None
_looks_sequences_manager = None


def init_app(database_path, looks_sequences_manager):
    """Initialize blueprint with required dependencies."""
    global _database_path, _looks_sequences_manager
    _database_path = database_path
    _looks_sequences_manager = looks_sequences_manager


@migrate_bp.route('/api/migrate/scenes-to-looks', methods=['POST'])
def migrate_scenes_to_looks_route():
    """Migrate legacy scenes to new looks format"""
    from looks_sequences import migrate_scenes_to_looks
    report = migrate_scenes_to_looks(_database_path, _looks_sequences_manager)
    return jsonify({'success': True, 'report': report})

@migrate_bp.route('/api/migrate/chases-to-sequences', methods=['POST'])
def migrate_chases_to_sequences_route():
    """Migrate legacy chases to new sequences format"""
    from looks_sequences import migrate_chases_to_sequences
    report = migrate_chases_to_sequences(_database_path, _looks_sequences_manager)
    return jsonify({'success': True, 'report': report})

@migrate_bp.route('/api/migrate/all', methods=['POST'])
def migrate_all_route():
    """Run full migration from legacy to new format"""
    report = run_full_migration(_database_path)
    return jsonify({'success': True, 'report': report})
