"""
AETHER Core — Session & STT Blueprint
Routes: /api/session/*, /api/stt
Dependencies: dmx_state, content_manager, get_whisper_model
"""

import os
import logging
import tempfile
from flask import Blueprint, jsonify, request

session_bp = Blueprint('session', __name__)

_dmx_state = None
_content_manager = None
_get_whisper_model = None


def init_app(dmx_state, content_manager, get_whisper_model_fn):
    """Initialize blueprint with required dependencies."""
    global _dmx_state, _content_manager, _get_whisper_model
    _dmx_state = dmx_state
    _content_manager = content_manager
    _get_whisper_model = get_whisper_model_fn


@session_bp.route('/api/session/resume', methods=['GET'])
def get_resume_session():
    """[F09] Check if there's a previous session to resume"""
    last_session = getattr(_dmx_state, 'last_session', None)
    last_sessions = getattr(_dmx_state, 'last_sessions', [])
    saved_at = getattr(_dmx_state, '_last_saved_at', None)
    if last_session:
        return jsonify({
            'has_session': True,
            'playback': last_session,
            'all_sessions': last_sessions,
            'saved_at': saved_at
        })
    return jsonify({'has_session': False, 'playback': None})

@session_bp.route('/api/session/resume', methods=['POST'])
def resume_session():
    """Resume the previous session's playback"""
    last_session = getattr(_dmx_state, 'last_session', None)
    if not last_session:
        return jsonify({'success': False, 'error': 'No session to resume'})

    playback_type = last_session.get('type')
    playback_id = last_session.get('id')
    universe = last_session.get('universe', 1)

    try:
        if playback_type == 'scene':
            result = _content_manager.play_scene(playback_id, universe=universe)
        elif playback_type == 'chase':
            result = _content_manager.play_chase(playback_id, universe=universe)
        else:
            return jsonify({'success': False, 'error': f'Unknown type: {playback_type}'})

        # Clear the last session after resuming
        _dmx_state.last_session = None
        return jsonify({'success': True, 'resumed': last_session})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@session_bp.route('/api/session/dismiss', methods=['POST'])
def dismiss_session():
    """Dismiss the resume prompt without resuming"""
    _dmx_state.last_session = None
    return jsonify({'success': True})

@session_bp.route('/api/stt', methods=['POST'])
def speech_to_text():
    """
    Transcribe audio using local Whisper model.
    Expects multipart form with 'audio' file (webm/wav/mp3).
    Returns: { "text": "transcribed text" } or { "error": "message" }
    """
    model = _get_whisper_model()
    if model is None:
        return jsonify({'error': 'STT not available - Whisper not installed'}), 503

    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    audio_file = request.files['audio']
    if not audio_file.filename:
        return jsonify({'error': 'Empty audio file'}), 400

    # Save to temp file for Whisper
    tmp_path = None
    try:
        # Get file extension from filename or default to webm
        ext = os.path.splitext(audio_file.filename)[1] or '.webm'
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            audio_file.save(tmp.name)
            tmp_path = tmp.name

        # Transcribe with Whisper
        segments, info = model.transcribe(tmp_path, beam_size=1, language='en')
        text = ' '.join(seg.text.strip() for seg in segments)

        # Clean up temp file
        os.unlink(tmp_path)

        logging.info(f"🎤 STT: '{text}' (duration: {info.duration:.1f}s)")
        return jsonify({'text': text})

    except Exception as e:
        logging.error(f"❌ STT error: {e}")
        # Clean up temp file on error
        try:
            if tmp_path:
                os.unlink(tmp_path)
        except:
            pass
        return jsonify({'error': f'Transcription failed: {str(e)}'}), 500
