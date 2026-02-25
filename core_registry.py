"""
AETHER Core Registry - Shared Instance Registry

All extracted modules import from here to access cross-dependencies.
aether-core.py populates these during startup.

This pattern avoids circular imports while allowing extracted modules
to reference each other. All attributes are None until aether-core.py
initializes them, but by the time any method is called during normal
operation, everything is wired up.
"""

# ── Core managers ──
dmx_state = None          # DMXStateManager instance
node_manager = None       # NodeManager instance
content_manager = None    # ContentManager instance
playback_manager = None   # PlaybackManager instance
arbitration = None        # ArbitrationManager instance

# ── Playback engines ──
chase_engine = None       # ChaseEngine instance
show_engine = None        # ShowEngine instance
effects_engine = None     # DynamicEffectsEngine instance (from effects_engine.py)

# ── Infrastructure ──
socketio = None           # Flask-SocketIO instance
merge_layer = None        # MergeLayer instance
channel_classifier = None # ChannelClassifier instance
render_engine = None      # RenderEngine instance

# ── Database ──
get_db = None             # Function to get thread-local DB connection

# ── Utilities ──
audit_log = None          # Function for persistent audit logging
beta_log = None           # Function for beta analytics logging

# ── Schedulers ──
schedule_runner = None    # ScheduleRunner instance
timer_runner = None       # TimerRunner instance

# ── Constants (set during startup) ──
AETHER_UDPJSON_PORT = 6455
AETHER_CONFIG_PORT = 8888
AETHER_DISCOVERY_PORT = 9999
DMX_STATE_FILE = None     # Path to dmx_state.json
DATA_DIR = None           # Path to data directory
DB_PATH = None            # Path to SQLite database
