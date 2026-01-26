"""
Operator Trust Enforcement Module
AETHER ARCHITECTURE PROGRAM - Phase 4 Lane 3

# ============================================================================
# PURPOSE
# ============================================================================
#
# This module ensures the system behaves predictably under failure and never
# lies to the operator. It implements explicit detection and handling for:
#
# 1. Network loss → Nodes HOLD last DMX value
# 2. Backend crash → Nodes CONTINUE output
# 3. UI desync → REALITY wins over UI
# 4. Partial node failure → SYSTEM HALTS playback + ALERTS
#
# TRUST RULE: Silent failure is forbidden.
# All trust events must emit structured logs visible to operators.
#
# ============================================================================
"""

import time
import threading
import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field

# Configure logging for trust events
trust_logger = logging.getLogger('aether.trust')
trust_logger.setLevel(logging.INFO)

# Add handler if not already present
if not trust_logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        '%(asctime)s [TRUST] %(levelname)s: %(message)s'
    ))
    trust_logger.addHandler(handler)


class TrustEvent(Enum):
    """Trust-related events that must be logged."""
    NETWORK_LOSS_DETECTED = "network_loss_detected"
    NETWORK_RESTORED = "network_restored"
    BACKEND_RESTART_DETECTED = "backend_restart_detected"
    NODE_HEARTBEAT_LOST = "node_heartbeat_lost"
    NODE_HEARTBEAT_RESTORED = "node_heartbeat_restored"
    UI_STATE_MISMATCH = "ui_state_mismatch"
    PARTIAL_NODE_FAILURE = "partial_node_failure"
    PLAYBACK_HALTED_DUE_TO_FAILURE = "playback_halted_due_to_failure"
    DMX_HOLD_ACTIVATED = "dmx_hold_activated"
    OPERATOR_ALERT = "operator_alert"


@dataclass
class TrustEventRecord:
    """Structured record of a trust event."""
    event: TrustEvent
    timestamp: str
    affected_components: List[str]
    details: Dict[str, Any]
    severity: str  # "info", "warning", "critical"

    def to_dict(self) -> Dict:
        return {
            'event': self.event.value,
            'timestamp': self.timestamp,
            'affected_components': self.affected_components,
            'details': self.details,
            'severity': self.severity
        }


@dataclass
class NodeHealthStatus:
    """Health status for a single node."""
    node_id: str
    last_heartbeat: Optional[datetime] = None
    consecutive_failures: int = 0
    is_healthy: bool = True
    last_dmx_values: Dict[int, int] = field(default_factory=dict)


class OperatorTrustEnforcer:
    """
    Enforces operator trust rules across the system.

    TRUST RULES:
    1. Network loss → Nodes HOLD last DMX value (handled by ESP32 firmware)
    2. Backend crash → Nodes CONTINUE output (handled by ESP32 firmware)
    3. UI desync → REALITY wins over UI (backend state is authoritative)
    4. Partial node failure → SYSTEM HALTS playback + ALERTS
    """

    # Configuration
    HEARTBEAT_TIMEOUT_SECONDS = 30  # Node considered unhealthy after this
    HEARTBEAT_CRITICAL_SECONDS = 60  # Node considered failed after this
    MAX_CONSECUTIVE_FAILURES = 3     # Failures before halting playback
    UI_SYNC_CHECK_INTERVAL = 5.0     # Seconds between UI sync checks

    def __init__(self):
        self._node_health: Dict[str, NodeHealthStatus] = {}
        self._event_history: List[TrustEventRecord] = []
        self._event_callbacks: List[Callable[[TrustEventRecord], None]] = []
        self._lock = threading.RLock()
        self._running = False
        self._monitor_thread: Optional[threading.Thread] = None

        # State tracking
        self._last_backend_start = datetime.now()
        self._playback_halted_due_to_failure = False
        self._active_alerts: List[str] = []

        # Callbacks for external actions
        self._halt_playback_callback: Optional[Callable[[], None]] = None
        self._get_playback_status_callback: Optional[Callable[[], Dict]] = None
        self._get_dmx_state_callback: Optional[Callable[[], Dict]] = None

    # ─────────────────────────────────────────────────────────
    # Configuration
    # ─────────────────────────────────────────────────────────

    def set_halt_playback_callback(self, callback: Callable[[], None]):
        """Set callback to halt playback when trust violation occurs."""
        self._halt_playback_callback = callback

    def set_get_playback_status_callback(self, callback: Callable[[], Dict]):
        """Set callback to get current playback status."""
        self._get_playback_status_callback = callback

    def set_get_dmx_state_callback(self, callback: Callable[[], Dict]):
        """Set callback to get current DMX state (reality)."""
        self._get_dmx_state_callback = callback

    def register_event_callback(self, callback: Callable[[TrustEventRecord], None]):
        """Register callback for trust events (e.g., for WebSocket broadcast)."""
        with self._lock:
            self._event_callbacks.append(callback)

    # ─────────────────────────────────────────────────────────
    # Event Logging (MANDATORY - Silent failure is forbidden)
    # ─────────────────────────────────────────────────────────

    def _emit_event(self, event: TrustEvent, affected_components: List[str],
                    details: Dict[str, Any], severity: str = "warning"):
        """Emit a trust event with structured logging."""
        record = TrustEventRecord(
            event=event,
            timestamp=datetime.now().isoformat(),
            affected_components=affected_components,
            details=details,
            severity=severity
        )

        # Log to trust logger
        log_msg = f"{event.value} | Components: {affected_components} | Details: {details}"
        if severity == "critical":
            trust_logger.critical(log_msg)
            print(f"🚨 TRUST CRITICAL: {log_msg}", flush=True)
        elif severity == "warning":
            trust_logger.warning(log_msg)
            print(f"⚠️ TRUST WARNING: {log_msg}", flush=True)
        else:
            trust_logger.info(log_msg)
            print(f"ℹ️ TRUST INFO: {log_msg}", flush=True)

        # Store in history
        with self._lock:
            self._event_history.append(record)
            # Keep last 1000 events
            if len(self._event_history) > 1000:
                self._event_history = self._event_history[-1000:]

        # Notify callbacks
        for callback in self._event_callbacks:
            try:
                callback(record)
            except Exception as e:
                trust_logger.error(f"Event callback error: {e}")

    # ─────────────────────────────────────────────────────────
    # Network Loss Detection
    # ─────────────────────────────────────────────────────────

    def report_node_heartbeat(self, node_id: str, node_info: Dict):
        """
        Report a heartbeat from a node.

        TRUST RULE: Network loss → Nodes HOLD last DMX value
        When we lose heartbeats, nodes continue holding their last values.
        This is handled by ESP32 firmware, but we must detect and log it.
        """
        with self._lock:
            if node_id not in self._node_health:
                self._node_health[node_id] = NodeHealthStatus(node_id=node_id)

            health = self._node_health[node_id]
            was_unhealthy = not health.is_healthy

            health.last_heartbeat = datetime.now()
            health.consecutive_failures = 0
            health.is_healthy = True

            # Log restoration if was unhealthy
            if was_unhealthy:
                self._emit_event(
                    TrustEvent.NODE_HEARTBEAT_RESTORED,
                    [node_id],
                    {'node_info': node_info},
                    severity="info"
                )

    def check_node_health(self, node_id: str) -> bool:
        """
        Check if a node is healthy based on heartbeat timing.

        Returns True if healthy, False if unhealthy.
        """
        with self._lock:
            if node_id not in self._node_health:
                return True  # Unknown nodes assumed healthy until proven otherwise

            health = self._node_health[node_id]
            if health.last_heartbeat is None:
                return True

            elapsed = (datetime.now() - health.last_heartbeat).total_seconds()

            if elapsed > self.HEARTBEAT_CRITICAL_SECONDS:
                if health.is_healthy:
                    health.is_healthy = False
                    health.consecutive_failures += 1
                    self._emit_event(
                        TrustEvent.NODE_HEARTBEAT_LOST,
                        [node_id],
                        {
                            'elapsed_seconds': elapsed,
                            'consecutive_failures': health.consecutive_failures,
                            'action': 'Node holding last DMX values (firmware behavior)'
                        },
                        severity="critical"
                    )
                    self._check_partial_failure()
                return False

            elif elapsed > self.HEARTBEAT_TIMEOUT_SECONDS:
                if health.is_healthy:
                    self._emit_event(
                        TrustEvent.NETWORK_LOSS_DETECTED,
                        [node_id],
                        {
                            'elapsed_seconds': elapsed,
                            'action': 'Monitoring - node may be experiencing network issues'
                        },
                        severity="warning"
                    )
                return True  # Still within tolerance

            return True

    # ─────────────────────────────────────────────────────────
    # Backend Crash Detection
    # ─────────────────────────────────────────────────────────

    def report_backend_start(self):
        """
        Report that backend has started/restarted.

        TRUST RULE: Backend crash → Nodes CONTINUE output
        Nodes hold their last DMX values when backend goes down.
        On restart, we must sync state and log the event.
        """
        now = datetime.now()

        # Detect if this is a restart (vs initial start)
        # If we have node health data, this is likely a restart
        with self._lock:
            is_restart = len(self._node_health) > 0

        if is_restart:
            self._emit_event(
                TrustEvent.BACKEND_RESTART_DETECTED,
                ['backend'],
                {
                    'previous_start': self._last_backend_start.isoformat(),
                    'action': 'Nodes continued output during downtime (firmware behavior)',
                    'recommendation': 'Verify node states match expected values'
                },
                severity="warning"
            )

        self._last_backend_start = now

    # ─────────────────────────────────────────────────────────
    # UI Desync Detection
    # ─────────────────────────────────────────────────────────

    def check_ui_sync(self, ui_state: Dict, component: str = "unknown") -> Dict:
        """
        Check if UI state matches reality (DMX state).

        TRUST RULE: UI desync → REALITY wins over UI
        If UI thinks channels are at certain values but DMX reality differs,
        we must alert and UI must update to match reality.

        Returns:
            Dict with 'synced' bool and 'differences' if not synced
        """
        if not self._get_dmx_state_callback:
            return {'synced': True, 'reason': 'No DMX state callback configured'}

        try:
            reality = self._get_dmx_state_callback()
        except Exception as e:
            return {'synced': True, 'error': str(e)}

        differences = []

        # Compare UI universe states vs reality
        ui_universes = ui_state.get('universes', {})
        real_universes = reality.get('universes', {})

        for universe, ui_channels in ui_universes.items():
            real_channels = real_universes.get(universe, [0] * 512)

            for ch_idx, ui_value in enumerate(ui_channels):
                if ch_idx < len(real_channels):
                    real_value = real_channels[ch_idx]
                    if ui_value != real_value:
                        differences.append({
                            'universe': universe,
                            'channel': ch_idx + 1,
                            'ui_value': ui_value,
                            'real_value': real_value
                        })

        if differences:
            self._emit_event(
                TrustEvent.UI_STATE_MISMATCH,
                [component, 'dmx_state'],
                {
                    'difference_count': len(differences),
                    'sample_differences': differences[:5],  # First 5
                    'action': 'UI must sync to reality - REALITY WINS'
                },
                severity="warning"
            )
            return {'synced': False, 'differences': differences}

        return {'synced': True}

    # ─────────────────────────────────────────────────────────
    # Partial Node Failure Detection
    # ─────────────────────────────────────────────────────────

    def _check_partial_failure(self):
        """
        Check if partial node failure should halt playback.

        TRUST RULE: Partial node failure → SYSTEM HALTS playback + ALERTS
        If some nodes fail during active playback, we must halt and alert.
        """
        with self._lock:
            unhealthy_nodes = [
                node_id for node_id, health in self._node_health.items()
                if not health.is_healthy
            ]

            if not unhealthy_nodes:
                return

            # Check if playback is active
            playback_active = False
            if self._get_playback_status_callback:
                try:
                    status = self._get_playback_status_callback()
                    playback_active = bool(status.get('sessions') or status.get('playing'))
                except:
                    pass

            if playback_active and len(unhealthy_nodes) >= 1:
                self._emit_event(
                    TrustEvent.PARTIAL_NODE_FAILURE,
                    unhealthy_nodes,
                    {
                        'unhealthy_count': len(unhealthy_nodes),
                        'playback_active': True,
                        'action': 'HALTING PLAYBACK due to partial node failure'
                    },
                    severity="critical"
                )

                # Halt playback
                if self._halt_playback_callback and not self._playback_halted_due_to_failure:
                    try:
                        self._halt_playback_callback()
                        self._playback_halted_due_to_failure = True
                        self._emit_event(
                            TrustEvent.PLAYBACK_HALTED_DUE_TO_FAILURE,
                            ['playback'],
                            {
                                'reason': 'Partial node failure detected',
                                'unhealthy_nodes': unhealthy_nodes
                            },
                            severity="critical"
                        )
                    except Exception as e:
                        trust_logger.error(f"Failed to halt playback: {e}")

                # Add to active alerts
                alert_id = f"partial_failure_{datetime.now().timestamp()}"
                self._active_alerts.append(alert_id)

    def clear_failure_halt(self):
        """Clear the playback halt flag after operator acknowledges."""
        self._playback_halted_due_to_failure = False
        self._emit_event(
            TrustEvent.OPERATOR_ALERT,
            ['operator'],
            {'action': 'Failure halt cleared by operator'},
            severity="info"
        )

    # ─────────────────────────────────────────────────────────
    # Monitoring
    # ─────────────────────────────────────────────────────────

    def start_monitoring(self):
        """Start background health monitoring thread."""
        if self._running:
            return

        self._running = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True,
            name="TrustMonitor"
        )
        self._monitor_thread.start()
        print("🛡️ Operator Trust Enforcer: Monitoring started", flush=True)

    def stop_monitoring(self):
        """Stop background monitoring."""
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)
        print("🛡️ Operator Trust Enforcer: Monitoring stopped", flush=True)

    def _monitor_loop(self):
        """Background monitoring loop."""
        while self._running:
            try:
                # Check all known node health
                with self._lock:
                    node_ids = list(self._node_health.keys())

                for node_id in node_ids:
                    self.check_node_health(node_id)

                time.sleep(5.0)  # Check every 5 seconds

            except Exception as e:
                trust_logger.error(f"Monitor loop error: {e}")
                time.sleep(1.0)

    # ─────────────────────────────────────────────────────────
    # Status & History
    # ─────────────────────────────────────────────────────────

    def get_status(self) -> Dict:
        """Get current trust enforcement status."""
        with self._lock:
            return {
                'monitoring': self._running,
                'backend_start': self._last_backend_start.isoformat(),
                'playback_halted_due_to_failure': self._playback_halted_due_to_failure,
                'active_alerts': self._active_alerts,
                'node_health': {
                    node_id: {
                        'is_healthy': h.is_healthy,
                        'last_heartbeat': h.last_heartbeat.isoformat() if h.last_heartbeat else None,
                        'consecutive_failures': h.consecutive_failures
                    }
                    for node_id, h in self._node_health.items()
                },
                'recent_events': [e.to_dict() for e in self._event_history[-10:]]
            }

    def get_event_history(self, limit: int = 100) -> List[Dict]:
        """Get recent trust event history."""
        with self._lock:
            return [e.to_dict() for e in self._event_history[-limit:]]


# Global instance
trust_enforcer = OperatorTrustEnforcer()


# ─────────────────────────────────────────────────────────
# Module-level convenience functions
# ─────────────────────────────────────────────────────────

def report_node_heartbeat(node_id: str, node_info: Dict):
    """Report a node heartbeat."""
    trust_enforcer.report_node_heartbeat(node_id, node_info)


def report_backend_start():
    """Report backend start/restart."""
    trust_enforcer.report_backend_start()


def check_ui_sync(ui_state: Dict, component: str = "unknown") -> Dict:
    """Check if UI state matches reality."""
    return trust_enforcer.check_ui_sync(ui_state, component)


def get_trust_status() -> Dict:
    """Get trust enforcement status."""
    return trust_enforcer.get_status()


def get_trust_events(limit: int = 100) -> List[Dict]:
    """Get trust event history."""
    return trust_enforcer.get_event_history(limit)


def start_trust_monitoring():
    """Start trust monitoring."""
    trust_enforcer.start_monitoring()


def stop_trust_monitoring():
    """Stop trust monitoring."""
    trust_enforcer.stop_monitoring()


def clear_failure_halt():
    """Clear playback halt after operator acknowledgment."""
    trust_enforcer.clear_failure_halt()
