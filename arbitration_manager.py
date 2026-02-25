"""
AETHER Arbitration Manager - Priority-based DMX output control

Extracted from aether-core.py for modularity.
Uses core_registry for cross-module references.
"""

import threading
from datetime import datetime
import core_registry as reg


class ArbitrationManager:
    """
    Priority-based arbitration for DMX output control.
    Priority: BLACKOUT(100) > MANUAL(80) > EFFECT(60) > LOOK(50) > SEQUENCE(45) > CHASE(40) > SCENE(20) > IDLE(0)

    SSOT ENFORCEMENT: All DMX write attempts must check arbitration first.
    Rejected writes are tracked for diagnostics.

    [F08] TOKEN-BASED TOCTOU FIX:
    acquire() returns a monotonic token (int). Each new acquire increments the
    token. can_write() and validate_token() check that the caller's token matches
    the current token — if another engine acquired in between, the token is stale
    and the write is rejected. This prevents the race where Engine A acquires,
    Engine B force-acquires, and Engine A's subsequent write goes through unchecked.
    """
    PRIORITY = {'blackout': 100, 'manual': 80, 'effect': 60, 'look': 50, 'sequence': 45, 'chase': 40, 'scene': 20, 'idle': 0}

    def __init__(self):
        self.current_owner = 'idle'
        self.current_id = None
        self.blackout_active = False
        self.last_change = None
        self.lock = threading.Lock()
        self.history = []
        # [F08] Monotonic token — incremented on every successful acquire
        self._token = 0
        # SSOT diagnostics tracking
        self.rejected_writes = []  # Track rejected acquire attempts
        self.stale_writes = 0  # [F08] Count of writes rejected due to stale token
        self.last_writer = None  # Last service that successfully wrote
        self.last_scene_id = None  # Last scene played
        self.last_scene_time = None  # When last scene was played
        self.writes_per_service = {}  # Count writes per service type

    def acquire(self, owner_type, owner_id=None, force=False):
        """Acquire arbitration. Returns token (int > 0) on success, 0 on failure.

        [F08] The returned token must be passed to can_write() or validate_token()
        before writing DMX. A stale token (from a previous acquire) will be rejected.
        For backward compatibility, the token is also truthy (non-zero = success).
        """
        with self.lock:
            now = datetime.now().isoformat()
            if self.blackout_active and owner_type != 'blackout':
                self._track_rejection(owner_type, owner_id, 'blackout_active', now)
                return 0
            new_pri = self.PRIORITY.get(owner_type, 0)
            cur_pri = self.PRIORITY.get(self.current_owner, 0)
            if force or new_pri >= cur_pri:
                old = self.current_owner
                self.current_owner = owner_type
                self.current_id = owner_id
                self.last_change = now
                self.last_writer = owner_type
                # [F08] Increment token — invalidates all previous tokens
                self._token += 1
                token = self._token
                # Track scene plays specifically
                if owner_type == 'scene':
                    self.last_scene_id = owner_id
                    self.last_scene_time = now
                # Track writes per service
                self.writes_per_service[owner_type] = self.writes_per_service.get(owner_type, 0) + 1
                self.history.append({'time': now, 'from': old, 'to': owner_type, 'id': owner_id, 'action': 'acquire', 'token': token})
                if len(self.history) > 50: self.history = self.history[-50:]
                # [F16] Persistent audit trail
                if reg.audit_log:
                    reg.audit_log('arb_acquire', owner=owner_type, id=owner_id, prev=old, token=token, force=force)
                print(f"🎯 Arbitration: {old} → {owner_type} (token={token})", flush=True)
                return token
            self._track_rejection(owner_type, owner_id, f'priority_too_low (current: {self.current_owner})', now)
            return 0

    def validate_token(self, token):
        """[F08] Check if a token is still valid (i.e., no one acquired since).

        Returns True if the token matches the current token, False if stale.
        """
        with self.lock:
            if token == self._token:
                return True
            self.stale_writes += 1
            return False

    def _track_rejection(self, owner_type, owner_id, reason, timestamp):
        """Track rejected write attempts for diagnostics"""
        self.rejected_writes.append({
            'time': timestamp,
            'requester': owner_type,
            'requester_id': owner_id,
            'reason': reason,
            'current_owner': self.current_owner
        })
        if len(self.rejected_writes) > 20:
            self.rejected_writes = self.rejected_writes[-20:]
        # [F16] Persist rejection
        if reg.audit_log:
            reg.audit_log('arb_reject', requester=owner_type, id=owner_id, reason=reason, current=self.current_owner)
        print(f"⚠️ Arbitration REJECTED: {owner_type} (reason: {reason})", flush=True)

    def release(self, owner_type=None):
        with self.lock:
            if owner_type is None or self.current_owner == owner_type:
                old = self.current_owner
                self.current_owner = 'idle'
                self.current_id = None
                self._token += 1  # [F08] Invalidate tokens on release too
                self.last_change = datetime.now().isoformat()
                self.history.append({'time': self.last_change, 'from': old, 'to': 'idle', 'action': 'release', 'token': self._token})
                if len(self.history) > 50: self.history = self.history[-50:]
                # [F16] Persist release
                if reg.audit_log:
                    reg.audit_log('arb_release', prev=old, token=self._token)

    def set_blackout(self, active):
        with self.lock:
            self.blackout_active = active
            self.current_owner = 'blackout' if active else 'idle'
            self._token += 1  # [F08] Blackout invalidates all tokens
            self.last_change = datetime.now().isoformat()
            # [F16] Persist blackout state changes
            if reg.audit_log:
                reg.audit_log('arb_blackout', active=active, token=self._token)
            print(f"{'⬛ BLACKOUT ACTIVE' if active else '🔓 Blackout released'} (token={self._token})", flush=True)

    def get_status(self):
        with self.lock:
            return {
                'current_owner': self.current_owner,
                'current_id': self.current_id,
                'blackout_active': self.blackout_active,
                'last_change': self.last_change,
                'token': self._token,
                'stale_writes': self.stale_writes,
                'last_writer': self.last_writer,
                'last_scene_id': self.last_scene_id,
                'last_scene_time': self.last_scene_time,
                'writes_per_service': dict(self.writes_per_service),
                'rejected_writes': self.rejected_writes[-5:],  # Last 5 rejections
                'history': self.history[-10:]
            }

    def can_write(self, owner_type, token=None):
        """Check if owner_type can currently write.

        [F08] If token is provided, also validates it hasn't been superseded.
        This prevents TOCTOU races where another engine acquired between
        the caller's acquire() and this can_write() check.
        """
        with self.lock:
            if self.blackout_active and owner_type != 'blackout': return False
            owner_ok = self.current_owner == owner_type or self.current_owner == 'idle'
            if not owner_ok:
                return False
            # [F08] Token validation — reject stale tokens
            if token is not None and token != self._token:
                self.stale_writes += 1
                return False
            return True
