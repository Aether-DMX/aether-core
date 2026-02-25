"""
AETHER Schedulers - ScheduleRunner and TimerRunner

Extracted from aether-core.py for modularity.
Uses core_registry for cross-module references.
"""

import time
import logging
import threading
from datetime import datetime
from croniter import croniter
import core_registry as reg


class ScheduleRunner:
    """Background scheduler that runs cron-based lighting events"""

    def __init__(self):
        self.schedules = []
        self.running = False
        self.thread = None

    def start(self):
        """Start the schedule runner background thread"""
        if self.running:
            return
        self.running = True
        self.update_schedules()
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()
        print("📅 Schedule runner started")

    def stop(self):
        """Stop the schedule runner"""
        self.running = False
        print("📅 Schedule runner stopped")

    def update_schedules(self):
        """Reload schedules from database"""
        conn = reg.get_db()
        c = conn.cursor()
        c.execute('SELECT * FROM schedules WHERE enabled = 1')
        rows = c.fetchall()
        conn.close()
        self.schedules = []
        for row in rows:
            self.schedules.append({
                'schedule_id': row[0], 'name': row[1], 'cron': row[2],
                'action_type': row[3], 'action_id': row[4]
            })
        print(f"📅 Loaded {len(self.schedules)} active schedules")

    def _parse_cron(self, cron_str):
        """Check if cron expression matches the current minute.

        [F14] Uses croniter library instead of fragile hand-rolled parser.
        Supports full cron syntax: ranges (1-5), lists (1,3,5), steps (*/5),
        combined expressions (1-5,10,15), day-of-week names, and more.
        """
        try:
            now = datetime.now()
            # croniter.match() returns True if 'now' matches the cron schedule
            return croniter.match(cron_str, now)
        except (ValueError, KeyError, TypeError) as e:
            print(f"⚠️ Cron parse error for '{cron_str}': {e}")
            return False

    def _run_loop(self):
        """Background loop checking schedules every minute"""
        last_check_minute = -1
        while self.running:
            try:
                now = datetime.now()
                # Only check once per minute
                if now.minute != last_check_minute:
                    last_check_minute = now.minute
                    for sched in self.schedules:
                        if self._parse_cron(sched['cron']):
                            print(f"⏰ Triggering schedule: {sched['name']}")
                            self.run_schedule(sched['schedule_id'])
                time.sleep(5)  # Check every 5 seconds for minute change
            except Exception as e:
                # [N17 fix] Outer guard prevents thread death on unexpected errors
                logging.error(f"❌ Schedule runner loop error: {e}")
                time.sleep(10)  # Back off before retrying

    def run_schedule(self, schedule_id):
        """Execute a schedule's action.

        RACE CONDITION NOTE: Schedules respect the arbitration system.
        If a higher priority source (effect, manual, etc.) is active,
        the schedule's action may be denied by arbitration. This is correct
        behavior - schedules shouldn't override live performances.
        """
        conn = reg.get_db()
        c = conn.cursor()
        c.execute('SELECT * FROM schedules WHERE schedule_id = ?', (schedule_id,))
        row = c.fetchone()
        if not row:
            conn.close()
            return {'success': False, 'error': 'Schedule not found'}

        action_type = row[3]
        action_id = row[4]
        schedule_name = row[1]

        # Check arbitration status before executing (for logging)
        current_owner = reg.arbitration.current_owner
        if current_owner not in ('idle', 'scene', 'chase'):
            print(f"📅 Schedule '{schedule_name}' triggered but {current_owner} has control", flush=True)

        # Update last_run
        c.execute('UPDATE schedules SET last_run = ? WHERE schedule_id = ?',
                  (datetime.now().isoformat(), schedule_id))
        conn.commit()
        conn.close()

        # Execute action (underlying methods handle arbitration)
        try:
            if action_type == 'scene':
                result = reg.content_manager.play_scene(action_id)
            elif action_type == 'chase':
                result = reg.content_manager.play_chase(action_id)
            elif action_type == 'blackout':
                # Blackout is highest priority - always executes
                result = reg.content_manager.blackout()
            else:
                result = {'success': False, 'error': f'Unknown action type: {action_type}'}

            success = result.get('success', False)
            if success:
                print(f"📅 Schedule '{schedule_name}' executed successfully", flush=True)
            else:
                print(f"📅 Schedule '{schedule_name}' denied: {result.get('error', 'unknown')}", flush=True)
            return result
        except Exception as e:
            print(f"❌ Schedule error: {e}")
            return {'success': False, 'error': str(e)}


class TimerRunner:
    """Background runner for countdown timers that can trigger actions"""

    def __init__(self):
        self.active_timers = {}  # timer_id -> threading.Timer
        self.lock = threading.Lock()

    def start_timer(self, timer_id, remaining_ms):
        """Start a countdown timer in the background"""
        with self.lock:
            # Cancel existing timer if any
            if timer_id in self.active_timers:
                self.active_timers[timer_id].cancel()

            # Schedule the completion callback
            delay_seconds = remaining_ms / 1000.0
            timer = threading.Timer(delay_seconds, self._on_timer_complete, args=[timer_id])
            timer.daemon = True
            timer.start()
            self.active_timers[timer_id] = timer
            print(f"⏱️ Timer '{timer_id}' started: {remaining_ms}ms")

    def stop_timer(self, timer_id):
        """Stop/cancel a running timer"""
        with self.lock:
            if timer_id in self.active_timers:
                self.active_timers[timer_id].cancel()
                del self.active_timers[timer_id]
                print(f"⏱️ Timer '{timer_id}' stopped")

    def _on_timer_complete(self, timer_id):
        """Called when a timer completes"""
        print(f"⏱️ Timer '{timer_id}' completed!")

        # Update database status
        conn = reg.get_db()
        c = conn.cursor()
        c.execute('SELECT action_type, action_id, name FROM timers WHERE timer_id = ?', (timer_id,))
        row = c.fetchone()

        if row:
            action_type, action_id, name = row
            c.execute('''UPDATE timers SET status='completed', remaining_ms=0
                WHERE timer_id=?''', (timer_id,))
            conn.commit()

            # Execute action if defined
            if action_type and action_id:
                try:
                    if action_type == 'look':
                        print(f"⏱️ Timer '{name}' triggering look: {action_id}")
                    elif action_type == 'sequence':
                        print(f"⏱️ Timer '{name}' triggering sequence: {action_id}")
                    elif action_type == 'blackout':
                        print(f"⏱️ Timer '{name}' triggering blackout")
                except Exception as e:
                    print(f"❌ Timer action error: {e}")

            # Broadcast completion via WebSocket [N01 fix]
            if reg.socketio:
                reg.socketio.emit('timer_complete', {
                    'timer_id': timer_id,
                    'name': name
                })

        conn.close()

        # Remove from active timers
        with self.lock:
            if timer_id in self.active_timers:
                del self.active_timers[timer_id]
