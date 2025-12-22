#!/usr/bin/env python3
"""
AETHER AI SSOT - Local Single Source of Truth for Operator AI
"""
import sqlite3
import json
import hashlib
import os
from datetime import datetime, timedelta

AI_DB_PATH = os.path.expanduser("~/aether-ai.db")

def get_ai_db():
    conn = sqlite3.connect(AI_DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_ai_database():
    """Initialize AI SSOT tables."""
    conn = get_ai_db()
    c = conn.cursor()
    
    c.execute('''CREATE TABLE IF NOT EXISTS session_memory (
        key TEXT PRIMARY KEY, value TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        expires_at TIMESTAMP
    )''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS user_preferences (
        pref_key TEXT PRIMARY KEY, pref_value TEXT,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS outcome_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        correlation_id TEXT, intent_type TEXT,
        suggested_scope TEXT, user_scope TEXT,
        action TEXT, execution_success BOOLEAN,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )''')
    
    conn.commit()
    conn.close()

def init_ai_database_part2():
    conn = get_ai_db()
    c = conn.cursor()
    
    c.execute('''CREATE TABLE IF NOT EXISTS llm_cache (
        cache_key TEXT PRIMARY KEY, response TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        expires_at TIMESTAMP
    )''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS audit_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        correlation_id TEXT, event_type TEXT, data TEXT,
        llm_called BOOLEAN DEFAULT 0, llm_reason TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS llm_budget (
        period TEXT PRIMARY KEY, calls_made INTEGER DEFAULT 0,
        max_calls INTEGER DEFAULT 50,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )''')
    
    conn.commit()
    conn.close()
    print("✅ AI SSOT database initialized")

# Merge init functions
def init_ai_db():
    init_ai_database()
    init_ai_database_part2()

# === Session Memory ===
def set_session(key, value, ttl_minutes=60):
    conn = get_ai_db()
    expires = datetime.now() + timedelta(minutes=ttl_minutes)
    conn.execute('INSERT OR REPLACE INTO session_memory VALUES (?,?,?,?)',
        (key, json.dumps(value), datetime.now(), expires))
    conn.commit()
    conn.close()

def get_session(key):
    conn = get_ai_db()
    row = conn.execute('SELECT * FROM session_memory WHERE key=? AND expires_at>?',
        (key, datetime.now())).fetchone()
    conn.close()
    return json.loads(row['value']) if row else None

# === User Preferences ===
def set_preference(key, value):
    conn = get_ai_db()
    conn.execute('INSERT OR REPLACE INTO user_preferences VALUES (?,?,?)',
        (key, json.dumps(value), datetime.now()))
    conn.commit()
    conn.close()

def get_preference(key, default=None):
    conn = get_ai_db()
    row = conn.execute('SELECT pref_value FROM user_preferences WHERE pref_key=?',
        (key,)).fetchone()
    conn.close()
    return json.loads(row['pref_value']) if row else default

def get_all_preferences():
    conn = get_ai_db()
    rows = conn.execute('SELECT * FROM user_preferences').fetchall()
    conn.close()
    return {r['pref_key']: json.loads(r['pref_value']) for r in rows}

# === Outcome History (Learning) ===
def record_outcome(corr_id, intent_type, suggested, user_scope, action, success):
    conn = get_ai_db()
    conn.execute('''INSERT INTO outcome_history 
        (correlation_id,intent_type,suggested_scope,user_scope,action,execution_success)
        VALUES (?,?,?,?,?,?)''',
        (corr_id, intent_type, json.dumps(suggested), json.dumps(user_scope), action, success))
    conn.commit()
    conn.close()

def get_outcomes(intent_type=None, limit=50):
    conn = get_ai_db()
    if intent_type:
        rows = conn.execute(
            'SELECT * FROM outcome_history WHERE intent_type=? ORDER BY id DESC LIMIT ?',
            (intent_type, limit)).fetchall()
    else:
        rows = conn.execute(
            'SELECT * FROM outcome_history ORDER BY id DESC LIMIT ?', (limit,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]

# === LLM Cache ===
def cache_key(intent, state_hash, params):
    data = json.dumps({'intent': intent, 'state': state_hash, 'params': params}, sort_keys=True)
    return hashlib.sha256(data.encode()).hexdigest()[:32]

def set_cache(key, response, ttl_hours=24):
    conn = get_ai_db()
    expires = datetime.now() + timedelta(hours=ttl_hours)
    conn.execute('INSERT OR REPLACE INTO llm_cache VALUES (?,?,?,?)',
        (key, json.dumps(response), datetime.now(), expires))
    conn.commit()
    conn.close()

def get_cache(key):
    conn = get_ai_db()
    row = conn.execute('SELECT * FROM llm_cache WHERE cache_key=? AND expires_at>?',
        (key, datetime.now())).fetchone()
    conn.close()
    return json.loads(row['response']) if row else None

# === LLM Budget ===
def check_budget(hourly_limit=20, daily_limit=100):
    conn = get_ai_db()
    now = datetime.now()
    hour_key = now.strftime('%Y-%m-%d-%H')
    day_key = now.strftime('%Y-%m-%d')
    
    hr = conn.execute('SELECT calls_made FROM llm_budget WHERE period=?', (hour_key,)).fetchone()
    dy = conn.execute('SELECT calls_made FROM llm_budget WHERE period=?', (day_key,)).fetchone()
    conn.close()
    
    h = hr['calls_made'] if hr else 0
    d = dy['calls_made'] if dy else 0
    return {'allowed': h < hourly_limit and d < daily_limit,
            'hourly': {'used': h, 'limit': hourly_limit},
            'daily': {'used': d, 'limit': daily_limit}}

def increment_budget():
    conn = get_ai_db()
    now = datetime.now()
    for p in [now.strftime('%Y-%m-%d-%H'), now.strftime('%Y-%m-%d')]:
        conn.execute('''INSERT INTO llm_budget (period, calls_made) VALUES (?, 1)
            ON CONFLICT(period) DO UPDATE SET calls_made = calls_made + 1''', (p,))
    conn.commit()
    conn.close()

# === Audit Log ===
def audit(corr_id, event_type, data, llm_called=False, llm_reason=None):
    conn = get_ai_db()
    conn.execute('''INSERT INTO audit_log 
        (correlation_id, event_type, data, llm_called, llm_reason)
        VALUES (?,?,?,?,?)''',
        (corr_id, event_type, json.dumps(data), llm_called, llm_reason))
    conn.commit()
    conn.close()

def get_audit_log(limit=100):
    conn = get_ai_db()
    rows = conn.execute('SELECT * FROM audit_log ORDER BY id DESC LIMIT ?', (limit,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]
