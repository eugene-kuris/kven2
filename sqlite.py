import sqlite3
import threading
import os
import logging
import json
import asyncio

logger = logging.getLogger(__name__)

import sys
sys.path.append('/opt/kven2')
from config import settings

DB_PATH = settings.DB_PATH
db_lock = threading.Lock()

def get_connection():
    os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA wal_autocheckpoint=500")
    return conn

# --- SYNC CORE HELPERS ---

def _sync_init_db():
    logger.info("Initializing DB schema...")
    conn = get_connection()
    with db_lock:
        try:
            # Existing Schema
            conn.execute("""CREATE TABLE IF NOT EXISTS me_store (
                id INTEGER PRIMARY KEY AUTOINCREMENT, content TEXT NOT NULL, 
                type TEXT DEFAULT 'static', priority INTEGER DEFAULT 50, 
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )""")
            conn.execute("""CREATE TABLE IF NOT EXISTS ch_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT, query TEXT, response TEXT,
                ts DATETIME DEFAULT CURRENT_TIMESTAMP
            )""")
            conn.execute("""CREATE TABLE IF NOT EXISTS semantic_memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL, kind TEXT DEFAULT 'semantic',
                importance REAL DEFAULT 0.5, tags TEXT DEFAULT '[]',
                epistemic_type TEXT DEFAULT 'Observation',
                source TEXT DEFAULT 'model_inference',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_used TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                decay_rate REAL DEFAULT 0.96, deleted INTEGER DEFAULT 0,
                usage_count INTEGER DEFAULT 0
            )""")
            conn.execute("""CREATE TABLE IF NOT EXISTS episodic_memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL, kind TEXT DEFAULT 'episodic',
                importance REAL DEFAULT 0.3, tags TEXT DEFAULT '[]',
                epistemic_type TEXT DEFAULT 'Observation',
                source TEXT DEFAULT 'model_inference',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_used TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                decay_rate REAL DEFAULT 0.85, deleted INTEGER DEFAULT 0,
                usage_count INTEGER DEFAULT 0
            )""")
            conn.execute("""CREATE TABLE IF NOT EXISTS projects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                goal TEXT NOT NULL,
                status TEXT DEFAULT 'active'
            )""")
            conn.execute("""CREATE TABLE IF NOT EXISTS active_state (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                state_json TEXT NOT NULL DEFAULT '{}',
                confidence REAL DEFAULT 0.9,
                salience REAL DEFAULT 1.0,
                evidence_count INTEGER DEFAULT 0,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )""")
            conn.execute("INSERT OR IGNORE INTO active_state (id, state_json) VALUES (1, '{}')")
            conn.execute("""CREATE TABLE IF NOT EXISTS state_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                snapshot_json TEXT NOT NULL
            )""")
            try:
                conn.execute("INSERT OR IGNORE INTO projects (id, name, goal) VALUES (1, 'Kven2 Engineering', 'Implementing a self-learning memory system for the LLM Agent.')")
            except:
                pass
            conn.commit()
        except Exception as e:
            logger.error(f"Schema migration failed: {e}", exc_info=True)
        finally:
            conn.close()

def _sync_save_active_state(state):
    conn = get_connection()
    with db_lock:
        try:
            json_str = json.dumps(state, ensure_ascii=False)
            conn.execute(
                "UPDATE active_state SET state_json = ?, updated_at = CURRENT_TIMESTAMP, confidence = ?, salience = ? WHERE id = 1",
                (json_str, state.get('confidence', 0.9), state.get('salience', 1.0))
            )
            conn.commit()
            return True
        except Exception as e:
            logger.error(f"[ACTIVE_STATE] Save failed: {e}")
            return False
        finally:
            conn.close()

def _sync_load_active_state():
    conn = get_connection()
    with db_lock:
        try:
            row = conn.execute("SELECT state_json, confidence, salience FROM active_state WHERE id = 1").fetchone()
            if row and row[0]:
                state = json.loads(row[0])
                state['confidence'] = row[1] if row[1] is not None else 0.9
                state['salience'] = row[2] if row[2] is not None else 1.0
                return state
            return {}
        except Exception as e:
            logger.warning(f"[ACTIVE_STATE] Load failed: {e}")
            return {}
        finally:
            conn.close()

def _sync_save_history_snapshot(state):
    conn = get_connection()
    with db_lock:
        try:
            conn.execute("INSERT INTO state_history (snapshot_json) VALUES (?)", 
                         (json.dumps(state, ensure_ascii=False),))
            conn.commit()
        except Exception as e:
            logger.error(f"History save failed: {e}")
        finally:
            conn.close()

def _sync_insert_memory(content, kind, importance, tags, decay_rate, table_name="semantic_memory", 
                  epistemic_type="Observation", source="model_inference"):
    conn = get_connection()
    with db_lock:
        try:
            conn.execute(f"""INSERT INTO {table_name} 
                            (content, kind, importance, tags, decay_rate, deleted, usage_count, epistemic_type, source)
                            VALUES (?, ?, ?, ?, ?, 0, 0, ?, ?)""", 
                         (content, kind, importance, tags, decay_rate, epistemic_type, source))
            new_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
            conn.commit()
            return new_id
        except Exception as e:
            logger.error(f"[SQLITE_INSERT] Error: {e}")
            return None
        finally:
            conn.close()

def _sync_get_semantic_context(limit):
    conn = get_connection()
    with db_lock:
        try:
            rows = conn.execute("SELECT content FROM semantic_memory ORDER BY importance DESC LIMIT ?", (limit,)).fetchall()
            return "\n".join([r[0] for r in rows])
        except Exception as e:
            logger.error(f"[SQLITE] Semantic Context load failed: {e}")
            return ""
        finally:
            conn.close()

def _sync_get_recent_episodic(limit):
    conn = get_connection()
    with db_lock:
        try:
            rows = conn.execute("SELECT content FROM episodic_memory ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
            return [r[0] for r in rows]
        except Exception as e:
            logger.error(f"[SQLITE] Episodic load failed: {e}")
            return []
        finally:
            conn.close()

def _sync_get_project_context(project_id):
    conn = get_connection()
    with db_lock:
        try:
            row = conn.execute("SELECT goal FROM projects WHERE id = ?", (project_id,)).fetchone()
            if row:
                return f"CURRENT PROJECT: {row[0]}"
            return "NO ACTIVE PROJECT DEFINED."
        except Exception as e:
            logger.error(f"[PROJECT] Load failed: {e}")
            return ""
        finally:
            conn.close()

def _sync_query(sql, params=()):
    """Generic query function to support HNSW re-ranking."""
    conn = get_connection()
    with db_lock:
        try:
            cursor = conn.execute(sql, params)
            return cursor.fetchall()
        except Exception as e:
            logger.error(f"[SQLITE] Query failed: {e}")
            return []
        finally:
            conn.close()

# --- ASYNC PUBLIC API ---

async def init_db():
    await asyncio.to_thread(_sync_init_db)

async def save_active_state(state: dict) -> bool:
    return await asyncio.to_thread(_sync_save_active_state, state)

async def load_active_state() -> dict:
    return await asyncio.to_thread(_sync_load_active_state)

async def save_history_snapshot(state: dict):
    await asyncio.to_thread(_sync_save_history_snapshot, state)

async def insert_memory(content: str, kind: str, importance: float, tags: str, decay_rate: float, table_name="semantic_memory", 
                  epistemic_type="Observation", source="model_inference") -> int:
    return await asyncio.to_thread(_sync_insert_memory, content, kind, importance, tags, decay_rate, table_name, epistemic_type, source)

async def get_semantic_context(limit=5):
    return await asyncio.to_thread(_sync_get_semantic_context, limit)

async def get_recent_episodic(limit=10):
    return await asyncio.to_thread(_sync_get_recent_episodic, limit)

async def get_project_context(project_id: int):
    return await asyncio.to_thread(_sync_get_project_context, project_id)

async def query(sql, params=()):
    return await asyncio.to_thread(_sync_query, sql, params)
