"""
SQLite storage layer for the multi-agent orchestration pipeline. Extends the
existing hatespeech.db (raw sqlite3, no ORM) rather than introducing a new
storage system - matches fastapp1.py's init_db()/log_analysis() style exactly.
"""
import json
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

DB_NAME = "hatespeech.db"


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn


def init_agent_tables() -> None:
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()

    c.execute(
        """CREATE TABLE IF NOT EXISTS clusters
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  platform TEXT,
                  representative_text TEXT,
                  centroid TEXT,
                  member_count INTEGER DEFAULT 0,
                  campaign_flag INTEGER DEFAULT 0,
                  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                  updated_at DATETIME DEFAULT CURRENT_TIMESTAMP)"""
    )

    c.execute(
        """CREATE TABLE IF NOT EXISTS cluster_members
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  cluster_id INTEGER NOT NULL REFERENCES clusters(id),
                  text TEXT,
                  embedding TEXT,
                  similarity_to_centroid REAL,
                  added_at DATETIME DEFAULT CURRENT_TIMESTAMP)"""
    )

    c.execute(
        """CREATE TABLE IF NOT EXISTS case_files
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  text TEXT,
                  category TEXT,
                  confidence REAL,
                  language TEXT,
                  username TEXT,
                  platform TEXT,
                  district TEXT DEFAULT 'Unknown',
                  cluster_id INTEGER REFERENCES clusters(id),
                  campaign_flag INTEGER DEFAULT 0,
                  legal_provisions TEXT,
                  sarcasm_score REAL,
                  sarcasm_flag INTEGER DEFAULT 0,
                  requires_human_review INTEGER NOT NULL DEFAULT 1 CHECK (requires_human_review = 1),
                  status TEXT DEFAULT 'pending',
                  created_at DATETIME DEFAULT CURRENT_TIMESTAMP)"""
    )

    c.execute(
        """CREATE TABLE IF NOT EXISTS review_queue
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  case_file_id INTEGER NOT NULL REFERENCES case_files(id),
                  priority TEXT DEFAULT 'medium',
                  status TEXT DEFAULT 'open',
                  decision TEXT,
                  decision_notes TEXT,
                  decided_by TEXT,
                  decided_at DATETIME,
                  created_at DATETIME DEFAULT CURRENT_TIMESTAMP)"""
    )

    # Migration-safe addition to the existing users table (same pattern as
    # fastapp1.py's init_db() ALTER TABLE / OperationalError handling).
    try:
        c.execute("ALTER TABLE users ADD COLUMN district TEXT DEFAULT 'Unknown'")
    except sqlite3.OperationalError:
        pass

    conn.commit()
    conn.close()


# --- Clustering ---------------------------------------------------------

def _cosine(a: List[float], b: List[float]) -> float:
    va, vb = np.asarray(a), np.asarray(b)
    denom = (np.linalg.norm(va) * np.linalg.norm(vb)) + 1e-9
    return float(np.dot(va, vb) / denom)


def assign_to_nearest_cluster(
    embedding: List[float],
    text: str,
    platform: Optional[str],
    similarity_threshold: float = 0.80,
) -> Tuple[int, int]:
    """Attach `text` to the nearest existing cluster centroid above threshold,
    or create a new one. Returns (cluster_id, member_count)."""
    conn = _connect()
    c = conn.cursor()
    c.execute(
        "SELECT id, centroid, member_count FROM clusters WHERE platform = ? OR platform IS NULL",
        (platform,),
    )
    rows = c.fetchall()

    best_id = None
    best_sim = -1.0
    for row in rows:
        centroid = json.loads(row["centroid"])
        sim = _cosine(embedding, centroid)
        if sim > best_sim:
            best_sim = sim
            best_id = row["id"]

    if best_id is not None and best_sim >= similarity_threshold:
        cluster_id = best_id
        c.execute("SELECT centroid, member_count FROM clusters WHERE id = ?", (cluster_id,))
        row = c.fetchone()
        old_centroid = json.loads(row["centroid"])
        old_count = row["member_count"]
        new_count = old_count + 1
        # Running mean update of the centroid.
        new_centroid = (
            (np.asarray(old_centroid) * old_count + np.asarray(embedding)) / new_count
        ).tolist()
        c.execute(
            "UPDATE clusters SET centroid = ?, member_count = ?, updated_at = ? WHERE id = ?",
            (json.dumps(new_centroid), new_count, datetime.now(), cluster_id),
        )
        c.execute(
            "INSERT INTO cluster_members (cluster_id, text, embedding, similarity_to_centroid) VALUES (?, ?, ?, ?)",
            (cluster_id, text, json.dumps(embedding), best_sim),
        )
    else:
        c.execute(
            "INSERT INTO clusters (platform, representative_text, centroid, member_count) VALUES (?, ?, ?, ?)",
            (platform, text, json.dumps(embedding), 1),
        )
        cluster_id = c.lastrowid
        new_count = 1
        c.execute(
            "INSERT INTO cluster_members (cluster_id, text, embedding, similarity_to_centroid) VALUES (?, ?, ?, ?)",
            (cluster_id, text, json.dumps(embedding), 1.0),
        )

    conn.commit()
    conn.close()
    return cluster_id, new_count


def mark_cluster_campaign(cluster_id: int) -> None:
    conn = _connect()
    conn.execute("UPDATE clusters SET campaign_flag = 1 WHERE id = ?", (cluster_id,))
    conn.commit()
    conn.close()


# --- Case files / review queue ------------------------------------------

def insert_case_file(
    text: str,
    category: Optional[str],
    confidence: Optional[float],
    language: Optional[str],
    username: str,
    platform: str,
    cluster_id: Optional[int],
    campaign_flag: bool,
    legal_provisions: List[Dict[str, Any]],
    sarcasm_score: Optional[float],
    sarcasm_flag: bool,
) -> int:
    # requires_human_review is a hardcoded SQL literal (1), never a bound
    # parameter - no caller can pass a different value through this function,
    # and the table's CHECK (requires_human_review = 1) constraint rejects
    # any insert that tried to. This is deliberate - see escalation_agent.py.
    conn = _connect()
    c = conn.cursor()
    c.execute(
        """INSERT INTO case_files
           (text, category, confidence, language, username, platform, cluster_id,
            campaign_flag, legal_provisions, sarcasm_score, sarcasm_flag, requires_human_review)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)""",
        (
            text,
            category,
            confidence,
            language,
            username,
            platform,
            cluster_id,
            1 if campaign_flag else 0,
            json.dumps(legal_provisions or []),
            sarcasm_score,
            1 if sarcasm_flag else 0,
        ),
    )
    case_file_id = c.lastrowid
    conn.commit()
    conn.close()
    return case_file_id


def insert_review_queue_entry(case_file_id: int, priority: str = "medium") -> int:
    conn = _connect()
    c = conn.cursor()
    c.execute(
        "INSERT INTO review_queue (case_file_id, priority, status) VALUES (?, ?, 'open')",
        (case_file_id, priority),
    )
    review_queue_id = c.lastrowid
    conn.commit()
    conn.close()
    return review_queue_id


def get_review_queue(
    status: Optional[str] = "open",
    platform: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
) -> Tuple[int, List[Dict[str, Any]]]:
    conn = _connect()
    c = conn.cursor()

    where = []
    params: List[Any] = []
    if status:
        where.append("rq.status = ?")
        params.append(status)
    if platform:
        where.append("cf.platform = ?")
        params.append(platform)
    where_clause = f"WHERE {' AND '.join(where)}" if where else ""

    c.execute(f"SELECT COUNT(*) FROM review_queue rq JOIN case_files cf ON cf.id = rq.case_file_id {where_clause}", params)
    total = c.fetchone()[0]

    c.execute(
        f"""SELECT rq.id, rq.case_file_id, rq.priority, rq.status, rq.decision,
                   rq.decision_notes, rq.decided_by, rq.decided_at, rq.created_at,
                   cf.text, cf.category, cf.confidence, cf.language, cf.username,
                   cf.platform, cf.district, cf.cluster_id, cf.campaign_flag,
                   cf.legal_provisions, cf.sarcasm_score, cf.sarcasm_flag
            FROM review_queue rq
            JOIN case_files cf ON cf.id = rq.case_file_id
            {where_clause}
            ORDER BY rq.id DESC
            LIMIT ? OFFSET ?""",
        [*params, limit, offset],
    )
    rows = []
    for row in c.fetchall():
        d = dict(row)
        try:
            d["legal_matches"] = json.loads(d.pop("legal_provisions") or "[]")
        except (TypeError, json.JSONDecodeError):
            d["legal_matches"] = []
        rows.append(d)
    conn.close()
    return total, rows


def record_review_decision(
    review_queue_id: int, decision: str, decided_by: Optional[str], notes: Optional[str]
) -> bool:
    conn = _connect()
    c = conn.cursor()
    c.execute("SELECT case_file_id FROM review_queue WHERE id = ?", (review_queue_id,))
    row = c.fetchone()
    if row is None:
        conn.close()
        return False
    case_file_id = row["case_file_id"]

    c.execute(
        """UPDATE review_queue
           SET status = 'closed', decision = ?, decision_notes = ?, decided_by = ?, decided_at = ?
           WHERE id = ?""",
        (decision, notes, decided_by, datetime.now(), review_queue_id),
    )
    new_status = "resolved" if decision == "confirm_violation" else "dismissed"
    c.execute("UPDATE case_files SET status = ? WHERE id = ?", (new_status, case_file_id))
    conn.commit()
    conn.close()
    return True


# --- Stats -----------------------------------------------------------------

def get_district_stats() -> List[Dict[str, Any]]:
    conn = _connect()
    c = conn.cursor()
    c.execute(
        """SELECT district,
                  SUM(CASE WHEN category = 'hate' THEN 1 ELSE 0 END) AS hate,
                  SUM(CASE WHEN category = 'offensive' THEN 1 ELSE 0 END) AS offensive,
                  SUM(CASE WHEN category = 'neutral' THEN 1 ELSE 0 END) AS neutral,
                  COUNT(*) AS total
           FROM case_files
           GROUP BY district"""
    )
    data = [dict(row) for row in c.fetchall()]
    conn.close()
    return data


def get_platform_stats() -> List[Dict[str, Any]]:
    conn = _connect()
    c = conn.cursor()
    c.execute(
        """SELECT COALESCE(platform, 'Unknown') AS platform,
                  SUM(CASE WHEN category = 'hate' THEN 1 ELSE 0 END) AS hate,
                  SUM(CASE WHEN category = 'offensive' THEN 1 ELSE 0 END) AS offensive,
                  SUM(CASE WHEN category = 'neutral' THEN 1 ELSE 0 END) AS neutral,
                  COUNT(*) AS total
           FROM logs
           GROUP BY platform"""
    )
    data = [dict(row) for row in c.fetchall()]
    conn.close()
    return data
