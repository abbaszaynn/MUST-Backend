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

    # Monitored-target columns: the Facebook page to scrape and the latest scrape
    # state, so the dashboard can show whether Apify is actually running.
    for col, ddl in (
        ("profile_url", "TEXT"),
        ("last_scrape_status", "TEXT"),
        ("last_scrape_at", "TEXT"),
        ("last_scrape_note", "TEXT"),
    ):
        try:
            c.execute(f"ALTER TABLE users ADD COLUMN {col} {ddl}")
        except sqlite3.OperationalError:
            pass

    c.execute(
        """CREATE TABLE IF NOT EXISTS scrape_runs
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_id INTEGER NOT NULL REFERENCES users(id),
                  apify_run_id TEXT,
                  status TEXT DEFAULT 'STARTING',
                  posts INTEGER DEFAULT 0,
                  flagged INTEGER DEFAULT 0,
                  error TEXT,
                  started_at TEXT,
                  finished_at TEXT)"""
    )
    try:
        c.execute("ALTER TABLE scrape_runs ADD COLUMN comments INTEGER DEFAULT 0")
    except sqlite3.OperationalError:
        pass

    # Every post and comment collected from a monitored page, with where it came
    # from and what the pipeline concluded. item_key (the post/comment URL, or a
    # text hash) de-duplicates re-scrapes so the same comment is never re-flagged.
    c.execute(
        """CREATE TABLE IF NOT EXISTS scraped_items
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  target_id INTEGER NOT NULL REFERENCES users(id),
                  scrape_run_id INTEGER REFERENCES scrape_runs(id),
                  kind TEXT NOT NULL,
                  item_key TEXT NOT NULL,
                  author TEXT,
                  text TEXT,
                  url TEXT,
                  parent_url TEXT,
                  posted_at TEXT,
                  category TEXT,
                  confidence REAL,
                  language TEXT,
                  case_file_id INTEGER REFERENCES case_files(id),
                  created_at TEXT,
                  UNIQUE (target_id, item_key))"""
    )

    # A backend restart kills the watcher threads, so any run still marked active
    # can never finish - close it out, or it would block new scrapes forever.
    now = _now()
    c.execute(
        """UPDATE scrape_runs SET status = 'FAILED', finished_at = ?,
                  error = 'The backend restarted before this run finished.'
           WHERE status IN ('STARTING', 'RUNNING', 'PROCESSING')""",
        (now,),
    )
    c.execute(
        """UPDATE users SET last_scrape_status = 'FAILED',
                  last_scrape_note = 'The backend restarted before the last scrape finished.'
           WHERE last_scrape_status IN ('STARTING', 'RUNNING', 'PROCESSING')"""
    )

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
    district: str = "Unknown",
) -> int:
    # requires_human_review is a hardcoded SQL literal (1), never a bound
    # parameter - no caller can pass a different value through this function,
    # and the table's CHECK (requires_human_review = 1) constraint rejects
    # any insert that tried to. This is deliberate - see escalation_agent.py.
    conn = _connect()
    c = conn.cursor()
    c.execute(
        """INSERT INTO case_files
           (text, category, confidence, language, username, platform, district, cluster_id,
            campaign_flag, legal_provisions, sarcasm_score, sarcasm_flag, requires_human_review)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)""",
        (
            text,
            category,
            confidence,
            language,
            username,
            platform,
            district or "Unknown",
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


# --- Monitored targets & scrape runs ------------------------------------------

ACTIVE_SCRAPE_STATES = ("STARTING", "RUNNING", "PROCESSING")
_RUN_FIELDS = {"apify_run_id", "status", "posts", "comments", "flagged", "error", "finished_at"}


def _now() -> str:
    # ISO seconds, which browsers parse reliably with new Date().
    return datetime.now().isoformat(timespec="seconds")


def add_target(username: str, profile_url: str, district: str = "Unknown", platform: str = "Facebook") -> int:
    """Lock a page as a monitored target. Re-adding an existing name updates its URL and district.

    The district is assigned by the officer: Facebook page posts carry no location,
    so this is the page's area of focus, not a location detected from the posts.
    """
    district = district or "Unknown"
    conn = _connect()
    c = conn.cursor()
    c.execute("SELECT id FROM users WHERE username = ?", (username,))
    row = c.fetchone()
    if row:
        user_id = row["id"]
        c.execute(
            "UPDATE users SET profile_url = ?, platform = ?, district = ? WHERE id = ?",
            (profile_url, platform, district, user_id),
        )
    else:
        c.execute(
            """INSERT INTO users (username, platform, risk_score, last_active, profile_url, district, last_scrape_status)
               VALUES (?, ?, 0, ?, ?, ?, 'NEVER')""",
            (username, platform, _now(), profile_url, district),
        )
        user_id = c.lastrowid
    conn.commit()
    conn.close()
    return user_id


def get_target(user_id: int) -> Optional[Dict[str, Any]]:
    conn = _connect()
    row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def active_scrape_for(user_id: int) -> bool:
    conn = _connect()
    row = conn.execute(
        f"SELECT 1 FROM scrape_runs WHERE user_id = ? AND status IN ({','.join('?' * len(ACTIVE_SCRAPE_STATES))}) LIMIT 1",
        (user_id, *ACTIVE_SCRAPE_STATES),
    ).fetchone()
    conn.close()
    return row is not None


def create_scrape_run(user_id: int) -> int:
    conn = _connect()
    c = conn.cursor()
    now = _now()
    c.execute("INSERT INTO scrape_runs (user_id, status, started_at) VALUES (?, 'STARTING', ?)", (user_id, now))
    run_id = c.lastrowid
    c.execute(
        "UPDATE users SET last_scrape_status = 'STARTING', last_scrape_at = ?, last_scrape_note = ? WHERE id = ?",
        (now, "Starting the Apify run...", user_id),
    )
    conn.commit()
    conn.close()
    return run_id


def update_scrape_run(run_id: int, **fields) -> None:
    fields = {k: v for k, v in fields.items() if k in _RUN_FIELDS}
    if not fields:
        return
    conn = _connect()
    conn.execute(
        f"UPDATE scrape_runs SET {', '.join(f'{k} = ?' for k in fields)} WHERE id = ?",
        (*fields.values(), run_id),
    )
    conn.commit()
    conn.close()


def set_target_scrape_state(user_id: int, status: str, note: str) -> None:
    conn = _connect()
    conn.execute(
        "UPDATE users SET last_scrape_status = ?, last_scrape_note = ?, last_scrape_at = ? WHERE id = ?",
        (status, note, _now(), user_id),
    )
    conn.commit()
    conn.close()


def refresh_target_risk(user_id: int) -> None:
    """Risk score = share of everything scraped from this page - its posts and the
    comments left under them - that was classified hate or offensive."""
    conn = _connect()
    c = conn.cursor()
    c.execute(
        """SELECT COUNT(*) AS total,
                  SUM(CASE WHEN category IN ('hate', 'offensive') THEN 1 ELSE 0 END) AS flagged
           FROM scraped_items WHERE target_id = ?""",
        (user_id,),
    )
    row = c.fetchone()
    total, flagged = row["total"] or 0, row["flagged"] or 0
    risk = round(100.0 * flagged / total, 1) if total else 0.0
    c.execute("UPDATE users SET risk_score = ?, last_active = ? WHERE id = ?", (risk, _now(), user_id))
    conn.commit()
    conn.close()


def list_scrape_runs(user_id: Optional[int] = None, limit: int = 20) -> List[Dict[str, Any]]:
    conn = _connect()
    sql = """SELECT sr.*, u.username, u.profile_url FROM scrape_runs sr
             JOIN users u ON u.id = sr.user_id"""
    params: List[Any] = []
    if user_id is not None:
        sql += " WHERE sr.user_id = ?"
        params.append(user_id)
    sql += " ORDER BY sr.id DESC LIMIT ?"
    params.append(limit)
    rows = [dict(r) for r in conn.execute(sql, params).fetchall()]
    conn.close()
    return rows


def scraped_item_exists(target_id: int, item_key: str) -> bool:
    conn = _connect()
    row = conn.execute(
        "SELECT 1 FROM scraped_items WHERE target_id = ? AND item_key = ? LIMIT 1", (target_id, item_key)
    ).fetchone()
    conn.close()
    return row is not None


def insert_scraped_item(
    target_id: int,
    scrape_run_id: int,
    kind: str,
    item_key: str,
    author: Optional[str],
    text: str,
    url: Optional[str],
    parent_url: Optional[str],
    posted_at: Optional[str],
    category: Optional[str],
    confidence: Optional[float],
    language: Optional[str],
    case_file_id: Optional[int],
) -> None:
    conn = _connect()
    conn.execute(
        """INSERT OR IGNORE INTO scraped_items
           (target_id, scrape_run_id, kind, item_key, author, text, url, parent_url, posted_at,
            category, confidence, language, case_file_id, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (target_id, scrape_run_id, kind, item_key, author, text, url, parent_url, posted_at,
         category, confidence, language, case_file_id, _now()),
    )
    conn.commit()
    conn.close()


def get_target_items(target_id: int, limit: int = 300) -> List[Dict[str, Any]]:
    """Posts and comments scraped from a page, with the review state of any that were flagged."""
    conn = _connect()
    rows = conn.execute(
        """SELECT si.id, si.kind, si.author, si.text, si.url, si.parent_url, si.posted_at,
                  si.category, si.confidence, si.language, si.case_file_id, si.created_at,
                  rq.status AS review_status
           FROM scraped_items si
           LEFT JOIN review_queue rq ON rq.case_file_id = si.case_file_id
           WHERE si.target_id = ?
           ORDER BY si.id DESC
           LIMIT ?""",
        (target_id, limit),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]
