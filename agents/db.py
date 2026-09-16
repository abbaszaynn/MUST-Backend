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


# --- Apify collection record --------------------------------------------------

def _parse_ts(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        # fastapp1.py's older rows use "YYYY-MM-DD HH:MM:SS[.ffffff]".
        for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
            try:
                return datetime.strptime(value, fmt)
            except ValueError:
                continue
    return None


def get_scrape_overview(limit: int = 100) -> Dict[str, Any]:
    """Everything the Apify Records page reports: what each run collected, what it
    cost at list prices, how long it took, and how often runs happen.

    Cost is estimated locally from agents/pricing.py - it is not read back from
    Apify's billing API, so it is a list-price estimate, not an invoice.
    """
    from agents.pricing import (
        COMMENT_PRICE_USD,
        COMMENTS_PER_POST,
        PKR_PER_USD,
        POST_PRICE_USD,
        START_FEE_USD,
        estimate_run_cost_usd,
    )

    conn = _connect()
    c = conn.cursor()

    rows = [
        dict(r)
        for r in c.execute(
            """SELECT sr.id, sr.user_id, sr.apify_run_id, sr.status, sr.posts,
                      COALESCE(sr.comments, 0) AS comments, sr.flagged, sr.error,
                      sr.started_at, sr.finished_at,
                      u.username, u.profile_url, u.district
               FROM scrape_runs sr
               JOIN users u ON u.id = sr.user_id
               ORDER BY sr.id DESC
               LIMIT ?""",
            (limit,),
        ).fetchall()
    ]

    # How many of each run's items the pipeline actually classified, and how they landed.
    per_run: Dict[int, Dict[str, int]] = {}
    for r in c.execute(
        """SELECT scrape_run_id AS run_id,
                  COUNT(*) AS items,
                  SUM(CASE WHEN category = 'hate' THEN 1 ELSE 0 END) AS hate,
                  SUM(CASE WHEN category = 'offensive' THEN 1 ELSE 0 END) AS offensive,
                  SUM(CASE WHEN category = 'neutral' THEN 1 ELSE 0 END) AS neutral,
                  SUM(CASE WHEN category IS NULL THEN 1 ELSE 0 END) AS unclassified
           FROM scraped_items
           WHERE scrape_run_id IS NOT NULL
           GROUP BY scrape_run_id"""
    ).fetchall():
        d = dict(r)
        per_run[d.pop("run_id")] = {k: (v or 0) for k, v in d.items()}

    starts: List[datetime] = []
    durations: List[float] = []
    for row in rows:
        started, finished = _parse_ts(row["started_at"]), _parse_ts(row["finished_at"])
        row["duration_seconds"] = (
            round((finished - started).total_seconds()) if started and finished else None
        )
        if row["duration_seconds"] is not None:
            durations.append(row["duration_seconds"])
        if started:
            starts.append(started)
        row["cost_usd"] = round(estimate_run_cost_usd(row["posts"], row["comments"]), 4)
        row["cost_pkr"] = round(row["cost_usd"] * PKR_PER_USD, 2)
        row["items"] = per_run.get(row["id"], {})

    # Cadence: scrapes are started by an officer today, so this describes the
    # observed spacing between runs, not a configured schedule.
    starts_sorted = sorted(starts)
    gaps = [
        (b - a).total_seconds()
        for a, b in zip(starts_sorted, starts_sorted[1:])
        if (b - a).total_seconds() > 0
    ]

    totals_posts = sum(r["posts"] or 0 for r in rows)
    totals_comments = sum(r["comments"] or 0 for r in rows)
    total_cost_usd = round(sum(r["cost_usd"] for r in rows), 4)

    item_totals = c.execute(
        """SELECT COUNT(*) AS total,
                  SUM(CASE WHEN kind = 'post' THEN 1 ELSE 0 END) AS posts,
                  SUM(CASE WHEN kind = 'comment' THEN 1 ELSE 0 END) AS comments,
                  SUM(CASE WHEN category = 'hate' THEN 1 ELSE 0 END) AS hate,
                  SUM(CASE WHEN category = 'offensive' THEN 1 ELSE 0 END) AS offensive,
                  SUM(CASE WHEN category = 'neutral' THEN 1 ELSE 0 END) AS neutral,
                  SUM(CASE WHEN category IS NULL THEN 1 ELSE 0 END) AS unclassified
           FROM scraped_items"""
    ).fetchone()

    conn.close()

    stored = {k: (v or 0) for k, v in dict(item_totals).items()}
    flagged = stored["hate"] + stored["offensive"]

    return {
        "runs": rows,
        "totals": {
            "runs": len(rows),
            "succeeded": sum(1 for r in rows if r["status"] == "SUCCEEDED"),
            "failed": sum(1 for r in rows if r["status"] == "FAILED"),
            "active": sum(1 for r in rows if r["status"] in ACTIVE_SCRAPE_STATES),
            "posts_collected": totals_posts,
            "comments_collected": totals_comments,
            "items_stored": stored["total"],
            "posts_stored": stored["posts"],
            "comments_stored": stored["comments"],
            "hate": stored["hate"],
            "offensive": stored["offensive"],
            "neutral": stored["neutral"],
            "unclassified": stored["unclassified"],
            "flagged": flagged,
            "flag_rate": round(100.0 * flagged / stored["total"], 1) if stored["total"] else 0.0,
            "comments_per_post": (
                round(totals_comments / totals_posts, 1) if totals_posts else 0.0
            ),
        },
        "cost": {
            "total_usd": total_cost_usd,
            "total_pkr": round(total_cost_usd * PKR_PER_USD, 2),
            "per_flagged_usd": round(total_cost_usd / flagged, 4) if flagged else None,
            "post_price_usd": POST_PRICE_USD,
            "comment_price_usd": COMMENT_PRICE_USD,
            "start_fee_usd": START_FEE_USD,
            "pkr_per_usd": PKR_PER_USD,
            "basis": (
                "Estimated locally from Apify's published pay-per-event prices, "
                "not read back from Apify's billing API."
            ),
        },
        "cadence": {
            "scheduled": False,
            "trigger": "Started by an officer from User Monitoring (no automatic schedule is configured).",
            "comments_requested_per_post": COMMENTS_PER_POST,
            "avg_run_seconds": round(sum(durations) / len(durations)) if durations else None,
            "longest_run_seconds": max(durations) if durations else None,
            "avg_gap_seconds": round(sum(gaps) / len(gaps)) if gaps else None,
            "first_run_at": starts_sorted[0].isoformat(timespec="seconds") if starts_sorted else None,
            "last_run_at": starts_sorted[-1].isoformat(timespec="seconds") if starts_sorted else None,
        },
    }


# --- Clustering (campaign detection) ------------------------------------------

def list_clusters(limit: int = 60, min_members: int = 1) -> List[Dict[str, Any]]:
    """Clusters with their members, newest-largest first.

    This is the simplified embedding + cosine-similarity grouping documented in
    clustering_agent.py - a placeholder for a full ULTRA integration, not ULTRA
    itself. Callers must present it as such.
    """
    conn = _connect()
    c = conn.cursor()
    clusters = [
        dict(r)
        for r in c.execute(
            """SELECT id, platform, representative_text, member_count, campaign_flag,
                      created_at, updated_at
               FROM clusters
               WHERE member_count >= ?
               ORDER BY member_count DESC, id DESC
               LIMIT ?""",
            (min_members, limit),
        ).fetchall()
    ]
    if not clusters:
        conn.close()
        return []

    ids = [c_["id"] for c_ in clusters]
    placeholders = ",".join("?" * len(ids))

    members: Dict[int, List[Dict[str, Any]]] = {i: [] for i in ids}
    for r in c.execute(
        f"""SELECT cluster_id, id, text, similarity_to_centroid, added_at
            FROM cluster_members
            WHERE cluster_id IN ({placeholders})
            ORDER BY added_at ASC""",
        ids,
    ).fetchall():
        d = dict(r)
        members[d["cluster_id"]].append(d)

    # Case files carry the district/username, so a cluster can be tied to real cases.
    cases: Dict[int, List[Dict[str, Any]]] = {i: [] for i in ids}
    for r in c.execute(
        f"""SELECT cf.cluster_id, cf.id, cf.username, cf.platform, cf.district,
                   cf.category, cf.confidence, cf.created_at, rq.id AS review_queue_id,
                   rq.status AS review_status
            FROM case_files cf
            LEFT JOIN review_queue rq ON rq.case_file_id = cf.id
            WHERE cf.cluster_id IN ({placeholders})
            ORDER BY cf.id DESC""",
        ids,
    ).fetchall():
        d = dict(r)
        cases[d["cluster_id"]].append(d)

    conn.close()

    for cl in clusters:
        cl["members"] = members.get(cl["id"], [])
        cl["cases"] = cases.get(cl["id"], [])
        authors = {case["username"] for case in cl["cases"] if case["username"]}
        districts = {case["district"] for case in cl["cases"] if case["district"]}
        cl["distinct_authors"] = len(authors)
        cl["districts"] = sorted(districts)
    return clusters


# --- Sarcasm heuristic --------------------------------------------------------

def get_sarcasm_overview(limit: int = 50) -> Dict[str, Any]:
    """What the sarcasm heuristic has actually recorded.

    The node only runs for classifications inside the ambiguous confidence band,
    so `scored` is normally far smaller than the total number of case files -
    that is expected behaviour, not missing data.
    """
    conn = _connect()
    c = conn.cursor()

    totals = dict(
        c.execute(
            """SELECT COUNT(*) AS case_files,
                      COUNT(sarcasm_score) AS scored,
                      SUM(CASE WHEN sarcasm_flag = 1 THEN 1 ELSE 0 END) AS flagged
               FROM case_files"""
        ).fetchone()
    )

    recent = [
        dict(r)
        for r in c.execute(
            """SELECT cf.id, cf.text, cf.category, cf.confidence, cf.language,
                      cf.username, cf.platform, cf.district, cf.sarcasm_score,
                      cf.sarcasm_flag, cf.created_at, rq.status AS review_status
               FROM case_files cf
               LEFT JOIN review_queue rq ON rq.case_file_id = cf.id
               WHERE cf.sarcasm_score IS NOT NULL
               ORDER BY cf.id DESC
               LIMIT ?""",
            (limit,),
        ).fetchall()
    ]
    conn.close()

    return {
        "totals": {
            "case_files": totals["case_files"] or 0,
            "scored": totals["scored"] or 0,
            "flagged": totals["flagged"] or 0,
        },
        "recent": recent,
    }
