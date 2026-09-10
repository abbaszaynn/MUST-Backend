"""
Scrapes for locked monitoring targets (public Facebook pages), run in the background.

A scrape has two Apify phases:
  1. Posts    - apify/facebook-posts-scraper collects the page's recent posts.
  2. Comments - apify/facebook-comments-scraper collects the comments under those
                posts. On news pages the posts are usually neutral reporting; hate
                speech spreads in the comments, so this is the phase that matters.

Every post and comment goes through the full agent pipeline. Posts are attributed
to the page, comments to the person who wrote them, and both carry the page's
officer-assigned district - so flagged comments land in the review queue under the
right author and region. Each item is also recorded in scraped_items (which page,
which post, what the pipeline decided), de-duplicated so re-scraping never
re-flags the same comment.

start_target_scrape() only *starts* the posts run and returns; a daemon thread
does the rest. Every step is mirrored onto the target's row (last_scrape_status /
last_scrape_note), which is how the dashboard shows Apify is running, and the
backend keeps answering other requests meanwhile. Each Apify run carries a hard
maxTotalChargeUsd cap, so a run can never spend more than its budget.

An hourly schedule can later call start_target_scrape() for every locked target.
"""
import asyncio
import hashlib
import logging
import os
import threading
import time

import requests

import agents.db as db
import agents.orchestrator as orchestrator
from agents.ingestion_agent import MAX_RESULTS, ApifyError, apify_config

API = "https://api.apify.com/v2"
POLL_SECONDS = 8
MAX_WAIT_SECONDS = 600
TERMINAL = {"SUCCEEDED", "FAILED", "ABORTED", "TIMED-OUT"}

COMMENTS_ACTOR_DEFAULT = "apify/facebook-comments-scraper"
COMMENTS_PER_POST = 20

# Free-tier prices of the two official actors, used only to size the hard cost cap
# sent with each run. Apify stops a run once it reaches the cap.
POST_PRICE_USD = 0.005
COMMENT_PRICE_USD = 0.0025
START_FEE_USD = 0.01


def _auth(token: str):
    return {"Authorization": f"Bearer {token}"}


def _fail(run_id: int, user_id: int, message: str) -> None:
    db.update_scrape_run(run_id, status="FAILED", error=message, finished_at=db._now())
    db.set_target_scrape_state(user_id, "FAILED", message)


def _http_message(resp, actor: str) -> str:
    return {
        401: "Apify rejected the token (401). Check APIFY_API_TOKEN.",
        402: "The Apify account is out of credit (402).",
        404: f"Apify actor '{actor}' was not found (404).",
    }.get(resp.status_code, f"Apify returned HTTP {resp.status_code}: {resp.text[:200]}")


def _start_run(actor: str, payload: dict, token: str, max_charge_usd: float):
    resp = requests.post(
        f"{API}/acts/{actor.replace('/', '~')}/runs",
        json=payload,
        headers=_auth(token),
        params={"maxTotalChargeUsd": f"{max_charge_usd:.2f}"},
        timeout=30,
    )
    if resp.status_code >= 400:
        raise ApifyError(_http_message(resp, actor))
    data = resp.json()["data"]
    return data["id"], data["defaultDatasetId"]


def _wait(apify_run_id: str, token: str) -> str:
    status = "RUNNING"
    deadline = time.time() + MAX_WAIT_SECONDS
    while time.time() < deadline:
        time.sleep(POLL_SECONDS)
        r = requests.get(f"{API}/actor-runs/{apify_run_id}", headers=_auth(token), timeout=30)
        r.raise_for_status()
        status = r.json()["data"]["status"]
        if status in TERMINAL:
            return status
    return "WAIT-TIMEOUT"


def _items(dataset_id: str, token: str, limit: int) -> list:
    r = requests.get(
        f"{API}/datasets/{dataset_id}/items",
        params={"clean": "true", "limit": limit},
        headers=_auth(token),
        timeout=60,
    )
    r.raise_for_status()
    return r.json()


def _key(url, text: str) -> str:
    return url or "sha1:" + hashlib.sha1(text.encode("utf-8")).hexdigest()


def start_target_scrape(user_id: int, limit: int = 10) -> int:
    target = db.get_target(user_id)
    if not target:
        raise ApifyError("That target no longer exists.")
    if not target.get("profile_url"):
        raise ApifyError("This target has no Facebook page URL. Remove it and add it again with one.")
    if db.active_scrape_for(user_id):
        raise ApifyError("A scrape for this target is already running.")
    token, actor, _ = apify_config()
    if not token:
        raise ApifyError("Live scraping is off: APIFY_API_TOKEN is not set in MUST_backend/.env.")

    n = max(1, min(int(limit), MAX_RESULTS))
    run_id = db.create_scrape_run(user_id)
    try:
        apify_run_id, dataset_id = _start_run(
            actor,
            {"startUrls": [{"url": target["profile_url"]}], "resultsLimit": n},
            token,
            max_charge_usd=n * POST_PRICE_USD + START_FEE_USD,
        )
    except ApifyError as e:
        _fail(run_id, user_id, str(e))
        raise
    except requests.RequestException as e:
        _fail(run_id, user_id, f"Could not reach Apify: {e}")
        raise ApifyError(f"Could not reach Apify: {e}") from e

    db.update_scrape_run(run_id, apify_run_id=apify_run_id, status="RUNNING")
    db.set_target_scrape_state(user_id, "RUNNING", "Apify is scraping the page's posts...")
    threading.Thread(
        target=_watch,
        args=(run_id, target, apify_run_id, dataset_id, token, n),
        daemon=True,
        name=f"scrape-{run_id}",
    ).start()
    return run_id


def _watch(run_id, target, apify_run_id, dataset_id, token, n):
    user_id = target["id"]
    try:
        # --- Phase 1: posts ---
        status = _wait(apify_run_id, token)
        if status != "SUCCEEDED":
            _fail(run_id, user_id, f"The posts run ended with status {status}.")
            return

        post_urls, post_rows = [], []
        for item in _items(dataset_id, token, n):
            url = item.get("url") or item.get("topLevelUrl") or item.get("facebookUrl")
            if url:
                post_urls.append(url)  # collect comments even on posts already seen
            text = item.get("text") or item.get("message") or ""
            if not text:
                continue
            key = _key(url, text)
            if db.scraped_item_exists(user_id, key):
                continue
            post_rows.append({
                "kind": "post", "key": key, "username": target["username"],
                "author": (item.get("user") or {}).get("name") or item.get("pageName") or target["username"],
                "text": text, "url": url, "parent_url": None, "posted_at": item.get("time"),
            })

        db.update_scrape_run(run_id, status="PROCESSING")
        db.set_target_scrape_state(user_id, "PROCESSING", f"Classifying {len(post_rows)} new posts...")
        flagged = asyncio.run(_classify(run_id, target, post_rows))

        # --- Phase 2: comments (where hate speech actually spreads) ---
        comment_rows, comments_note = [], ""
        if post_urls and COMMENTS_PER_POST > 0:
            try:
                comments_actor = os.environ.get("APIFY_COMMENTS_ACTOR_ID", "").strip() or COMMENTS_ACTOR_DEFAULT
                db.update_scrape_run(run_id, status="RUNNING")
                db.set_target_scrape_state(user_id, "RUNNING", f"Apify is scraping comments on {len(post_urls)} posts...")
                c_run, c_dataset = _start_run(
                    comments_actor,
                    {
                        "startUrls": [{"url": u} for u in post_urls],
                        "resultsLimit": COMMENTS_PER_POST,
                        "includeNestedComments": False,
                    },
                    token,
                    max_charge_usd=len(post_urls) * COMMENTS_PER_POST * COMMENT_PRICE_USD + START_FEE_USD,
                )
                c_status = _wait(c_run, token)
                if c_status != "SUCCEEDED":
                    comments_note = f" The comments run ended with status {c_status}; partial results kept."
                # A run stopped at its cost cap still has the comments it collected.
                raw_comments = _items(c_dataset, token, len(post_urls) * COMMENTS_PER_POST) if c_status in TERMINAL else []
                for item in raw_comments:
                    text = item.get("text") or ""
                    if not text:
                        continue
                    url = item.get("commentUrl") or item.get("url")
                    key = _key(url, text)
                    if db.scraped_item_exists(user_id, key):
                        continue
                    author = (
                        item.get("profileName")
                        or (item.get("author") or {}).get("name")
                        or (item.get("user") or {}).get("name")
                        or "Facebook user"
                    )
                    comment_rows.append({
                        "kind": "comment", "key": key, "username": author, "author": author,
                        "text": text, "url": url,
                        # inputUrl is the exact post URL we submitted, so it always matches the post.
                        "parent_url": item.get("inputUrl") or item.get("facebookUrl") or item.get("postUrl"),
                        "posted_at": item.get("date"),
                    })
                db.update_scrape_run(run_id, status="PROCESSING")
                db.set_target_scrape_state(user_id, "PROCESSING", f"Classifying {len(comment_rows)} new comments...")
                flagged += asyncio.run(_classify(run_id, target, comment_rows))
            except Exception as e:
                logging.exception(f"Comments phase of scrape run {run_id} failed")
                comments_note = f" Comments could not be collected: {e}"

        db.update_scrape_run(
            run_id, status="SUCCEEDED", posts=len(post_rows), comments=len(comment_rows),
            flagged=flagged, finished_at=db._now(),
        )
        db.refresh_target_risk(user_id)
        db.set_target_scrape_state(
            user_id,
            "SUCCEEDED",
            f"{len(post_rows)} new posts and {len(comment_rows)} new comments scraped, "
            f"{flagged} flagged for review." + comments_note,
        )
    except Exception as e:  # never leave a run stuck in an active state
        logging.exception(f"Scrape run {run_id} failed")
        _fail(run_id, user_id, f"Scrape failed: {e}")


async def _classify(run_id, target, rows) -> int:
    flagged = 0
    district = target.get("district") or "Unknown"
    for r in rows:
        state = await orchestrator.run_pipeline(
            text=r["text"], username=r["username"], platform="Facebook", source="apify", district=district
        )
        # Text the language gate rejects (emoji-only, unsupported language) is still
        # recorded, with no category, so the officer sees it was collected but not judged.
        db.insert_scraped_item(
            target_id=target["id"], scrape_run_id=run_id, kind=r["kind"], item_key=r["key"],
            author=r["author"], text=r["text"], url=r["url"], parent_url=r["parent_url"],
            posted_at=r["posted_at"],
            category=None if state.get("error") else state.get("category"),
            confidence=None if state.get("error") else state.get("confidence"),
            language=None if state.get("error") else state.get("language"),
            case_file_id=state.get("case_file_id"),
        )
        if state.get("requires_human_review"):
            flagged += 1
    return flagged
