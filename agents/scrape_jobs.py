"""
Scrapes for locked monitoring targets, run in the background.

start_target_scrape() only *starts* an Apify run (about a second) and returns.
A daemon thread then polls the run, and when it succeeds, downloads the posts
and pushes each through the full agent pipeline attributed to the target (and
tagged with the target's officer-assigned district), so flagged posts land in
the review queue and the target's risk score updates.

Every step is written to the scrape_runs table and mirrored onto the target's
row (last_scrape_status / last_scrape_note), which is how the dashboard shows
that Apify is actually running - and, unlike the synchronous /ingest-and-process
call, the backend keeps answering other requests while a scrape is in flight.

An hourly schedule can later call start_target_scrape() for every locked target;
nothing here assumes the trigger was a person clicking a button.
"""
import asyncio
import logging
import threading
import time

import requests

import agents.db as db
import agents.orchestrator as orchestrator
from agents.ingestion_agent import MAX_RESULTS, ApifyError, apify_config, parse_items

API = "https://api.apify.com/v2"
POLL_SECONDS = 8
MAX_WAIT_SECONDS = 600
TERMINAL = {"SUCCEEDED", "FAILED", "ABORTED", "TIMED-OUT"}


def _auth(token: str):
    return {"Authorization": f"Bearer {token}"}


def _fail(run_id: int, user_id: int, message: str) -> None:
    db.update_scrape_run(run_id, status="FAILED", error=message, finished_at=db._now())
    db.set_target_scrape_state(user_id, "FAILED", message)


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
        resp = requests.post(
            f"{API}/acts/{actor.replace('/', '~')}/runs",
            json={"startUrls": [{"url": target["profile_url"]}], "resultsLimit": n},
            headers=_auth(token),
            timeout=30,
        )
    except requests.RequestException as e:
        _fail(run_id, user_id, f"Could not reach Apify: {e}")
        raise ApifyError(f"Could not reach Apify: {e}") from e

    if resp.status_code >= 400:
        message = {
            401: "Apify rejected the token (401). Check APIFY_API_TOKEN.",
            402: "The Apify account is out of credit (402).",
            404: f"Apify actor '{actor}' was not found (404). Check APIFY_ACTOR_ID.",
        }.get(resp.status_code, f"Apify returned HTTP {resp.status_code}: {resp.text[:200]}")
        _fail(run_id, user_id, message)
        raise ApifyError(message)

    data = resp.json()["data"]
    db.update_scrape_run(run_id, apify_run_id=data["id"], status="RUNNING")
    db.set_target_scrape_state(user_id, "RUNNING", "Apify is scraping the page...")
    threading.Thread(
        target=_watch,
        args=(
            run_id,
            user_id,
            target["username"],
            target.get("district") or "Unknown",
            data["id"],
            data["defaultDatasetId"],
            token,
            n,
        ),
        daemon=True,
        name=f"scrape-{run_id}",
    ).start()
    return run_id


def _watch(run_id, user_id, username, district, apify_run_id, dataset_id, token, n):
    try:
        status = "RUNNING"
        deadline = time.time() + MAX_WAIT_SECONDS
        while time.time() < deadline:
            time.sleep(POLL_SECONDS)
            r = requests.get(f"{API}/actor-runs/{apify_run_id}", headers=_auth(token), timeout=30)
            r.raise_for_status()
            status = r.json()["data"]["status"]
            if status in TERMINAL:
                break

        if status != "SUCCEEDED":
            _fail(
                run_id,
                user_id,
                f"The Apify run ended with status {status}."
                if status in TERMINAL
                else "Gave up waiting for Apify after 10 minutes.",
            )
            return

        db.update_scrape_run(run_id, status="PROCESSING")
        db.set_target_scrape_state(user_id, "PROCESSING", "Classifying the scraped posts...")
        r = requests.get(
            f"{API}/datasets/{dataset_id}/items",
            params={"clean": "true", "limit": n},
            headers=_auth(token),
            timeout=60,
        )
        r.raise_for_status()
        posts = list(parse_items(r.json()))
        flagged = asyncio.run(_classify(posts, username, district))

        db.update_scrape_run(run_id, status="SUCCEEDED", posts=len(posts), flagged=flagged, finished_at=db._now())
        db.refresh_target_risk(user_id, username)
        db.set_target_scrape_state(
            user_id,
            "SUCCEEDED",
            f"{len(posts)} post{'' if len(posts) == 1 else 's'} scraped, {flagged} flagged for review.",
        )
    except Exception as e:  # never leave a run stuck in an active state
        logging.exception(f"Scrape run {run_id} failed")
        _fail(run_id, user_id, f"Scrape failed: {e}")


async def _classify(posts, username, district) -> int:
    """Attribute every post to the target so its logs drive the target's risk score."""
    flagged = 0
    for post in posts:
        state = await orchestrator.run_pipeline(
            text=post["text"], username=username, platform="Facebook", source="apify", district=district
        )
        if state.get("requires_human_review"):
            flagged += 1
    return flagged
