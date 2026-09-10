"""
Pluggable ingestion source. The rest of the pipeline does not care where text
came from - stream_posts() always yields the same {"text","username","platform",
"source"} shape regardless of origin, so a future news/tip/other-platform source
can be added without touching any downstream agent.

Apify path: starts a run of the configured actor and waits for its results
(run-sync-get-dataset-items), so a demo needs no manual run on apify.com first.
Default actor is the official apify/facebook-posts-scraper (public pages only).

Settings (MUST_backend/.env, loaded by agents/__init__.py):
  APIFY_API_TOKEN   - required for live scraping
  APIFY_ACTOR_ID    - optional, defaults to apify/facebook-posts-scraper
  APIFY_START_URLS  - comma-separated public Facebook page URLs

source="apify" surfaces Apify errors to the caller; source="auto" falls back to
the sample posts, and the response's source_used says which one actually ran.
"""
import itertools
import logging
import os

import requests

DEFAULT_ACTOR = "apify/facebook-posts-scraper"

# Cost guardrail. resultsLimit is applied per page URL, and the default actor
# bills about USD 0.005 per post on Apify's free tier - so one demo run against
# one page costs at most ~USD 0.25.
MAX_RESULTS = 50

# Expanded from fastapp1.py's 5-sentence /scrape mock so the sample fallback
# exercises every graph branch (neutral, offensive, hate, sarcasm markers, and
# a repeated hate-speech line across 3 usernames to trigger campaign_flag).
SAMPLE_POSTS = [
    {"text": "I love this beautiful day!", "username": "cool_guy", "platform": "Facebook"},
    {"text": "Just had a great meal with family.", "username": "cool_guy", "platform": "Facebook"},
    {"text": "These people are ruining our country, they should be kicked out.", "username": "angry_user1", "platform": "Facebook"},
    {"text": "These people are ruining our country, they should be kicked out.", "username": "angry_user2", "platform": "Facebook"},
    {"text": "These people are ruining our country, they should be kicked out.", "username": "angry_user3", "platform": "Facebook"},
    {"text": "Why are they so stupid? I hate them.", "username": "angry_bird", "platform": "Facebook"},
    {"text": "Oh sure, great job ruining everything, real geniuses over there.", "username": "sarcastic1", "platform": "Facebook"},
    {"text": "Working hard on my new project.", "username": "dev_guy", "platform": "Facebook"},
]


class ApifyError(RuntimeError):
    """A live-ingestion failure with a message safe to show in the dashboard."""


def apify_config():
    """Read at call time, not import time, so tests and restarts see current values."""
    token = os.environ.get("APIFY_API_TOKEN", "").strip()
    actor = os.environ.get("APIFY_ACTOR_ID", "").strip() or DEFAULT_ACTOR
    urls = [u.strip() for u in os.environ.get("APIFY_START_URLS", "").split(",") if u.strip()]
    return token, actor, urls


def stream_posts(limit: int = 20, source: str = "auto", urls=None):
    """Generator yielding {"text","username","platform","source"} dicts."""
    if source != "sample":
        token, actor, default_urls = apify_config()
        if source == "apify" or token:
            try:
                posts = list(_fetch_from_apify(limit, urls or default_urls, token, actor))
            except ApifyError as e:
                if source == "apify":
                    raise
                logging.error(f"Apify ingestion failed, falling back to sample data: {e}")
            else:
                yield from posts
                return
    yield from _stream_from_sample(limit)


def _fetch_from_apify(limit: int, urls, token: str, actor: str):
    if not token:
        raise ApifyError("APIFY_API_TOKEN is not set. Add it to MUST_backend/.env and restart the backend.")
    if not urls:
        raise ApifyError(
            "No Facebook page URLs to scrape. Set APIFY_START_URLS in MUST_backend/.env "
            "or pass urls in the request."
        )

    n = max(1, min(int(limit), MAX_RESULTS))
    endpoint = f"https://api.apify.com/v2/acts/{actor.replace('/', '~')}/run-sync-get-dataset-items"
    payload = {"startUrls": [{"url": u} for u in urls], "resultsLimit": n}
    try:
        # Token in a header, not the query string, so it never lands in access logs.
        resp = requests.post(
            endpoint,
            json=payload,
            headers={"Authorization": f"Bearer {token}"},
            params={"timeout": 240},
            timeout=300,
        )
    except requests.RequestException as e:
        raise ApifyError(f"Could not reach Apify: {e}") from e

    if resp.status_code == 401:
        raise ApifyError("Apify rejected the token (401). Check APIFY_API_TOKEN in MUST_backend/.env.")
    if resp.status_code == 404:
        raise ApifyError(f"Apify actor '{actor}' was not found (404). Check APIFY_ACTOR_ID.")
    if resp.status_code == 408:
        raise ApifyError("The Apify run took longer than 5 minutes. Try fewer page URLs or a lower limit.")
    if resp.status_code >= 400:
        raise ApifyError(f"Apify returned HTTP {resp.status_code}: {resp.text[:200]}")

    yield from parse_items(resp.json()[: n * len(urls)])


def parse_items(items):
    """Normalise Apify Facebook dataset items into pipeline posts (skips items with no text)."""
    for item in items:
        text = item.get("text") or item.get("message") or item.get("post_text") or ""
        user_data = item.get("user") or item.get("author") or {}
        username = (
            user_data.get("name")
            or user_data.get("username")
            or item.get("userName")
            or item.get("pageName")
            or "Facebook User"
        )
        if text:
            yield {"text": text, "username": username, "platform": "Facebook", "source": "apify"}


def _stream_from_sample(limit: int):
    for post in itertools.islice(itertools.cycle(SAMPLE_POSTS), limit):
        yield {**post, "source": "sample"}
