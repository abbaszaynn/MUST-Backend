"""
Pluggable ingestion source. The rest of the pipeline does not care where text
came from - stream_posts() always yields the same {"text","username","platform",
"source"} shape regardless of origin, so a future news/tip/other-platform source
can be added without touching any downstream agent.

Config via APIFY_API_TOKEN / APIFY_ACTOR_ID env vars (both blank until filled
in). Targets a Facebook Pages/Groups scraper actor, matching the existing
/api/webhooks/apify receiver's field-extraction assumptions (text/message/
post_text, user.name/author.username/userName, platform="Facebook"). Falls
back to local sample data whenever Apify isn't configured or the call fails,
so local dev/testing never depends on a live token.
"""
import itertools
import logging
import os

import requests

APIFY_API_TOKEN = os.environ.get("APIFY_API_TOKEN", "")
APIFY_ACTOR_ID = os.environ.get("APIFY_ACTOR_ID", "")

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


def stream_posts(limit: int = 20, source: str = "auto"):
    """Generator yielding {"text","username","platform","source"} dicts."""
    if source in ("auto", "apify") and APIFY_API_TOKEN and APIFY_ACTOR_ID:
        try:
            yield from _stream_from_apify(limit)
            return
        except Exception as e:
            logging.error(f"Apify ingestion failed, falling back to sample data: {e}")
    yield from _stream_from_sample(limit)


def _stream_from_apify(limit: int):
    url = f"https://api.apify.com/v2/acts/{APIFY_ACTOR_ID}/runs/last/dataset/items"
    resp = requests.get(url, params={"token": APIFY_API_TOKEN, "limit": limit}, timeout=30)
    resp.raise_for_status()
    for item in resp.json():
        text = item.get("text") or item.get("message") or item.get("post_text") or ""
        user_data = item.get("user") or item.get("author") or {}
        username = (
            user_data.get("name")
            or user_data.get("username")
            or item.get("userName")
            or "Facebook User"
        )
        if text:
            yield {"text": text, "username": username, "platform": "Facebook", "source": "apify"}


def _stream_from_sample(limit: int):
    for post in itertools.islice(itertools.cycle(SAMPLE_POSTS), limit):
        yield {**post, "source": "sample"}
