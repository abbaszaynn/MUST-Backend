"""
Apify cost model, in one place because two callers need the same numbers:
scrape_jobs.py sizes each run's hard maxTotalChargeUsd cap from it, and the
Apify Records page reports what the collection has cost so far.

These are Apify's published pay-per-event prices for the two official Facebook
actors. Every figure derived from them is an ESTIMATE computed locally from
what we collected - it is not an amount billed back from Apify's billing API.
Anything shown to an officer from this module must say so.
"""

POST_PRICE_USD = 0.005
COMMENT_PRICE_USD = 0.0025
START_FEE_USD = 0.01

# Comments requested per post (see scrape_jobs.COMMENTS_PER_POST).
COMMENTS_PER_POST = 20

# The rate used throughout the Home Department proposal
# (PKR 29,894,666 = USD 106,767). Kept here so the dashboard and the costed
# proposal never quote two different conversions.
PKR_PER_USD = 280.0


def estimate_run_cost_usd(posts: int, comments: int) -> float:
    """What one scrape run cost, at list prices.

    A run that collected comments used two Apify actors (posts, then comments),
    so it paid two start fees; a posts-only run paid one.
    """
    posts = posts or 0
    comments = comments or 0
    phases = 2 if comments > 0 else 1
    return (
        posts * POST_PRICE_USD
        + comments * COMMENT_PRICE_USD
        + phases * START_FEE_USD
    )


def usd_to_pkr(usd: float) -> float:
    return usd * PKR_PER_USD
