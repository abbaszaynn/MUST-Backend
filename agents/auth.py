"""
Lightweight shared-secret gate for the public deployment. This is a basic access
gate, not real per-user authentication - it deters casual/opportunistic access
(search engines, scanners, someone stumbling on the URL) but a shared key sent by
a public frontend is inherently visible to anyone inspecting that frontend's
network requests. Treat it as a doorbell, not a lock, and don't put anything
here that needs stronger protection without adding real per-user auth first.

If SENTINEL_API_KEY is unset (e.g. local dev), the gate is a no-op so nothing
breaks without configuration - it only enforces once the operator sets the
secret (Space Settings -> Repository secrets for a Hugging Face deployment).
"""
import os
import secrets

from fastapi import Header, HTTPException


def require_api_key(x_api_key: str | None = Header(default=None)):
    expected = os.environ.get("SENTINEL_API_KEY", "")
    if not expected:
        return  # no key configured - permissive local-dev default
    if not x_api_key or not secrets.compare_digest(x_api_key, expected):
        raise HTTPException(status_code=401, detail="Missing or invalid X-API-Key header")
