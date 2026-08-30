---
title: Sentinel GB Backend
emoji: 🛰️
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# Sentinel — GB Hate Speech Monitoring Backend

FastAPI backend for the Sentinel multi-agent hate-speech monitoring pipeline (Gilgit-Baltistan).
Wraps a fine-tuned XLM-RoBERTa classifier with a LangGraph orchestration layer: sarcasm
heuristic, clustering-based campaign detection, legal-provision mapping (PPC/PECA/Constitution),
and a human-gated escalation/review-queue system — no agent in this pipeline ever takes
autonomous enforcement action.

## Endpoints

- `POST /analyze` — raw single-item classification (existing, unmodified contract)
- `POST /process` — single item through the full agent pipeline
- `POST /ingest-and-process` — batch ingestion + full pipeline
- `GET /review-queue`, `POST /review-queue/{id}/decision` — human review workflow
- `GET /stats/districts`, `GET /stats/platforms`, `GET /legal-reference`
- `POST /api/webhooks/apify` — Apify scraper webhook receiver

## Configuration (Space secrets)

Set under Space Settings → Repository secrets:

- `SENTINEL_API_KEY` — **required for this to be a public Space.** A shared-secret gate
  (see `agents/auth.py`) on every sensitive endpoint (`/trends`, `/flagged`, `/monitoring`,
  `/live-feed`, `/scrape`, `/process`, `/ingest-and-process`, `/review-queue*`,
  `/stats/*`, `/api/webhooks/apify`). Only `/analyze`, `GET /`, and `/legal-reference`
  stay open. If this secret is unset, the gate is a no-op and everything is open — this
  is a basic access gate against casual/opportunistic access, not real per-user auth, since
  the key is visible in the frontend's client bundle. The frontend's
  `NEXT_PUBLIC_SENTINEL_API_KEY` must match this value exactly.
- `APIFY_API_TOKEN` / `APIFY_ACTOR_ID` — optional; the ingestion agent falls back to
  sample data if unset.

## Storage note

Case files, review-queue entries, and logs are stored in a local SQLite database
(`hatespeech.db`) created on first startup. On the free CPU tier this storage is
**ephemeral** — it resets whenever the Space rebuilds or restarts. Enable persistent
storage in Space settings if case history needs to survive restarts.
