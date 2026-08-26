"""
Eval harness for the multi-agent pipeline. Run from MUST_backend/:

    python -m eval.run_eval

Reports (informational, not a hard pass/fail gate - the fixture is a small
hand-built proxy set, not the original model test split, so its accuracy
figure is not a reproduction of the published 95.7% XLM-RoBERTa result):
  - classifier accuracy on the fixture set
  - sarcasm-agent trigger-routing correctness (did it fire exactly when the
    ambiguous confidence band was hit?)
  - clustering agent's campaign-detection on a synthetic 3+ near-duplicate group
  - legal-mapping agent's match rate on hate/offensive fixture items
  - escalation invariant: every case file that reached escalation has
    requires_human_review == True, checked in both the returned state and
    the persisted case_files row (a direct test of the structural guarantee
    in agents/escalation_agent.py)

The one hard assertion is the escalation invariant - everything else is
reported, since accuracy/trigger-rate/match-rate numbers are meant to guide
future improvement, not gate this pass.
"""
import asyncio
import json
import os
import sqlite3

from agents.orchestrator import run_pipeline, SARCASM_LOW, SARCASM_HIGH

FIXTURE_PATH = os.path.join(os.path.dirname(__file__), "fixtures", "labeled_samples.json")
REPORT_PATH = os.path.join(os.path.dirname(__file__), "last_run_report.json")
DB_NAME = "hatespeech.db"


def _case_file_requires_review(case_file_id: int) -> bool:
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT requires_human_review FROM case_files WHERE id = ?", (case_file_id,))
    row = c.fetchone()
    conn.close()
    return bool(row and row[0] == 1)


async def main():
    with open(FIXTURE_PATH, "r", encoding="utf-8") as f:
        fixtures = json.load(f)

    results = []
    for item in fixtures:
        state = await run_pipeline(text=item["text"], username="eval_user", platform="Eval", source="eval")
        results.append({"fixture": item, "state": state})

    # --- classifier accuracy (informational) ---
    scored = [r for r in results if not r["state"].get("error")]
    correct = sum(1 for r in scored if r["state"].get("category") == r["fixture"]["expected_category"])
    accuracy = correct / len(scored) if scored else 0.0

    # --- sarcasm trigger routing correctness ---
    sarcasm_expected = [r for r in scored if SARCASM_LOW <= r["state"].get("confidence_frac", 0.0) <= SARCASM_HIGH]
    sarcasm_fired_correctly = sum(1 for r in sarcasm_expected if r["state"].get("sarcasm_score") is not None)
    sarcasm_trigger_rate = (
        sarcasm_fired_correctly / len(sarcasm_expected) if sarcasm_expected else None
    )

    # --- clustering campaign detection ---
    campaign_group = [r for r in results if r["fixture"].get("group") == "campaign_test"]
    campaign_detected = any(r["state"].get("campaign_flag") for r in campaign_group)

    # --- legal mapping match rate ---
    flagged_hate_offensive = [
        r for r in scored if r["state"].get("category") in ("hate", "offensive") and r["state"].get("requires_human_review")
    ]
    mapped = sum(
        1
        for r in flagged_hate_offensive
        if r["state"].get("legal_matches")
        and r["state"]["legal_matches"][0].get("id") != "unmapped"
    )
    legal_match_rate = mapped / len(flagged_hate_offensive) if flagged_hate_offensive else None

    # --- escalation invariant (hard assertion) ---
    escalated = [r for r in results if r["state"].get("case_file_id") is not None]
    for r in escalated:
        assert r["state"].get("requires_human_review") is True, (
            f"requires_human_review was not True in-state for case_file_id="
            f"{r['state'].get('case_file_id')}"
        )
        assert _case_file_requires_review(r["state"]["case_file_id"]), (
            f"requires_human_review was not 1 in the case_files row for "
            f"case_file_id={r['state']['case_file_id']}"
        )

    report = {
        "fixture_size": len(fixtures),
        "classifier_accuracy_on_fixture": round(accuracy, 3),
        "accuracy_note": "Small hand-built proxy fixture, not the original model test split - not a reproduction of the published 95.7% figure.",
        "sarcasm_trigger_rate": round(sarcasm_trigger_rate, 3) if sarcasm_trigger_rate is not None else None,
        "sarcasm_ambiguous_band_size": len(sarcasm_expected),
        "clustering_campaign_detected": campaign_detected,
        "legal_mapping_match_rate": round(legal_match_rate, 3) if legal_match_rate is not None else None,
        "legal_mapping_flagged_items": len(flagged_hate_offensive),
        "escalation_invariant_checked_count": len(escalated),
        "escalation_invariant_passed": True,
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("=== Sentinel pipeline eval report ===")
    for k, v in report.items():
        print(f"{k}: {v}")
    print(f"\nFull report written to {REPORT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
