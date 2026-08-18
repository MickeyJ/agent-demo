"""Seed the mini-RAG knowledge base with mock company-wiki docs (idempotent —
ingest replaces by title). Facts are distinctive on purpose so retrieval demos
are unambiguous: the model cannot know Dicecorp's mascot by pretraining.

Usage: uv run python scripts/seed_docs.py
"""

from __future__ import annotations

from playtime.rag import ingest, list_docs

DOCS = {
    "Dicecorp — Company Overview": """\
Dicecorp builds precision randomness infrastructure. Founded in 2019 in Reno,
Nevada by Ada Vantis and Jun Okafor, the company employs 42 people.

Our flagship product is D20 Cloud, an API that serves cryptographically fair
dice rolls. The free tier allows 1,000 rolls per month; the Pro plan is $29
per month for unlimited rolls and webhook delivery.

The company mascot is an axolotl named Crit. Crit lives in a tank in the Reno
office lobby and has his own Slack channel, #crit-pics.""",
    "HR — Time Off Policy": """\
Full-time Dicecorp employees receive 23 days of paid time off per year, plus
the week between Christmas and New Year's when the office closes entirely.

PTO requests go through the Lodestone HR portal and need manager approval only
for stretches longer than 5 consecutive days. Unused PTO rolls over, capped at
10 carried days.

Sick days are unlimited and tracked separately — do not log them as PTO.""",
    "IT — Office Network": """\
The office WiFi network is DICECORP-5G; the guest network is DICECORP-GUEST
with password rollforinitiative (all lowercase, one word).

VPN access uses Tailscale — install it from the self-service portal and sign
in with your @dicecorp.com Google account. The old OpenVPN endpoint was
decommissioned in March 2026.

Printer: the 3rd-floor printer is named GAZEBO. It jams if you feed it labels.""",
    "Engineering — On-call Runbook": """\
The on-call rotation is weekly, Monday 09:00 to Monday 09:00, managed in
PagerDuty. Primary gets paged first; secondary after 10 unacknowledged minutes.

Sev1 (API down): page the IC immediately, open a #incident channel, status
page update within 15 minutes. The D20 Cloud API's error budget is 43 minutes
of downtime per month.

Post-incident reviews happen within 3 business days and are blameless.""",
    "Product — D20 Cloud API": """\
D20 Cloud serves fair dice rolls over HTTPS. Endpoint: POST /v1/roll with
JSON body {"sides": 20, "count": 2}. Responses include a verifiable fairness
proof under the "attest" key.

Rate limits: free tier 10 requests/second, Pro tier 200 requests/second.
Latency SLO is p99 under 80 ms from US regions.

The April 2026 fairness audit by Untrusted Third Party LLC passed with zero
findings.""",
    "Facilities — Lunch & Snacks": """\
Catered lunch arrives Tuesdays and Thursdays at 12:15. Dietary preferences are
collected in the #lunch channel pinned form.

The espresso machine (nicknamed The Kraken) is maintained by whoever broke it
last. Descale it monthly; instructions are taped inside the left cabinet.

Snack budget is $400 per month per floor. Crit's brine shrimp do not come out
of the snack budget.""",
}


def main() -> None:
    for title, text in DOCS.items():
        report = ingest(title, text)
        print(f"  ingested {report['title']!r}: {report['chunks']} chunks")
    print("\nknowledge base now holds:")
    for d in list_docs():
        print(f"  {d['title']} ({d['chunks']} chunks)")


if __name__ == "__main__":
    main()
