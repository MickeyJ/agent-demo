"""End-to-end checks for the demo web UI — run against a live instance.

Usage:
    uv run python scripts/e2e_web.py [base_url] [--concurrency N]

Defaults to http://127.0.0.1:9020. Exercises the real stack (LLM, embeddings,
graph, SQLite), so it needs the router up; each check prints PASS/FAIL and the
script exits non-zero on any failure.

Checks: static shell → RAG ingest/search → a chat turn that must retrieve →
approval decline and approve paths → context state → N concurrent chat turns
on distinct threads (the local-model concurrency story).
"""

from __future__ import annotations

import json
import sys
import threading
import time
import uuid

import httpx

BASE = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else "http://127.0.0.1:9020"
CONCURRENCY = int(sys.argv[sys.argv.index("--concurrency") + 1]) if "--concurrency" in sys.argv else 4

client = httpx.Client(base_url=BASE, timeout=180.0)
failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f" — {detail}" if detail and not ok else ""))
    if not ok:
        failures.append(name)


def sse_events(path: str, body: dict) -> list[dict]:
    """POST and parse the full SSE stream into a list of event dicts."""
    events = []
    with client.stream("POST", path, json=body) as resp:
        resp.raise_for_status()
        buf = ""
        for chunk in resp.iter_text():
            buf += chunk
            while "\n\n" in buf:
                frame, buf = buf.split("\n\n", 1)
                if frame.startswith("data: "):
                    events.append(json.loads(frame[6:]))
    return events


def turn(thread: str, message: str) -> list[dict]:
    return sse_events("/api/chat", {"thread": thread, "message": message})


def texts(events: list[dict], type_: str) -> str:
    return " ".join(e.get("text", "") for e in events if e["type"] == type_)


def main() -> None:
    run_id = uuid.uuid4().hex[:6]
    doc_title = f"E2E Probe {run_id}"
    codeword = f"XYLOPHONE-{run_id}"

    print(f"e2e against {BASE}\n")

    # 1. static shell
    print("[shell]")
    check("GET / serves the page", "<title>agent-playtime</title>" in client.get("/").text)
    check("app.js served", client.get("/static/app.js").status_code == 200)
    check("style.css served", client.get("/static/style.css").status_code == 200)

    # 2. RAG pipeline via the API (the paste-text path)
    print("[rag]")
    r = client.post("/api/docs", data={"title": doc_title, "text": f"The secret probe codeword is {codeword}.\n\nProbe docs are temporary."})
    check("ingest pasted text", r.status_code == 200 and r.json()["chunks"] >= 1)
    hits = client.get("/api/docs/search", params={"q": "secret probe codeword"}).json()["hits"]
    check("raw search finds it", any(codeword in h["text"] for h in hits))
    check("doc listed", any(d["title"] == doc_title for d in client.get("/api/docs").json()["docs"]))

    # 3. a chat turn that must retrieve the fresh doc
    print("[chat+rag]")
    t = f"e2e-{run_id}"
    ev = turn(t, "what is the secret probe codeword? search the knowledge base.")
    check("agent called search_docs", any(e["type"] == "tool_call" and e["name"] == "search_docs" for e in ev))
    check("answer contains the codeword", codeword in texts(ev, "assistant"))
    check("stream ends with done", ev and ev[-1]["type"] == "done")

    # 4. approval: decline, then approve
    print("[approvals]")
    ev = turn(t, f"send an email to probe@example.com with subject Probe and body {codeword}")
    check("approval card raised", any(e["type"] == "approval" for e in ev))
    ev = sse_events("/api/resume", {"thread": t, "decision": "decline"})
    check("decline yields a reply, no send", bool(texts(ev, "assistant")) and "saved to" not in texts(ev, "tool_result"))
    ev = turn(t, "ok send that same email after all")
    if any(e["type"] == "approval" for e in ev):
        ev = sse_events("/api/resume", {"thread": t, "decision": "approve"})
        check("approve executes send_email", "saved to" in texts(ev, "tool_result"))
    else:
        check("approval card raised on retry", False, f"events: {[e['type'] for e in ev]}")

    # 5. context state
    print("[state]")
    s = client.get(f"/api/state/{t}").json()
    check("state has window+budget", s["window_tokens"] > 0 and s["budget"] > 0)
    check("transcript survives", any(m["kind"] == "human" for m in s["messages"]))

    # 6. concurrency — distinct threads hammering the model at once
    print(f"[concurrency x{CONCURRENCY}]")
    results: dict[int, tuple[bool, float]] = {}

    def worker(i: int) -> None:
        t0 = time.time()
        try:
            ev = turn(f"e2e-{run_id}-c{i}", f"roll {i + 2} dice and tell me the total")
            ok = any(e["type"] == "tool_call" and e["name"] == "roll_dice" for e in ev) and bool(
                texts(ev, "assistant")
            )
        except Exception:
            ok = False
        results[i] = (ok, time.time() - t0)

    t0 = time.time()
    threads = [threading.Thread(target=worker, args=(i,)) for i in range(CONCURRENCY)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    wall = time.time() - t0
    per = ", ".join(f"{results[i][1]:.1f}s" for i in sorted(results))
    check(
        f"all {CONCURRENCY} concurrent turns completed",
        all(ok for ok, _ in results.values()),
        f"per-turn: {per}",
    )
    print(f"        wall {wall:.1f}s, per-turn [{per}]")

    # cleanup
    client.delete(f"/api/docs/{doc_title}")

    print(f"\n{'ALL PASS' if not failures else f'{len(failures)} FAILURE(S): {failures}'}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
