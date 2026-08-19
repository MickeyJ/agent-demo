"""Stage 7: the demo web UI — FastAPI wrapped around the same graph the REPL uses.

Nothing agent-side changed for this stage; that's the point. The graph, tools,
context machinery, and checkpoints are channel-agnostic — the web UI and the
CLI REPL are two skins over the same var/checkpoints.sqlite threads (open a
thread in both and they see the same conversation).

Turn delivery is Server-Sent Events: one POST /api/chat streams the whole turn
as JSON events (tool_call / tool_result / assistant / trim / approval / done),
which is run_turn() from main.py translated to the wire. An approval pauses the
graph exactly like the REPL's y/N prompt — the card renders as buttons, and
POST /api/resume streams the continuation.

Run: uv run playtime-web  →  http://127.0.0.1:9020 (host-only, demo, no auth)
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.types import Command
from pydantic import BaseModel

from playtime.context import count_message_tokens
from playtime.graph import build_graph, stamp, thread_config
from playtime.rag import delete_doc, ingest, list_docs, search
from playtime.settings import get_settings

STATIC = Path(__file__).parent / "static"
NO_CACHE = {"Cache-Control": "no-cache, must-revalidate"}

app = FastAPI(title="agent-playtime")
graph = build_graph()


# --- chat --------------------------------------------------------------------


class ChatIn(BaseModel):
    thread: str
    message: str


class ResumeIn(BaseModel):
    thread: str
    decision: str  # "approve" | "decline"


def _turn_events(thread: str, payload):
    """The REPL's run_turn, emitting JSON events instead of printing."""
    card = None
    for step in graph.stream(payload, thread_config(thread), stream_mode="updates"):
        for node, update in step.items():
            if node == "__interrupt__":
                card = update[0].value
                continue
            if not update:
                continue
            if node == "prepare_context":
                if "summary" in update:
                    yield {"type": "trim", "folded": len(update.get("messages", []))}
                continue
            for m in update.get("messages", []):
                if isinstance(m, AIMessage):
                    for c in m.tool_calls:
                        yield {"type": "tool_call", "name": c["name"], "args": c["args"]}
                    if m.text:
                        yield {
                            "type": "assistant",
                            "text": m.text,
                            "ts": m.additional_kwargs.get("ts"),
                        }
                elif isinstance(m, ToolMessage):
                    yield {"type": "tool_result", "name": m.name, "text": m.text}
    if card is not None:
        yield {"type": "approval", "card": card["card"]}
    yield {"type": "done"}


def _sse(events):
    for ev in events:
        yield f"data: {json.dumps(ev)}\n\n"


@app.post("/api/chat")
def chat(body: ChatIn):
    payload = {"messages": [stamp(HumanMessage(body.message))]}
    return StreamingResponse(
        _sse(_turn_events(body.thread, payload)), media_type="text/event-stream"
    )


@app.post("/api/resume")
def resume(body: ResumeIn):
    decision = "approve" if body.decision == "approve" else "decline"
    return StreamingResponse(
        _sse(_turn_events(body.thread, Command(resume=decision))),
        media_type="text/event-stream",
    )


# --- threads & context state ---------------------------------------------------


@app.get("/api/threads")
def threads():
    with sqlite3.connect("var/checkpoints.sqlite") as conn:
        try:
            rows = conn.execute("SELECT DISTINCT thread_id FROM checkpoints").fetchall()
        except sqlite3.OperationalError:  # no table yet: fresh install
            rows = []
    return {"threads": sorted(r[0] for r in rows)}


@app.get("/api/state/{thread}")
def state(thread: str):
    """Everything the Context tab shows, plus the window messages so the UI can
    rebuild a transcript on thread switch. Window only — evicted turns live on
    solely inside the summary; that lossiness is the stage 4 lesson, visible."""
    snap = graph.get_state(thread_config(thread))
    vals = snap.values or {}
    msgs = vals.get("messages", [])
    return {
        "pinned": vals.get("pinned", {}),
        "summary": vals.get("summary", ""),
        "budget": get_settings().ctx_window_budget,
        "window_tokens": count_message_tokens(msgs) if msgs else 0,
        "pending_approval": "approval" in (snap.next or ()),
        "messages": [
            {
                "kind": type(m).__name__.removesuffix("Message").lower(),
                "tokens": count_message_tokens([m]),
                "text": m.text,
                "ts": m.additional_kwargs.get("ts"),
                "tool_calls": [
                    {"name": c["name"], "args": c["args"]}
                    for c in (getattr(m, "tool_calls", None) or [])
                ],
            }
            for m in msgs
        ],
    }


# --- docs (the RAG pipeline, exposed) ------------------------------------------


@app.get("/api/docs")
def docs_list():
    return {"docs": list_docs()}


@app.post("/api/docs")
def docs_add(
    file: UploadFile | None = File(None),
    title: str = Form(""),
    text: str = Form(""),
):
    """Paste text or upload a .txt/.md file — same pipeline either way."""
    if file is not None and file.filename:
        if not file.filename.lower().endswith((".txt", ".md", ".markdown")):
            raise HTTPException(400, "only .txt / .md files in this demo")
        text = file.file.read().decode("utf-8", errors="replace")
        title = title.strip() or Path(file.filename).stem
    if not text.strip():
        raise HTTPException(400, "no text: paste some or attach a file")
    return ingest(title.strip() or "untitled", text)


@app.delete("/api/docs/{title}")
def docs_delete(title: str):
    if not delete_doc(title):
        raise HTTPException(404, f"no doc titled {title!r}")
    return {"deleted": title}


@app.get("/api/docs/search")
def docs_search(q: str):
    """Raw retrieval, no agent — the Docs tab's test-search box."""
    return {"hits": search(q, k=3)}


# --- static shell ---------------------------------------------------------------


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html", headers=NO_CACHE)


class NoCacheStatic(StaticFiles):
    """Demo-grade cache policy: always revalidate. A browser holding an old
    app.js against a new style.css renders a broken page (the bubbles lose
    their alignment) — for files this small, correctness beats caching."""

    def file_response(self, *args, **kw):
        resp = super().file_response(*args, **kw)
        resp.headers.update(NO_CACHE)
        return resp


app.mount("/static", NoCacheStatic(directory=STATIC), name="static")


def main() -> None:
    uvicorn.run(app, host="127.0.0.1", port=get_settings().web_port, log_level="warning")


if __name__ == "__main__":
    main()
