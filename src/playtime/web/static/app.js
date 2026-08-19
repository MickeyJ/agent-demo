/* agent-playtime demo UI — vanilla JS, ~250 lines, no dependencies.
   One idea to keep in view: this file only *renders* what the backend streams;
   every agent concept (tools, approvals, trimming) arrives as an SSE event. */

const $ = (id) => document.getElementById(id);
let busy = false;

/* Each browser gets its own sticky thread (concurrent visitors must not share
   t1 — two people interleaving one checkpoint thread is chaos). */
let thread = localStorage.getItem("playtime-thread");
if (!thread) {
  thread = "w-" + Math.random().toString(36).slice(2, 7);
  localStorage.setItem("playtime-thread", thread);
}
const rememberThread = () => localStorage.setItem("playtime-thread", thread);

/* ---------- tiny helpers ---------- */

const esc = (s) =>
  s.replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;");

// minimal markdown: **bold**, `code` — enough for chat replies
const md = (s) =>
  esc(s)
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/`([^`]+)`/g, "<code>$1</code>");

function el(html) {
  const t = document.createElement("template");
  t.innerHTML = html.trim();
  return t.content.firstChild;
}

/* Epoch seconds → "8/18 1:07:23 pm" in the viewer's own timezone. */
function fmtTs(ts) {
  const d = ts ? new Date(ts * 1000) : new Date();
  const pad = (n) => String(n).padStart(2, "0");
  const h24 = d.getHours();
  const h = h24 % 12 || 12;
  return `${d.getMonth() + 1}/${d.getDate()} ${h}:${pad(d.getMinutes())}:${pad(
    d.getSeconds()
  )} ${h24 < 12 ? "am" : "pm"}`;
}

/* A chat bubble is a timestamp line + the bubble itself. */
function bubble(who, html, ts) {
  const row = el(`<div class="turn ${who}"><div class="ts">${fmtTs(ts)}</div></div>`);
  row.appendChild(el(`<div class="msg ${who}">${html}</div>`));
  return row;
}

function addToChat(node) {
  $("messages").appendChild(node);
  $("messages").scrollTop = $("messages").scrollHeight;
  return node;
}

/* ---------- rendering turn events ---------- */

const pendingChips = []; // tool chips awaiting their result, in call order

function renderEvent(ev) {
  if (ev.type === "tool_call") {
    const chip = addToChat(
      el(`<details class="tool"><summary>${esc(ev.name)}(${esc(
        JSON.stringify(ev.args)
      )})</summary><pre>running…</pre></details>`)
    );
    pendingChips.push(chip);
  } else if (ev.type === "tool_result") {
    const chip = pendingChips.shift();
    if (chip) {
      chip.querySelector("pre").textContent = ev.text;
      chip.classList.add("done");
    }
  } else if (ev.type === "assistant") {
    addToChat(bubble("bot", md(ev.text), ev.ts));
  } else if (ev.type === "trim") {
    addToChat(
      el(`<div class="trim-note">folded ${ev.folded} old messages into the summary</div>`)
    );
  } else if (ev.type === "approval") {
    renderApprovalCard(ev.card);
  }
}

function renderApprovalCard(card, resolved = null) {
  const node = addToChat(
    el(`<div class="approval">
      <strong>Approval needed</strong>
      <div class="card-line">${esc(card)}</div>
      <div class="buttons">
        <button class="approve">Approve</button>
        <button class="decline">Decline</button>
      </div>
    </div>`)
  );
  const settle = (decision) => {
    node.classList.add("resolved");
    node.querySelector(".buttons").replaceWith(
      el(`<div class="outcome">${decision === "approve" ? "approved" : "declined"}</div>`)
    );
    streamTurn("/api/resume", { thread, decision });
  };
  node.querySelector(".approve").onclick = () => settle("approve");
  node.querySelector(".decline").onclick = () => settle("decline");
  return node;
}

/* ---------- SSE plumbing ---------- */

async function streamTurn(url, body) {
  busy = true;
  $("send").disabled = true;
  try {
    const resp = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      let idx;
      while ((idx = buf.indexOf("\n\n")) >= 0) {
        const frame = buf.slice(0, idx);
        buf = buf.slice(idx + 2);
        if (frame.startsWith("data: ")) renderEvent(JSON.parse(frame.slice(6)));
      }
    }
  } catch (e) {
    addToChat(el(`<div class="trim-note">error: ${esc(String(e))}</div>`));
  } finally {
    busy = false;
    $("send").disabled = false;
    pendingChips.length = 0;
    refreshContext();
  }
}

/* ---------- composer ---------- */

$("composer").onsubmit = (e) => {
  e.preventDefault();
  const text = $("prompt").value.trim();
  if (!text || busy) return;
  $("prompt").value = "";
  addToChat(bubble("user", esc(text)));
  streamTurn("/api/chat", { thread, message: text });
};

/* ---------- threads ---------- */

async function loadThreads() {
  const { threads } = await (await fetch("/api/threads")).json();
  if (!threads.includes(thread)) threads.push(thread); // ours may be unborn
  $("thread-select").innerHTML = threads
    .sort()
    .map((t) => `<option ${t === thread ? "selected" : ""}>${esc(t)}</option>`)
    .join("");
}

$("thread-select").onchange = async (e) => {
  thread = e.target.value;
  rememberThread();
  await renderTranscript();
};

$("new-thread").onclick = async () => {
  thread = "w-" + Math.random().toString(36).slice(2, 7);
  rememberThread();
  await loadThreads();
  $("messages").innerHTML = "";
  refreshContext();
  $("prompt").focus();
};

$("panel-toggle").onclick = () => $("sidebar").classList.toggle("open");

async function renderTranscript() {
  const s = await (await fetch(`/api/state/${encodeURIComponent(thread)}`)).json();
  $("messages").innerHTML = "";
  if (s.summary)
    addToChat(el(`<div class="trim-note">older turns live in the summary (Context tab)</div>`));
  for (const m of s.messages) {
    if (m.kind === "human") addToChat(bubble("user", esc(m.text), m.ts));
    else if (m.kind === "ai") {
      for (const c of m.tool_calls)
        addToChat(
          el(`<details class="tool done"><summary>${esc(c.name)}(${esc(
            JSON.stringify(c.args)
          )})</summary><pre>(see next result)</pre></details>`)
        );
      if (m.text) addToChat(bubble("bot", md(m.text), m.ts));
    } else if (m.kind === "tool") {
      const chips = [...$("messages").querySelectorAll("details.tool")];
      const open = chips.reverse().find((c) => c.querySelector("pre").textContent === "(see next result)");
      if (open) open.querySelector("pre").textContent = m.text;
    }
  }
  if (s.pending_approval)
    addToChat(el(`<div class="trim-note">an approval is pending on this thread — resend or answer it below if a card is shown</div>`));
  applyContext(s);
}

/* ---------- context tab ---------- */

function applyContext(s) {
  const pct = Math.min(100, Math.round((s.window_tokens / s.budget) * 100));
  $("gauge-fill").style.width = pct + "%";
  $("gauge-fill").classList.toggle("high", pct > 85);
  $("gauge-label").textContent = `${s.window_tokens} / ${s.budget} tokens (~${pct}%), ${s.messages.length} messages in window`;
  $("pinned").innerHTML = Object.keys(s.pinned).length
    ? Object.entries(s.pinned).map(([k, v]) => `<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`).join("")
    : `none yet — ask the agent to remember something`;
  $("summary").textContent = s.summary || "empty — appears once old turns get trimmed";
}

async function refreshContext() {
  const s = await (await fetch(`/api/state/${encodeURIComponent(thread)}`)).json();
  applyContext(s);
  loadThreads(); // a first message may have created the thread
}

/* ---------- tabs ---------- */

for (const btn of document.querySelectorAll(".tab"))
  btn.onclick = () => {
    document.querySelectorAll(".tab, .tab-panel").forEach((n) => n.classList.remove("active"));
    btn.classList.add("active");
    $(`tab-${btn.dataset.tab}`).classList.add("active");
    if (btn.dataset.tab === "docs") refreshDocs();
  };

/* ---------- docs tab ---------- */

async function refreshDocs() {
  const { docs } = await (await fetch("/api/docs")).json();
  $("doc-list").innerHTML = docs.length
    ? docs
        .map(
          (d) => `<li><span>${esc(d.title)}</span>
            <span class="chunks">${d.chunks} chunk${d.chunks === 1 ? "" : "s"}</span>
            <button data-del="${esc(d.title)}">✕</button></li>`
        )
        .join("")
    : `<li class="muted">no documents yet</li>`;
  for (const b of $("doc-list").querySelectorAll("button[data-del]"))
    b.onclick = async () => {
      await fetch(`/api/docs/${encodeURIComponent(b.dataset.del)}`, { method: "DELETE" });
      refreshDocs();
    };
}

$("doc-form").onsubmit = async (e) => {
  e.preventDefault();
  const fd = new FormData();
  fd.append("title", $("doc-title").value);
  fd.append("text", $("doc-text").value);
  const f = $("doc-file").files[0];
  if (f) fd.append("file", f);
  $("doc-status").textContent = "chunking + embedding…";
  const resp = await fetch("/api/docs", { method: "POST", body: fd });
  if (resp.ok) {
    const r = await resp.json();
    $("doc-status").textContent = `✓ ingested "${r.title}" as ${r.chunks} chunk${r.chunks === 1 ? "" : "s"}`;
    $("doc-title").value = $("doc-text").value = "";
    $("doc-file").value = "";
    refreshDocs();
  } else {
    $("doc-status").textContent = `✗ ${(await resp.json()).detail}`;
  }
};

let searchTimer;
$("search-q").oninput = () => {
  clearTimeout(searchTimer);
  searchTimer = setTimeout(async () => {
    const q = $("search-q").value.trim();
    if (!q) return ($("search-hits").innerHTML = "");
    const { hits } = await (await fetch(`/api/docs/search?q=${encodeURIComponent(q)}`)).json();
    $("search-hits").innerHTML = hits
      .map(
        (h) => `<div class="hit">
          <span class="score">${h.score.toFixed(2)}</span>
          <span class="src">${esc(h.title)} §${h.seq}</span>
          <p>${esc(h.text)}</p></div>`
      )
      .join("");
  }, 300);
};
$("search-form").onsubmit = (e) => e.preventDefault();

/* ---------- boot ---------- */

(async () => {
  await loadThreads();
  await renderTranscript();
})();
