import asyncio
import json
import os
import queue
import re
import threading
from pathlib import Path
from typing import Any, Callable

from flask import Flask, Response, jsonify, render_template_string, request, stream_with_context

try:
    from backboard import BackboardClient
except ImportError:  # pragma: no cover - handled at runtime with a readable error
    BackboardClient = None


APP_DIR = Path(__file__).resolve().parent
AGENTS_FILE = APP_DIR / "agents.json"

PRO_NAME = "PRO AGENT"
CON_NAME = "CONS AGENT"
JUDGE_NAME = "JUDGE"


PRO_PROMPT = """
You are PRO AGENT in Debate Arena. Your job is to argue FOR the user's submitted
question. Determine the strongest, most charitable meaning of the pro position,
then present exactly two strong arguments supporting that pro side.

Every argument must include 3-4 credible source links. Use web research when it
is enabled for the turn. Return only valid JSON with this schema:
{
  "side": "pro",
  "position_summary": "...",
  "arguments": [
    {
      "title": "...",
      "claim": "...",
      "evidence": "...",
      "sources": [{"title": "...", "url": "https://..."}]
    }
  ]
}
""".strip()

CON_PROMPT = """
You are CONS AGENT in Debate Arena. Your job is to argue AGAINST the user's
submitted question. Determine the strongest, most charitable meaning of the con
position, then present exactly two strong arguments supporting that con side.

Every argument must include 3-4 credible source links. Use web research when it
is enabled for the turn. Return only valid JSON with this schema:
{
  "side": "con",
  "position_summary": "...",
  "arguments": [
    {
      "title": "...",
      "claim": "...",
      "evidence": "...",
      "sources": [{"title": "...", "url": "https://..."}]
    }
  ]
}
""".strip()

JUDGE_PROMPT = """
You are JUDGE in Debate Arena. Review the arguments and evidence from both
sides. Decide which side wins using a morally black-and-white framework: identify
the clearer right-vs-wrong position and choose the side whose argument is more
morally correct, not merely the side with the most popular opinion.

Return only valid JSON with this schema:
{
  "winner": "pro" | "con",
  "why_they_won": "...",
  "moral_reasoning": "...",
  "why_opposing_side_lost": "...",
  "public_popularity_estimate": "..."
}
""".strip()


HTML = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Debate Arena</title>
  <style>
    :root {
      --bg: #f4f7fb;
      --panel: rgba(255,255,255,.86);
      --text: #132033;
      --muted: #627084;
      --accent: #6557ff;
      --accent-2: #18b6a4;
      --danger: #d43f5e;
      --border: rgba(36,48,72,.14);
      --shadow: 0 24px 80px rgba(26,35,61,.15);
    }
    body.dark {
      --bg: #0c1020;
      --panel: rgba(20,27,48,.84);
      --text: #eef3ff;
      --muted: #a9b4ca;
      --accent: #8b7cff;
      --accent-2: #24d3bc;
      --danger: #ff6685;
      --border: rgba(220,230,255,.16);
      --shadow: 0 24px 90px rgba(0,0,0,.34);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, Segoe UI, sans-serif;
      min-height: 100vh;
      color: var(--text);
      background:
        radial-gradient(circle at top left, rgba(101,87,255,.25), transparent 34rem),
        radial-gradient(circle at bottom right, rgba(24,182,164,.20), transparent 30rem),
        var(--bg);
      transition: background .25s ease, color .25s ease;
    }
    .wrap { width: min(1120px, calc(100% - 32px)); margin: 0 auto; padding: 32px 0 56px; }
    header { display: flex; justify-content: space-between; align-items: center; gap: 16px; margin-bottom: 36px; }
    .brand { display: flex; align-items: center; gap: 14px; }
    .logo { width: 48px; height: 48px; border-radius: 16px; background: linear-gradient(135deg, var(--accent), var(--accent-2)); box-shadow: var(--shadow); }
    h1 { font-size: clamp(2.2rem, 6vw, 5rem); letter-spacing: -.07em; margin: 0; line-height: .92; }
    .tagline { color: var(--muted); margin: 8px 0 0; font-size: 1.05rem; }
    .theme-toggle, button {
      border: 1px solid var(--border);
      color: var(--text);
      background: var(--panel);
      border-radius: 999px;
      padding: 12px 16px;
      cursor: pointer;
      font-weight: 700;
      backdrop-filter: blur(14px);
    }
    .hero, .card {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 30px;
      box-shadow: var(--shadow);
      backdrop-filter: blur(18px);
    }
    .hero { padding: clamp(22px, 5vw, 42px); margin-bottom: 22px; }
    form { display: grid; grid-template-columns: 1fr auto; gap: 14px; align-items: start; margin-top: 24px; }
    textarea {
      min-height: 84px;
      resize: vertical;
      border: 1px solid var(--border);
      border-radius: 22px;
      padding: 18px 20px;
      font: inherit;
      color: var(--text);
      background: rgba(255,255,255,.55);
      outline: none;
    }
    body.dark textarea { background: rgba(5,8,18,.44); }
    .start {
      background: linear-gradient(135deg, var(--accent), var(--accent-2));
      color: white;
      border: 0;
      padding: 18px 22px;
      border-radius: 22px;
      min-height: 84px;
      box-shadow: 0 18px 46px rgba(101,87,255,.28);
    }
    .status {
      display: flex;
      align-items: center;
      gap: 10px;
      margin-top: 18px;
      color: var(--muted);
      min-height: 28px;
      font-weight: 650;
    }
    .pulse { width: 10px; height: 10px; border-radius: 50%; background: var(--accent-2); box-shadow: 0 0 0 8px rgba(24,182,164,.12); }
    .hidden { display: none !important; }
    .grid { display: grid; grid-template-columns: repeat(2, minmax(0,1fr)); gap: 18px; margin-top: 18px; }
    .card { padding: 24px; }
    .summary { display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin-top: 18px; }
    .metric { border: 1px solid var(--border); border-radius: 22px; padding: 18px; background: rgba(255,255,255,.28); }
    body.dark .metric { background: rgba(255,255,255,.05); }
    .metric b { display: block; font-size: .8rem; text-transform: uppercase; letter-spacing: .08em; color: var(--muted); margin-bottom: 8px; }
    .metric span { font-size: 1.15rem; font-weight: 850; }
    h2, h3 { margin: 0 0 12px; letter-spacing: -.03em; }
    .arg { padding: 16px 0; border-top: 1px solid var(--border); }
    .arg:first-of-type { border-top: 0; padding-top: 0; }
    .arg-title { font-weight: 850; margin-bottom: 8px; }
    .sources { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 12px; }
    .sources a { color: var(--accent); border: 1px solid var(--border); border-radius: 999px; padding: 7px 10px; text-decoration: none; font-size: .88rem; background: rgba(255,255,255,.28); }
    .error { color: var(--danger); background: rgba(212,63,94,.10); border: 1px solid rgba(212,63,94,.35); padding: 16px; border-radius: 18px; margin-top: 16px; }
    pre { white-space: pre-wrap; word-break: break-word; }
    @media (max-width: 760px) { form, .grid, .summary { grid-template-columns: 1fr; } .start { min-height: auto; } header { align-items: start; } }
  </style>
</head>
<body>
  <div class="wrap">
    <header>
      <div class="brand"><div class="logo"></div><div><strong>Multi-Agent Web App</strong><div class="tagline">Pro vs. Con vs. Moral Judge</div></div></div>
      <button class="theme-toggle" id="themeToggle" type="button">Toggle dark/light</button>
    </header>

    <section class="hero">
      <h1>Debate Arena</h1>
      <p class="tagline">Submit a question. Two research agents build the strongest cases, then a judge declares a morally grounded winner.</p>
      <form id="debateForm">
        <textarea id="topic" placeholder="Example: Are AI Agents necessary?" required></textarea>
        <button class="start" id="startBtn" type="submit">Start the debate!</button>
      </form>
      <div class="status" id="status"><span class="pulse hidden" id="pulse"></span><span id="statusText">Ready.</span></div>
      <div id="error" class="error hidden"></div>
    </section>

    <section id="results" class="hidden">
      <div class="card">
        <h2>Results</h2>
        <div class="summary">
          <div class="metric"><b>Winning side</b><span id="winner">—</span></div>
          <div class="metric"><b>Public agreement estimate</b><span id="popularity">—</span></div>
          <div class="metric"><b>Agents</b><span>PRO · CONS · JUDGE</span></div>
        </div>
        <h3 style="margin-top:22px">Why the winner won</h3>
        <p id="why"></p>
        <h3>Moral reasoning</h3>
        <p id="moral"></p>
      </div>
      <div class="grid">
        <div class="card"><h2>Arguments For</h2><div id="proArgs"></div></div>
        <div class="card"><h2>Arguments Against</h2><div id="conArgs"></div></div>
      </div>
    </section>
  </div>

  <script>
    const form = document.getElementById('debateForm');
    const topic = document.getElementById('topic');
    const statusText = document.getElementById('statusText');
    const pulse = document.getElementById('pulse');
    const errorBox = document.getElementById('error');
    const results = document.getElementById('results');
    const startBtn = document.getElementById('startBtn');

    const savedTheme = localStorage.getItem('debate-theme');
    if (savedTheme === 'dark' || (!savedTheme && matchMedia('(prefers-color-scheme: dark)').matches)) document.body.classList.add('dark');
    document.getElementById('themeToggle').onclick = () => {
      document.body.classList.toggle('dark');
      localStorage.setItem('debate-theme', document.body.classList.contains('dark') ? 'dark' : 'light');
    };

    function setBusy(isBusy) {
      startBtn.disabled = isBusy;
      pulse.classList.toggle('hidden', !isBusy);
    }
    function escapeHtml(text) {
      return String(text ?? '').replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));
    }
    function renderArgs(el, data) {
      const args = data?.arguments || [];
      el.innerHTML = args.map(arg => {
        const sources = (arg.sources || []).map((src, idx) => {
          const url = typeof src === 'string' ? src : src.url;
          const title = typeof src === 'string' ? `Source ${idx + 1}` : (src.title || `Source ${idx + 1}`);
          return url ? `<a href="${escapeHtml(url)}" target="_blank" rel="noopener noreferrer">${escapeHtml(title)}</a>` : '';
        }).join('');
        return `<div class="arg"><div class="arg-title">${escapeHtml(arg.title || arg.claim || 'Argument')}</div><p>${escapeHtml(arg.claim || '')}</p><p>${escapeHtml(arg.evidence || '')}</p><div class="sources">${sources}</div></div>`;
      }).join('') || '<p>No structured arguments returned.</p><pre>' + escapeHtml(data?.raw || '') + '</pre>';
    }
    function renderFinal(payload) {
      const final = payload.final || {};
      const judge = final.judge || {};
      document.getElementById('winner').textContent = (final.winner || judge.winner || 'unknown').toUpperCase();
      document.getElementById('popularity').textContent = final.public_popularity_estimate || judge.public_popularity_estimate || 'Not estimated';
      document.getElementById('why').textContent = final.why_they_won || judge.why_they_won || '';
      document.getElementById('moral').textContent = judge.moral_reasoning || '';
      renderArgs(document.getElementById('proArgs'), final.pro);
      renderArgs(document.getElementById('conArgs'), final.con);
      results.classList.remove('hidden');
    }

    form.addEventListener('submit', (event) => {
      event.preventDefault();
      errorBox.classList.add('hidden');
      results.classList.add('hidden');
      setBusy(true);
      statusText.textContent = 'Preparing Debate Arena agents...';
      const params = new URLSearchParams({ topic: topic.value.trim() });
      const events = new EventSource(`/api/debate/stream?${params.toString()}`);
      events.onmessage = (event) => {
        const payload = JSON.parse(event.data);
        if (payload.type === 'status') statusText.textContent = payload.message;
        if (payload.type === 'final') {
          renderFinal(payload);
          statusText.textContent = 'Debate complete.';
          setBusy(false);
          events.close();
        }
        if (payload.type === 'error') {
          errorBox.textContent = payload.message;
          errorBox.classList.remove('hidden');
          statusText.textContent = 'Debate failed.';
          setBusy(false);
          events.close();
        }
      };
      events.onerror = () => {
        errorBox.textContent = 'The live debate connection failed. Check the server logs and try again.';
        errorBox.classList.remove('hidden');
        statusText.textContent = 'Connection error.';
        setBusy(false);
        events.close();
      };
    });
  </script>
</body>
</html>
"""


app = Flask(__name__)


class DebateError(Exception):
    """User-readable Debate Arena error."""


def api_key() -> str:
    key = os.environ.get("BACKBOARD_API_KEY")
    if not key:
        raise DebateError("BACKBOARD_API_KEY is not set. Export it before starting Debate Arena.")
    return key


def get_client() -> Any:
    if BackboardClient is None:
        raise DebateError("backboard-sdk is not installed. Run: pip install -r requirements.txt")
    return BackboardClient(api_key=api_key())


def assistant_id(assistant: Any) -> str:
    value = getattr(assistant, "assistant_id", None) or getattr(assistant, "assistantId", None) or assistant["assistant_id"]
    return str(value)


async def get_or_create_assistant(client: Any, name: str, system_prompt: str) -> str:
    try:
        existing = await client.list_assistants(name=name, limit=1)
        if existing:
            return assistant_id(existing[0])
    except Exception:
        # If listing is unavailable for any reason, fall back to direct creation.
        pass

    created = await client.create_assistant(name=name, system_prompt=system_prompt)
    return assistant_id(created)


async def ensure_agents(client: Any) -> dict[str, str]:
    if AGENTS_FILE.exists():
        try:
            with AGENTS_FILE.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            if all(data.get(key) for key in ("pro", "con", "judge")):
                return {key: str(data[key]) for key in ("pro", "con", "judge")}
        except (json.JSONDecodeError, OSError):
            pass

    data = {
        "pro": await get_or_create_assistant(client, PRO_NAME, PRO_PROMPT),
        "con": await get_or_create_assistant(client, CON_NAME, CON_PROMPT),
        "judge": await get_or_create_assistant(client, JUDGE_NAME, JUDGE_PROMPT),
    }
    with AGENTS_FILE.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    return data


def extract_json(text: str) -> dict[str, Any]:
    if not text:
        return {"raw": ""}
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.S)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
    return {"raw": text, "arguments": []}


async def ask_agent(client: Any, prompt: str, assistant: str, *, web_search: str = "off") -> dict[str, Any]:
    response = await client.send_message(
        prompt,
        assistant_id=assistant,
        web_search=web_search,
        json_output=True,
    )
    content = getattr(response, "content", "")
    parsed = extract_json(content)
    parsed.setdefault("raw", content)
    return parsed


async def run_debate(topic: str, status: Callable[[str], None]) -> dict[str, Any]:
    if not topic.strip():
        raise DebateError("Please enter a debate question.")

    client = get_client()
    status("Preparing Debate Arena agents...")
    agents = await ensure_agents(client)

    pro_prompt = f"User question: {topic}\n\nArgue FOR this question following your Debate Arena instructions."
    con_prompt = f"User question: {topic}\n\nArgue AGAINST this question following your Debate Arena instructions."

    status(f"Agent 1 is finding reasons for: {topic}")
    pro = await ask_agent(client, pro_prompt, agents["pro"], web_search="Auto")

    status(f"Agent 2 is finding reasons against: {topic}")
    con = await ask_agent(client, con_prompt, agents["con"], web_search="Auto")

    status("Judge is judging the moral strength of both sides...")
    judge_prompt = json.dumps(
        {
            "topic": topic,
            "pro_arguments": pro,
            "con_arguments": con,
            "task": "Choose pro or con as winner and explain why the winning side is more morally correct.",
        },
        indent=2,
    )
    judge = await ask_agent(client, judge_prompt, agents["judge"])

    return {
        "topic": topic,
        "winner": judge.get("winner", "unknown"),
        "why_they_won": judge.get("why_they_won", ""),
        "public_popularity_estimate": judge.get("public_popularity_estimate", "Not estimated"),
        "pro": pro,
        "con": con,
        "judge": judge,
        "agents": agents,
    }


def sse(data: dict[str, Any]) -> str:
    return f"data: {json.dumps(data)}\n\n"


@app.get("/")
def index() -> str:
    return render_template_string(HTML)


@app.get("/health")
def health() -> Response:
    return jsonify({"ok": True, "agents_file_exists": AGENTS_FILE.exists()})


@app.post("/api/debate")
def debate_json() -> Response:
    body = request.get_json(silent=True) or {}
    topic = str(body.get("topic", ""))
    try:
        result = asyncio.run(run_debate(topic, lambda _message: None))
        return jsonify(result)
    except Exception as exc:
        message = str(exc) if isinstance(exc, DebateError) else f"Debate failed: {exc}"
        return jsonify({"error": message}), 400


@app.get("/api/debate/stream")
def debate_stream() -> Response:
    topic = request.args.get("topic", "")
    messages: queue.Queue[dict[str, Any] | None] = queue.Queue()

    def publish(message: str) -> None:
        messages.put({"type": "status", "message": message})

    def worker() -> None:
        try:
            result = asyncio.run(run_debate(topic, publish))
            messages.put({"type": "final", "final": result})
        except Exception as exc:
            message = str(exc) if isinstance(exc, DebateError) else f"Debate failed: {exc}"
            messages.put({"type": "error", "message": message})
        finally:
            messages.put(None)

    threading.Thread(target=worker, daemon=True).start()

    @stream_with_context
    def generate():
        while True:
            item = messages.get()
            if item is None:
                break
            yield sse(item)

    return Response(generate(), mimetype="text/event-stream")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5000")), debug=False)
