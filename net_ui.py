"""Streamlit UI for the agentic network troubleshooter (net_agent.py).

Chat drives it: "troubleshoot 10.10.1.20 to 172.20.5.10". The agent decides the
per-vendor commands itself; every device command needs approval here first.

    uv run streamlit run net_ui.py
"""
import asyncio
import concurrent.futures
import json
import re
import time
import uuid

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

import net_agent

_POOL = concurrent.futures.ThreadPoolExecutor(max_workers=1)


def run_async(coro):
    return _POOL.submit(asyncio.run, coro).result()


GREETING = ("Hello. I can help with source-to-destination troubleshooting. "
            "Try: troubleshoot 10.10.1.20 to 172.20.5.10")

st.set_page_config(page_title="Network Troubleshooting Agent", page_icon="🛰️",
                   layout="wide")
st.title("🛰️ Network Troubleshooting Agent")

# keep the right-hand panel pinned while the conversation scrolls
st.markdown("""
<style>
div[data-testid="stHorizontalBlock"] > div:nth-child(2) > div {
    position: sticky;
    top: 3rem;
    align-self: flex-start;
}
</style>
""", unsafe_allow_html=True)

ss = st.session_state
ss.setdefault("checkpointer", MemorySaver())
ss.setdefault("thread_id", str(uuid.uuid4()))
ss.setdefault("trace", [("assistant", GREETING)])   # [(kind, text)]
ss.setdefault("pending", None)
ss.setdefault("seen", 0)                           # messages already rendered
ss.setdefault("streamed", 0)                       # trace entries already streamed
ss.setdefault("show_details", False)               # hide reasoning / tool calls
ss.setdefault("busy", False)                       # a step is currently running
ss.setdefault("decision", None)                    # approve/reject awaiting execution


def stream_words(text, delay=0.012):
    """Yield the text word by word so st.write_stream renders it progressively."""
    for word in str(text).split(" "):
        yield word + " "
        time.sleep(delay)


# the labels the agent is told to emit in its final report
_FIELDS = ("Source", "Destination", "Ping", "Path", "Result", "Evidence",
           "Cause", "Next step", "Next Step")
_FIELD_RE = re.compile(r"\b(" + "|".join(_FIELDS) + r")\s*:\s*", re.I)


_VERDICT_ICON = {"SUCCESS": "✅", "REACHABLE": "✅",
                 "FAILED": "❌", "NOT REACHABLE": "❌",
                 "INCONCLUSIVE": "⚠️", "NOT RUN": "⚪"}


def _render_structured(d: dict) -> bool:
    """Render the agreed final-answer object. Returns False if it isn't one."""
    keys = {"source", "destination", "ping", "path", "result"}
    if not keys & set(d):
        return False

    c1, c2 = st.columns(2)
    if d.get("source"):
        c1.markdown(f"**Source**  \n{d['source']}")
    if d.get("destination"):
        c2.markdown(f"**Destination**  \n{d['destination']}")

    for key, label in (("ping", "Ping"), ("result", "Result")):
        if d.get(key):
            v = str(d[key]).upper()
            st.markdown(f"**{label}** {_VERDICT_ICON.get(v, '•')} {d[key]}")

    path = d.get("path")
    if path:
        st.markdown("**Path**")
        st.code(" -> ".join(str(p) for p in path) if isinstance(path, list)
                else str(path), language="text")

    ev = d.get("evidence")
    if ev:
        st.markdown("**Evidence**")
        for line in (ev if isinstance(ev, list) else [ev]):
            st.markdown(f"- {line}")

    for key, label in (("cause", "Cause"), ("next_step", "Next step")):
        if d.get(key) and str(d[key]).lower() not in ("none", "-"):
            st.markdown(f"**{label}**  \n{d[key]}")

    extra = {k: v for k, v in d.items()
             if k not in ("source", "destination", "ping", "path", "result",
                          "evidence", "cause", "next_step") and v}
    if extra:
        for k, v in extra.items():
            st.markdown(f"**{k.replace('_', ' ').title()}**  \n{v}")
    return True


def _render_generic(d: dict, level: int = 0):
    """Lay out any other JSON object the agent returns as labelled values."""
    for key, value in d.items():
        label = str(key).replace("_", " ").strip().title()
        pad = "&nbsp;" * (4 * level)
        if isinstance(value, dict):
            st.markdown(f"{pad}**{label}**")
            _render_generic(value, level + 1)
        elif isinstance(value, list):
            st.markdown(f"{pad}**{label}**")
            for item in value:
                st.markdown(f"{pad}- {item}")
        elif value not in (None, "", []):
            st.markdown(f"{pad}**{label}**  \n{pad}{value}")


def render_report(text: str):
    """Render the agent's answer.

    Preferred form is the agreed JSON object; falls back to splitting on
    'Label : value' text, and finally to plain text with line breaks kept.
    """
    text = str(text or "").strip()

    if text.startswith("{") or text.startswith("["):
        try:
            obj = json.loads(text)
        except Exception:
            obj = None
        if isinstance(obj, dict):
            if _render_structured(obj):          # the troubleshooting schema
                return
            _render_generic(obj)                 # some other object -> key/value
            return
        if isinstance(obj, list):
            for item in obj:
                st.markdown(f"- {item}")
            return

    parts = _FIELD_RE.split(text)
    if len(parts) < 3:                       # not a report -> keep line breaks
        st.markdown(text.replace("\n", "  \n"))
        return

    if parts[0].strip():                     # any preamble before the first label
        st.markdown(parts[0].strip().replace("\n", "  \n"))

    for label, value in zip(parts[1::2], parts[2::2]):
        value = value.strip().rstrip(",;")
        if not value:
            continue
        label = label.strip().title()
        if label == "Path":
            st.markdown(f"**{label}**")
            st.code(value, language="text")
        elif label in ("Ping", "Result"):
            v = value.upper()
            icon = "✅" if ("SUCCESS" in v or "REACHABLE" in v
                            and "NOT REACHABLE" not in v) else \
                   "❌" if ("FAIL" in v or "NOT REACHABLE" in v) else "⚠️"
            st.markdown(f"**{label}** {icon} {value}")
        else:
            st.markdown(f"**{label}**  \n{value.replace(chr(10), '  ' + chr(10))}")

CLIP = getattr(net_agent, "LLM_MODE", "") == "clipboard"


# NOTE: no st.session_state access in here -- this runs in a worker thread.
async def _run(checkpointer, thread_id, question, resume=None):
    app = await net_agent.build_agent(checkpointer=checkpointer)
    cfg = {"configurable": {"thread_id": thread_id}}
    if resume is None:
        return await app.ainvoke({"messages": [("user", question)], "loops": 0}, cfg)
    return await app.ainvoke(Command(resume=resume), cfg)


def _absorb(state):
    """Turn new graph messages into trace entries; capture any pending approval."""
    msgs = state.get("messages", [])
    for m in msgs[ss.seen:]:
        if isinstance(m, HumanMessage):
            continue
        if isinstance(m, ToolMessage):
            ss.trace.append(("tool", f"{m.name} →\n{m.content}"))
        elif isinstance(m, AIMessage) and m.tool_calls:
            if m.content:                                   # reasoning for this step
                ss.trace.append(("think", str(m.content)))
            for tc in m.tool_calls:
                ss.trace.append(("call", f"{tc['name']}({tc['args']})"))
        elif isinstance(m, AIMessage) and m.content:
            th = (m.additional_kwargs or {}).get("thought")
            if th:
                ss.trace.append(("think", str(th)))
            ss.trace.append(("assistant", m.content))
    ss.seen = len(msgs)
    ss.pending = state["__interrupt__"][0].value if "__interrupt__" in state else None
    ss.gstate = {k: state.get(k) for k in
                 ("ping_ok", "hops", "path", "commands_run", "devices", "loops")}


def render(kind, text, stream=False):
    if kind == "user":
        st.chat_message("user").write(text)
    elif kind == "assistant":
        with st.chat_message("assistant"):
            plain = not str(text).strip().startswith("{") \
                and not _FIELD_RE.search(str(text))
            if stream and plain:
                st.write_stream(stream_words(text))     # chit-chat: stream it
            else:
                render_report(text)                     # report: lay it out
    elif kind == "think":
        with st.chat_message("assistant"):
            with st.expander("💭 reasoning", expanded=True):
                if stream:
                    st.write_stream(stream_words(text, 0.006))
                else:
                    st.markdown(text)
    elif kind == "call":
        st.chat_message("assistant").info(f"🔧 {text}")
    elif kind == "tool":
        with st.chat_message("assistant"):
            st.code(text, language="text")


_ICON = {"done": "✅", "fail": "❌", "wait": "🕒", "run": "🔄", "idle": "⚪"}


def workflow_steps():
    """(status, label, detail) per step, derived from the trace + graph state."""
    gs = ss.get("gstate") or {}
    calls = [t for k, t in ss.trace if k == "call"]
    busy = ss.get("busy")
    lookups = [c for c in calls if "device_details" in c or "get_device" in c]
    devcmds = [c for c in calls if "execute_query_on_server" in c]

    out = []
    if lookups:
        out.append(("done", "CMDB lookup",
                    f"{len(lookups)} device record(s) fetched"))
    else:
        out.append((("run" if busy else "idle"), "CMDB lookup", ""))

    ping = gs.get("ping_ok")
    if ping is True:
        out.append(("done", "Reachability (ping)", "Ping result: reachable"))
    elif ping is False:
        out.append(("fail", "Reachability (ping)", "Ping result: failed"))
    elif ss.pending and "ping" in str(ss.pending.get("command", "")).lower():
        out.append(("wait", "Reachability (ping)", "awaiting your approval"))
    else:
        out.append((("run" if busy and devcmds else "idle"),
                    "Reachability (ping)", ""))

    hops = gs.get("hops") or []
    if hops:
        out.append(("done", "Path discovery", f"{len(hops)} hop(s) found"))
    elif ss.pending and "trace" in str(ss.pending.get("command", "")).lower():
        out.append(("wait", "Path discovery", "awaiting your approval"))
    else:
        out.append(("idle", "Path discovery", ""))

    extra = [c for c in devcmds if "show" in c.lower()]
    if extra:
        out.append(("done", "Deeper checks", f"{len(extra)} show command(s) run"))

    answered = any(k == "assistant" for k, _ in ss.trace[1:])
    out.append((("done" if answered else "idle"), "Conclusion", ""))
    return out


chat_col, side_col = st.columns([1.4, 1], gap="large")

with chat_col:
    st.subheader("Conversation")
    if CLIP:
        st.caption("Clipboard relay — when the banner shows, paste into Copilot and "
                   "copy back its JSON reply. One paste per agent step.")
    # entries added since the last render stream in; older ones render instantly
    shown = ss.trace if ss.get("show_details") else \
        [(k, t) for k, t in ss.trace if k in ("user", "assistant")]
    for i, (kind, text) in enumerate(shown):
        render(kind, text, stream=(i >= ss.streamed))
    ss.streamed = len(shown)

    with st.form("ask", clear_on_submit=True):
        q = st.text_input("Ask", placeholder="troubleshoot 10.10.1.20 to 172.20.5.10",
                          label_visibility="collapsed")
        sent = st.form_submit_button("Send", type="primary")

with side_col:
    st.subheader("Device command approval")
    approve = reject = False
    if ss.decision is not None and ss.pending:
        # already clicked -> buttons gone, show what is happening instead
        p = ss.pending
        if ss.decision:
            st.info(f"⚙️ **Running on {p.get('device_ip')} ({p.get('region')})**\n\n"
                    f"`{p.get('command')}`")
        else:
            st.error(f"⛔ Rejected: `{p.get('command')}`")
    elif ss.pending:
        p = ss.pending
        st.warning("The agent wants to run this on a device.")
        st.code(f"device_ip : {p.get('device_ip')}\n"
                f"region    : {p.get('region')}\n"
                f"command   : {p.get('command')}", language="text")
        a, r = st.columns(2)
        approve = a.button("✅ Approve", use_container_width=True)
        reject = r.button("❌ Reject", use_container_width=True)
    else:
        st.caption("No command pending. Read-only commands only; each one is "
                   "validated in code and needs your approval.")

    st.checkbox("Show agent reasoning and tool calls", key="show_details",
                help="Off = only the question and the final answer")

    st.divider()
    st.subheader("Workflow progress")
    _STATE = {"done": "complete", "fail": "error", "run": "running",
              "wait": "running", "idle": "running"}
    for status, label, detail in workflow_steps():
        if status == "idle":                       # not started -> plain, no spinner
            st.markdown(f"⚪ {label}")
            continue
        with st.status(label, state=_STATE[status],
                       expanded=bool(detail)) as box:
            if detail:
                st.write(detail)
            if status == "wait":
                box.update(label=f"{label} — awaiting your approval")

    st.divider()
    gs = ss.get("gstate") or {}
    finished = (not ss.pending) and ss.decision is None and not ss.busy \
        and any(k == "assistant" for k, _ in ss.trace[1:])

    if finished:
        st.subheader("📋 Summary report")
        ping = gs.get("ping_ok")
        verdict = ("REACHABLE" if ping else
                   "NOT REACHABLE / INCONCLUSIVE" if ping is False else "NOT TESTED")
        st.markdown(f"**Question:** {ss.get('question', '-')}")
        st.markdown(f"**Ping:** {'SUCCESS' if ping else ('FAILED' if ping is False else 'not run')}"
                    f" — parsed independently of the model")
        st.markdown(f"**Verdict (parsed):** {verdict}")
        if gs.get("path"):
            st.markdown("**Path:**")
            st.code(gs["path"], language="text")
        hops = gs.get("hops") or []
        if hops:
            st.markdown("**Hops:**")
            st.code("\n".join(
                f"{h['n']:>2}  " + ("* * * (no response)" if h["timeout"]
                                    else (h.get("host") or h.get("ip") or ""))
                for h in hops), language="text")
        cmds = gs.get("commands_run") or []
        if cmds:
            st.markdown(f"**Commands run ({len(cmds)}):**")
            for c in cmds:
                st.markdown(f"{'✅' if c.get('approved') else '⛔'} "
                            f"`{c.get('command')}` on {c.get('device_ip')}")
        answers = [t for k, t in ss.trace if k == "assistant"][1:]
        if answers:
            with st.expander("Agent's full report", expanded=True):
                render_report(answers[-1])
        st.caption(f"tool loops: {gs.get('loops') or 0} / {net_agent.MAX_TOOL_LOOPS}")
    else:
        st.subheader("Graph state")
        ping = gs.get("ping_ok")
        st.markdown(f"**Ping:** {'✅ reachable' if ping else ('❌ failed' if ping is False else '⚪ not run')}")
        st.markdown(f"**Hops:** {len(gs.get('hops') or [])}")
        if gs.get("path"):
            st.code(gs["path"], language="text")
        if gs.get("commands_run"):
            st.caption("Commands run")
            for c in gs["commands_run"]:
                st.markdown(f"{'✅' if c.get('approved') else '⛔'} `{c.get('command')}` "
                            f"on {c.get('device_ip')}")
        st.caption(f"tool loops: {gs.get('loops') or 0} / {net_agent.MAX_TOOL_LOOPS}")

    st.divider()
    st.caption("MCP servers")
    st.code("\n".join(net_agent.MCP_SERVERS.keys()), language="text")
    if st.button("New conversation"):
        ss.thread_id = str(uuid.uuid4())
        ss.checkpointer = MemorySaver()
        ss.trace, ss.pending, ss.seen = [("assistant", GREETING)], None, 0
        ss.streamed = 0
        if hasattr(net_agent.llm, "reset_conversation"):
            net_agent.llm.reset_conversation()
        st.rerun()


def _drive(resume=None):
    cue, status = st.empty(), st.empty()
    running = ss.pending if resume is True else None

    if running:                       # an approved command is about to execute
        status.info(f"⚙️ Running on {running.get('device_ip')} "
                    f"({running.get('region')}): `{running.get('command')}` …")
        label = "executing command on the device…"
    elif CLIP:
        cue.warning("📋 **Prompt is on your clipboard** — paste it into Copilot, "
                    "then copy the reply.")
        label = "waiting for clipboard…"
    else:
        label = "thinking…"

    ss.busy = True
    with st.spinner(label):
        try:
            # read session_state HERE (main thread), pass values into the worker
            _absorb(run_async(_run(ss.checkpointer, ss.thread_id,
                                   ss.get("question", ""), resume)))
        except Exception as e:
            ss.trace.append(("assistant", f"error: {e}"))
    ss.busy = False
    cue.empty()
    status.empty()
    st.rerun()


if sent and q.strip():
    ss.question = q
    ss.thread_id = str(uuid.uuid4())
    ss.checkpointer = MemorySaver()
    ss.trace.append(("user", q))
    ss.seen = 0
    _drive()

# click -> record + rerun so the buttons disappear immediately, THEN execute
if approve:
    ss.decision = True
    st.rerun()
if reject:
    ss.decision = False
    st.rerun()

if ss.decision is not None:
    d, ss.decision = ss.decision, None
    _drive(resume=d)
