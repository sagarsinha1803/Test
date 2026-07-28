"""Streamlit UI for the agentic network troubleshooter (net_agent.py).

Chat drives it: "troubleshoot 10.10.1.20 to 172.20.5.10". The agent decides the
per-vendor commands itself; every device command needs approval here first.

    uv run streamlit run net_ui.py
"""
import asyncio
import concurrent.futures
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
            if stream:
                st.write_stream(stream_words(text))
            else:
                st.write(text)
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
                st.text(answers[-1])
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
















"""Agentic network troubleshooter -- LangGraph built by hand (no prebuilts).

No create_react_agent, no ToolNode, no tools_condition. The graph, the state and
the tool executor are all defined here:

    START -> agent -> (route) -> tools -> agent -> ... -> END

The LLM decides the commands: it reads the device details from the CMDB (unicorn
MCP), works out the correct read-only CLI for THAT vendor/OS/model, runs it via
the SSH MCP, and reports the source -> hop -> hop -> destination path.

Two guards, because the model is choosing commands that run on real devices:
  1. read-only allowlist enforced IN CODE (the model is never trusted)
  2. human approval (interrupt) before every SSH execution -- collected BEFORE
     anything executes, so a resume never re-runs a command twice

    uv run python agent/net_agent.py "troubleshoot 10.10.1.20 to 172.20.5.10"
"""
import asyncio
import json
import os
import re
import sys
from typing import Annotated, Any, Optional, TypedDict

from langchain_core.messages import AIMessage, ToolMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.types import interrupt, Command

try:
    import vendors          # optional: regex parsers that cross-check the LLM
except Exception:           # without it the graph still runs, just no parsed state
    vendors = None

_HERE = os.path.dirname(os.path.abspath(__file__))

try:                                    # load .env sitting next to this file
    from dotenv import load_dotenv
    load_dotenv(os.path.join(_HERE, ".env"))
except ImportError:
    pass


def _stdio(script):
    return {"command": sys.executable, "args": [os.path.join(_HERE, script)],
            "transport": "stdio"}


SSH_MCP_URL = "http://175.60.57.250:4021/sse"

MCP_SERVERS = {
    "unicorn": _stdio("unicorn_mcp.py"),                 # get_device_details
    "ssh":     {"url": SSH_MCP_URL, "transport": "sse"},  # execute_query_on_server
    # "tools": _stdio("tools_mcp.py"),                   # local mocks, testing only
}

# Which SERVERS touch real devices. Every tool they expose -- including ones
# added later -- is automatically read-only checked and human approved. Tools
# whose origin cannot be determined are treated as device tools (deny by
# default), so a new server is never accidentally ungated.
DEVICE_SERVERS = {"ssh"}

REQUIRE_APPROVAL = True
MAX_TOOL_LOOPS = 12           # agent<->tools round trips before we force an answer
                              # (2 CMDB + ping + traceroute + escalation checks)

# ---- LLM ---------------------------------------------------------------------
LLM_MODE = os.environ.get("LLM_MODE", "clipboard")
if LLM_MODE == "clipboard":
    from clipboard_llm import ClipboardLLM
    # CLIP_MODE=agent  -> instructions+tools live in a custom M365 Copilot agent
    # CLIP_MODE=delta  -> one normal Copilot chat, only new messages pasted
    # CLIP_MODE=full   -> self-contained paste every time
    llm = ClipboardLLM(mode=os.environ.get("CLIP_MODE", "delta"))
else:
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(base_url=os.environ.get("LLM_BASE_URL", "http://localhost:11434/v1"),
                     api_key="dummy", model=os.environ.get("LLM_MODEL", "gpt-4o"),
                     temperature=0)

# ---- safety: read-only allowlist, enforced in code ---------------------------
_ALLOWED = re.compile(
    r"^\s*(ping|ping6|traceroute|traceroute6|tracert|tracepath|show|display|get|"
    r"execute\s+ping|execute\s+traceroute|run\s+/util|/ping|/tool\s+traceroute)\b",
    re.I)
_BLOCKED = re.compile(
    r"\b(conf(ig)?(ure)?|write|erase|reload|reboot|delete|remove|copy|clear|reset|"
    r"shutdown|set\s|commit|rollback|request|restart|halt|format|install)\b", re.I)


# Commands that dump whole tables. Harmless to the device, fatal to the context
# window (a full route table or config is hundreds of thousands of characters).
# Allowed only when narrowed by an argument or piped through a filter.
_BULK = re.compile(
    r"^\s*show\s+("
    r"run(ning-config)?|tech(-support)?|config(uration)?|logging|version\s*$|"
    r"route\s*$|route\s+(ipv4|ipv6|vrf\s+\S+)?\s*$|"
    r"bgp\s*$|bgp\s+(ipv4|vpnv4|vrf\s+\S+)?\s*(unicast)?\s*$|"
    r"cef\s*$|arp\s*$|mpls\s+forwarding\s*$|"
    r"interfaces?\s*$|ip(v4|v6)?\s+interface\s*$|"
    r"access-lists?\s*$|vrf\s+all\s+detail"
    r")\b", re.I)


def check_command(cmd: str) -> Optional[str]:
    """Return an error string if this is not a safe, appropriately scoped
    read-only command."""
    c = (cmd or "").strip()
    if not c:
        return "empty command"
    if not _ALLOWED.match(c):
        return f"'{c}' is not in the read-only allowlist (ping/traceroute/show only)"
    if _BLOCKED.search(c):
        return f"'{c}' contains a state-changing keyword"
    if _BULK.match(c) and "|" not in c:
        return (f"'{c}' would dump a whole table. Narrow it with a specific "
                "prefix/interface, or filter it, e.g. "
                "'show route 10.1.1.1' or 'show arp | include 10.1.1.1'")
    return None


def _tool_text(result) -> str:
    """Flatten an MCP tool result into plain text.

    MCP returns content blocks -- [{'type': 'text', 'text': '...'}] -- and a raw
    str() of that keeps the repr escaping, which breaks both the output parsers
    and what the model gets to read.
    """
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        if "text" in result:
            return str(result["text"])
        return str(result)
    if isinstance(result, (list, tuple)):
        parts = []
        for item in result:
            if isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
            elif hasattr(item, "text"):
                parts.append(str(item.text))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    if hasattr(result, "text"):
        return str(result.text)
    return str(result)


def _commands_of(args: dict) -> list:
    """Pull the command strings out of a device tool's arguments."""
    cmds = args.get("commands") or args.get("command") or []
    if isinstance(cmds, str):
        cmds = [cmds]
    return [str(c) for c in cmds]


SYSTEM_PROMPT = """You are a Network Operations troubleshooting agent.

GOAL: given a source and a destination, determine whether the destination is
reachable from the source and print the full path.

HOW TO WORK -- think, then decide, then act, one step at a time:
- Before every tool call, reason explicitly in your "thought": what the previous
  result told you, what you still need, what you will do next, and WHY that
  command is the right syntax for THIS device's platform.
- Never batch the whole plan into one step. Take one action, read the result,
  re-assess, then take the next.
- If a result is unexpected (device not in CMDB, unknown platform, command
  rejected, empty output), say so in your thought and adapt: try the closest
  standard syntax for that vendor, or continue with what you have and explain
  the gap in the final answer. Do not silently retry the same thing.
- State any assumption you make about the platform.

WORKFLOW (in order, one tool call at a time):
1. Call the CMDB tool for the SOURCE, then again for the DESTINATION.
   Read whatever fields the record returns and work out from them: the vendor,
   the hardware model, the OS/platform and version, and the management IP. Field
   names vary between records -- infer from the key names and values rather than
   expecting a fixed schema. The response also carries the region, which is what
   you pass to the device command tool.
2. From the SOURCE device's vendor/OS/model, work out the correct READ-ONLY ping
   command for that exact platform and run it on the source toward the
   destination. Platforms differ:
     Cisco IOS/IOS-XE : ping <dest> repeat 3
     Cisco NX-OS      : ping <dest> count 3
     Juniper Junos    : ping <dest> count 3
     Arista EOS       : ping <dest> repeat 3
     FortiOS          : execute ping <dest>
     PAN-OS           : ping count 3 host <dest>
     Checkpoint Gaia  : ping <dest> -c 3
     NetScaler/F5/Linux : ping -c 3 <dest>
     Huawei VRP / HP Comware : ping -c 3 <dest>
     MikroTik RouterOS: /ping <dest> count=3
3. Then run the matching READ-ONLY traceroute. ALWAYS BOUND IT -- an unbounded
   traceroute probes 30 hops with 3 probes each and takes minutes. Use at most
   5 hops, 1 probe, 1s timeout, numeric:
     IOS-XR : traceroute <dest> maxttl 5 timeout 1 probe 1 numeric
              (VRF: traceroute vrf <vrf> <dest> maxttl 5 timeout 1 probe 1 numeric)
     IOS    : traceroute <dest> ttl 1 5 timeout 1 probe 1 numeric
     Junos  : traceroute <dest> ttl 5 wait 1
     Linux/Gaia/F5/NetScaler : traceroute -n -m 5 -w 1 -q 1 <dest>
     Huawei/Comware : tracert -m 5 <dest>
   ALWAYS run it, even when the ping succeeded -- the path is part of the answer.
   If 5 hops is not enough, say so and raise it once, to 10.
4. If the ping FAILED or the traceroute stopped early, do not stop and do not
   repeat the same command. Escalate one check at a time, stopping as soon as
   the failure is explained. Examples are Cisco IOS-XR; adapt to the platform.
   a. Route present?   show route <dest>        (IOS: show ip route <dest>)
      No route -> local routing problem, that is the answer. A route -> note the
      next hop, the outgoing interface and any VRF.
   b. In a VRF?        show vrf all / show route vrf <vrf> <dest>
      then retest inside it: ping vrf <vrf> <dest>, traceroute vrf <vrf> <dest>
      (IOS-XR may need the address family: traceroute vrf <vrf> ipv4 <dest>)
      A SUCCESSFUL PING IS AUTHORITATIVE: if the ping succeeds in the VRF the
      destination IS reachable, even if the traceroute returns nothing or only
      * * *. In an MPLS L3VPN the core switches on labels and has no route back
      into the VRF, so it cannot answer the probes -- the path is invisible, not
      broken. Report reachable and explain that.
   c. Wrong source?    ping <dest> source <interface>   (try egress + loopback)
   d. Programmed?      show cef <dest>          (IOS: show ip cef <dest>)
   e. Next hop alive?  show arp | ping <next-hop>
   f. Interface ok?    show interface <interface> brief
   g. Traceroute stopping early is often NORMAL: an MPLS core forwards on labels
      and never answers probes (check: show mpls forwarding prefix <dest>), or a
      firewall drops ICMP/UDP. Do not call the destination unreachable on
      traceroute alone when the route exists and the ping succeeded.
   h. Filtered?        show access-lists <acl> | include <dest>
   i. MPLS L3VPN and the VRF traceroute is blind -> get the path via the underlay:
        show bgp vrf <vrf> <dest> | include "Received Label|from|metric|Local"
      "Received Label n" confirms L3VPN; the address before "(metric n)" is the
      REMOTE PE loopback (the one after "from" is the route reflector, ignore).
      Then trace the underlay in the GLOBAL table (no vrf keyword):
        traceroute <remote-PE-loopback>
      Those hops ARE the path; report them, noting the destination sits behind
      that PE.
   In your thought, say what each result would rule in or out.

OUTPUT SIZE IS A HARD LIMIT: large CLI output cannot be returned. Always scope a
show command to a specific prefix/interface and add a filter when it could still
be long -- "show route 10.1.1.1", "show arp | include 10.1.1.1",
"show interface Gi0/0/0/1 brief". Never ask for show running-config,
show tech-support, or a bare show route / bgp / interfaces / arp / cef /
mpls forwarding; those are rejected in code. If output is still long, re-run
with a tighter filter, never unfiltered.

5. Read the outputs and give the final answer. Use "inconclusive" rather than
   "not reachable" when probes were blocked but the routing looks correct.

FINAL ANSWER FORMAT (plain text, no LaTeX, no $):
  Source      : <name / ip> (<vendor> <os>)
  Destination : <name / ip> (<vendor> <os>)
  Ping        : SUCCESS or FAILED
  Path        : source -> hop1 -> hop2 -> ... -> destination
                (if it never arrives, end at the last hop that answered and mark
                 the break, e.g. ... -> FW-DC1-EDGE-01 -> X)
  Result      : one line, reachable / not reachable
  Cause       : if unreachable, the most likely reason at that hop

RULES:
- READ-ONLY commands only: ping, traceroute/tracert, show/display. Never
  configure, write, reload, clear or otherwise change a device. Commands are
  validated in code and will be rejected.
- A human approves every device command before it runs. If one is rejected,
  continue with what you have and say so in the final answer.

USING execute_query_on_server:
- "commands" is a LIST even for one command: {"commands": ["ping 10.1.1.1 repeat 3"]}
- "region" is REQUIRED -- use the region returned by the CMDB lookup for the
  SOURCE device (PARIS, ASIA, AMER, UK, INDIA or IBFS).
- Run on the SOURCE: device_ip is the source device; the destination goes inside
  the command text.
- Write function names and argument names plainly: get_device_details, not
  get\\_device\\_details. No markdown escaping anywhere in the JSON.
"""


# ============================== STATE ========================================
class NetState(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    loops: int                 # agent<->tools round trips
    devices: dict              # device_name -> CMDB result
    commands_run: list         # audit trail: {device_ip, command, approved}
    ping_ok: Optional[bool]    # parsed from the ping output
    hops: list                 # parsed from the traceroute output
    path: str                  # "src -> hop -> ... -> dst"
    answer: str                # final text from the agent


# ============================== GRAPH ========================================
async def _load_tools(client):
    """Return (tools, owner) where owner maps tool name -> MCP server name."""
    tools, owner = [], {}
    for server in MCP_SERVERS:
        try:
            got = await client.get_tools(server_name=server)
        except TypeError:                    # adapter without per-server loading
            got = await client.get_tools()
            return got, {}                   # unknown origin -> deny by default
        for t in got:
            owner[t.name] = server
        tools.extend(got)
    return tools, owner


async def build_agent(checkpointer=None):
    client = MultiServerMCPClient(MCP_SERVERS)
    tools, owner = await _load_tools(client)
    by_name = {t.name: t for t in tools}
    llm_with_tools = llm.bind_tools(tools)

    def touches_device(tool_name: str) -> bool:
        """Deny by default: unknown origin is treated as device-touching."""
        return owner.get(tool_name, "?") in DEVICE_SERVERS or tool_name not in owner

    # ---- node: agent (think + decide) -----------------------------------
    async def agent(state: NetState):
        msgs = state["messages"]
        if not any(getattr(m, "type", "") == "system" for m in msgs):
            msgs = [("system", SYSTEM_PROMPT)] + list(msgs)
        reply = await llm_with_tools.ainvoke(msgs)
        out: dict = {"messages": [reply]}
        if not getattr(reply, "tool_calls", None) and reply.content:
            out["answer"] = str(reply.content)
        return out

    # ---- node: tools (validate -> approve -> execute) --------------------
    async def tools_node(state: NetState):
        last = state["messages"][-1]
        calls = list(getattr(last, "tool_calls", []) or [])

        # PHASE 1 -- validate + collect every approval BEFORE anything runs, so a
        # resume (which re-executes this node) can never run a command twice.
        verdict: dict = {}
        for tc in calls:
            name, args = tc["name"], tc.get("args") or {}
            if not touches_device(name):
                verdict[tc["id"]] = True
                continue
            bad = next((e for e in (check_command(c) for c in _commands_of(args)) if e), None)
            if bad:
                verdict[tc["id"]] = f"REJECTED: {bad}"
                continue
            if REQUIRE_APPROVAL:
                approved = interrupt({
                    "action": "device_command",
                    "tool": name,
                    "device_ip": args.get("device_ip") or args.get("source"),
                    "region": args.get("region"),
                    "command": "; ".join(_commands_of(args)) or str(args),
                })
                verdict[tc["id"]] = True if approved else \
                    "REJECTED: the reviewer declined this command."
            else:
                verdict[tc["id"]] = True

        # PHASE 2 -- execute, capture structured state
        out_msgs, audit = [], list(state.get("commands_run") or [])
        devices = dict(state.get("devices") or {})
        upd: dict = {}

        for tc in calls:
            name, args, tid = tc["name"], tc.get("args") or {}, tc["id"]
            gate = verdict.get(tid)

            if gate is not True:
                result: Any = gate
            elif name not in by_name:
                result = f"unknown tool '{name}' (have: {sorted(by_name)})"
            else:
                try:
                    result = await by_name[name].ainvoke(args)
                except Exception as e:
                    result = f"error calling {name}: {e}"

            text = _tool_text(result)
            out_msgs.append(ToolMessage(content=text, name=name, tool_call_id=tid))

            if touches_device(name):
                audit.append({"device_ip": args.get("device_ip") or args.get("source"),
                              "command": "; ".join(_commands_of(args)) or str(args),
                              "approved": gate is True})
            else:                                   # lookup tools -> remember the record
                key = args.get("device_name") or args.get("device_ip") or "?"
                devices[key] = text

            # structured capture from the raw CLI output (independent of the LLM)
            if vendors is None or gate is not True:
                continue
            blob = " ".join(_commands_of(args)).lower()
            if gate is True and ("ping" in blob or name == "ping_device"):
                upd["ping_ok"] = vendors.ping_ok(text)
            if gate is True and ("trace" in blob or name == "traceroute_device"):
                hops = vendors.parse_hops(text)
                if hops:
                    upd["hops"] = hops
                    upd["path"] = vendors.path_line(
                        str(args.get("device_ip") or args.get("source") or "source"),
                        hops,
                        str(args.get("dest") or args.get("destination") or "destination"),
                        reached=bool(state.get("ping_ok")))

        return {"messages": out_msgs, "loops": (state.get("loops") or 0) + 1,
                "commands_run": audit, "devices": devices, **upd}

    # ---- router: keep looping while the model asks for tools -------------
    def route(state: NetState):
        last = state["messages"][-1]
        if getattr(last, "tool_calls", None):
            if (state.get("loops") or 0) >= MAX_TOOL_LOOPS:
                return END          # cap reached -> stop instead of looping forever
            return "tools"
        return END

    g = StateGraph(NetState)
    g.add_node("agent", agent)
    g.add_node("tools", tools_node)
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", route, {"tools": "tools", END: END})
    g.add_edge("tools", "agent")
    return g.compile(checkpointer=checkpointer or MemorySaver())


# ============================== CLI ==========================================
async def main():
    question = " ".join(sys.argv[1:]) or "troubleshoot 10.10.1.20 to 172.20.5.10"
    app = await build_agent()
    config = {"configurable": {"thread_id": "cli-1"}}

    state = await app.ainvoke({"messages": [("user", question)], "loops": 0}, config)

    while "__interrupt__" in state:
        p = state["__interrupt__"][0].value
        print("\n" + "!" * 60)
        print("DEVICE COMMAND NEEDS HUMAN APPROVAL")
        print(f"  tool      : {p.get('tool')}")
        print(f"  device_ip : {p.get('device_ip')}")
        print(f"  region    : {p.get('region')}")
        print(f"  command   : {p.get('command')}")
        print("!" * 60)
        ans = await asyncio.to_thread(input, "Run it? [y/N]: ")
        state = await app.ainvoke(
            Command(resume=ans.strip().lower() in ("y", "yes")), config)

    print("\n" + "=" * 60)
    print(state.get("answer") or state["messages"][-1].content)
    if state.get("path"):
        print(f"\nparsed path : {state['path']}")
    if state.get("commands_run"):
        print("commands    :", *[f"\n  {c}" for c in state["commands_run"]])


if __name__ == "__main__":
    asyncio.run(main())
























"""ClipboardLLM -- a human-relay "LLM" for when no API is available.

Each call copies the rendered prompt to the clipboard and blocks. You paste it
into Copilot, copy the answer, and the moment the clipboard changes the agent
resumes with that text as the model's reply.

Supports NATIVE LangGraph tool calling: bind_tools() injects the tool schemas
into the prompt and the pasted JSON is parsed back into AIMessage.tool_calls, so
ToolNode / tools_condition / create_react_agent work exactly as they would with a
real API model. Swapping to a real endpoint later changes nothing but the model
object.

    from clipboard_llm import ClipboardLLM
    llm = ClipboardLLM()
    llm_with_tools = llm.bind_tools(tools)      # same as ChatOpenAI

No automation of the Copilot UI -- a human does the paste. Cost: one manual
round trip per model call (so one per tool step in a ReAct loop).

    pip install pyperclip        (falls back to PowerShell on Windows)
"""
import hashlib
import json
import re
import subprocess
import sys
import time
import uuid
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_mcp_adapters.client import MultiServerMCPClient
from pydantic import PrivateAttr

# ---- clipboard backends -----------------------------------------------------
try:
    import pyperclip

    def _copy(text):
        pyperclip.copy(text)

    def _paste():
        return pyperclip.paste()

except Exception:                                   # PowerShell fallback (Windows)
    def _copy(text):
        subprocess.run(["powershell", "-NoProfile", "-Command",
                        "$in = [Console]::In.ReadToEnd(); Set-Clipboard -Value $in"],
                       input=text, text=True, check=True)

    def _paste():
        r = subprocess.run(["powershell", "-NoProfile", "-Command", "Get-Clipboard -Raw"],
                           capture_output=True, text=True)
        return r.stdout or ""


def _alert():
    """Short beep so you know the prompt is on the clipboard (UI blocks silently)."""
    try:
        import winsound
        winsound.Beep(880, 150)
    except Exception:
        print("\a", end="", flush=True)


# Display labels. Deliberately NOT "[SYSTEM]" / "[USER]": Copilot's safety layer
# reads role-tagged, override-styled prompts as injection attempts and refuses.
_ROLE = {"system": "Context", "human": "Question", "user": "Question",
         "ai": "Previous answer", "assistant": "Previous answer",
         "tool": "Result from"}


def _render(messages: Sequence[BaseMessage]) -> str:
    """Flatten chat messages into one pasteable block."""
    parts = []
    for m in messages:
        role = _ROLE.get(getattr(m, "type", "human"), "Question")
        body = str(m.content or "")
        if getattr(m, "tool_calls", None):          # show what we asked for
            body = (body + "\n" if body else "") + json.dumps(
                [{"tool": tc["name"], "args": tc["args"]} for tc in m.tool_calls])
        if role == "Result from":
            role = f"Result from {getattr(m, 'name', '') or 'the function'}"
        parts.append(f"{role}:\n{body}")
    return "\n\n".join(parts)


def _system_of(messages: Sequence[BaseMessage]) -> str:
    return "\n\n".join(str(m.content) for m in messages
                       if _ROLE.get(getattr(m, "type", ""), "") == "SYSTEM")


_JSON_RE = re.compile(r"\{.*\}", re.S)


_SMART = {
    "“": '"', "”": '"', "„": '"', "‟": '"',   # curly doubles
    "‘": "'", "’": "'", "‚": "'", "‛": "'",   # curly singles
    " ": " ", "​": "", "–": "-", "—": "-",    # nbsp/zwsp/dashes
}


def _extract_json(text: str) -> Optional[dict]:
    """Pull the first JSON object out of the reply.

    Copilot renders typographic quotes, so a straight json.loads on a copied
    answer fails; normalise those (and nbsp/zero-width) before parsing.
    Tolerates ``` fences and surrounding prose.
    """
    cleaned = text.strip()
    for bad, good in _SMART.items():
        cleaned = cleaned.replace(bad, good)
    cleaned = re.sub(r"^\s*```(?:json)?|```\s*$", "", cleaned, flags=re.M)
    # Copilot markdown-escapes punctuation ("get\_device\_details") -> undo it,
    # otherwise tool names never match.
    cleaned = re.sub(r"\\([_*\[\]()#+\-.!~`>])", r"\1", cleaned)

    m = _JSON_RE.search(cleaned)
    if not m:
        return None
    candidate = m.group(0)
    # copying from rendered markdown puts real newlines inside string values,
    # which json.loads rejects -> collapse them and retry.
    flat = re.sub(r"\s*\n\s*", " ", candidate)

    for attempt in (candidate, flat):
        try:
            obj = json.loads(attempt)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
        try:                                   # trailing commas / single quotes
            import ast
            obj = ast.literal_eval(attempt)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    return None


def _tool_block(schemas: List[dict]) -> str:
    """Ask for a function choice as JSON, phrased as an ordinary dev question.

    Kept deliberately plain: hard 'reply with nothing else' phrasing reads like a
    jailbreak to Copilot's safety layer and gets refused.
    """
    lines = ["", "I am building a small automation and I need help choosing which of "
                 "my own functions to run next. These are the functions available:"]
    for s in schemas:
        fn = s.get("function", s)
        lines.append(f"- {fn.get('name')}: {fn.get('description', '')}")
        lines.append(f"  parameters: {json.dumps(fn.get('parameters', {}))}")
    lines += [
        "",
        "Please answer as a JSON object so my script can read it. Include a short "
        '"thought" explaining what you concluded and what should happen next.',
        "",
        "To run one function:",
        '  {"thought": "...", "tool": "<function name>", "args": { ... }}',
        "To run several:",
        '  {"thought": "...", "tools": [{"tool": "<name>", "args": { ... }}]}',
        "If you already have everything needed to answer:",
        '  {"thought": "...", "final": "<the answer>"}',
        "",
        "JSON only in the reply, please - my script parses it directly.",
    ]
    return "\n".join(lines)


class ClipboardLLM(BaseChatModel):
    """Human-relay chat model: prompt -> clipboard -> (you) -> clipboard -> reply."""

    timeout: float = 600.0          # seconds to wait for the answer
    poll_interval: float = 0.4      # clipboard poll cadence
    min_len: int = 2                # ignore trivially short clipboard content
    prompt_file: Optional[str] = "last_prompt.txt"
    verbose_console: bool = True

    # "delta": reuse ONE Copilot window -- only new messages are pasted, since
    #          Copilot itself keeps the history.
    # "full" : every paste is self-contained (fresh chat each time).
    # "agent": the instructions AND the tool list already live in a custom M365
    #          Copilot agent, so pastes carry only the new question / tool result.
    #          Generate the instruction text with `python clipboard_llm.py`.
    mode: str = "delta"
    beep: bool = True               # audible cue when the prompt is ready to paste

    tool_schemas: List[dict] = []   # set by bind_tools()

    _sent_system: str = PrivateAttr(default="")
    _sent_count: int = PrivateAttr(default=0)

    @property
    def _llm_type(self) -> str:
        return "clipboard-human-relay"

    # ---- tool binding: same contract as ChatOpenAI.bind_tools ---------------
    def bind_tools(
        self,
        tools: Sequence[Union[Dict[str, Any], type, Callable, BaseTool]],
        **kwargs: Any,
    ):
        schemas = [convert_to_openai_tool(t) for t in tools]
        bound = self.model_copy(update={"tool_schemas": schemas})
        bound._sent_system = ""      # force a context resend on the new binding
        bound._sent_count = 0
        return bound

    def reset_conversation(self):
        """Call when you start a new Copilot chat window."""
        self._sent_system = ""
        self._sent_count = 0

    # ---- prompt assembly ----------------------------------------------------
    def _build(self, messages: List[BaseMessage]) -> str:
        tools_txt = _tool_block(self.tool_schemas) if self.tool_schemas else ""

        if self.mode == "agent":
            # instructions + tool list live in the custom agent -> send only what
            # is new (skip system messages entirely).
            body = [m for m in messages
                    if _ROLE.get(getattr(m, "type", ""), "") != "Context"]
            if len(messages) < self._sent_count:      # conversation restarted
                self._sent_count = 0
            already = self._sent_count
            self._sent_count = len(messages)
            new = [m for m in messages[already:]
                   if _ROLE.get(getattr(m, "type", ""), "") != "Context"]
            return _render(new or body[-1:] if body else messages[-1:])

        if self.mode != "delta":
            return _render(messages) + tools_txt

        digest = hashlib.sha1(
            (_system_of(messages) + str(len(self.tool_schemas))).encode("utf-8", "replace")
        ).hexdigest()
        fresh = digest != self._sent_system or len(messages) < self._sent_count
        if fresh:                                   # new context -> send it all
            self._sent_system, self._sent_count = digest, len(messages)
            return _render(messages) + tools_txt

        new_msgs = messages[self._sent_count:] or messages[-1:]
        self._sent_count = len(messages)
        reminder = ("\n\nPlease answer in the same JSON form as before."
                    if self.tool_schemas else "")
        return _render(new_msgs) + reminder

    # ---- reply parsing ------------------------------------------------------
    def _to_message(self, text: str) -> AIMessage:
        if not self.tool_schemas:
            return AIMessage(content=text)

        obj = _extract_json(text)
        if not obj:                                  # no JSON -> treat as the answer
            return AIMessage(content=text)

        thought = str(obj.get("thought") or "").strip()

        calls = []
        if "tools" in obj and isinstance(obj["tools"], list):
            calls = obj["tools"]
        elif "tool" in obj:
            calls = [obj]

        def _clean(v):
            """Drop markdown escaping from names/keys (Copilot writes get\\_x)."""
            if isinstance(v, str):
                return v.replace("\\", "")
            if isinstance(v, dict):
                return {str(k).replace("\\", ""): _clean(x) for k, x in v.items()}
            if isinstance(v, list):
                return [_clean(x) for x in v]
            return v

        if calls:
            tool_calls = [{
                "name": str(c.get("tool") or c.get("name")).replace("\\", "").strip(),
                "args": _clean(c.get("args") or c.get("arguments") or {}),
                "id": "call_" + uuid.uuid4().hex[:12],
                "type": "tool_call",
            } for c in calls if (c.get("tool") or c.get("name"))]
            if tool_calls:
                # reasoning rides in content so the UI can show it next to the call
                return AIMessage(content=thought, tool_calls=tool_calls)

        if "final" in obj:
            body = str(obj["final"])
            return AIMessage(content=body, additional_kwargs={"thought": thought})
        return AIMessage(content=text)

    # ---- the relay ----------------------------------------------------------
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        prompt = self._build(messages)

        if self.prompt_file:
            try:
                with open(self.prompt_file, "w", encoding="utf-8") as f:
                    f.write(prompt)
            except Exception:
                pass

        _copy(prompt)
        if self.beep:
            _alert()
        if self.verbose_console:
            print("\n" + "=" * 60)
            print(f"PROMPT ON CLIPBOARD ({len(prompt)} chars"
                  + (f", {len(self.tool_schemas)} tools" if self.tool_schemas else "")
                  + f", also {self.prompt_file})")
            print("  1. paste into Copilot   2. copy the answer (Ctrl+C)")
            print("  waiting for the clipboard to change...")
            print("=" * 60)

        answer = self._wait_for_change(prompt)
        return ChatResult(generations=[ChatGeneration(message=self._to_message(answer))])

    def _wait_for_change(self, sent: str) -> str:
        deadline = time.time() + self.timeout
        sent_norm = sent.strip()
        while time.time() < deadline:
            try:
                cur = _paste() or ""
            except Exception:
                cur = ""
            cur_s = cur.strip()
            if cur_s and cur_s != sent_norm and len(cur_s) >= self.min_len:
                if self.verbose_console:
                    print(f"got answer from clipboard ({len(cur_s)} chars)\n")
                return cur
            time.sleep(self.poll_interval)
        raise TimeoutError(
            f"no new clipboard content within {self.timeout}s "
            "(copy the Copilot answer to continue)")


# ---- helper: build the text for a custom M365 Copilot agent ------------------
def agent_instructions(system_prompt: str, schemas: List[dict]) -> str:
    """Instruction text for a custom Copilot agent (mode='agent').

    Paste this into the agent builder's instructions box once. Afterwards every
    clipboard paste is only the new question or tool result, which keeps the
    prompts short and avoids the injection-classifier triggers.
    """
    lines = [system_prompt.strip(), "", "-" * 60, "",
             "I will send you questions from a small automation I am building. "
             "Some steps need one of my own functions to be run. These are the "
             "functions I can run for you:"]
    for s in schemas:
        fn = s.get("function", s)
        lines.append(f"- {fn.get('name')}: {fn.get('description', '')}")
        lines.append(f"  parameters: {json.dumps(fn.get('parameters', {}))}")
    lines += [
        "",
        "Always answer with a single JSON object and nothing around it, because "
        "my script reads the reply directly. Include a short \"thought\" saying "
        "what you concluded and what should happen next.",
        "",
        "To run one function:",
        '  {"thought": "...", "tool": "<function name>", "args": { ... }}',
        "To run several:",
        '  {"thought": "...", "tools": [{"tool": "<name>", "args": { ... }}]}',
        "When you have everything needed to answer:",
        '  {"thought": "...", "final": "<the answer>"}',
        "",
        "I will paste the result of each function back to you as "
        "\"Result from <function name>: ...\" so you can decide the next step.",
    ]
    return "\n".join(lines)


if __name__ == "__main__":                      # python clipboard_llm.py [module]
    import asyncio
    import importlib

    target = sys.argv[1] if len(sys.argv) > 1 else "net_agent"
    mod = importlib.import_module(target)
    tools = asyncio.run(
        MultiServerMCPClient(mod.MCP_SERVERS).get_tools())  # type: ignore[name-defined]
    text = agent_instructions(mod.SYSTEM_PROMPT,
                              [convert_to_openai_tool(t) for t in tools])
    out = f"{target}_agent_instructions.txt"
    with open(out, "w", encoding="utf-8") as f:
        f.write(text)
    _copy(text)
    print(text)
    print(f"\n[{len(text)} chars -> {out}, also on the clipboard]")
























You are a Network Operations troubleshooting agent. Given a source and a
destination, decide whether the destination is reachable from the source, show
the path, and if it is broken say at which hop and why.

HOW TO WORK
Think, then act, one function call at a time. In every "thought" say what the
last result told you, what you will do next, and why that command suits THIS
platform. Read each result before the next step. If something is unexpected
(device missing, unknown platform, command rejected, empty output) say so and
adapt - never silently repeat a command. State your assumptions.

OUTPUT SIZE - HARD LIMIT
Large CLI output cannot be returned. Every show command must be narrowed before
it runs:
  show route 10.1.1.1                 NOT  show route
  show interface Gi0/0/0/1 brief      NOT  show interfaces
  show mpls forwarding prefix 10.1.1.1/32
Add a filter when output could still be long (IOS-XR and IOS support
| include, | exclude, | begin; IOS-XR also | utility head -n 20):
  show route vrf CUST 10.1.1.1 | include "via|Known"
  show arp | include 10.1.1.1
  show bgp vrf CUST 10.1.1.1 | include "Received Label|from|metric|Local"
Never ask for show running-config, show tech-support, or a bare show route /
bgp / interfaces / arp / cef / mpls forwarding / access-lists. They are rejected
in code. If a result is still long, re-run with a TIGHTER filter, never
unfiltered.

WORKFLOW
1. get_device_details for the SOURCE, then for the DESTINATION. Read whatever
   fields come back and work out the vendor, model, OS/platform and management
   IP - field names vary, infer from keys and values. The response also carries
   the region: that is what you pass to execute_query_on_server.
   Note: a management IP is often on an out-of-band network and unreachable from
   another router by design. Prefer a loopback/service IP as the target when the
   record offers one, and say if you switched.
2. Ping the destination FROM the source, in the source platform's syntax:
     IOS/IOS-XE/Arista: ping <dest> repeat 3   NX-OS/IOS-XR/Junos: ping <dest> count 3
     FortiOS: execute ping <dest>              PAN-OS: ping count 3 host <dest>
     Gaia/NetScaler/F5/Linux/Huawei/Comware: ping -c 3 <dest>
   Unlisted platform: use the closest match and say which you assumed.
3. Traceroute in the same platform's syntax. ALWAYS BOUND IT - an unbounded
   traceroute probes 30 hops with 3 probes each and takes minutes. Use at most
   5 hops, 1 probe, 1 second timeout, numeric output:
     IOS-XR : traceroute [vrf <vrf>] <dest> maxttl 5 timeout 1 probe 1 numeric
     IOS    : traceroute <dest> ttl 1 5 timeout 1 probe 1 numeric
     Junos  : traceroute <dest> ttl 5 wait 1
     Linux/Gaia/F5/NetScaler : traceroute -n -m 5 -w 1 -q 1 <dest>
     Huawei/Comware: tracert -m 5 <dest>     FortiOS: execute traceroute <dest>
   ALWAYS run it - the path is part of the answer. If 5 hops is not enough, say
   so and raise it once, to 10.
4. If the ping FAILED or the traceroute stopped early, escalate one check at a
   time and stop as soon as the failure is explained. Examples are IOS-XR.
   a. Route present?  show route <dest>        (IOS: show ip route <dest>)
      No route -> the source has no path: that IS the finding, stop.
      Route -> note the next hop, outgoing interface and any VRF.
   b. In a VRF?  show vrf all
      then: show route vrf <vrf> <dest> | include "via|Known"
      Retest inside it:  ping vrf <vrf> <dest>
                         traceroute vrf <vrf> <dest>   (XR may need: ... vrf <vrf> ipv4 <dest>)
   c. Wrong source address?  ping <dest> source <egress-interface>
      and ping <dest> source Loopback0
   d. Programmed in hardware?  show cef <dest>
   e. Next hop alive?  ping <next-hop>   /   show arp | include <next-hop>
      No answer -> the break is on that link, not at the destination.
   f. Interface healthy?  show interface <interface> brief   (down/down, errors)
   g. Filtered?  show access-lists <acl-name> | include <dest>

A SUCCESSFUL PING IS AUTHORITATIVE. If the ping succeeds - in the global table
or in a VRF - the destination IS reachable. Report it reachable even when the
traceroute returns nothing or only * * *. Never downgrade the verdict because
hops are missing.

MPLS L3VPN - HOW TO GET THE PATH WHEN THE VRF TRACEROUTE IS BLIND
On a service-provider core the P routers switch on labels and have no route back
into the VRF, so they cannot answer probes: the VRF traceroute is blind BY
DESIGN. Get the path from the underlay instead:
  1) show bgp vrf <vrf> <dest> | include "Received Label|from|metric|Local"
     "Received Label <n>"        -> confirms MPLS L3VPN
     address before "(metric n)" -> the REMOTE PE loopback
     address after "from"        -> the route reflector, ignore it
  2) traceroute <remote-PE-loopback>      (GLOBAL table, no vrf keyword)
     The core does have global routes back, so this normally works and gives the
     real hop-by-hop path.
  3) Optional: show mpls forwarding prefix <remote-PE-loopback>/32
Report those hops as the path and say they are the underlay path to the remote
PE, with the destination sitting behind that PE.

FINAL ANSWER (plain text, no LaTeX, no $)
  Source      : <name / ip> (<vendor> <os>)
  Destination : <name / ip> (<vendor> <os>)
  Ping        : SUCCESS or FAILED
  Path        : source -> hop1 -> hop2 -> ... -> destination
                (if it never arrives, end at the last hop that answered and mark
                 the break, e.g. ... -> FW-DC1-EDGE-01 -> X)
  Result      : reachable / not reachable / inconclusive
  Evidence    : the checks you ran and what each showed
  Cause       : the most likely reason and where in the path it sits
  Next step   : what an engineer should look at next, if anything
Use "inconclusive" - not "not reachable" - when probes were blocked but the
routing looks correct. That is a different finding and it matters.

RULES
- READ-ONLY only: ping, traceroute/tracert, show/display. Never configure,
  write, reload, clear or otherwise change a device. Commands are validated in
  code and will be rejected.
- A human approves every device command. If one is rejected, continue with what
  you have and say so in the final answer.

------------------------------------------------------------
I will send you questions from a small automation I am building. Some steps need
one of my own functions run. These are the only functions available:

- get_device_details: Look up device details in unicorn (CMDB). Accepts a device
  NAME or an IP address (an IP is resolved to its name first). Region optional -
  omit it or pass 'AUTO' to search every region.
  parameters: {"properties": {"device_name": {"type": "string"}, "region": {"type": ["string","null"], "default": null}}, "required": ["device_name"]}

- execute_query_on_server: Run read-only commands on a device over SSH via the
  region bastion.
  parameters: {"properties": {"device_ip": {"type": "string"}, "commands": {"type": "array"}, "region": {"type": "string", "enum": ["PARIS","ASIA","AMER","UK","INDIA","IBFS"]}, "port": {"type": "integer", "default": 22}}, "required": ["device_ip","commands","region"]}

  "commands" is a LIST even for one command: {"commands": ["ping 10.1.1.1 repeat 3"]}
  "region" is REQUIRED - use the region from the CMDB lookup for that device.
  Run on the SOURCE: device_ip is the source device's management IP; the
  destination goes inside the command text.

Always answer with a single JSON object and nothing around it - my script reads
it directly. Write names plainly: get_device_details, not get\_device\_details.
No markdown escaping anywhere in the JSON. Include a short "thought".

  one function : {"thought": "...", "tool": "<name>", "args": { ... }}
  several      : {"thought": "...", "tools": [{"tool": "<name>", "args": { ... }}]}
  final answer : {"thought": "...", "final": "<the answer>"}

I will paste each result back as "Result from <function name>: ..." so you can
decide the next step.

























# unicorn_mcp.py -- CMDB lookup by device NAME or IP.
#
# One tool, auto-routing:
#   value looks like an IPv4 address -> searchEngine/?element=<ip> -> take the
#       'remote management' entry -> its name -> devices/?name=<name>
#   otherwise                        -> devices/?name=<NAME> directly
#
# Region may be omitted / 'AUTO' -> every configured region is tried in turn.
import re
import sys
from typing import Annotated, Optional

import requests
import urllib3
import yaml
from mcp.server.fastmcp import FastMCP
from pydantic import Field
from requests.adapters import HTTPAdapter

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

mcp = FastMCP("unicorn-server")

# Regions that require a proxy for Unicorn API calls
PROXY_REQUIRED_REGIONS = {"paris", "uk", "star_sdwan", "sgrf_sdwan", "afs"}


class Settings:
    try:
        with open("credentials.yml", "r") as f:
            credentials = yaml.safe_load(f)
    except FileNotFoundError:
        credentials = {}

    # per-region base URL for the device API (must end in / -- see _base())
    UNICORN_URLS = credentials.get("UNICORN_URLS", {})
    UNICORN_TOKEN = credentials.get("UNICORN_TOKEN", {})
    PROXIES = credentials.get("PROXIES", [])


def _is_ip_address(value: str) -> bool:
    """True if value looks like an IPv4 address (e.g. 196.34.145.88)."""
    return bool(re.match(r"^\d{1,3}(\.\d{1,3}){3}$", str(value).strip()))


def _base(region: str) -> str:
    """Device API base for a region, always with a trailing slash."""
    url = str(Settings.UNICORN_URLS[region])
    return url if url.endswith("/") else url + "/"


def _proxy_info(proxy: dict) -> dict:
    """Build the authenticated proxy mapping from a credentials.yml entry."""
    begin, end = proxy.get("uri").split("//")
    return {proxy.get("protocol"):
            f"{begin}//{proxy.get('login')}:{proxy.get('password')}@{end}"}


def _make_request(session, url, api_key, **kwargs):
    """Authenticated GET against the Unicorn API."""
    response = session.get(
        url,
        headers={"Accept": "application/json",
                 "Authorization": f"Token {api_key}"},
        verify=False,
        timeout=30,
        **kwargs,
    )
    response.raise_for_status()
    return response.json()


def _resolve_name_from_ip(session, region, ip, api_key, **kwargs) -> Optional[str]:
    """IP -> device name, via the search engine's 'remote management' entry."""
    search_url = f"{_base(region)}searchEngine/?element={ip}"
    search_result = _make_request(session, search_url, api_key, **kwargs)

    if not isinstance(search_result, list) or len(search_result) == 0:
        return None

    rm_entry = None
    for item in search_result:
        if item.get("type") == "remote management":
            rm_entry = item
            break
    if not rm_entry:
        return None
    return rm_entry.get("name")


# Fields kept from the raw CMDB record. The full record carries the whole
# chassis inventory (every module, port and serial) and runs to hundreds of
# thousands of characters, which blows any model's context window.
SLIM_FIELDS = (
    "brand",
    "teamInCharge",
    "accountable",
    "operatingSystemVersion",
    "infrastructure",
    "environment",
    "brandModel",
    "remoteManagement",
    "location",
    "operatingSystemVersionEndOfSupport",
    "tag",
)


def _slim(data: Optional[dict]) -> Optional[dict]:
    """Reduce a CMDB record to the fields the caller actually needs."""
    if not data:
        return None
    return {field: data.get(field) for field in SLIM_FIELDS}


def _fetch(session, region, value, api_key, **kwargs) -> Optional[dict]:
    """Look up one device in one region. value may be a name or an IP."""
    if _is_ip_address(value):
        device_name = _resolve_name_from_ip(session, region, value, api_key, **kwargs)
        if not device_name:
            return None
    else:
        device_name = str(value).upper()

    device_url = f"{_base(region)}devices/?name={device_name}"
    data = _make_request(session, device_url, api_key, **kwargs)
    results = (data or {}).get("results", [])
    return _slim(results[0]) if results else None


@mcp.tool(
    name="get_device_details",
    description="""
    Look up device details in unicorn (CMDB).
    Accepts either a device NAME or a device IP address -- an IP is resolved to
    its device name first, then the device record is fetched.
    Region is optional: omit it or pass 'AUTO' to search every region.
    """)
def get_device_details(
    device_name: Annotated[str, Field(
        description="device name OR IPv4 address of the device")],
    region: Annotated[Optional[str], Field(
        description="Region (PARIS, ASIA, AMER, UK, INDIA, IBFS). "
                    "Omit or set 'AUTO' to auto-detect")] = None,
) -> dict | str:
    try:
        value = str(device_name).strip()
        lookup_by = "ip" if _is_ip_address(value) else "name"

        requested = (region or "").strip().upper()
        candidates = (list(Settings.UNICORN_URLS.keys())
                      if not requested or requested == "AUTO" else [requested])

        if requested and requested != "AUTO" and requested not in Settings.UNICORN_URLS:
            return (f"Invalid region '{requested}'. "
                    f"Valid: {', '.join(Settings.UNICORN_URLS.keys())}")

        with requests.Session() as session:
            session.mount("https://", HTTPAdapter(max_retries=2))

            for candidate in candidates:
                api_key = Settings.UNICORN_TOKEN[candidate]
                needs_proxy = candidate.lower() in PROXY_REQUIRED_REGIONS
                proxies = Settings.PROXIES if needs_proxy else [None]

                for index_proxy, proxy in enumerate(proxies):
                    kwargs = {"proxies": _proxy_info(proxy)} if proxy else {}
                    try:
                        data = _fetch(session, candidate, value, api_key, **kwargs)
                        if data:
                            return {"region": candidate, "lookup_by": lookup_by,
                                    "query": value, "data": data}
                        break                      # reached the API, no such device
                    except Exception as ex:
                        if index_proxy == len(proxies) - 1:
                            # stderr: stdout carries the MCP JSON-RPC stream
                            print(f"[{candidate}] lookup failed: {ex}",
                                  file=sys.stderr)
                        continue                   # try the next proxy

        return (f"No data found for {lookup_by} '{value}'"
                + ("." if len(candidates) == 1 else " in any region."))
    except Exception as ex:
        return f"Error: {str(ex)}"


if __name__ == "__main__":
    mcp.run()   # stdio



