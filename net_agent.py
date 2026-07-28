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

FINAL ANSWER
For a SOURCE-TO-DESTINATION TROUBLESHOOTING request, "final" must be an OBJECT
with exactly these keys, never free prose:
  {"thought": "...", "final": {
     "source":      "<name> / <ip> (<vendor> <os>)",
     "destination": "<name> / <ip> (<vendor> <os>)",
     "ping":        "SUCCESS" | "FAILED" | "NOT RUN",
     "path":        ["<source>", "<hop1>", "<hop2>", "<destination or X>"],
     "result":      "REACHABLE" | "NOT REACHABLE" | "INCONCLUSIVE",
     "evidence":    ["<one line per check you ran and what it showed>"],
     "cause":       "<most likely reason and where in the path it sits>",
     "next_step":   "<what an engineer should look at next, or 'none'>"
  }}
"path" is a LIST of hops in order, ending in "X" when traffic never arrives.
Use "INCONCLUSIVE" -- not "NOT REACHABLE" -- when probes were blocked but the
routing looks correct. Include every key even if a value is "none".

For ANY OTHER question (a device lookup, an uptime check, a general question),
do NOT use that schema. "final" is then either a plain string or a small object
of your own keys -- answer normally, keep it short, and use markdown (bullets,
**bold**) so it renders cleanly.

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
