"""Basic source->dest reachability agent (LangGraph, deterministic) with a
human-in-the-loop gate before EVERY SSH tool call.

Wired to the user's real MCPs:
  - unicorn : get_device_details(device_name, region)  -> {"region","data"} | str
  - ssh     : execute_query_on_server(device_ip, commands[list], region, port)

Flow:
  1. cmdb       Unicorn -> device details (source + dest); grab SOURCE region
  2. ping       SSH     -> ping <dest> ON source   (paused for approval)
  3. traceroute SSH     -> traceroute <dest>, only if ping failed (paused)
  4. report     LLM     -> plain-text summary + likely cause

The SSH gate uses LangGraph interrupt(): the graph pauses, the caller (CLI or
Streamlit UI) approves/rejects, then resumes with Command(resume=bool).

    uv run python agent/path_agent.py --source 10.10.1.20 --dest 172.20.5.10
"""
import argparse
import asyncio
import json
import os
import re
import sys
from typing import Optional, TypedDict

from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command

# ---- CONFIG: point at YOUR unicorn + ssh MCP files --------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))


def _stdio(script):
    return {"command": sys.executable, "args": [os.path.join(_HERE, script)],
            "transport": "stdio"}


MCP_SERVERS = {
    "unicorn": _stdio("unicorn_mcp.py"),
    # "ssh":   _stdio("troubleshoot_agent_mcp.py"),   # commented: isolate unicorn first
}

UNICORN_TOOL = "get_device_details"        # {"device_name", "region"}
SSH_TOOL     = "execute_query_on_server"    # {"device_ip", "commands":[..], "region", "port"}

MODEL = "gpt-4o"
llm = ChatOpenAI(base_url="http://localhost:11434/v1", api_key="dummy",
                 model=MODEL, temperature=0)

REQUIRE_APPROVAL = True   # human-in-the-loop gate before every SSH tool call
MAX_HOPS = 3              # cap traceroute hops so it can't run long
PING_COUNT = 3            # number of ping probes (Cisco default is 5)


# ---- helpers to normalize MCP results (may arrive as JSON strings) ----------
def _as_obj(res):
    if isinstance(res, (dict, list)):
        return res
    if isinstance(res, str):
        try:
            return json.loads(res)
        except Exception:
            return res
    return res


def _region_of(unicorn_res):
    o = _as_obj(unicorn_res)
    if isinstance(o, dict) and o.get("region"):
        return str(o["region"]).upper()
    if isinstance(unicorn_res, str):
        m = re.search(r'"region"\s*:\s*"([^"]+)"', unicorn_res)
        if m:
            return m.group(1).upper()
    return None


def _ssh_stdout(ssh_res):
    o = _as_obj(ssh_res)
    if isinstance(o, list):
        return "\n".join(str(r.get("stdout", "")) for r in o if isinstance(r, dict))
    if isinstance(o, dict):
        return str(o.get("stdout") or o.get("error") or o)
    return str(ssh_res)


def _ping_ok(out: str) -> bool:
    o = out.lower()
    if "success rate is 100" in o or "0% packet loss" in o:  return True
    if re.search(r"success rate is [1-9]\d? percent", o):    return True
    return False


class State(TypedDict, total=False):
    source: str
    dest: str
    src_region: Optional[str]
    cmdb: dict
    ping_ok: bool
    ping_raw: str
    hops: list
    failed_hop: Optional[str]
    traceroute_raw: str
    report: str


async def build(checkpointer=None):
    client = MultiServerMCPClient(MCP_SERVERS)
    tools = await client.get_tools()
    by_name = {t.name: t for t in tools}

    async def call(name, args):
        tool = by_name.get(name)
        if not tool:
            return f"[tool '{name}' not found; have: {list(by_name)}]"
        try:
            return await tool.ainvoke(args)
        except Exception as e:
            return f"[error calling {name}: {e}]"

    async def ssh_run(device_ip, region, command):
        if not region:
            return "[error: no region for source device - unicorn did not return one]"
        if REQUIRE_APPROVAL:
            # pause the graph; caller resumes with Command(resume=True/False)
            approved = interrupt({
                "action": "ssh_command",
                "device_ip": device_ip,
                "region": region,
                "command": command,
            })
            if not approved:
                return "[aborted by reviewer]"
        res = await call(SSH_TOOL, {"device_ip": device_ip, "commands": [command],
                                    "region": region, "port": 22})
        return _ssh_stdout(res)

    async def cmdb(state: State):
        src = await call(UNICORN_TOOL, {"device_name": state["source"], "region": "AUTO"})
        dst = await call(UNICORN_TOOL, {"device_name": state["dest"], "region": "AUTO"})
        return {"cmdb": {"source": str(src), "dest": str(dst)},
                "src_region": _region_of(src)}

    async def ping(state: State):
        out = await ssh_run(state["source"], state.get("src_region"),
                            f"ping {state['dest']} repeat {PING_COUNT}")
        return {"ping_raw": out, "ping_ok": _ping_ok(out)}

    async def traceroute(state: State):
        # Cisco: 'ttl 1 <max>' bounds hops; short timeout/probe so it returns fast
        cmd = f"traceroute {state['dest']} ttl 1 {MAX_HOPS} timeout 1 probe 1"
        out = await ssh_run(state["source"], state.get("src_region"), cmd)
        prompt = ('From this traceroute, return ONLY JSON: '
                  '{"hops":[...ordered names/ips...], "failed_after":"<last responding '
                  'hop before timeouts, or null>"}.\n\n' + out)
        raw = llm.invoke(prompt).content
        m = re.search(r"\{.*\}", raw, re.S)
        data = json.loads(m.group(0)) if m else {"hops": [], "failed_after": None}
        return {"traceroute_raw": out, "hops": data.get("hops", []),
                "failed_hop": data.get("failed_after")}

    async def report(state: State):
        evidence = {
            "source": state["source"], "dest": state["dest"],
            "source_region": state.get("src_region"),
            "cmdb": state.get("cmdb"),
            "ping_ok": state.get("ping_ok"),
            "path_hops": state.get("hops"),
            "failed_after_hop": state.get("failed_hop"),
        }
        prompt = (
            "You are a network engineer. Given this evidence (JSON), write a short "
            "plain-text report: is the destination reachable from the source, the path "
            "and where it stops (if any), and the most likely cause. No LaTeX, no $.\n\n"
            + json.dumps(evidence, indent=2, default=str))
        return {"report": llm.invoke(prompt).content}

    def after_ping(state: State):
        return "report" if state.get("ping_ok") else "traceroute"

    g = StateGraph(State)
    for name, fn in [("cmdb", cmdb), ("ping", ping),
                     ("traceroute", traceroute), ("report", report)]:
        g.add_node(name, fn)
    g.add_edge(START, "cmdb")
    g.add_edge("cmdb", "ping")
    g.add_conditional_edges("ping", after_ping,
                            {"report": "report", "traceroute": "traceroute"})
    g.add_edge("traceroute", "report")
    g.add_edge("report", END)
    return g.compile(checkpointer=checkpointer or MemorySaver())


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--dest", required=True)
    args = ap.parse_args()

    app = await build()
    config = {"configurable": {"thread_id": "cli-1"}}
    state = await app.ainvoke({"source": args.source, "dest": args.dest}, config)

    # resume loop: keep approving each paused SSH action until the graph finishes
    while "__interrupt__" in state:
        payload = state["__interrupt__"][0].value
        print("\n" + "!" * 60)
        print("SSH ACTION NEEDS HUMAN APPROVAL")
        print(f"  device_ip : {payload['device_ip']}")
        print(f"  region    : {payload['region']}")
        print(f"  command   : {payload['command']}")
        print("!" * 60)
        ans = await asyncio.to_thread(input, "Run this on the device? [y/N]: ")
        approved = ans.strip().lower() in ("y", "yes")
        state = await app.ainvoke(Command(resume=approved), config)

    print("\n" + "=" * 60)
    print(state.get("report", "(no report)"))


if __name__ == "__main__":
    asyncio.run(main())



















"""Streamlit UI for path_agent with human-in-the-loop SSH approval.

    python -m pip install --user streamlit        (or: uv pip install streamlit)
    python -m streamlit run path_ui.py            (or: uv run streamlit run path_ui.py)

Needs the vscode.lm bridge on :11434 and the unicorn/ssh MCP files alongside.

The graph pauses (interrupt) before every SSH command. The UI shows the pending
action with Approve / Reject buttons, then resumes the graph.
"""
import asyncio
import uuid

import streamlit as st
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from path_agent import build

st.set_page_config(page_title="Network Troubleshooter", page_icon="🛠️")
st.title("🛠️ Source → Dest Troubleshooter")

ss = st.session_state
ss.setdefault("checkpointer", MemorySaver())   # persists across reruns in this session
ss.setdefault("thread_id", None)
ss.setdefault("phase", "idle")                 # idle | await_approval | done
ss.setdefault("pending", None)                 # the interrupt payload
ss.setdefault("report", None)


async def _run(resume=None):
    """Build the agent (fresh MCP connections) against the persisted checkpointer,
    then start or resume the graph. Returns the resulting state."""
    app = await build(checkpointer=ss.checkpointer)
    config = {"configurable": {"thread_id": ss.thread_id}}
    if resume is None:
        return await app.ainvoke({"source": ss.src, "dest": ss.dst}, config)
    return await app.ainvoke(Command(resume=resume), config)


def _apply(state):
    if "__interrupt__" in state:
        ss.pending = state["__interrupt__"][0].value
        ss.phase = "await_approval"
    else:
        ss.report = state.get("report", "(no report)")
        ss.pending = None
        ss.phase = "done"


# ---- input form -------------------------------------------------------------
with st.form("inputs"):
    c1, c2 = st.columns(2)
    src = c1.text_input("Source IP / device", "10.10.1.20")
    dst = c2.text_input("Destination IP / device", "172.20.5.10")
    run = st.form_submit_button("Run troubleshooting")

if run:
    ss.src, ss.dst = src, dst
    ss.thread_id = str(uuid.uuid4())     # fresh run
    ss.checkpointer = MemorySaver()
    ss.report = None
    _apply(asyncio.run(_run()))

# ---- approval gate ----------------------------------------------------------
if ss.phase == "await_approval" and ss.pending:
    p = ss.pending
    st.warning("SSH action needs approval before it runs on the device.")
    st.code(f"device_ip : {p['device_ip']}\n"
            f"region    : {p['region']}\n"
            f"command   : {p['command']}", language="text")
    a, r = st.columns(2)
    if a.button("✅ Approve", use_container_width=True):
        _apply(asyncio.run(_run(resume=True)))
        st.rerun()
    if r.button("❌ Reject", use_container_width=True):
        _apply(asyncio.run(_run(resume=False)))
        st.rerun()

# ---- final report -----------------------------------------------------------
if ss.phase == "done" and ss.report:
    st.subheader("Troubleshooting Report")
    st.text(ss.report)













# cat troubleshoot_agent_mcp.py
# Reconstructed from your screenshots -- verify against your original.
"""
Don't run this MCP server directly from the base machine.
Escalation will arise.
"""

from environs import Env
from fastmcp import FastMCP
from pydantic import Field
from typing import Annotated, Literal
from sqlalchemy import text
import paramiko
import yaml


class Settings:
    # _basedir = os.path.abspath(os.path.dirname(__file__))

    TRAP_HTTP_EXCEPTIONS = True
    ERROR_404_HELP = True
    BUNDLE_ERRORS = True

    try:
        with open("credentials.yml", "r") as f:
            credentials = yaml.safe_load(f)
    except FileNotFoundError:
        credentials = {}

    SSH_JUMPHOST_DETAILS = credentials.get("SSH_JUMPHOST_DETAILS", {})
    DEVICE_DETAILS_SSH = credentials.get("DEVICE_DETAILS_SSH", {})


mcp = FastMCP("device-troubleshooting-server")


@mcp.prompt(
    name="system_prompt",
    description="System prompt for the device troubleshooter agent.")
def system_prompt() -> str:
    return """
    You are a Network CLI Assistant

    Your job is to interact with network devices by:
      - Interpreting natural language requests
      - Converting them into safe, read-only CLI commands
      - Executing them using a secure tool
      - Returning human-friendly summaries of the results

    -------------------------------------------------------------
    Device Details (provided by user):
    {device_details}

    -------------------------------------------------------------
    Workflow:

      Intent Detection:
        - Understand the user's request (e.g., check CPU, OSPF, interface status)

      System Identification:
        - Use the device details provided by the user.

      Cache Check:
        - If (device_ip, region, command) has been executed before, return cached result.

      Command Generation:
        - Convert the request into a valid read-only CLI command.
        - If vendor/OS is known, adjust the command format accordingly.

      Execution:
        - execute_query_on_server(device_ip: str, region: str, command: str, port: int = 22)

      Output Interpretation:
        - Summarize CLI output in clear, user-friendly language.

      Session Management:
        - Retain session state for follow-up queries.
        - Use 'logout' to clear session context.

    -------------------------------------------------------------
    Security Rules:
      Only execute read-only commands: show, ping, traceroute
      Never run configuration or destructive commands: configure terminal, reload, set, etc.
    """


import socket
from paramiko import SSHClient, AutoAddPolicy
from paramiko.ssh_exception import AuthenticationException, SSHException
import logging
import time

logging.getLogger("paramiko").setLevel(logging.CRITICAL)


class BastionError(Exception):
    pass


class Bastion:
    def __init__(self, region, timeout=15, keepalive=30):
        self.host = Settings.SSH_JUMPHOST_DETAILS[region]["IP"]
        self.user = Settings.SSH_JUMPHOST_DETAILS[region]["USERNAME"]
        self.key = Settings.SSH_JUMPHOST_DETAILS[region]["KEY_PATH"]
        self.port = Settings.SSH_JUMPHOST_DETAILS[region].get("PORT", 22)
        self.region = region
        self.password = None
        self.timeout = timeout
        self.keepalive = keepalive
        self.client: SSHClient | None = None

    def open(self):
        try:
            c = SSHClient()
            c.set_missing_host_key_policy(AutoAddPolicy())
            c.connect(
                hostname=self.host,
                port=self.port,
                username=self.user,
                key_filename=self.key,
                password=self.password,
                look_for_keys=False,
                allow_agent=False,
                timeout=self.timeout,
            )
            c.get_transport().set_keepalive(self.keepalive)
            self.client = c
        except (AuthenticationException, SSHException,
                socket.timeout, OSError) as e:
            self.close()
            raise BastionError(f"Failed to connect bastion: {e}")

    def open_channel(self, target_host, target_port=22):
        if not self.client:
            raise BastionError("Bastion not opened")
        transport = self.client.get_transport()
        dest = (target_host, target_port)
        src = (self.host, 0)
        return transport.open_channel("direct-tcpip", dest, src)

    def close(self):
        try:
            if self.client:
                self.client.close()
        finally:
            self.client = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, *_):
        self.close()
        return False


def _run_composite_sequence(
    client: SSHClient,
    parts: list[str],
    per_cmd_wait: float = 0.3,
    read_window: float = 1.5,
) -> dict:
    """
    Execute a sequence of commands in a single interactive shell.
    ignore_output_for: set of command strings whose output should be discarded
    (case-insensitive). Example: {'bash','exit'}
    """
    ignore_output_for = {"bash", "exit"}   # default ignore list
    norm_ignore = {c.lower().strip() for c in ignore_output_for}

    shell = client.invoke_shell()
    kept_chunks: list[str] = []
    try:
        for part in parts:
            cmd_sent = part.strip()
            shell.send(part + '\n')
            time.sleep(per_cmd_wait)
            end = time.time() + read_window
            chunk = ''
            while time.time() < end:
                while shell.recv_ready():
                    chunk += shell.recv(4096).decode(errors='ignore')
                time.sleep(0.05)
            if cmd_sent.lower() not in norm_ignore:
                kept_chunks.append(chunk)
        if parts[-1].strip().lower() != 'exit':
            shell.send('exit\n')
            time.sleep(0.2)
            exit_chunk = ''
            while shell.recv_ready():
                exit_chunk += shell.recv(4096).decode(errors='ignore')
            if 'exit' not in norm_ignore:
                kept_chunks.append(exit_chunk)
        rc = 0
    except Exception as e:
        kept_chunks.append(f"[COMPOSITE_ERROR] {e}")
        rc = -1
    finally:
        try:
            shell.close()
        except Exception:
            pass
    return {
        "stdout": ''.join(kept_chunks),
        "stderr": "",
        "rc": rc,
    }


@mcp.tool(
    name='execute_query_on_server',
    description='This tool execute read only commands on the networking device',
)
def execute_query_on_server(
    device_ip: Annotated[str, Field(description="IP address of the device")],
    commands: Annotated[list, Field(description="list of command to execute on the device")],
    region: Annotated[Literal['PARIS', 'ASIA', 'AMER', 'UK', 'INDIA', 'IBFS'],
                      Field(description="Region device belongs to")],
    port: Annotated[int, Field(description='port of the device', default=22)] = 22,
):
    """
    device = { 'host': 'x.x.x.x', 'user': 'username',
               'password': 'pwd' | 'key': 'path', 'port': 22 }
    """
    region = region.lower()
    channel = None
    client = SSHClient()
    client.set_missing_host_key_policy(AutoAddPolicy())
    try:
        with Bastion(region.lower()) as bastion:
            channel = bastion.open_channel(device_ip, port)
            client.connect(
                hostname=device_ip,
                port=port,
                username=Settings.DEVICE_DETAILS_SSH[bastion.region]["username"],
                password=Settings.DEVICE_DETAILS_SSH[bastion.region]["password"],
                key_filename=None,
                sock=channel,
                look_for_keys=False,
                allow_agent=False,
                timeout=30,
            )
            results = []
            for cmd in commands:
                if isinstance(cmd, str) and '**' in cmd:
                    parts = [p for p in cmd.split('**') if p.strip()]
                    composite_res = _run_composite_sequence(client, parts)
                    results.append({
                        "cmd": cmd,
                        "stdout": composite_res["stdout"],
                        "stderr": composite_res["stderr"],
                        "rc": composite_res["rc"],
                    })
                else:
                    stdin, stdout, stderr = client.exec_command(cmd, timeout=30)
                    out = stdout.read().decode(errors="ignore")
                    err = stderr.read().decode(errors="ignore")
                    rc = stdout.channel.recv_exit_status()
                    results.append({
                        "cmd": cmd,
                        "stdout": out,
                        "stderr": err,
                        "rc": rc,
                    })
            return results
    except Exception as e:
        return {"host": device_ip, "ok": False, "error": str(e)}
    finally:
        try:
            client.close()
        except Exception:
            pass
        if channel:
            try:
                channel.close()
            except Exception:
                pass


if __name__ == "__main__":
    print('nemo Server Started...')
    mcp.run(transport='stdio')



















# cat unicorn_server.py
# Reconstructed from your screenshots -- verify against your original.
from mcp.server.fastmcp import FastMCP
from pydantic import Field
from typing import Annotated, Literal
import requests
from requests.adapters import HTTPAdapter
from environs import Env
import yaml

env = Env()
env.read_env()

mcp = FastMCP("unicorn-server")


class Settings:
    # _basedir = os.path.abspath(os.path.dirname(__file__))

    TRAP_HTTP_EXCEPTIONS = True
    ERROR_404_HELP = True
    BUNDLE_ERRORS = True

    try:
        with open("credentials.yml", "r") as f:
            credentials = yaml.safe_load(f)
    except FileNotFoundError:
        credentials = {}

    UNICORN_URLS = credentials.get("UNICORN_URLS", {})
    UNICORN_TOKEN = credentials.get("UNICORN_TOKEN", {})
    PROXIES = credentials.get("PROXIES", {})

    SOGPT_CHAT_COMPLETION_URL = env("SOGPT_CHAT_COMPLETION_URL")

    CLIENT_ID = env("CLIENT_ID")
    CLIENT_SECRET = env("CLIENT_SECRET")
    X_APPLICATION = env("X-APPLICATION")
    X_KEY_NAME = env("X-KEY-NAME")

    UNITY_CLIENT_ID = env("UNITY_CLIENT_ID")
    UNITY_CLIENT_SECRET = env("UNITY_CLIENT_SECRET")
    UNITY_SCOPE = env("UNITY_SCOPE")
    UNITY_ENV = env("UNITY_ENV")


@mcp.tool(
    name="get_device_details",
    description="""
    This tool used to search the device details from unicorn (CMDB).
    It give the device details based on device name and region provided by the user
    """)
def get_unicorn_device_details(
    device_name: Annotated[str, Field(description="name of the device")],
    region: Annotated[str | None, Field(
        description="Region (PARIS, ASIA, AMER, UK, INDIA, IBFS). "
                    "Omit or set 'AUTO' to auto-detect")] = None,
) -> dict | str:
    try:
        with requests.Session() as session:
            session.mount('https://', HTTPAdapter(max_retries=3))

            def fetch(region_name: str) -> dict | None:
                for proxy in Settings.PROXIES:
                    try:
                        begin, end = proxy.get("uri").split("//")
                        proxy_info = {
                            proxy.get("protocol"):
                                f"{begin}//{proxy.get('login')}:{proxy.get('password')}@{end}"
                        }
                        url = f"{Settings.UNICORN_URLS[region_name]}/{device_name.lower()}"
                        headers = {
                            'Authorization': f"Token {Settings.UNICORN_TOKEN[region_name]}",
                            'Accept': 'application/json'
                        }
                        response = session.get(
                            url,
                            verify=False,
                            headers=headers,
                            proxies=proxy_info if region_name not in ['ASIA', 'AMER', 'INDIA'] else None,
                            timeout=(3, 60),
                        )
                        if response.status_code in (200, 201):
                            return response.json()
                    except Exception as ex:
                        # Continue with next proxy or region
                        print(f"[{region_name}] proxy attempt failed: {ex}")
                return None

            requested_region = (region or "").strip().upper()
            if not requested_region or requested_region == "AUTO":
                for candidate in Settings.UNICORN_URLS.keys():
                    data = fetch(candidate)
                    if data:
                        return {"region": candidate, "data": data}
                return "No data found in any region."
            else:
                if requested_region not in Settings.UNICORN_URLS:
                    return (f"Invalid region '{requested_region}'. "
                            f"Valid: {', '.join(Settings.UNICORN_URLS.keys())}")
                data = fetch(requested_region)
                return {"region": requested_region, "data": data} if data else "No data found!!"
    except Exception as ex:
        return f"Error: {str(ex)}"


if __name__ == "__main__":
    # mcp.run(
    #     transport="sse",
    #     host="175.60.57.250",
    #     port=4200,
    #     path="/unicorn_server",
    #     log_level="debug"
    # )
    mcp.run()
