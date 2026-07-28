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
            final = obj["final"]
            # a structured final answer is kept as JSON so the UI can lay it out
            body = (json.dumps(final, ensure_ascii=False)
                    if isinstance(final, (dict, list)) else str(final))
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
