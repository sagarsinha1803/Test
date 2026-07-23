const vscode = require('vscode');
const http = require('http');

const PORT = 11434;
let server;

function safeParse(s) {
  try { return typeof s === 'string' ? JSON.parse(s) : (s || {}); }
  catch { return {}; }
}

// OpenAI messages -> vscode.lm chat messages (handles tool_calls + tool results)
function toLMMessages(messages) {
  const out = [];
  for (const m of messages || []) {
    if (m.role === 'system' || m.role === 'user') {
      out.push(vscode.LanguageModelChatMessage.User(String(m.content ?? '')));
    } else if (m.role === 'assistant') {
      if (Array.isArray(m.tool_calls) && m.tool_calls.length) {
        const parts = [];
        if (m.content) parts.push(new vscode.LanguageModelTextPart(String(m.content)));
        for (const tc of m.tool_calls) {
          parts.push(new vscode.LanguageModelToolCallPart(
            tc.id, tc.function.name, safeParse(tc.function.arguments)));
        }
        out.push(vscode.LanguageModelChatMessage.Assistant(parts));
      } else {
        out.push(vscode.LanguageModelChatMessage.Assistant(String(m.content ?? '')));
      }
    } else if (m.role === 'tool') {
      // tool result must ride inside a User message in vscode.lm
      out.push(vscode.LanguageModelChatMessage.User([
        new vscode.LanguageModelToolResultPart(
          m.tool_call_id, [new vscode.LanguageModelTextPart(String(m.content ?? ''))])
      ]));
    }
  }
  return out;
}

// OpenAI tools -> vscode.lm LanguageModelChatTool[]
function toLMTools(tools) {
  if (!Array.isArray(tools) || !tools.length) return undefined;
  return tools.map(t => ({
    name: t.function.name,
    description: t.function.description || '',
    inputSchema: t.function.parameters || { type: 'object', properties: {} }
  }));
}

async function handleChat(body, res) {
  const { messages, model: family, tools } = JSON.parse(body);

  const all = await vscode.lm.selectChatModels({});
  if (!all.length) throw new Error(
    'no LM models from any extension. Install Copilot Chat, or an ext that registers vscode.lm');
  const model = (family && all.find(m => m.family === family)) || all[0];

  const lmMsgs = toLMMessages(messages);
  const lmTools = toLMTools(tools);

  const options = {};
  if (lmTools) {
    options.tools = lmTools;
    options.toolMode = vscode.LanguageModelChatToolMode.Auto;
  }

  const resp = await model.sendRequest(
    lmMsgs, options, new vscode.CancellationTokenSource().token);

  let text = '';
  const toolCalls = [];
  for await (const part of resp.stream) {
    if (part instanceof vscode.LanguageModelTextPart) {
      text += part.value;
    } else if (part instanceof vscode.LanguageModelToolCallPart) {
      toolCalls.push({
        id: part.callId,
        type: 'function',
        function: { name: part.name, arguments: JSON.stringify(part.input || {}) }
      });
    }
  }

  const message = { role: 'assistant', content: text || null };
  let finish = 'stop';
  if (toolCalls.length) {
    message.tool_calls = toolCalls;
    finish = 'tool_calls';
  }

  res.writeHead(200, { 'Content-Type': 'application/json' });
  res.end(JSON.stringify({
    id: 'chatcmpl-' + Date.now(),
    object: 'chat.completion',
    model: `${model.vendor}/${model.family}`,
    choices: [{ index: 0, message, finish_reason: finish }]
  }));
}

async function activate(ctx) {
  server = http.createServer((req, res) => {
    if (req.method !== 'POST') {
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ ok: true, hint: 'POST /v1/chat/completions' }));
      return;
    }
    let body = '';
    req.on('data', c => (body += c));
    req.on('end', async () => {
      try {
        await handleChat(body, res);
      } catch (e) {
        res.writeHead(500, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: String((e && e.message) || e) }));
      }
    });
  });

  server.on('error', e =>
    vscode.window.showErrorMessage(`TstVS bridge: ${e.message}`));

  server.listen(PORT, () =>
    vscode.window.showInformationMessage(`TstVS bridge on http://localhost:${PORT}`));

  ctx.subscriptions.push(
    { dispose: () => server && server.close() },
    vscode.commands.registerCommand('tstvs.status', () =>
      vscode.window.showInformationMessage(
        server && server.listening
          ? `TstVS bridge listening on ${PORT}`
          : 'TstVS bridge NOT listening')),
    vscode.commands.registerCommand('tstvs.models', async () => {
      const all = await vscode.lm.selectChatModels({});
      const list = all.map(m => `${m.vendor}/${m.family} (${m.name})`).join(', ') || 'NONE';
      vscode.window.showInformationMessage(`LM models: ${list}`);
    })
  );
}

function deactivate() {
  if (server) server.close();
}

module.exports = { activate, deactivate };




curl -X POST http://localhost:11434/v1/chat/completions -H "Content-Type: application/json" -d "{\"messages\":[{\"role\":\"user\",\"content\":\"say hi\"}]}"


curl -X POST http://localhost:11434/v1/chat/completions -H "Content-Type: application/json" -d "{\"model\":\"gpt-4o\",\"messages\":[{\"role\":\"user\",\"content\":\"say hi\"}]}"


{
  "name": "tstvs",
  "displayName": "TstVS LM Bridge",
  "description": "Expose VS Code Copilot LM as local OpenAI-compatible endpoint",
  "version": "0.0.1",
  "publisher": "local",
  "engines": { "vscode": "^1.95.0" },
  "main": "./extension.js",
  "activationEvents": [ "onStartupFinished" ],
  "contributes": {
    "commands": [
      { "command": "tstvs.status", "title": "TstVS: Bridge Status" },
      { "command": "tstvs.models", "title": "TstVS: List Copilot Models" }
    ]
  },
  "devDependencies": {
    "@types/node": "^26.1.1",
    "@types/vscode": "^1.125.0",
    "typescript": "^7.0.2"
  }
}













"""LangGraph tool-calling agent, tools served by an MCP server (FastMCP).

Tools come from tools_mcp.py via MultiServerMCPClient (add more MCPs to the dict).
MCP tools are async -> the graph is driven with ainvoke.

    python -m pip install --user langchain-openai langgraph langchain-mcp-adapters
    python agent.py       # needs the vscode.lm bridge running on :11434
"""
import asyncio
from typing import Annotated, TypedDict

from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

MODEL = "gpt-4o"       # family that supports tool-calling (List Copilot Models)
MAX_TOOL_LOOPS = 2     # cap tool rounds -> no infinite loop

llm = ChatOpenAI(
    base_url="http://localhost:11434/v1",
    api_key="dummy",       # bridge ignores it
    model=MODEL,
    temperature=0,
)

# register MCP servers here (stdio or http). Add your other MCPs to this dict.
MCP_SERVERS = {
    "tools": {
        "command": "python",
        "args": ["tools_mcp.py"],   # run agent.py from this folder
        "transport": "stdio",
    },
}


class State(TypedDict):
    messages: Annotated[list, add_messages]
    loops: int          # how many tool rounds have run


async def build_agent():
    """Spawn MCP servers, pull their tools, compile the LangGraph agent."""
    client = MultiServerMCPClient(MCP_SERVERS)
    tools = await client.get_tools()          # all MCP tools, merged
    llm_with_tools = llm.bind_tools(tools)
    tool_node = ToolNode(tools)

    def agent(state: State):
        return {"messages": [llm_with_tools.invoke(state["messages"])]}

    async def tools_step(state: State):
        result = await tool_node.ainvoke(state)
        result["loops"] = state.get("loops", 0) + 1   # count this tool round
        return result

    def route(state: State):
        last = state["messages"][-1]
        wants_tools = bool(getattr(last, "tool_calls", None))
        if wants_tools and state.get("loops", 0) < MAX_TOOL_LOOPS:
            return "tools"
        return END        # no tool_calls, or cap reached -> stop

    g = StateGraph(State)
    g.add_node("agent", agent)
    g.add_node("tools", tools_step)
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", route, {"tools": "tools", END: END})
    g.add_edge("tools", "agent")
    return g.compile()


async def main():
    app = await build_agent()
    out = await app.ainvoke({"messages": [
        ("system", "Reply in plain text. No LaTeX, no $ symbols."),
        ("user", "What is 23 * 7, and how many letters are in the word 'automation'?"),
    ]})
    for m in out["messages"]:
        m.pretty_print()


if __name__ == "__main__":
    asyncio.run(main())

