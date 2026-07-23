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
