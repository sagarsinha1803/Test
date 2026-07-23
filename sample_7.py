const vscode = require('vscode');
const http = require('http');

const PORT = 11434;
let server;

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
        const { messages, model: family } = JSON.parse(body);

        // list ALL models (any vendor), then pick requested family or first
        const all = await vscode.lm.selectChatModels({});
        if (!all.length) throw new Error(
          'no LM models from any extension. Install Copilot Chat, or an ext that registers vscode.lm');
        const model = (family && all.find(m => m.family === family)) || all[0];

        const lmMsgs = messages.map(m =>
          vscode.LanguageModelChatMessage.User(m.content));

        const resp = await model.sendRequest(
          lmMsgs, {}, new vscode.CancellationTokenSource().token);

        let text = '';
        for await (const chunk of resp.text) text += chunk;

        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({
          choices: [{ message: { role: 'assistant', content: text } }]
        }));
      } catch (e) {
        res.writeHead(500, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: String(e && e.message || e) }));
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
