// End-to-end: real agent loop through the proxy, streaming SSE, real tools executed.
// Run: node e2e.mjs
import fs from 'node:fs';
import path from 'node:path';

const PROXY = 'http://127.0.0.1:11435/v1/chat/completions';
const ROOT = process.env.BENCH_ROOT || process.cwd();

const TOOLS = [
  { type: 'function', function: { name: 'list_files', description: 'List files in a directory (max 3 levels)', parameters: { type: 'object', properties: { dir: { type: 'string' } }, required: [] } } },
  { type: 'function', function: { name: 'read_file', description: 'Read a text file', parameters: { type: 'object', properties: { path: { type: 'string' } }, required: ['path'] } } },
];

function runTool(name, args) {
  if (name === 'list_files') {
    const dir = path.resolve(ROOT, args.dir || '.');
    return fs.readdirSync(dir).map((f) => (fs.statSync(path.join(dir, f)).isDirectory() ? f + '/' : f)).join('\n');
  }
  if (name === 'read_file') {
    const p = path.resolve(ROOT, args.path);
    const text = fs.readFileSync(p, 'utf8');
    return text.split('\n').slice(0, 12).join('\n');
  }
  return `unknown tool ${name}`;
}

/** Stream one turn; returns { content, toolCalls, finish } the way Luban parses it. */
async function turn(messages) {
  const res = await fetch(PROXY, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model: 'qwen2.5-coder:7b', messages, tools: TOOLS, stream: true }),
  });
  if (!res.ok) throw new Error(`proxy ${res.status}: ${await res.text()}`);

  let content = '';
  const calls = [];
  let finish = null;
  const reader = res.body.getReader();
  const dec = new TextDecoder();
  let buf = '';
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    let nl;
    while ((nl = buf.indexOf('\n')) !== -1) {
      const line = buf.slice(0, nl).trim();
      buf = buf.slice(nl + 1);
      if (!line.startsWith('data:')) continue;
      const data = line.slice(5).trim();
      if (data === '[DONE]') continue;
      const chunk = JSON.parse(data);
      const choice = chunk.choices?.[0];
      if (!choice) continue;
      if (choice.delta?.content) content += choice.delta.content;
      for (const tc of choice.delta?.tool_calls || []) {
        const slot = (calls[tc.index] ??= { id: null, name: null, arguments: '' });
        if (tc.id) slot.id = tc.id;
        if (tc.function?.name) slot.name = tc.function.name;
        if (tc.function?.arguments) slot.arguments += tc.function.arguments;
      }
      if (choice.finish_reason) finish = choice.finish_reason;
    }
  }
  return { content, toolCalls: calls, finish };
}

const messages = [
  { role: 'system', content: 'You are Luban, a coding agent in a terminal. Be brief.' },
  { role: 'user', content: '看看 tetris2 目录下有什么文件，然后告诉我 index.js 开头几行是什么。' },
];

for (let step = 1; step <= 4; step++) {
  const t0 = Date.now();
  const r = await turn(messages);
  console.log(`\n--- step ${step} (${Date.now() - t0}ms) finish=${r.finish}`);
  if (r.toolCalls.length === 0) {
    console.log('assistant:', r.content.trim().slice(0, 300));
    break;
  }
  console.log('tool_calls:', r.toolCalls.map((c) => `${c.name}(${c.arguments})`).join(', '));
  messages.push({
    role: 'assistant',
    content: '',
    tool_calls: r.toolCalls.map((c) => ({ id: c.id, type: 'function', function: { name: c.name, arguments: c.arguments } })),
  });
  for (const c of r.toolCalls) {
    let args = {};
    try { args = JSON.parse(c.arguments || '{}'); } catch { /* keep {} */ }
    let result;
    try { result = runTool(c.name, args); } catch (e) { result = `ERROR: ${e.message}`; }
    console.log(`  -> ${c.name} result: ${JSON.stringify(result.slice(0, 120))}`);
    messages.push({ role: 'tool', tool_call_id: c.id, name: c.name, content: result });
  }
}
