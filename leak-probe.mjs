// Probe: does a prose-prefixed tool call survive streaming vs non-streaming?
// Run: node leak-probe.mjs [runs]
const PROXY = 'http://127.0.0.1:11435/v1/chat/completions';
const RUNS = Number(process.argv[2] || 6);

const TOOLS = [{ type: 'function', function: { name: 'read_file', description: 'Read a text file', parameters: { type: 'object', properties: { path: { type: 'string' } }, required: ['path'] } } }];
const MSGS = [
  { role: 'system', content: 'You are Luban, a coding agent. Use the provided tools when you need file access. Be brief.' },
  { role: 'user', content: '读一下 tetris2/index.js 的开头部分。' },
];
const body = (stream) => JSON.stringify({ model: 'qwen2.5-coder:7b', messages: MSGS, tools: TOOLS, stream });

const looksLikeCall = (t) => /\{\s*"name"\s*:/.test(t);

async function nonStream() {
  const res = await fetch(PROXY, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: body(false) });
  const j = await res.json();
  const m = j.choices?.[0]?.message ?? {};
  return { calls: m.tool_calls?.length || 0, content: m.content || '', finish: j.choices?.[0]?.finish_reason };
}

async function streamed() {
  const res = await fetch(PROXY, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: body(true) });
  let content = '';
  let calls = 0;
  let finish = null;
  const reader = res.body.getReader();
  const dec = new TextDecoder();
  let buf = '';
  for (;;) {
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
      let chunk;
      try { chunk = JSON.parse(data); } catch { continue; }
      const ch = chunk.choices?.[0];
      if (!ch) continue;
      if (ch.delta?.content) content += ch.delta.content;
      calls += (ch.delta?.tool_calls || []).length;
      if (ch.finish_reason) finish = ch.finish_reason;
    }
  }
  return { calls, content, finish };
}

let sOk = 0, sLeak = 0, nOk = 0, nLeak = 0;
for (let i = 0; i < RUNS; i++) {
  const n = await nonStream();
  if (n.calls) nOk++;
  else if (looksLikeCall(n.content)) { nLeak++; console.log(`  nonstream run${i + 1} LEAK content=${JSON.stringify(n.content.slice(0, 90))}`); }
  else console.log(`  nonstream run${i + 1} plain answer (no call, no leak)`);

  const s = await streamed();
  if (s.calls) sOk++;
  else if (looksLikeCall(s.content)) { sLeak++; console.log(`  stream    run${i + 1} LEAK content=${JSON.stringify(s.content.slice(0, 90))}`); }
  else console.log(`  stream    run${i + 1} plain answer (no call, no leak)`);
}
console.log(`\nnon-stream: ${nOk} tool_calls, ${nLeak} JSON-in-content  (of ${RUNS})`);
console.log(`stream:     ${sOk} tool_calls, ${sLeak} JSON-in-content  (of ${RUNS})`);
