// Tool-calling bench for qwen2.5-coder:7b through the thin proxy.
// Run: node bench.mjs [runsPerCase]
import fs from 'node:fs';
import path from 'node:path';

const PROXY = 'http://127.0.0.1:11435/v1/chat/completions';
const ROOT = process.env.BENCH_ROOT || process.cwd();
const RUNS = Number(process.argv[2] || 3);

const TOOLS = [
  { type: 'function', function: { name: 'list_files', description: 'List files in a directory (max 3 levels)', parameters: { type: 'object', properties: { dir: { type: 'string', description: 'subdirectory, omit for root' } }, required: [] } } },
  { type: 'function', function: { name: 'read_file', description: 'Read a text file. Must be called before editing it.', parameters: { type: 'object', properties: { path: { type: 'string' } }, required: ['path'] } } },
  { type: 'function', function: { name: 'search_code', description: 'Search the workspace for a keyword, returns path:line hits', parameters: { type: 'object', properties: { query: { type: 'string' } }, required: ['query'] } } },
  { type: 'function', function: { name: 'run_command', description: 'Run a shell command in the workspace', parameters: { type: 'object', properties: { command: { type: 'string' } }, required: ['command'] } } },
];

function runTool(name, args) {
  try {
    if (name === 'list_files') {
      const dir = path.resolve(ROOT, args.dir || '.');
      return fs.readdirSync(dir).map((f) => (fs.statSync(path.join(dir, f)).isDirectory() ? f + '/' : f)).join('\n');
    }
    if (name === 'read_file') return fs.readFileSync(path.resolve(ROOT, args.path), 'utf8').split('\n').slice(0, 8).join('\n');
    if (name === 'search_code') return 'tetris2/index.js:6: const GRID_WIDTH = 10;';
    if (name === 'run_command') return 'ok';
  } catch (e) { return `ERROR: ${e.message}`; }
  return `unknown tool ${name}`;
}

async function turn(messages, useTools = true) {
  const res = await fetch(PROXY, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model: 'qwen2.5-coder:7b', messages, ...(useTools ? { tools: TOOLS } : {}), stream: true }),
  });
  if (!res.ok) throw new Error(`proxy ${res.status}: ${await res.text()}`);
  let content = '', finish = null;
  const calls = [];
  const reader = res.body.getReader();
  const dec = new TextDecoder();
  let buf = '';
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    let nl;
    while ((nl = buf.indexOf('\n')) !== -1) {
      const line = buf.slice(0, nl).trim(); buf = buf.slice(nl + 1);
      if (!line.startsWith('data:')) continue;
      const data = line.slice(5).trim();
      if (data === '[DONE]') continue;
      let chunk; try { chunk = JSON.parse(data); } catch { continue; }
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

const SYS = 'You are Luban, a coding agent. Use the provided tools when you need file or shell access. Be brief.';

// expect: { name, argCheck?(args) } | null (= must not call a tool)
const CASES = [
  { id: 'list-dir', prompt: 'tetris2 目录下有哪些文件？用工具看。', expect: { name: 'list_files' } },
  { id: 'read-file', prompt: '读一下 tetris2/index.js 的开头部分。', expect: { name: 'read_file', argCheck: (a) => String(a.path || '').includes('tetris2/index.js') } },
  { id: 'search', prompt: '在代码里找 GRID_WIDTH 定义在哪个文件哪一行。', expect: { name: 'search_code', argCheck: (a) => String(a.query || '').toUpperCase().includes('GRID_WIDTH') } },
  { id: 'run-cmd', prompt: '在工作区执行 `node --version` 看看 Node 版本。', expect: { name: 'run_command', argCheck: (a) => String(a.command || '').includes('node') && String(a.command || '').includes('version') } },
  { id: 'no-tool', prompt: '解释一下 Tetris 里 T 型方块旋转 90 度后的形状，不要调用任何工具。', expect: null, useTools: true },
];

const stats = [];
for (const c of CASES) {
  let ok = 0, leak = 0, argBad = 0, lat = [];
  for (let r = 0; r < (c.id === 'no-tool' ? Math.max(1, RUNS - 1) : RUNS); r++) {
    const t0 = Date.now();
    let res;
    try {
      res = await turn([{ role: 'system', content: SYS }, { role: 'user', content: c.prompt }], c.useTools !== false);
    } catch (e) {
      console.log(`  ${c.id} run${r + 1} ERROR ${e.message}`);
      continue;
    }
    lat.push(Date.now() - t0);
    const calls = res.toolCalls;
    // leak = a tool-call-looking JSON left in the visible content
    if (c.expect && calls.length === 0 && /^\s*(```)?(json)?\s*\{\s*"name"/.test(res.content)) leak++;
    if (c.expect === null) {
      if (calls.length === 0 && res.content.trim()) ok++;
      else console.log(`  ${c.id} run${r + 1} UNEXPECTED CALL ${calls.map((x) => x.name).join(',')}`);
    } else {
      if (calls.length === 0) { console.log(`  ${c.id} run${r + 1} NO CALL content=${JSON.stringify(res.content.slice(0, 60))}`); continue; }
      const hit = calls.find((x) => x.name === c.expect.name);
      if (!hit) { console.log(`  ${c.id} run${r + 1} WRONG TOOL ${calls.map((x) => x.name).join(',')}`); continue; }
      let args = {}; let parsed = true;
      try { args = JSON.parse(hit.arguments || '{}'); } catch { parsed = false; }
      if (!parsed) { console.log(`  ${c.id} run${r + 1} BAD JSON ${hit.arguments}`); continue; }
      if (c.expect.argCheck && !c.expect.argCheck(args)) { argBad++; console.log(`  ${c.id} run${r + 1} BAD ARGS ${hit.arguments}`); continue; }
      ok++;
    }
  }
  const n = lat.length || 1;
  stats.push({ id: c.id, ok, n, argBad, leak, avg: Math.round(lat.reduce((a, b) => a + b, 0) / n) });
}

console.log('\ncase         ok/n   badArgs  leak  avgMs');
let totalOk = 0, totalN = 0;
for (const s of stats) {
  console.log(`${s.id.padEnd(12)} ${String(s.ok + '/' + s.n).padEnd(6)} ${String(s.argBad).padEnd(8)} ${String(s.leak).padEnd(5)} ${s.avg}`);
  totalOk += s.ok; totalN += s.n;
}
console.log(`\nTOTAL ${totalOk}/${totalN} = ${Math.round((totalOk / totalN) * 100)}%`);
