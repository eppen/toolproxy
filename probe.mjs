const SYS = `# Tool Use Protocol

You have access to the following tools. To call a tool, reply with ONLY one JSON object, no prose and no markdown fences:
{"name": "<tool_name>", "arguments": { ... }}

Rules:
- One JSON object per reply, nothing before or after it.
- Only call a tool when you actually need it; otherwise answer normally in plain language.
- Never invent tool names or results.

Tools:
1. list_files(dir?: string) — List files under a directory (recursively, 3 levels).
2. read_file(path: string) — Read a text file.
3. run_command(command: string) — Run a shell command in the workspace.`;

async function chat(messages) {
  const r = await fetch('http://127.0.0.1:11434/v1/chat/completions', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model: 'qwen2.5-coder:7b', messages, stream: false, max_tokens: 400, temperature: 0 })
  });
  const j = await r.json();
  return j.choices[0].message.content;
}

const msgs = [
  { role: 'system', content: SYS },
  { role: 'user', content: '工作目录 /private/tmp/p1test 里有一个 README.md，请告诉我它的内容。' },
];
const t0 = Date.now();
const a1 = await chat(msgs);
console.log('--- round1 (' + (Date.now() - t0) + 'ms) ---\n' + a1);
msgs.push({ role: 'assistant', content: a1 });
msgs.push({ role: 'user', content: 'TOOL RESULT (name=list_files):\nREADME.md\nsrc/\ntetris/\nnode_modules/' });
const t1 = Date.now();
const a2 = await chat(msgs);
console.log('--- round2 (' + (Date.now() - t1) + 'ms) ---\n' + a2);
msgs.push({ role: 'assistant', content: a2 });
msgs.push({ role: 'user', content: 'TOOL RESULT (name=read_file):\nThis is a sandbox for testing file tools.' });
const t2 = Date.now();
const a3 = await chat(msgs);
console.log('--- round3 (' + (Date.now() - t2) + 'ms) ---\n' + a3);
