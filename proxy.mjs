#!/usr/bin/env node
/**
 * Thin OpenAI-compatible proxy in front of Ollama.
 *
 * Why: qwen2.5-coder:7b (and friends) emit tool calls as bare JSON in `content`
 * instead of the `<tool_call>` tags Ollama's chat template expects, so Ollama's
 * /v1 returns content and no tool_calls. This proxy:
 *   1. injects a prompt-level tool protocol into the system message (request `tools`
 *      are turned into plain text the model can imitate),
 *   2. rewrites assistant tool_calls / role:tool history into plain text so the
 *      model keeps a coherent transcript,
 *   3. parses JSON objects out of the reply and promotes them to real tool_calls
 *      (both buffered and streaming responses).
 *
 * Env: PORT (11435), UPSTREAM (http://127.0.0.1:11434), DEFAULT_MODEL,
 *      PROXY_LOG (file path, optional), PROXY_DEBUG=1
 */
import http from 'node:http';
import fs from 'node:fs';

const PORT = Number(process.env.PORT || 11435);
const HOST = process.env.HOST || '127.0.0.1';
const UPSTREAM = (process.env.UPSTREAM || 'http://127.0.0.1:11434').replace(/\/+$/, '');
const DEFAULT_MODEL = process.env.DEFAULT_MODEL || 'qwen2.5-coder:7b';
const LOG_PATH = process.env.PROXY_LOG || '';
const DEBUG = process.env.PROXY_DEBUG === '1';

function log(...parts) {
  const line = `[${new Date().toISOString()}] ${parts.join(' ')}`;
  if (DEBUG) console.log(line);
  if (LOG_PATH) {
    try { fs.appendFileSync(LOG_PATH, line + '\n'); } catch { /* ignore */ }
  }
}

// ---------------------------------------------------------------- tool protocol

/** Render the request's `tools` array as an imitation-friendly text block. */
function renderToolProtocol(tools) {
  if (!Array.isArray(tools) || tools.length === 0) return '';
  const lines = tools.map((t) => {
    const fn = t?.function || t || {};
    const name = fn.name || 'unknown';
    const params = fn.parameters?.properties || {};
    const required = new Set(fn.parameters?.required || []);
    const argList = Object.entries(params).map(([k, v]) => {
      const type = v?.type === 'array' ? 'array' : (v?.type || 'string');
      const opt = required.has(k) ? '' : '?';
      return `${k}${opt}: ${type}`;
    }).join(', ');
    const desc = (fn.description || '').replace(/\s+/g, ' ').trim();
    return `- ${name}(${argList})${desc ? ' — ' + desc : ''}`;
  });

  return [
    '# Tool Use Protocol',
    '',
    'You can call tools. To call one, reply with ONLY one JSON object and nothing else',
    '(no prose, no markdown code fence):',
    '',
    '{"name": "<tool_name>", "arguments": {"<arg>": "<value>"}}',
    '',
    'Rules:',
    '- Exactly one JSON object per reply, starting with "{" and ending with "}".',
    '- "arguments" must be an object; use {} when the tool takes no arguments.',
    '- Call a tool only when you need it. Otherwise answer the user in plain language.',
    '- Never invent tool names or tool results. Wait for the tool result before answering.',
    '',
    'Available tools:',
    ...lines,
  ].join('\n');
}

/** Normalize one call object into { name, argumentsJson }. */
function normalizeCall(obj) {
  const src = obj?.function && typeof obj.function === 'object' ? obj.function : obj;
  const name = typeof src?.name === 'string' ? src.name : null;
  if (!name) return null;
  let args = src.arguments ?? src.parameters ?? src.args ?? {};
  if (typeof args === 'string') {
    const trimmed = args.trim();
    if (!trimmed) args = '{}';
    else return { name, argumentsJson: trimmed };
  }
  return { name, argumentsJson: JSON.stringify(args ?? {}) };
}

/** Index of the matching `}` for the `{` at `start`, or -1. String-aware. */
function matchBrace(text, start) {
  let depth = 0, inStr = false, esc = false;
  for (let i = start; i < text.length; i++) {
    const ch = text[i];
    if (inStr) {
      if (esc) esc = false;
      else if (ch === '\\') esc = true;
      else if (ch === '"') inStr = false;
      continue;
    }
    if (ch === '"') inStr = true;
    else if (ch === '{') depth++;
    else if (ch === '}') { depth--; if (depth === 0) return i; }
  }
  return -1;
}

/**
 * Pull tool calls out of a plain-text reply.
 * Only objects whose name is a real, registered tool are promoted, so ordinary
 * JSON in prose is left alone.
 */
function extractToolCalls(text, toolNames) {
  const calls = [];
  if (!text) return calls;
  let i = 0;
  while (i < text.length) {
    const brace = text.indexOf('{', i);
    if (brace === -1) break;
    const end = matchBrace(text, brace);
    if (end === -1) break;
    const slice = text.slice(brace, end + 1);
    i = end + 1;
    let parsed;
    try { parsed = JSON.parse(slice); } catch { continue; }
    if (!parsed || typeof parsed !== 'object') continue;
    const call = normalizeCall(parsed);
    if (!call) continue;
    if (toolNames.size > 0 && !toolNames.has(call.name)) continue;
    calls.push(call);
  }
  return calls;
}

// ---------------------------------------------------------------- messages

/**
 * Rewrite history into plain role/content turns the small model can follow:
 * assistant tool_calls become the JSON it is supposed to emit, role:tool
 * results become user turns carrying the result text.
 */
function convertMessages(messages, toolNames) {
  const out = [];
  for (const msg of messages || []) {
    if (!msg || typeof msg !== 'object') continue;
    const role = msg.role;
    const text = typeof msg.content === 'string'
      ? msg.content
      : Array.isArray(msg.content)
        ? msg.content.map((p) => (typeof p === 'string' ? p : p?.text || '')).join('')
        : '';

    if (role === 'tool') {
      const name = msg.name || 'tool';
      out.push({ role: 'user', content: `TOOL RESULT (name=${name}):\n${text}` });
      continue;
    }

    if (role === 'assistant') {
      const calls = (msg.tool_calls || [])
        .map((c) => normalizeCall(c))
        .filter(Boolean);
      if (calls.length > 0) {
        out.push({
          role: 'assistant',
          content: calls.map((c) => JSON.stringify({ name: c.name, arguments: safeParse(c.argumentsJson) })).join('\n'),
        });
        continue;
      }
      if (text) out.push({ role: 'assistant', content: text });
      continue;
    }

    if (role === 'system' || role === 'developer') {
      // handled by the caller (protocol is appended to the first system message)
      out.push({ role: 'system', content: text });
      continue;
    }

    out.push({ role: 'user', content: text });
  }
  return out;
}

function safeParse(s) {
  try { return JSON.parse(s); } catch { return {}; }
}

// ---------------------------------------------------------------- upstream

async function callUpstream(body, stream) {
  const res = await fetch(`${UPSTREAM}/v1/chat/completions`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ ...body, stream }),
  });
  if (!res.ok) {
    const errText = await res.text().catch(() => '');
    throw new Error(`upstream ${res.status}: ${errText.slice(0, 500)}`);
  }
  return res;
}

// ---------------------------------------------------------------- handlers

async function handleChat(req, res, raw) {
  let payload;
  try { payload = JSON.parse(raw || '{}'); } catch {
    res.writeHead(400, { 'Content-Type': 'application/json' });
    return res.end(JSON.stringify({ error: { message: 'invalid JSON body' } }));
  }

  const tools = Array.isArray(payload.tools) ? payload.tools : [];
  const toolNames = new Set(tools.map((t) => t?.function?.name || t?.name).filter(Boolean));
  const wantStream = payload.stream === true;

  const converted = convertMessages(payload.messages, toolNames);

  // Append the protocol to the system prompt (create one if none exists).
  const protocol = renderToolProtocol(tools);
  if (protocol) {
    const sys = converted.find((m) => m.role === 'system');
    if (sys) sys.content = `${sys.content}\n\n${protocol}`;
    else converted.unshift({ role: 'system', content: protocol });
  }

  const upstreamBody = {
    ...payload,
    model: payload.model || DEFAULT_MODEL,
    messages: converted,
  };
  delete upstreamBody.tools;
  delete upstreamBody.tool_choice;
  delete upstreamBody.parallel_tool_calls;
  delete upstreamBody.stream;
  delete upstreamBody.stream_options;

  const t0 = Date.now();
  if (!wantStream) {
    const up = await callUpstream(upstreamBody, false);
    const json = await up.json();
    const message = json.choices?.[0]?.message || {};
    const content = message.content || '';
    const calls = toolNames.size > 0 ? extractToolCalls(content, toolNames) : [];
    const outMessage = { role: 'assistant', content: calls.length ? '' : content };
    let finish = json.choices?.[0]?.finish_reason || 'stop';
    if (calls.length) {
      outMessage.tool_calls = calls.map((c, idx) => ({
        id: `call_${Date.now().toString(36)}_${idx}`,
        type: 'function',
        function: { name: c.name, arguments: c.argumentsJson },
      }));
      finish = 'tool_calls';
    }
    log(`POST ${payload.model || DEFAULT_MODEL} stream=0 tools=${toolNames.size} calls=${calls.length} ${Date.now() - t0}ms`);
    res.writeHead(200, { 'Content-Type': 'application/json' });
    return res.end(JSON.stringify({
      id: json.id || `chatcmpl-proxy-${Date.now()}`,
      object: 'chat.completion',
      created: Math.floor(Date.now() / 1000),
      model: upstreamBody.model,
      choices: [{ index: 0, message: outMessage, finish_reason: finish }],
      usage: json.usage || undefined,
    }));
  }

  // ---- streaming: buffer until we can tell text from a tool call, then relay.
  const up = await callUpstream(upstreamBody, true);
  const id = `chatcmpl-proxy-${Date.now()}`;
  const created = Math.floor(Date.now() / 1000);
  const modelName = upstreamBody.model;

  res.writeHead(200, {
    'Content-Type': 'text/event-stream; charset=utf-8',
    'Cache-Control': 'no-cache',
    Connection: 'keep-alive',
    'X-Accel-Buffering': 'no',
  });

  const send = (delta, finish = null) => {
    res.write(`data: ${JSON.stringify({
      id, object: 'chat.completion.chunk', created, model: modelName,
      choices: [{ index: 0, delta, finish_reason: finish }],
    })}\n\n`);
  };

  send({ role: 'assistant', content: '' });

  let buffer = '';
  let mode = 'undecided'; // undecided | text | maybe-tool
  let emitted = 0;

  const reader = up.body.getReader();
  const decoder = new TextDecoder();
  let pending = '';

  const consumeLine = (line) => {
    const trimmed = line.trim();
    if (!trimmed.startsWith('data:')) return;
    const data = trimmed.slice(5).trim();
    if (!data || data === '[DONE]') return;
    let chunk;
    try { chunk = JSON.parse(data); } catch { return; }
    const delta = chunk.choices?.[0]?.delta || {};
    const piece = delta.content;
    if (!piece) return;

    if (mode === 'text') { send({ content: piece }); emitted += piece.length; return; }

    buffer += piece;
    if (mode === 'undecided') {
      const head = buffer.trimStart();
      if (!head) return;
      if (head.startsWith('{') || head.startsWith('`')) { mode = 'maybe-tool'; return; }
      mode = 'text';
      send({ content: buffer });
      emitted += buffer.length;
      buffer = '';
    }
  };

  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      pending += decoder.decode(value, { stream: true });
      let nl;
      while ((nl = pending.indexOf('\n')) !== -1) {
        const line = pending.slice(0, nl);
        pending = pending.slice(nl + 1);
        consumeLine(line);
      }
    }
    if (pending) consumeLine(pending);

    let calls = [];
    if (mode === 'maybe-tool' && toolNames.size > 0) {
      calls = extractToolCalls(buffer, toolNames);
    }

    if (calls.length > 0) {
      const toolCalls = calls.map((c, idx) => ({
        index: idx,
        id: `call_${Date.now().toString(36)}_${idx}`,
        type: 'function',
        function: { name: c.name, arguments: c.argumentsJson },
      }));
      send({ tool_calls: toolCalls });
      send({}, 'tool_calls');
      log(`POST ${modelName} stream=1 tools=${toolNames.size} calls=${calls.length} ${Date.now() - t0}ms`);
    } else {
      if (buffer) { send({ content: buffer }); emitted += buffer.length; }
      send({}, 'stop');
      log(`POST ${modelName} stream=1 tools=${toolNames.size} calls=0 chars=${emitted} ${Date.now() - t0}ms`);
    }
    res.write('data: [DONE]\n\n');
    res.end();
  } catch (err) {
    log(`stream error: ${err.message}`);
    try {
      send({ content: `\n[proxy error: ${err.message}]` });
      send({}, 'stop');
      res.write('data: [DONE]\n\n');
    } catch { /* client already gone */ }
    res.end();
  }
}

async function handleModels(res) {
  try {
    const up = await fetch(`${UPSTREAM}/v1/models`);
    const json = await up.json();
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify(json));
  } catch (err) {
    res.writeHead(502, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ error: { message: err.message } }));
  }
}

const server = http.createServer((req, res) => {
  const url = new URL(req.url || '/', `http://${req.headers.host || 'localhost'}`);

  if (req.method === 'GET' && (url.pathname === '/health' || url.pathname === '/')) {
    res.writeHead(200, { 'Content-Type': 'application/json' });
    return res.end(JSON.stringify({ ok: true, upstream: UPSTREAM, defaultModel: DEFAULT_MODEL }));
  }
  if (req.method === 'GET' && url.pathname.endsWith('/models')) return handleModels(res);

  if (req.method !== 'POST') {
    res.writeHead(405, { 'Content-Type': 'application/json' });
    return res.end(JSON.stringify({ error: { message: 'method not allowed' } }));
  }

  const chunks = [];
  req.on('data', (c) => chunks.push(c));
  req.on('end', async () => {
    const raw = Buffer.concat(chunks).toString('utf8');
    try {
      if (url.pathname.endsWith('/chat/completions')) await handleChat(req, res, raw);
      else {
        res.writeHead(404, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: { message: `no route for ${url.pathname}` } }));
      }
    } catch (err) {
      log(`handler error: ${err.stack || err.message}`);
      if (!res.headersSent) res.writeHead(502, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: { message: String(err.message || err) } }));
    }
  });
});

if (process.env.PROXY_NO_LISTEN !== '1') {
  server.listen(PORT, HOST, () => {
    console.log(`tool-proxy listening on http://${HOST}:${PORT}/v1  -> ${UPSTREAM}`);
  });
}

export { renderToolProtocol, extractToolCalls, convertMessages, normalizeCall, matchBrace };
