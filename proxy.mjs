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

const NO_JSON = Symbol('no-json');

/** JSON.parse that reports failure with NO_JSON instead of throwing. */
function tryJson(text) {
  try { return JSON.parse(text); } catch { return NO_JSON; }
}

/** Separator/case-insensitive form so readFile, read_file and read-file compare equal. */
function canonName(s) {
  return String(s ?? '').replace(/[^A-Za-z0-9]/g, '').toLowerCase();
}

/**
 * Map a name the model emitted onto a registered tool.
 * Exact match wins, then a unique separator/case-insensitive match. A name that
 * half-matches several tools stays unresolved — we never redirect a call to a
 * different tool than the one the model asked for.
 */
function resolveToolName(name, toolNames) {
  if (typeof name !== 'string' || !name) return null;
  if (!toolNames || toolNames.size === 0) return name;
  if (toolNames.has(name)) return name;
  const canon = canonName(name);
  let hit = null;
  for (const n of toolNames) {
    if (canonName(n) !== canon) continue;
    if (hit !== null) return null;
    hit = n;
  }
  return hit;
}

function schemaForTool(toolSchemas, name) {
  if (!toolSchemas) return null;
  if (toolSchemas instanceof Map) return toolSchemas.get(name) || null;
  return toolSchemas[name] || null;
}

/**
 * Repair one argument value against its JSON Schema.
 * Small models hand back "3" for an integer, "[a,b]" for an array, or an object
 * serialized as a string. Only the shape is fixed — values are never invented.
 */
function coerceValue(value, schema) {
  if (value === undefined || !schema || typeof schema !== 'object') return value;

  if (Array.isArray(schema.enum) && schema.enum.length > 0) {
    const exact = schema.enum.find((e) => e === value);
    if (exact !== undefined) return exact;
    const asText = String(value);
    const loose = schema.enum.find((e) => String(e) === asText);
    if (loose !== undefined) return loose;
  }

  const types = Array.isArray(schema.type) ? schema.type : schema.type ? [schema.type] : [];

  if (types.includes('array') || schema.items) {
    let arr = value;
    if (typeof arr === 'string') {
      const parsed = tryJson(arr);
      arr = parsed === NO_JSON ? arr : parsed;
    }
    if (typeof arr === 'string') {
      const trimmed = arr.trim();
      // "[a, b]" and "['a','b']" are the usual not-quite-JSON hand-rolled arrays
      const inner = trimmed.startsWith('[') && trimmed.endsWith(']') ? trimmed.slice(1, -1) : trimmed;
      arr = inner.trim() === ''
        ? []
        : inner.split(',').map((s) => s.trim().replace(/^["']|["']$/g, ''));
    }
    if (!Array.isArray(arr)) arr = [arr];
    return schema.items ? arr.map((v) => coerceValue(v, schema.items)) : arr;
  }

  if (types.includes('object') || schema.properties) {
    let obj = value;
    if (typeof obj === 'string') {
      const parsed = tryJson(obj);
      if (parsed === NO_JSON) return value;
      obj = parsed;
    }
    if (!obj || typeof obj !== 'object' || Array.isArray(obj)) return obj;
    if (!schema.properties) return obj;
    const out = { ...obj };
    for (const [key, sub] of Object.entries(schema.properties)) {
      if (key in out) out[key] = coerceValue(out[key], sub);
    }
    return out;
  }

  if (types.includes('integer') || types.includes('number')) {
    const n = typeof value === 'number' ? value : Number(String(value).trim());
    if (!Number.isFinite(n)) return value;
    return types.includes('integer') ? Math.trunc(n) : n;
  }

  if (types.includes('boolean')) {
    if (typeof value === 'boolean') return value;
    const t = String(value).trim().toLowerCase();
    if (['true', '1', 'yes'].includes(t)) return true;
    if (['false', '0', 'no'].includes(t)) return false;
    return value;
  }

  if (types.includes('string')) {
    if (typeof value === 'string') return value;
    if (value !== null && typeof value === 'object') return JSON.stringify(value);
    return String(value);
  }

  return value;
}

/** Coerce a whole arguments object against a tool's parameter schema. */
function coerceArgumentsJson(argumentsJson, schema) {
  if (!schema) return argumentsJson;
  const args = tryJson(argumentsJson || '{}');
  if (args === NO_JSON || !args || typeof args !== 'object' || Array.isArray(args)) return argumentsJson;
  const coerced = coerceValue(args, { ...schema, type: schema.type || 'object' });
  return JSON.stringify(coerced);
}

/** Index of the matching closer for the opener at `start`, or -1. String-aware. */
function matchPair(text, start) {
  const opener = text[start];
  const closer = opener === '{' ? '}' : opener === '[' ? ']' : null;
  if (!closer) return -1;
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
    else if (ch === opener) depth++;
    else if (ch === closer) { depth--; if (depth === 0) return i; }
  }
  return -1;
}

/** Index of the matching `}` for the `{` at `start`, or -1. String-aware. */
function matchBrace(text, start) {
  return matchPair(text, start);
}

/** Read one JSON object/array starting at or after `from`, or null. */
function scanJsonValue(text, from) {
  let i = from;
  while (i < text.length && /\s/.test(text[i])) i++;
  if (i >= text.length) return null;
  const open = text[i];
  if (open !== '{' && open !== '[') return null;
  const end = matchPair(text, i);
  if (end === -1) return null;
  const value = tryJson(text.slice(i, end + 1));
  if (value === NO_JSON) return null;
  return { value, start: i, end: end + 1 };
}

/** Every JSON object/array in a fragment, in order. */
function parseJsonValues(text) {
  const out = [];
  let i = 0;
  const src = text || '';
  while (i < src.length) {
    const scanned = scanJsonValue(src, i);
    if (scanned) {
      out.push(scanned.value);
      i = scanned.end;
      continue;
    }
    const next = src.slice(i).search(/[{\[]/);
    if (next === -1) break;
    i += next + 1;
  }
  return out;
}

/** Parse Qwen 3 style `<parameter=key>value</parameter>` bodies into an object. */
function parseXmlParams(body) {
  const args = {};
  const re = /<parameter=([A-Za-z0-9_.\-]+)\s*>([\s\S]*?)(?:<\/parameter\s*>|(?=<parameter=)|$)/g;
  let m;
  while ((m = re.exec(body || '')) !== null) {
    const raw = (m[2] || '').trim();
    const parsed = tryJson(raw);
    args[m[1]] = parsed === NO_JSON ? raw : parsed;
  }
  if (Object.keys(args).length > 0) return args;
  // No parameter tags: the body may just be the JSON arguments object.
  const json = tryJson((body || '').trim());
  return json !== NO_JSON && json && typeof json === 'object' && !Array.isArray(json) ? json : {};
}

/** Anthropic-style `<parameter name="key">value</parameter>` bodies. */
function parseInvokeParams(body) {
  const args = {};
  const re = /<parameter\s+name=["']([^"']+)["']\s*>([\s\S]*?)(?:<\/parameter\s*>|$)/g;
  let m;
  while ((m = re.exec(body || '')) !== null) {
    const raw = (m[2] || '').trim();
    const parsed = tryJson(raw);
    args[m[1]] = parsed === NO_JSON ? raw : parsed;
  }
  return args;
}

// Formats that carry a tool call in markup rather than bare JSON. Each is
// matched, promoted, and then blanked out so the bare-JSON pass cannot see it twice.
const FUNCTION_XML_RE = /<function=([A-Za-z0-9_.\-]+)\s*>([\s\S]*?)(?:<\/function\s*>|(?=<\/tool_call\s*>|<function=|<\|python_tag\|>|\[TOOL_CALLS\]|$))/g;
const INVOKE_RE = /<invoke\s+name=["']([^"']+)["']\s*>([\s\S]*?)(?:<\/invoke\s*>|$)/g;
const TOOL_CALL_BLOCK_RE = /<tool_call\s*>([\s\S]*?)(?:<\/tool_call\s*>|$)/g;
const MISTRAL_MARKER = '[TOOL_CALLS]';
const PYTHON_TAG = '<|python_tag|>';

/** Markers a reply can start with when it is (or may become) a tool call. */
const TOOL_START_MARKERS = ['<tool_call', '<function=', '<|python_tag|>', MISTRAL_MARKER];

/**
 * Could this buffered prefix still turn into a tool call?
 * Used while streaming so prose is flushed early and only real candidates wait.
 */
function couldBeToolStart(text) {
  const head = String(text || '').trimStart();
  if (!head) return true;
  if (head[0] === '{' || head[0] === '`') return true;
  if (TOOL_START_MARKERS.some((m) => m.startsWith(head) || head.startsWith(m))) return true;
  if (head[0] === '[' && /^\[\s*\{\s*"/.test(head)) return true;
  return false;
}

/**
 * Pull tool calls out of a plain-text reply.
 * Handles bare/fenced JSON, `<tool_call>` wrappers, Qwen 3 `<function=...>`
 * XML, Anthropic-style `<invoke>`, Mistral `[TOOL_CALLS]` and Llama
 * `<|python_tag|>`. Only names matching a registered tool are promoted, so
 * ordinary JSON in prose is left alone. Optional `toolSchemas` (Map or plain
 * object of name -> parameters schema) repairs argument shapes.
 */
function extractToolCalls(text, toolNames, toolSchemas) {
  const calls = [];
  if (!text) return calls;
  let work = String(text);

  const push = (obj) => {
    const call = normalizeCall(obj);
    if (!call) return;
    const name = resolveToolName(call.name, toolNames);
    if (!name) return;
    calls.push({ name, argumentsJson: coerceArgumentsJson(call.argumentsJson, schemaForTool(toolSchemas, name)) });
  };

  const blank = (from, to) => {
    work = work.slice(0, from) + ' '.repeat(to - from) + work.slice(to);
  };

  // 1. Qwen 3 XML: <function=name> <parameter=key>value</parameter> </function>
  for (const m of work.matchAll(FUNCTION_XML_RE)) push({ name: m[1], arguments: parseXmlParams(m[2]) });
  work = work.replace(FUNCTION_XML_RE, (m) => ' '.repeat(m.length));

  // 2. Anthropic-style: <invoke name="x"> <parameter name="k">v</parameter> </invoke>
  for (const m of work.matchAll(INVOKE_RE)) push({ name: m[1], arguments: parseInvokeParams(m[2]) });
  work = work.replace(INVOKE_RE, (m) => ' '.repeat(m.length));

  // 3. <tool_call> ... </tool_call> wrappers around a JSON object or array
  for (const m of work.matchAll(TOOL_CALL_BLOCK_RE)) {
    for (const v of parseJsonValues(m[1])) push(v);
  }
  work = work.replace(TOOL_CALL_BLOCK_RE, (m) => ' '.repeat(m.length));

  // 4. Mistral [TOOL_CALLS] followed by a JSON array
  let at;
  while ((at = work.indexOf(MISTRAL_MARKER)) !== -1) {
    const scanned = scanJsonValue(work, at + MISTRAL_MARKER.length);
    if (scanned) {
      for (const v of Array.isArray(scanned.value) ? scanned.value : [scanned.value]) push(v);
    }
    blank(at, scanned ? scanned.end : at + MISTRAL_MARKER.length);
  }

  // 5. Llama <|python_tag|> followed by a JSON object
  while ((at = work.indexOf(PYTHON_TAG)) !== -1) {
    const scanned = scanJsonValue(work, at + PYTHON_TAG.length);
    if (scanned) {
      for (const v of Array.isArray(scanned.value) ? scanned.value : [scanned.value]) push(v);
    }
    blank(at, scanned ? scanned.end : at + PYTHON_TAG.length);
  }

  // 6. Bare or fenced JSON left over
  let i = 0;
  while (i < work.length) {
    const brace = work.indexOf('{', i);
    if (brace === -1) break;
    const end = matchBrace(work, brace);
    if (end === -1) break;
    i = end + 1;
    const parsed = tryJson(work.slice(brace, end + 1));
    if (parsed === NO_JSON || !parsed || typeof parsed !== 'object') continue;
    push(parsed);
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
  const toolSchemas = new Map();
  for (const t of tools) {
    const fn = t?.function || t || {};
    if (fn.name) toolSchemas.set(fn.name, fn.parameters || null);
  }
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
    const calls = toolNames.size > 0 ? extractToolCalls(content, toolNames, toolSchemas) : [];
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
    // Still deciding: keep buffering while the head could still become a tool
    // call in any supported format, otherwise stream it out as prose.
    if (couldBeToolStart(buffer)) { mode = 'maybe-tool'; return; }

    mode = 'text';
    send({ content: buffer });
    emitted += buffer.length;
    buffer = '';
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
      calls = extractToolCalls(buffer, toolNames, toolSchemas);
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

export {
  renderToolProtocol, extractToolCalls, convertMessages, normalizeCall, matchBrace,
  coerceValue, resolveToolName, couldBeToolStart,
};
