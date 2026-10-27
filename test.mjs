// Unit tests for the tool-call extraction layer. Run: node test.mjs
process.env.PROXY_NO_LISTEN = '1';
const { extractToolCalls, convertMessages, renderToolProtocol } = await import('./proxy.mjs');

const NAMES = new Set(['read_file', 'list_files', 'run_command']);
let pass = 0, fail = 0;

function check(label, cond, extra) {
  if (cond) { pass++; console.log(`  ok   ${label}`); }
  else { fail++; console.log(`  FAIL ${label}${extra ? ' -> ' + JSON.stringify(extra) : ''}`); }
}

console.log('extractToolCalls');
let c = extractToolCalls('{"name": "read_file", "arguments": {"path": "a.js"}}', NAMES);
check('bare JSON', c.length === 1 && c[0].name === 'read_file' && JSON.parse(c[0].argumentsJson).path === 'a.js', c);

c = extractToolCalls('```json\n{"name": "list_files", "arguments": {}}\n```', NAMES);
check('fenced JSON', c.length === 1 && c[0].name === 'list_files', c);

c = extractToolCalls('{"type":"function","function":{"name":"run_command","arguments":"{\\"command\\":\\"ls\\"}"}}', NAMES);
check('openai nested shape + string args', c.length === 1 && c[0].name === 'run_command' && JSON.parse(c[0].argumentsJson).command === 'ls', c);

c = extractToolCalls('{"name": "read_file", "arguments": {"path": "a{b}.js"}}', NAMES);
check('braces inside string literal', c.length === 1 && JSON.parse(c[0].argumentsJson).path === 'a{b}.js', c);

c = extractToolCalls('{"name":"read_file","arguments":{"path":"x"}}\n{"name":"list_files","arguments":{}}', NAMES);
check('two calls in one reply', c.length === 2, c);

c = extractToolCalls('{"name": "delete_everything", "arguments": {}}', NAMES);
check('unknown tool name is ignored', c.length === 0, c);

c = extractToolCalls('{"name": "read_file"}', NAMES);
check('missing arguments defaults to {}', c.length === 1 && c[0].argumentsJson === '{}', c);

c = extractToolCalls('Sure! {"name": "list_files", "arguments": {}}', NAMES);
check('prose before JSON still parses', c.length === 1, c);

c = extractToolCalls('{"response": "files_list": ["a"]}', NAMES);
check('malformed JSON ignored', c.length === 0, c);

c = extractToolCalls('{"name": "read_file", "arguments": {"path": "unterminated}', NAMES);
check('unterminated JSON ignored', c.length === 0, c);

c = extractToolCalls('普通文本回复，没有工具调用。', NAMES);
check('plain text yields nothing', c.length === 0, c);

console.log('convertMessages');
const conv = convertMessages([
  { role: 'system', content: 'sys' },
  { role: 'user', content: 'read a.js' },
  { role: 'assistant', content: '', tool_calls: [{ id: 'x', type: 'function', function: { name: 'read_file', arguments: '{"path":"a.js"}' } }] },
  { role: 'tool', tool_call_id: 'x', name: 'read_file', content: 'FILE BODY' },
], NAMES);
check('assistant tool_calls -> plain JSON turn', conv[2].role === 'assistant' && JSON.parse(conv[2].content).name === 'read_file', conv[2]);
check('role:tool -> user TOOL RESULT turn', conv[3].role === 'user' && conv[3].content.includes('FILE BODY') && conv[3].content.includes('read_file'), conv[3]);
check('no role:tool survives to upstream', conv.every((m) => ['system', 'user', 'assistant'].includes(m.role)));

const conv2 = convertMessages([
  { role: 'user', content: [{ type: 'text', text: 'part1 ' }, { type: 'text', text: 'part2' }] },
], NAMES);
check('array content is flattened', conv2[0].content === 'part1 part2', conv2[0]);

console.log('renderToolProtocol');
const proto = renderToolProtocol([
  { type: 'function', function: { name: 'read_file', description: 'Read a file', parameters: { type: 'object', properties: { path: { type: 'string' } }, required: ['path'] } } },
  { type: 'function', function: { name: 'list_files', parameters: { type: 'object', properties: { dir: { type: 'string' } } } } },
]);
check('lists both tools with required marker', proto.includes('read_file(path: string)') && proto.includes('list_files(dir?: string)'), proto);
check('empty tools -> empty protocol', renderToolProtocol([]) === '');

console.log(`\n${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);
