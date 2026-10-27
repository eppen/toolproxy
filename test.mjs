// Unit tests for the tool-call extraction layer. Run: node test.mjs
process.env.PROXY_NO_LISTEN = '1';
const { extractToolCalls, convertMessages, renderToolProtocol, couldBeToolStart } = await import('./proxy.mjs');

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

// ---------------------------------------------------------------- markup formats

const SCHEMAS = new Map([
  ['read_file', { type: 'object', properties: { path: { type: 'string' } }, required: ['path'] }],
  ['list_files', { type: 'object', properties: { dir: { type: 'string' }, max: { type: 'integer' } } }],
  ['run_command', {
    type: 'object',
    properties: {
      command: { type: 'string' },
      timeout: { type: 'integer' },
      interactive: { type: 'boolean' },
      args: { type: 'array', items: { type: 'string' } },
    },
  }],
]);

console.log('markup formats');
c = extractToolCalls('<tool_call>\n{"name": "read_file", "arguments": {"path": "a.js"}}\n</tool_call>', NAMES);
check('qwen <tool_call> wrapper', c.length === 1 && c[0].name === 'read_file', c);

c = extractToolCalls('<tool_call>{"name":"read_file","arguments":{"path":"a.js"}}</tool_call><tool_call>{"name":"list_files","arguments":{}}</tool_call>', NAMES);
check('two <tool_call> wrappers', c.length === 2 && c[1].name === 'list_files', c);

c = extractToolCalls('<tool_call>\n{"name": "read_file", "arguments": {"path": "a.js"}}\n</tool_call>', NAMES);
check('wrapper does not double count via bare JSON', c.length === 1, c);

c = extractToolCalls('<function=read_file>\n<parameter=path>\nsrc/a.js\n</parameter>\n</function>', NAMES);
check('qwen3 <function=...> XML', c.length === 1 && JSON.parse(c[0].argumentsJson).path === 'src/a.js', c);

c = extractToolCalls('<tool_call>\n<function=run_command>\n<parameter=command>ls -la</parameter>\n</function>\n</tool_call>', NAMES);
check('qwen3 XML nested in <tool_call>', c.length === 1 && JSON.parse(c[0].argumentsJson).command === 'ls -la', c);

c = extractToolCalls('[TOOL_CALLS] [{"name": "run_command", "arguments": {"command": "ls"}}]', NAMES);
check('mistral [TOOL_CALLS] array', c.length === 1 && c[0].name === 'run_command', c);

c = extractToolCalls('<|python_tag|>{"name": "list_files", "arguments": {}}<|eom_id|>', NAMES);
check('llama <|python_tag|>', c.length === 1 && c[0].name === 'list_files', c);

c = extractToolCalls('<invoke name="read_file"><parameter name="path">x.js</parameter></invoke>', NAMES);
check('anthropic <invoke>', c.length === 1 && JSON.parse(c[0].argumentsJson).path === 'x.js', c);

c = extractToolCalls('```json\n{"name": "list_files", "arguments": {}}\n```', NAMES);
check('fenced JSON still parses once', c.length === 1, c);

c = extractToolCalls('先说一句，然后 <tool_call>{"name":"list_files","arguments":{}}</tool_call> 结束', NAMES);
check('prose around markup is tolerated', c.length === 1, c);

console.log('name repair');
c = extractToolCalls('{"name": "readFile", "arguments": {"path": "a.js"}}', NAMES);
check('camelCase snapped to registered snake_case', c.length === 1 && c[0].name === 'read_file', c);

c = extractToolCalls('{"name": "read-file", "arguments": {"path": "a.js"}}', NAMES);
check('kebab-case snapped', c.length === 1 && c[0].name === 'read_file', c);

c = extractToolCalls('{"name": "deleteFile", "arguments": {}}', NAMES);
check('unrelated name still rejected', c.length === 0, c);

c = extractToolCalls('{"name": "read_file", "arguments": {}}', NAMES);
check('exact match preferred', c.length === 1 && c[0].name === 'read_file', c);

console.log('schema coercion');
c = extractToolCalls('{"name": "run_command", "arguments": {"command": "ls", "timeout": "30"}}', NAMES, SCHEMAS);
check('"30" -> integer', JSON.parse(c[0].argumentsJson).timeout === 30, c);

c = extractToolCalls('{"name": "run_command", "arguments": {"command": "ls", "interactive": "false"}}', NAMES, SCHEMAS);
check('"false" -> boolean', JSON.parse(c[0].argumentsJson).interactive === false, c);

c = extractToolCalls('{"name": "run_command", "arguments": {"command": "ls", "args": "[-l, -a]"}}', NAMES, SCHEMAS);
check('comma string -> array', JSON.stringify(JSON.parse(c[0].argumentsJson).args) === '["-l","-a"]', c);

c = extractToolCalls('{"name": "run_command", "arguments": {"command": "ls", "args": "[\\"-l\\",\\"-a\\"]"}}', NAMES, SCHEMAS);
check('JSON-encoded array string -> array', JSON.stringify(JSON.parse(c[0].argumentsJson).args) === '["-l","-a"]', c);

c = extractToolCalls('{"name": "run_command", "arguments": {"command": "ls", "args": "-l"}}', NAMES, SCHEMAS);
check('single string -> one-element array', JSON.stringify(JSON.parse(c[0].argumentsJson).args) === '["-l"]', c);

c = extractToolCalls('{"name": "list_files", "arguments": {"max": 3}}', NAMES, SCHEMAS);
check('valid arguments are left alone', JSON.parse(c[0].argumentsJson).max === 3, c);

c = extractToolCalls('{"name": "run_command", "arguments": {"command": "ls", "timeout": "30"}}', NAMES);
check('no schema -> no coercion', JSON.parse(c[0].argumentsJson).timeout === '30', c);

c = extractToolCalls('{"name": "run_command", "arguments": {"command": 42}}', NAMES, SCHEMAS);
check('number -> string when schema says string', JSON.parse(c[0].argumentsJson).command === '42', c);

c = extractToolCalls('<function=run_command><parameter=timeout>15</parameter><parameter=command>ls</parameter></function>', NAMES, SCHEMAS);
check('XML params coerced too', JSON.parse(c[0].argumentsJson).timeout === 15, c);

console.log('stream start heuristics');
check('prose is not a tool start', couldBeToolStart('Sure, here you go') === false);
check('html is not a tool start', couldBeToolStart('<html>') === false);
check('markdown footnote is not a tool start', couldBeToolStart('[1] see below') === false);
check('bare JSON is a tool start', couldBeToolStart('{ "name"') === true);
check('fence is a tool start', couldBeToolStart('```json') === true);
check('<tool_call> is a tool start', couldBeToolStart('<tool_call>') === true);
check('<function= is a tool start', couldBeToolStart('<function=read_file>') === true);
check('[TOOL_CALLS] is a tool start', couldBeToolStart('[TOOL_CALLS] [') === true);
check('<|python_tag|> is a tool start', couldBeToolStart('<|python_tag|>{') === true);
check('partial marker keeps buffering', couldBeToolStart('<tool_c') === true);

console.log(`\n${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);
