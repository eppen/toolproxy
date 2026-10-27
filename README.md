# toolproxy

A ~400-line, zero-dependency OpenAI-compatible proxy that makes small Ollama models (e.g. `qwen2.5-coder:7b`) produce **real** `tool_calls` instead of leaking JSON into `content`.

```
your agent ──/v1/chat/completions──▶ toolproxy :11435 ──▶ Ollama :11434
```

## The problem

Many small / fine-tuned models emit a tool call as a bare JSON object in the message `content`:

```json
{"name": "read_file", "arguments": {"path": "src/index.js"}}
```

They never use the `<tool_call>` tags that Ollama's chat template expects, so Ollama's
`/v1/chat/completions` returns plain content and an empty `tool_calls` array. Agents that
parse `tool_calls` therefore see a model that "just talks about tools" and never acts.

This proxy sits in between and fixes it on both sides of the wire — no model changes, no
fine-tuning, no Ollama template hacking.

## What it does

1. **Injects a prompt-level tool protocol.** The request's `tools` array is rendered into a
   compact text block (`name(arg: type, ...) — description`) appended to the system message,
   so the model imitates a format it already knows how to produce.
2. **Rewrites history into plain turns.** `assistant.tool_calls` become the JSON text the model
   is supposed to emit, and `role: "tool"` results become `user` turns labeled
   `TOOL RESULT (name=...)`. The upstream model never sees roles it cannot follow.
3. **Promotes JSON back into `tool_calls`.** Replies are scanned for balanced JSON objects
   (string-aware brace matching) and any object whose `name` matches a registered tool is
   lifted into a proper `tool_calls` entry — in both buffered and streaming responses.
   Unknown names are ignored, so ordinary JSON in prose is left alone.

Because it strips `tools` / `tool_choice` / `parallel_tool_calls` before forwarding, the
upstream model never has to understand OpenAI's tool schema at all.

## Quick start

Requires Node 18+ (built-in `fetch`) and a running Ollama.

```bash
# terminal 1
ollama serve

# terminal 2
node proxy.mjs
# tool-proxy listening on http://127.0.0.1:11435/v1  ->  http://127.0.0.1:11434
```

Point your OpenAI-compatible client at `http://127.0.0.1:11435/v1` — that's it.

```bash
curl http://127.0.0.1:11435/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen2.5-coder:7b",
    "messages": [{"role": "user", "content": "list files in src/"}],
    "tools": [{
      "type": "function",
      "function": {
        "name": "list_files",
        "description": "List files in a directory",
        "parameters": {"type": "object", "properties": {"dir": {"type": "string"}}, "required": ["dir"]}
      }
    }]
  }'
```

### Environment

| var            | default                  | meaning                                  |
| -------------- | ------------------------ | ---------------------------------------- |
| `PORT`         | `11435`                  | listen port                              |
| `HOST`         | `127.0.0.1`              | bind address                             |
| `UPSTREAM`     | `http://127.0.0.1:11434` | Ollama base URL                          |
| `DEFAULT_MODEL`| `qwen2.5-coder:7b`       | used when the request omits `model`      |
| `PROXY_LOG`    | _(off)_                  | append one-line request/timing logs here |
| `PROXY_DEBUG`  | _(off)_                  | also log to stdout                       |

`GET /health` returns `{ ok, upstream, defaultModel }`; `GET /v1/models` is proxied to Ollama.

## Files

| file        | what it is                                                              |
| ----------- | ----------------------------------------------------------------------- |
| `proxy.mjs` | the proxy (single file, no dependencies)                                |
| `test.mjs`  | unit tests for the extraction / rewriting layer — `node test.mjs`       |
| `bench.mjs` | tool-calling bench over 5 scripted cases — `node bench.mjs [runs]`      |
| `e2e.mjs`   | a real multi-step agent loop with tools actually executed               |
| `probe.mjs` | minimal script to sanity-check raw Ollama behaviour without the proxy   |

`bench.mjs` and `e2e.mjs` run their `list_files` / `read_file` tools against a sandbox
directory, `BENCH_ROOT` (default: the current working directory). Adjust the prompts in
`CASES` to match files you actually have.

### Sample bench output

```
case         ok/n   badArgs  leak  avgMs
list-dir     3/3    0        0     1840
read-file    3/3    0        0     2105
search       3/3    0        0     1977
run-cmd      3/3    0        0     2560
no-tool      2/2    0        0     1602

TOTAL 14/14 = 100%
```

`leak` counts replies where a tool-call-shaped JSON stayed visible in `content` instead of
becoming a `tool_call` — the exact failure mode this proxy exists to remove.

## Limitations

- One JSON object per reply is the documented protocol; multiple calls are supported but
  discouraged for small models.
- Streaming responses are fully buffered until the proxy can tell prose from a tool call, so
  first-token latency for tool calls equals full-generation latency. Plain prose streams live.
- Tool schemas are flattened to `name(arg: type)` text, so nested object schemas lose detail
  beyond the top level.
- `/v1/completions`, embeddings, and image inputs are not handled.

## License

MIT
