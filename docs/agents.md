# Using zemsearch with a coding agent

zemsearch exposes its tools over MCP (stdio) and ships a companion skill. Wire
both into whichever agent you use.

## The server

Any agent that supports local **stdio MCP servers** can run:

```
/abs/path/zemsearch/.venv/bin/zemsearch serve --root .
```

`cwd: "."` together with `--root .` makes one entry index whatever workspace the
agent is opened in, so the same config works for every repository. The agent then
gets the four tools `semantic_search`, `get_context`, `reindex`, and
`index_status`.

## The skill

Copy — or point the agent at — [`skills/zemsearch/`](../skills/zemsearch/SKILL.md).
It teaches the agent *when* to reach for semantic search and how to turn hits into
targeted grep/read calls.

## opencode (reference setup)

Register the server in `~/.config/opencode/opencode.json`:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "mcp": {
    "zemsearch": {
      "type": "local",
      "command": [
        "/abs/path/zemsearch/.venv/bin/zemsearch",
        "serve",
        "--root",
        "."
      ],
      "cwd": ".",
      "environment": {
        "ZEMSEARCH_BINARY": "/home/you/.local/share/zemsearch/llama.cpp/build/bin/llama-server"
      },
      "enabled": true,
      "timeout": 120000
    }
  }
}
```

Load the skill globally by pointing opencode at the `skills/` directory:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "skills": {
    "paths": ["/abs/path/zemsearch/skills"]
  }
}
```

Restart opencode after editing (config is not hot-reloaded). Then, in a new repo,
ask the agent to *"initialize the semantic index for this repo"* before searching,
or let it discover the state itself — the server instructions tell it to call
`reindex` when `index_status` reports `initialized: false`.

To disable the server for one project, add
`"mcp": { "zemsearch": { "enabled": false } }` to that repo's `opencode.json`.

## Other agents

The pattern is identical everywhere: register a **local stdio MCP server** whose
command is `zemsearch` and args are `["serve", "--root", "."]` (with an optional
`ZEMSEARCH_BINARY` in the environment), then make the skill available to the
agent. The outer key differs per tool (many use `mcpServers`), but the local-server
entry — `command` + `args` + `env` — is the same shape.

Consult your agent's MCP documentation for the exact file and key; the server
command and the skill are the only zemsearch-specific parts.
