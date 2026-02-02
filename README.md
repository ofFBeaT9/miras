# MIRAS Memory API + BMAD

A per-project long-term memory system based on Google's MIRAS framework, 
designed as the memory backbone for BMAD-driven coding workflows.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        BMAD Layer                           │
│  Analyst → PM → Architect → Developer → QA                  │
│  (each agent reads/writes project memory via the bridge)    │
└──────────────────────┬──────────────────────────────────────┘
                       │  HTTP (BMADBridge)
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                    MIRAS Memory API                         │
│  FastAPI server on :8100                                    │
│                                                             │
│  POST /projects                    — create project         │
│  POST /projects/{id}/memorize      — store (if surprising)  │
│  POST /projects/{id}/recall        — search memory          │
│  POST /projects/{id}/context       — get LLM context window │
│  POST /projects/{id}/chat          — memory-augmented chat  │
│  POST /projects/{id}/consolidate   — decay + evict          │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              MIRAS Memory Engine (per project)              │
│                                                             │
│  ┌─────────────────┐  ┌──────────────────┐                 │
│  │ Surprise Calc    │  │ Retention Manager │                 │
│  │ (Attentional     │  │ (Retention Gate)  │                 │
│  │  Bias)           │  │                   │                 │
│  │ • L2 (default)   │  │ • Exponential     │                 │
│  │ • Huber (YAAD)   │  │ • Adaptive(Titans)│                 │
│  │ • Lp-norm(MONETA)│  │ • Sliding window  │                 │
│  │ • KL-div(MEMORA) │  │ • None            │                 │
│  └─────────────────┘  └──────────────────┘                 │
│  ┌─────────────────┐  ┌──────────────────┐                 │
│  │ Memory Updater   │  │ Project Memory   │                 │
│  │ (Memory Algo)    │  │ (Memory Arch)    │                 │
│  │ • SGD            │  │ • Vector         │                 │
│  │ • SGD+Momentum   │  │ • Matrix         │                 │
│  │ • Adam           │  │ • MLP (deep)     │                 │
│  └─────────────────┘  └──────────────────┘                 │
│                                                             │
│  Each project = isolated memory instance + own config       │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Start the API

```bash
cd miras-api
pip install -r requirements.txt
python api/server.py
# → Running on http://0.0.0.0:8100
# → Docs at http://localhost:8100/docs
```

### 2. Create a Project

```bash
# Create a project with Titans-default MIRAS config
curl -X POST http://localhost:8100/projects \
  -H "Content-Type: application/json" \
  -d '{
    "project_id": "my-saas-app",
    "name": "My SaaS Application", 
    "preset": "titans_default"
  }'
```

### 3. Install BMAD in your coding project

```bash
mkdir my-saas-app && cd my-saas-app
npx bmad-method install
```

### 4. Connect BMAD to MIRAS

```python
from bmad_integration import BMADBridge

bridge = BMADBridge(api_url="http://localhost:8100", project_id="my-saas-app")

# Analyst phase: store requirements
bridge.store_fact("analyst", "Target users are small business owners")
bridge.store_constraint("analyst", "Must run on $20/month infrastructure budget")

# Architect phase: load context, then store decisions
context = bridge.load_agent_context("architect", task="design the auth system")
print(context)  # → shows analyst's findings from memory

bridge.store_decision("architect", "JWT with refresh tokens, 15min access / 7day refresh")

# Dev phase: memory flows forward
dev_context = bridge.load_agent_context("dev", task="implement auth endpoints")
# → includes both analyst constraints AND architect decisions
```

### 5. Or use the CLI

```bash
# Store from command line
python -m bmad_integration.bridge --project my-saas-app \
  store --agent architect --type decision \
  --content "Using PostgreSQL with Prisma ORM" --tags database backend

# Recall
python -m bmad_integration.bridge --project my-saas-app \
  recall --query "database"

# Load context for an agent
python -m bmad_integration.bridge --project my-saas-app \
  context --agent dev --task "implement user CRUD"
```

## MIRAS Presets

| Preset | Best For | Attentional Bias | Retention |
|--------|----------|-----------------|-----------|
| `titans_default` | Most projects | L2 (standard MSE) | Adaptive decay |
| `yaad_robust` | Noisy/exploratory phases | Huber (outlier-resistant) | Adaptive decay |
| `moneta_strict` | Mature, stable projects | Lp-norm (strict) | Exponential decay |
| `memora_stable` | Long-running, many contributors | KL-divergence (stable) | Adaptive decay |
| `lightweight` | Small projects, fast iteration | L2 | Sliding window |

### Test results show different biases handle similarity differently:

```
l2        : surprise for similar content = 0.8264
huber     : surprise for similar content = 0.4132  ← most forgiving
lp_norm   : surprise for similar content = 0.8668  ← strictest
kl_div    : surprise for similar content = 0.2398  ← most stable
```

## BMAD Workflow with MIRAS Memory

```
Phase 1: Analysis
  └→ Analyst agent stores: requirements, constraints, user research
     └→ MIRAS: high surprise for novel requirements
         
Phase 2: Planning (PM)
  └→ PM loads analyst memories via /context
  └→ PM stores: user stories, acceptance criteria, priorities
     └→ MIRAS: surprise filters redundant stories
  └→ bridge.transition_phase("analysis", "planning")
     └→ Memory consolidation runs (decay old, keep critical)
         
Phase 3: Architecture
  └→ Architect loads ALL prior memories via /context
  └→ Architect stores: design decisions, trade-offs, component boundaries
     └→ MIRAS: momentum boosts related decisions when a breaking change occurs
         
Phase 4: Implementation
  └→ Developer loads architecture + constraint memories
  └→ Developer stores: code patterns, API contracts, bugs
  └→ Each story file can trigger /memorize for implementation notes
         
Phase 5: QA
  └→ QA agent loads everything
  └→ QA stores: test results, coverage gaps, bugs found
  └→ bridge.consolidate() between sprints
```

## API Reference

### Projects
- `POST /projects` — Create project with MIRAS config
- `GET /projects` — List all projects
- `GET /projects/{id}` — Project stats
- `DELETE /projects/{id}` — Delete project + memories
- `PUT /projects/{id}/config` — Update MIRAS settings
- `GET /presets` — List available presets

### Memory
- `POST /projects/{id}/memorize` — Store (surprise-gated)
- `POST /projects/{id}/memorize/bulk` — Bulk store (artifact ingestion)
- `POST /projects/{id}/recall` — Search with filters
- `DELETE /projects/{id}/memory/{mem_id}` — Explicit forget
- `POST /projects/{id}/consolidate` — Decay + evict cycle

### Integration
- `POST /projects/{id}/context` — Get formatted context for LLM injection
- `POST /projects/{id}/chat` — Memory-augmented LLM chat (needs ANTHROPIC_API_KEY)

Full interactive docs: `http://localhost:8100/docs`

## File Structure

```
miras-api/
├── miras_memory/
│   ├── __init__.py
│   ├── engine.py              ← Core MIRAS engine (4 design choices)
│   └── registry.py            ← Multi-project management
├── api/
│   ├── __init__.py
│   └── server.py              ← FastAPI REST server (:8100)
├── bmad_integration/
│   ├── __init__.py
│   ├── __main__.py            ← CLI entry point
│   ├── bridge.py              ← BMAD ↔ MIRAS connector + CLI
│   └── agents/
│       └── miras-architect.md ← Custom BMAD agent template
├── mcp_server.py              ← MCP server (official SDK, JSON-RPC stdio/SSE)
├── ai_connector.py            ← Sync memory to AI tool config files
├── tests/
│   ├── __init__.py
│   └── test_integration.py
├── requirements.txt
└── README.md
```

## MCP Server (Claude Code, Cursor, Claude Desktop, Cline)

The `mcp_server.py` is a proper Model Context Protocol server using the
official Anthropic MCP Python SDK. It communicates via JSON-RPC 2.0 over
stdio or SSE — the same protocol Claude Code, Cursor, and Claude Desktop expect.

### Claude Code setup

```bash
claude mcp add miras-memory -- python /path/to/miras-api/mcp_server.py --project my-project
```

Or add to `.claude.json`:

```json
{
  "mcpServers": {
    "miras-memory": {
      "command": "python",
      "args": ["/path/to/miras-api/mcp_server.py", "--project", "my-project"]
    }
  }
}
```

### Cursor setup (`.cursor/mcp.json`)

```json
{
  "mcpServers": {
    "miras-memory": {
      "command": "python",
      "args": ["/path/to/miras-api/mcp_server.py", "--project", "my-project"]
    }
  }
}
```

### Claude Desktop setup (`claude_desktop_config.json`)

Same JSON format as Cursor above.

### MCP Tools exposed

| Tool | Description |
|------|-------------|
| `recall_memory` | Search memory before making decisions |
| `store_memory` | Save decisions, facts, constraints |
| `get_project_context` | Load full context for a session |
| `store_code_pattern` | Record reusable patterns |
| `list_memories` | Browse all stored memories |
| `consolidate_memory` | Decay old memories, evict low-weight |
| `project_stats` | Memory count, capacity, breakdowns |

## AI Tool Connector (file sync for any tool)

For tools without MCP support, `ai_connector.py` syncs memory into
config files each tool reads:

```bash
# Sync to Claude Code (writes CLAUDE.md)
python ai_connector.py --project my-project sync --tool claude-code

# Sync to Cursor (writes .cursor/rules/miras-memory.mdc)
python ai_connector.py --project my-project sync --tool cursor

# Sync to all tools at once
python ai_connector.py --project my-project sync --tool all

# Auto-sync every 30s while coding
python ai_connector.py --project my-project watch --tool cursor --interval 30

# List supported tools
python ai_connector.py tools
```

Supported tools: `claude-code`, `cursor`, `windsurf`, `cline`, `aider`, `generic`, `all`

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MIRAS_STORAGE_DIR` | `./memory_store` | Where project memories are persisted |
| `PORT` | `8100` | API server port |
| `ANTHROPIC_API_KEY` | — | Required for `/chat` endpoint |
#   m i r a s  
 