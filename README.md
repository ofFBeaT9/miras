# MIRAS Memory API

A per-project long-term memory system based on Google's MIRAS framework, designed as the memory backbone for BMAD-driven coding workflows.

## Features

- **Surprise-based memory gating** - Only stores genuinely new information
- **Multiple presets** - Titans, YAAD, MONETA, MEMORA configurations
- **REST API** - Full CRUD on port 8100
- **MCP Server** - Native integration with Claude Code, Cursor, Cline
- **BMAD Bridge** - Seamless agent workflow integration
- **AI Tool Sync** - Export memory to any AI coding tool

## Quick Start

### 1. Install and Start

```bash
pip install -r requirements.txt
python api/server.py
# Server runs on http://localhost:8100
# Interactive docs at http://localhost:8100/docs
```

### 2. Create a Project

```bash
curl -X POST http://localhost:8100/projects \
  -H "Content-Type: application/json" \
  -d '{"project_id":"my-app","name":"My App","preset":"titans_default"}'
```

### 3. Store and Recall Memories

```bash
# Store a decision
curl -X POST http://localhost:8100/projects/my-app/memorize \
  -H "Content-Type: application/json" \
  -d '{"content":"Using PostgreSQL for the database","content_type":"decision"}'

# Recall memories
curl -X POST http://localhost:8100/projects/my-app/recall \
  -H "Content-Type: application/json" \
  -d '{"query":"database"}'

# Get LLM-ready context
curl -X POST http://localhost:8100/projects/my-app/context \
  -H "Content-Type: application/json" \
  -d '{"query":"database","max_tokens":2000}'
```

## Architecture

```
+-------------------------------------------------------------+
|                        BMAD Layer                           |
|  Analyst -> PM -> Architect -> Developer -> QA              |
|  (each agent reads/writes project memory via the bridge)    |
+-----------------------------+-------------------------------+
                              | HTTP (BMADBridge)
                              v
+-------------------------------------------------------------+
|                    MIRAS Memory API                         |
|  FastAPI server on :8100                                    |
|                                                             |
|  POST /projects                    - create project         |
|  POST /projects/{id}/memorize      - store (if surprising)  |
|  POST /projects/{id}/recall        - search memory          |
|  POST /projects/{id}/context       - get LLM context window |
|  POST /projects/{id}/chat          - memory-augmented chat  |
|  POST /projects/{id}/consolidate   - decay + evict          |
+-----------------------------+-------------------------------+
                              |
                              v
+-------------------------------------------------------------+
|              MIRAS Memory Engine (per project)              |
|                                                             |
|  +------------------+  +-------------------+                |
|  | Surprise Calc    |  | Retention Manager |                |
|  | (Attentional     |  | (Retention Gate)  |                |
|  |  Bias)           |  |                   |                |
|  | - L2 (default)   |  | - Exponential     |                |
|  | - Huber (YAAD)   |  | - Adaptive(Titans)|                |
|  | - Lp-norm(MONETA)|  | - Sliding window  |                |
|  | - KL-div(MEMORA) |  | - None            |                |
|  +------------------+  +-------------------+                |
|                                                             |
|  Each project = isolated memory instance + own config       |
+-------------------------------------------------------------+
```

## MIRAS Presets

| Preset | Best For | Attentional Bias | Retention |
|--------|----------|------------------|-----------|
| `titans_default` | Most projects | L2 (standard MSE) | Adaptive decay |
| `yaad_robust` | Noisy/exploratory phases | Huber (outlier-resistant) | Adaptive decay |
| `moneta_strict` | Mature, stable projects | Lp-norm (strict) | Exponential decay |
| `memora_stable` | Long-running, many contributors | KL-divergence (stable) | Adaptive decay |
| `lightweight` | Small projects, fast iteration | L2 | Sliding window |

## BMAD Integration

```python
from bmad_integration import BMADBridge

bridge = BMADBridge(api_url="http://localhost:8100", project_id="my-app")

# Analyst phase: store requirements
bridge.store_fact("analyst", "Target users are small business owners")
bridge.store_constraint("analyst", "Must run on $20/month infrastructure")

# Architect phase: load context, then store decisions
context = bridge.load_agent_context("architect", task="design auth system")
bridge.store_decision("architect", "JWT with refresh tokens")

# Dev phase: memory flows forward
dev_context = bridge.load_agent_context("dev", task="implement auth")
# Includes both analyst constraints AND architect decisions
```

### BMAD CLI

```bash
# Store a decision
python -m bmad_integration --project my-app \
  store --agent architect --type decision \
  --content "Using PostgreSQL with Prisma ORM"

# Recall memories
python -m bmad_integration --project my-app recall --query "database"

# Load agent context
python -m bmad_integration --project my-app context --agent dev --task "implement CRUD"
```

## MCP Server (Claude Code, Cursor, Cline)

### Claude Code Setup

```bash
claude mcp add miras-memory -- python /path/to/mcp_server.py --project my-project
```

Or add to `.claude.json`:

```json
{
  "mcpServers": {
    "miras-memory": {
      "command": "python",
      "args": ["/path/to/mcp_server.py", "--project", "my-project"]
    }
  }
}
```

### Cursor Setup (`.cursor/mcp.json`)

```json
{
  "mcpServers": {
    "miras-memory": {
      "command": "python",
      "args": ["/path/to/mcp_server.py", "--project", "my-project"]
    }
  }
}
```

### MCP Tools

| Tool | Description |
|------|-------------|
| `recall_memory` | Search memory before making decisions |
| `store_memory` | Save decisions, facts, constraints |
| `get_project_context` | Load full context for a session |
| `store_code_pattern` | Record reusable patterns |
| `list_memories` | Browse all stored memories |
| `consolidate_memory` | Decay old memories, evict low-weight |
| `project_stats` | Memory count, capacity, breakdowns |

## AI Tool Connector

For tools without MCP support, sync memory to config files:

```bash
# Sync to Claude Code (writes CLAUDE.md)
python ai_connector.py --project my-project sync --tool claude-code

# Sync to Cursor (writes .cursor/rules/miras-memory.mdc)
python ai_connector.py --project my-project sync --tool cursor

# Sync to all tools
python ai_connector.py --project my-project sync --tool all

# List supported tools
python ai_connector.py tools
```

Supported: `claude-code`, `cursor`, `windsurf`, `cline`, `aider`, `generic`

## API Reference

### Projects
- `POST /projects` - Create project with MIRAS config
- `GET /projects` - List all projects
- `GET /projects/{id}` - Project stats
- `DELETE /projects/{id}` - Delete project + memories
- `PUT /projects/{id}/config` - Update MIRAS settings

### Memory
- `POST /projects/{id}/memorize` - Store (surprise-gated)
- `POST /projects/{id}/memorize/bulk` - Bulk store
- `POST /projects/{id}/recall` - Search with filters
- `DELETE /projects/{id}/memory/{mem_id}` - Explicit forget
- `POST /projects/{id}/consolidate` - Decay + evict cycle

### Integration
- `POST /projects/{id}/context` - LLM-ready context
- `POST /projects/{id}/chat` - Memory-augmented chat (requires `ANTHROPIC_API_KEY`)

## Project Structure

```
miras-api/
├── miras_memory/
│   ├── engine.py          # Core MIRAS engine
│   └── registry.py        # Multi-project management
├── api/
│   └── server.py          # FastAPI REST server (:8100)
├── bmad_integration/
│   └── bridge.py          # BMAD <-> MIRAS connector
├── mcp_server.py          # MCP server for Claude/Cursor
├── ai_connector.py        # Sync to AI tool configs
├── tests/
│   └── test_integration.py
└── requirements.txt
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MIRAS_STORAGE_DIR` | `./memory_store` | Where memories are persisted |
| `PORT` | `8100` | API server port |
| `ANTHROPIC_API_KEY` | - | Required for `/chat` endpoint |

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run integration tests
python -m pytest tests/test_integration.py -v
```

## License

MIT
