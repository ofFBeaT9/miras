"""
MIRAS Memory — MCP Server (Model Context Protocol)

A proper MCP server using the official Anthropic MCP Python SDK.
Exposes MIRAS project memory as tools that Claude Code, Cursor,
Claude Desktop, Cline, and any MCP-compatible client can use directly.

Transports:
  - stdio  (default): Claude Code, Claude Desktop spawn this as a subprocess
  - sse:              Cursor, network clients connect via HTTP SSE

Usage:
  # stdio mode (for Claude Code / Claude Desktop)
  python mcp_server.py

  # SSE mode (for Cursor / network)
  python mcp_server.py --transport sse --port 8200

Claude Code setup:
  claude mcp add miras-memory -- python /path/to/miras-api/mcp_server.py --project YOUR_PROJECT_ID

  Or add to .claude.json:
  {
    "mcpServers": {
      "miras-memory": {
        "command": "python",
        "args": ["/path/to/miras-api/mcp_server.py", "--project", "my-project"]
      }
    }
  }

Cursor setup (.cursor/mcp.json):
  {
    "mcpServers": {
      "miras-memory": {
        "command": "python",
        "args": ["/path/to/miras-api/mcp_server.py", "--project", "my-project"]
      }
    }
  }

Claude Desktop setup (claude_desktop_config.json):
  {
    "mcpServers": {
      "miras-memory": {
        "command": "python",
        "args": ["/path/to/miras-api/mcp_server.py", "--project", "my-project"]
      }
    }
  }
"""

import sys
import os
import json
import argparse
import logging

# Parse args BEFORE mcp import (stdio mode uses stdin/stdout for JSON-RPC,
# so we must not print anything to stdout)
parser = argparse.ArgumentParser(description="MIRAS Memory MCP Server")
parser.add_argument("--project", default="default", help="MIRAS project ID")
parser.add_argument("--transport", default="stdio", choices=["stdio", "sse"],
                    help="MCP transport: stdio (default) or sse")
parser.add_argument("--port", type=int, default=8200, help="Port for SSE transport")
parser.add_argument("--api", default="", help="MIRAS API URL (if using remote API)")
parser.add_argument("--data-dir", default="", help="Directory for memory persistence")

args, unknown = parser.parse_known_args()

# Configure logging to stderr (NEVER stdout — it would corrupt JSON-RPC)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [MIRAS-MCP] %(message)s",
    stream=sys.stderr,
)
log = logging.getLogger("miras-mcp")

# Add parent dir to path so we can import miras_memory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mcp.server.fastmcp import FastMCP
from miras_memory.engine import ProjectMemory, MIRASConfig, PRESETS
from miras_memory.registry import ProjectRegistry

# ── Initialize ───────────────────────────────────────────────────────────

data_dir = args.data_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), "memory_store")
registry = ProjectRegistry(base_dir=data_dir)

# Auto-create the project if it doesn't exist
existing = registry.list_projects()
if args.project not in [p["project_id"] for p in existing]:
    registry.create_project(args.project, preset="titans_default")
    log.info(f"Created new project: {args.project}")
else:
    log.info(f"Loaded existing project: {args.project}")


def get_memory() -> ProjectMemory:
    """Get the current project's memory instance"""
    return registry.get_project(args.project)


# ── MCP Server ───────────────────────────────────────────────────────────

mcp = FastMCP(
    "miras-memory",
    host="0.0.0.0",
    port=args.port,
    instructions="""MIRAS Memory Server — Persistent project memory that survives across sessions.

Use these tools to maintain continuity in your coding work:
- recall_memory: ALWAYS check this before making decisions to see what's already been decided
- store_memory: Save important decisions, constraints, and discoveries
- get_project_context: Load full project context at the start of a session
- store_code_pattern: Record reusable patterns and implementation details
- search_and_store: Search for related memories, then store new info

Memory is surprise-gated: routine/duplicate info is automatically filtered out.
Only novel, high-surprise content persists. This prevents memory bloat.""",
)


@mcp.tool()
def recall_memory(query: str, top_k: int = 5) -> str:
    """Search project memory for relevant context.

    ALWAYS use this before making architectural decisions, choosing libraries,
    or implementing features. Past decisions and constraints may already exist.

    Args:
        query: What to search for (e.g. "authentication", "database choice", "deployment")
        top_k: Max results to return (default 5)

    Returns:
        Matching memories with type, surprise score, and content
    """
    mem = get_memory()
    results = mem.recall(query, top_k=top_k)

    if not results:
        return f"No memories found for '{query}'. This may be a new topic for this project."

    lines = []
    for r in results:
        lines.append(
            f"[{r.content_type}|surprise={r.surprise_score:.2f}|weight={r.relevance_weight:.2f}] "
            f"{r.content}"
            + (f" (tags: {', '.join(r.tags)})" if r.tags else "")
        )
    return "\n".join(lines)


@mcp.tool()
def store_memory(
    content: str,
    content_type: str = "decision",
    tags: str = "",
    source_agent: str = "dev",
) -> str:
    """Store an important decision, fact, constraint, or code pattern in project memory.

    The surprise metric automatically filters duplicate/routine content.
    Only novel information gets stored. Use this for:
    - Architecture decisions (e.g. "Using PostgreSQL with Prisma ORM")
    - Constraints discovered (e.g. "API rate limit is 100 req/min")
    - Important bugs found (e.g. "Race condition in auth token refresh")
    - Design patterns chosen (e.g. "Repository pattern for data access layer")

    Args:
        content: What to remember (be specific and concise)
        content_type: One of: decision, fact, constraint, code, context
        tags: Comma-separated tags (e.g. "auth,security,backend")
        source_agent: Who is storing this (analyst, pm, architect, dev, qa)

    Returns:
        Whether stored, the surprise score, and memory ID
    """
    mem = get_memory()
    tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []

    entry = mem.memorize(
        content=content,
        content_type=content_type,
        source_agent=source_agent,
        tags=tag_list,
    )

    if entry:
        return (
            f"✓ Stored (surprise={entry.surprise_score:.2f}, id={entry.id[:12]})\n"
            f"  Type: {content_type}, Tags: {tag_list or 'none'}\n"
            f"  Content: {content[:100]}{'...' if len(content) > 100 else ''}"
        )
    else:
        return (
            f"⊘ Not stored — content was too similar to existing memories "
            f"(below surprise threshold). This is expected for routine/duplicate info."
        )


@mcp.tool()
def get_project_context(query: str = "", max_tokens: int = 4000) -> str:
    """Get full project memory context. Use this at the start of a session
    or when switching to a new task area.

    Args:
        query: Optional focus area (e.g. "backend architecture"). Empty = all top memories.
        max_tokens: Approximate context size limit

    Returns:
        Formatted project memories for context injection
    """
    mem = get_memory()
    context = mem.get_context_window(query=query, max_tokens=max_tokens)

    if not context.strip():
        return (
            f"Project '{args.project}' has no memories yet.\n"
            f"Start storing decisions and discoveries with store_memory."
        )

    stats = mem.stats()
    config = stats.get("config", {})
    capacity = stats.get("capacity", config.get("capacity", "?"))
    header = (
        f"Project: {args.project} | "
        f"Memories: {stats.get('count', 0)}/{capacity} | "
        f"Avg surprise: {stats.get('avg_surprise', 0):.2f}"
    )
    return f"{header}\n{'=' * len(header)}\n{context}"


@mcp.tool()
def store_code_pattern(
    pattern_name: str,
    description: str,
    code_snippet: str = "",
    tags: str = "",
) -> str:
    """Store a reusable code pattern or implementation detail.

    Use this when you've figured out the right way to do something that
    might need to be repeated or referenced later.

    Args:
        pattern_name: Short name (e.g. "JWT refresh flow", "DB migration pattern")
        description: What this pattern does and when to use it
        code_snippet: Optional code example (keep short — key lines only)
        tags: Comma-separated tags

    Returns:
        Storage result
    """
    content = f"Pattern: {pattern_name}\n{description}"
    if code_snippet:
        content += f"\nExample:\n```\n{code_snippet}\n```"

    mem = get_memory()
    tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []
    tag_list.append("pattern")

    entry = mem.memorize(
        content=content,
        content_type="code",
        source_agent="dev",
        tags=tag_list,
    )

    if entry:
        return f"✓ Pattern '{pattern_name}' stored (surprise={entry.surprise_score:.2f})"
    else:
        return f"⊘ Pattern not stored — similar pattern may already exist. Use recall_memory to check."


@mcp.tool()
def list_memories(
    content_type: str = "",
    source_agent: str = "",
    limit: int = 20,
) -> str:
    """List all stored memories, optionally filtered by type or agent.

    Args:
        content_type: Filter by type: decision, fact, constraint, code, context (empty = all)
        source_agent: Filter by agent: analyst, pm, architect, dev, qa (empty = all)
        limit: Max results

    Returns:
        Numbered list of all memories
    """
    mem = get_memory()
    memories = mem.memories

    if content_type:
        memories = [m for m in memories if m.content_type == content_type]
    if source_agent:
        memories = [m for m in memories if m.source_agent == source_agent]

    # Sort by weight descending
    memories = sorted(memories, key=lambda m: m.relevance_weight, reverse=True)[:limit]

    if not memories:
        return "No memories match the filter."

    lines = []
    for i, m in enumerate(memories, 1):
        tags_str = f" [{', '.join(m.tags)}]" if m.tags else ""
        lines.append(
            f"{i}. [{m.content_type}|w={m.relevance_weight:.2f}|s={m.surprise_score:.2f}] "
            f"{m.content[:120]}{'...' if len(m.content) > 120 else ''}{tags_str}"
        )
    return "\n".join(lines)


@mcp.tool()
def consolidate_memory() -> str:
    """Run memory consolidation — decay old memories and evict low-weight ones.

    Use this between major project phases or when memory feels cluttered.
    Similar to how the brain consolidates during sleep.

    Returns:
        Before/after memory count
    """
    mem = get_memory()
    before = len(mem.memories)
    mem.consolidate()
    after = len(mem.memories)

    return (
        f"Consolidation complete: {before} → {after} memories "
        f"({before - after} forgotten)"
    )


@mcp.tool()
def project_stats() -> str:
    """Get memory statistics for the current project.

    Returns:
        Memory count, capacity, averages, breakdowns by type and agent
    """
    mem = get_memory()
    stats = mem.stats()

    # stats shape differs between empty and populated projects
    config = stats.get("config", {})
    capacity = stats.get("capacity", config.get("capacity", "?"))

    lines = [
        f"Project: {args.project}",
        f"Preset: {config.get('memory_architecture', 'unknown')}/{config.get('attentional_bias', '?')}",
        f"Memories: {stats.get('count', 0)}/{capacity}",
        f"Avg surprise: {stats.get('avg_surprise', 0):.3f}",
        f"Avg weight: {stats.get('avg_weight', 0):.3f}",
        f"By type: {json.dumps(stats.get('by_type', {}))}",
        f"By agent: {json.dumps(stats.get('by_agent', {}))}",
    ]
    return "\n".join(lines)


# ── Run ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if args.transport == "sse":
        log.info(f"Starting MCP server (SSE) on port {args.port}")
        mcp.run(transport="sse")
    else:
        log.info("Starting MCP server (stdio)")
        mcp.run(transport="stdio")
