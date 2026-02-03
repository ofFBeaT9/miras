"""
MIRAS ↔ AI Tool Connector

Syncs project memory into the config files that AI coding tools actually read:
  - Claude Code  → CLAUDE.md
  - Cursor       → .cursor/rules/miras-memory.mdc
  - Windsurf     → .windsurfrules
  - Cline/Roo    → .clinerules
  - Aider        → .aider.conf.yml context
  - Generic      → .ai-context/memory.md (any tool that reads project files)

Also provides an MCP (Model Context Protocol) server so tools with
MCP support can query memory directly without file syncing.

Usage:
  # One-time sync (run before coding session)
  python ai_connector.py sync --project tritone --tool claude-code

  # Watch mode (auto-sync every 30s while you code)
  python ai_connector.py watch --project tritone --tool cursor --interval 30

  # Sync to all tools at once
  python ai_connector.py sync --project tritone --tool all

  # Start MCP server (for tools that support it)
  python ai_connector.py mcp --project tritone --port 8200
"""

import os
import sys
import time
import json
import argparse
import requests
from pathlib import Path
from datetime import datetime
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


# ── Configuration ────────────────────────────────────────────────────────

DEFAULT_API = "http://localhost:8100"

# Where each tool looks for context
TOOL_CONFIGS = {
    "claude-code": {
        "file": "CLAUDE.md",
        "format": "markdown",
        "description": "Claude Code (CLI + IDE)",
        "supports_mcp": True,
    },
    "cursor": {
        "file": ".cursor/rules/miras-memory.mdc",
        "format": "mdc",
        "description": "Cursor IDE",
        "supports_mcp": True,
    },
    "windsurf": {
        "file": ".windsurfrules",
        "format": "markdown",
        "description": "Windsurf IDE",
        "supports_mcp": False,
    },
    "cline": {
        "file": ".clinerules",
        "format": "markdown",
        "description": "Cline / Roo Code",
        "supports_mcp": True,
    },
    "aider": {
        "file": ".aider.context.md",
        "format": "markdown",
        "description": "Aider CLI",
        "supports_mcp": False,
    },
    "generic": {
        "file": ".ai-context/memory.md",
        "format": "markdown",
        "description": "Generic (any tool that reads project files)",
        "supports_mcp": False,
    },
}


# ── Memory Fetcher ───────────────────────────────────────────────────────

class MemoryFetcher:
    """Pulls memory from the MIRAS API"""

    def __init__(self, api_url: str, project_id: str):
        self.api_url = api_url.rstrip("/")
        self.project_id = project_id

    def get_context(self, query: str = "", max_tokens: int = 6000) -> str:
        resp = requests.post(
            f"{self.api_url}/projects/{self.project_id}/context",
            json={"query": query, "max_tokens": max_tokens},
        )
        resp.raise_for_status()
        return resp.json().get("context", "")

    def get_stats(self) -> dict:
        resp = requests.get(f"{self.api_url}/projects/{self.project_id}")
        resp.raise_for_status()
        return resp.json()

    def recall(self, query: str, top_k: int = 10) -> list[dict]:
        resp = requests.post(
            f"{self.api_url}/projects/{self.project_id}/recall",
            json={"query": query, "top_k": top_k},
        )
        resp.raise_for_status()
        return resp.json().get("memories", [])

    def memorize(self, content: str, content_type: str = "context",
                 source_agent: str = "", tags: list[str] = None) -> dict:
        resp = requests.post(
            f"{self.api_url}/projects/{self.project_id}/memorize",
            json={
                "content": content,
                "content_type": content_type,
                "source_agent": source_agent,
                "tags": tags or [],
            },
        )
        resp.raise_for_status()
        return resp.json()


# ── Format Builders ──────────────────────────────────────────────────────

def build_markdown_context(context: str, stats: dict, project_id: str) -> str:
    """Standard markdown format for most tools"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    count = stats.get("count", 0)
    capacity = stats.get("capacity", 500)

    return f"""# Project Memory (MIRAS)

> Auto-synced from MIRAS Memory API at {now}
> Project: {project_id} | Memories: {count}/{capacity}
> To update: `python ai_connector.py sync --project {project_id}`

## Critical Context

The following is persistent project memory maintained by the MIRAS system.
Each memory is tagged with its type, surprise score (novelty when stored),
and current weight (relevance after decay). Use this context to maintain
continuity across sessions.

## Memories

{context if context else "(No memories stored yet. Feed documents to the MIRAS API to populate.)"}

## Memory Protocol

When you make important decisions or discover constraints:
1. State them clearly so they can be captured
2. Prefix critical decisions with [CRITICAL DECISION]
3. Reference past memories naturally — don't repeat stored info unnecessarily
4. Flag contradictions with stored memories when you notice them
"""


def build_mdc_context(context: str, stats: dict, project_id: str) -> str:
    """Cursor .mdc format (markdown with frontmatter)"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    count = stats.get("count", 0)

    return f"""---
description: MIRAS project memory for {project_id} — persistent context across sessions
globs:
alwaysApply: true
---

# Project Memory (MIRAS) — {count} memories

Auto-synced at {now}. This file is auto-generated — do not edit manually.

{context if context else "(No memories yet.)"}

When making decisions, state them clearly. Prefix critical ones with [CRITICAL DECISION].
Flag any contradictions with the above memories.
"""


def build_claude_md(context: str, stats: dict, project_id: str,
                    existing_claude_md: str = "") -> str:
    """
    For Claude Code: merge MIRAS memory into existing CLAUDE.md
    without overwriting the user's own instructions.
    """
    miras_section = build_markdown_context(context, stats, project_id)

    # Markers for the auto-managed section
    START_MARKER = "<!-- MIRAS-MEMORY-START -->"
    END_MARKER = "<!-- MIRAS-MEMORY-END -->"

    if existing_claude_md:
        # Replace existing MIRAS section or append
        if START_MARKER in existing_claude_md:
            before = existing_claude_md[:existing_claude_md.index(START_MARKER)]
            after_marker = existing_claude_md.find(END_MARKER)
            if after_marker != -1:
                after = existing_claude_md[after_marker + len(END_MARKER):]
            else:
                after = ""
            return f"{before}{START_MARKER}\n{miras_section}\n{END_MARKER}{after}"
        else:
            return f"{existing_claude_md}\n\n{START_MARKER}\n{miras_section}\n{END_MARKER}\n"
    else:
        return f"{START_MARKER}\n{miras_section}\n{END_MARKER}\n"


# ── Sync Engine ──────────────────────────────────────────────────────────

def sync_to_tool(tool_name: str, fetcher: MemoryFetcher, project_dir: str):
    """Sync MIRAS memory to a specific tool's config file"""
    if tool_name == "all":
        for name in TOOL_CONFIGS:
            sync_to_tool(name, fetcher, project_dir)
        return

    if tool_name not in TOOL_CONFIGS:
        print(f"  ✗ Unknown tool: {tool_name}")
        print(f"    Available: {', '.join(TOOL_CONFIGS.keys())}, all")
        return

    config = TOOL_CONFIGS[tool_name]
    filepath = Path(project_dir) / config["file"]

    # Fetch memory
    context = fetcher.get_context(max_tokens=6000)
    stats = fetcher.get_stats()

    # Build content based on format
    if tool_name == "claude-code":
        existing = filepath.read_text() if filepath.exists() else ""
        content = build_claude_md(context, stats, fetcher.project_id, existing)
    elif config["format"] == "mdc":
        content = build_mdc_context(context, stats, fetcher.project_id)
    else:
        content = build_markdown_context(context, stats, fetcher.project_id)

    # Write
    filepath.parent.mkdir(parents=True, exist_ok=True)
    filepath.write_text(content)

    count = stats.get("count", 0)
    print(f"  ✓ {config['description']}: synced {count} memories → {filepath}")


def watch_and_sync(tool_name: str, fetcher: MemoryFetcher,
                   project_dir: str, interval: int = 30):
    """Continuously sync memory to tool config files"""
    print(f"  Watching MIRAS memory for project '{fetcher.project_id}'")
    print(f"  Syncing to {tool_name} every {interval}s (Ctrl+C to stop)")
    print()

    try:
        while True:
            sync_to_tool(tool_name, fetcher, project_dir)
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n  Stopped watching.")


# ── MCP Server ───────────────────────────────────────────────────────────

def run_mcp_server(fetcher: MemoryFetcher, port: int = 8200):
    """
    Start the real MCP server (mcp_server.py).

    The actual MCP server is in mcp_server.py and uses the official
    Anthropic MCP Python SDK with proper JSON-RPC 2.0 protocol.

    This function is a convenience launcher. For direct usage:
      python mcp_server.py --project YOUR_PROJECT --transport sse --port 8200

    Claude Code / Cursor / Claude Desktop config:
    {
      "mcpServers": {
        "miras-memory": {
          "command": "python",
          "args": ["/path/to/miras-api/mcp_server.py", "--project", "YOUR_PROJECT"]
        }
      }
    }
    """
    import subprocess
    mcp_path = os.path.join(os.path.dirname(__file__), "mcp_server.py")

    if not os.path.exists(mcp_path):
        print(f"  ✗ MCP server not found at {mcp_path}")
        return

    cmd = [
        sys.executable, mcp_path,
        "--project", fetcher.project_id,
        "--transport", "sse",
        "--port", str(port),
    ]
    print(f"  Starting real MCP server: {' '.join(cmd)}")
    print(f"  Protocol: JSON-RPC 2.0 over SSE (Model Context Protocol)")
    print(f"  Endpoint: http://localhost:{port}")
    print()
    print(f"  For Claude Code (stdio mode), add to .claude.json:")
    print(f'    {{"mcpServers": {{"miras-memory": {{"command": "python",')
    print(f'      "args": ["{mcp_path}", "--project", "{fetcher.project_id}"]}}}}}}')

    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n  MCP server stopped.")


# ── Post-Session Capture ─────────────────────────────────────────────────

def capture_session(fetcher: MemoryFetcher, session_file: str, agent: str = "dev"):
    """
    After a coding session, feed the session log/diff into MIRAS memory.
    The surprise metric filters out routine changes and keeps important ones.
    """
    text = Path(session_file).read_text()

    # Split into reasonable chunks
    chunks = []
    current = []
    for line in text.split("\n"):
        current.append(line)
        if len(current) >= 30:
            chunks.append("\n".join(current))
            current = []
    if current:
        chunks.append("\n".join(current))

    stored = 0
    for chunk in chunks:
        result = fetcher.memorize(
            content=chunk,
            content_type="code",
            source_agent=agent,
            tags=["session-capture"],
        )
        if result.get("status") == "memorized":
            stored += 1

    print(f"  ✓ Captured session: {stored}/{len(chunks)} chunks stored (rest filtered by surprise)")


# ── CLI ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="MIRAS ↔ AI Tool Connector",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Sync memory to Claude Code's CLAUDE.md
  python ai_connector.py sync --project my-project --tool claude-code

  # Sync to all tools at once
  python ai_connector.py sync --project my-project --tool all

  # Auto-sync every 30s while coding
  python ai_connector.py watch --project my-project --tool cursor

  # Start MCP server for direct integration
  python ai_connector.py mcp --project my-project

  # Capture a coding session into memory
  python ai_connector.py capture --project my-project --file session.log

Supported tools: claude-code, cursor, windsurf, cline, aider, generic, all
        """,
    )

    parser.add_argument("--api", default=DEFAULT_API, help="MIRAS API URL")
    parser.add_argument("--project", default="", help="MIRAS project ID")
    parser.add_argument("--dir", default=".", help="Project directory (default: current)")

    sub = parser.add_subparsers(dest="command")

    # Sync
    s = sub.add_parser("sync", help="Sync memory to AI tool config files")
    s.add_argument("--tool", required=True,
                   help="Tool: claude-code, cursor, windsurf, cline, aider, generic, all")

    # Watch
    w = sub.add_parser("watch", help="Auto-sync on interval")
    w.add_argument("--tool", required=True)
    w.add_argument("--interval", type=int, default=30, help="Sync interval in seconds")

    # MCP
    m = sub.add_parser("mcp", help="Start MCP server for direct tool integration")
    m.add_argument("--port", type=int, default=8200)

    # Capture
    c = sub.add_parser("capture", help="Feed a session log into memory")
    c.add_argument("--file", required=True, help="Session log or diff file")
    c.add_argument("--agent", default="dev", help="BMAD agent role")

    # List tools
    sub.add_parser("tools", help="List supported AI tools")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    if args.command == "tools":
        print("Supported AI tools:\n")
        for name, cfg in TOOL_CONFIGS.items():
            mcp_badge = " [MCP]" if cfg["supports_mcp"] else ""
            print(f"  {name:15s} -> {cfg['file']:40s} ({cfg['description']}{mcp_badge})")
        print(f"\n  {'all':15s} -> Sync to all tools at once")
        return

    if not args.project:
        print("Error: --project is required for this command")
        return

    fetcher = MemoryFetcher(api_url=args.api, project_id=args.project)

    if args.command == "sync":
        print(f"Syncing MIRAS memory -> {args.tool}")
        sync_to_tool(args.tool, fetcher, args.dir)

    elif args.command == "watch":
        watch_and_sync(args.tool, fetcher, args.dir, args.interval)

    elif args.command == "mcp":
        run_mcp_server(fetcher, port=args.port)

    elif args.command == "capture":
        capture_session(fetcher, args.file, agent=args.agent)


if __name__ == "__main__":
    main()
