"""
BMAD <-> MIRAS Bridge

This module connects BMAD agents to the MIRAS memory API.
Each BMAD agent (analyst, pm, architect, dev, qa) automatically
reads from and writes to the project's MIRAS memory.

Usage:
    bridge = BMADBridge(api_url="http://localhost:8100", project_id="my-project")
    
    # Agent reads project memory before starting work
    context = bridge.load_agent_context("architect", task="design the auth system")
    
    # Agent stores decisions
    bridge.store_decision("architect", "Using JWT tokens with refresh rotation", tags=["auth", "security"])
    
    # Agent stores code artifacts
    bridge.store_code("dev", "def authenticate(token): ...", tags=["auth", "backend"])
    
    # Ingest full BMAD artifacts (PRD, architecture doc, etc.)
    bridge.ingest_artifact("architect", artifact_path="docs/architecture.md")
"""

import requests
import json
from pathlib import Path
from typing import Optional


class BMADBridge:
    """
    Bridge between BMAD agents and MIRAS memory API.
    
    Each BMAD agent role maps to specific memory behaviors:
    - analyst:    stores facts, constraints, requirements (high-surprise threshold)
    - pm:         stores user stories, acceptance criteria, priorities
    - architect:  stores decisions, trade-offs, system design 
    - dev:        stores code patterns, implementation notes, bugs
    - qa:         stores test results, quality gates, issues found
    """

    AGENT_CONFIGS = {
        "analyst": {
            "default_content_type": "fact",
            "default_tags": ["analysis", "requirements"],
            "context_focus": "requirements constraints stakeholders scope",
        },
        "pm": {
            "default_content_type": "decision",
            "default_tags": ["product", "stories"],
            "context_focus": "user stories priorities acceptance criteria PRD",
        },
        "architect": {
            "default_content_type": "decision",
            "default_tags": ["architecture", "design"],
            "context_focus": "architecture design patterns trade-offs system components",
        },
        "dev": {
            "default_content_type": "code",
            "default_tags": ["implementation", "code"],
            "context_focus": "implementation code functions APIs bugs",
        },
        "qa": {
            "default_content_type": "fact",
            "default_tags": ["testing", "quality"],
            "context_focus": "tests bugs quality coverage edge cases",
        },
        "scrum_master": {
            "default_content_type": "context",
            "default_tags": ["process", "sprint"],
            "context_focus": "sprint progress blockers velocity stories",
        },
    }

    def __init__(self, api_url: str = "http://localhost:8100", project_id: str = ""):
        self.api_url = api_url.rstrip("/")
        self.project_id = project_id

    def _url(self, path: str) -> str:
        return f"{self.api_url}/projects/{self.project_id}{path}"

    def _post(self, path: str, data: dict) -> dict:
        resp = requests.post(self._url(path), json=data)
        resp.raise_for_status()
        return resp.json()

    def _get(self, path: str) -> dict:
        resp = requests.get(self._url(path))
        resp.raise_for_status()
        return resp.json()

    # ── Context Loading ──────────────────────────────────────────────

    def load_agent_context(self, agent_role: str, task: str = "",
                           max_tokens: int = 4000) -> str:
        """
        Load relevant project memory for a BMAD agent.
        
        This is called BEFORE the agent starts working.
        Returns a formatted context string to inject into the agent's system prompt.
        """
        config = self.AGENT_CONFIGS.get(agent_role, {})
        query = f"{config.get('context_focus', '')} {task}".strip()

        result = self._post("/context", {
            "query": query,
            "max_tokens": max_tokens,
        })

        context = result.get("context", "")
        if not context:
            return f"(No project memories yet. This is a fresh project.)"

        return (
            f"## Project Memory (MIRAS) — Role: {agent_role}\n"
            f"## Task: {task}\n\n"
            f"{context}\n\n"
            f"---\n"
            f"Total memories in project: {result.get('approx_tokens', 0):.0f} tokens"
        )

    # ── Memory Storage ───────────────────────────────────────────────

    def store_decision(self, agent_role: str, decision: str,
                       tags: Optional[list[str]] = None,
                       force: bool = False) -> dict:
        """Store an architectural/product decision"""
        config = self.AGENT_CONFIGS.get(agent_role, {})
        all_tags = list(set((tags or []) + config.get("default_tags", [])))

        return self._post("/memorize", {
            "content": decision,
            "content_type": "decision",
            "tags": all_tags,
            "source_agent": agent_role,
            "force": force,
        })

    def store_fact(self, agent_role: str, fact: str,
                   tags: Optional[list[str]] = None) -> dict:
        """Store a factual finding (requirement, constraint, discovery)"""
        config = self.AGENT_CONFIGS.get(agent_role, {})
        all_tags = list(set((tags or []) + config.get("default_tags", [])))

        return self._post("/memorize", {
            "content": fact,
            "content_type": "fact",
            "tags": all_tags,
            "source_agent": agent_role,
        })

    def store_code(self, agent_role: str, code_note: str,
                   tags: Optional[list[str]] = None) -> dict:
        """Store a code pattern, implementation note, or bug"""
        all_tags = list(set((tags or []) + ["code", "implementation"]))

        return self._post("/memorize", {
            "content": code_note,
            "content_type": "code",
            "tags": all_tags,
            "source_agent": agent_role,
        })

    def store_constraint(self, agent_role: str, constraint: str,
                         tags: Optional[list[str]] = None,
                         force: bool = True) -> dict:
        """Store a project constraint (always forced — constraints are critical)"""
        all_tags = list(set((tags or []) + ["constraint"]))

        return self._post("/memorize", {
            "content": constraint,
            "content_type": "constraint",
            "tags": all_tags,
            "source_agent": agent_role,
            "force": force,
        })

    # ── Artifact Ingestion ───────────────────────────────────────────

    def ingest_artifact(self, agent_role: str, artifact_path: str = "",
                        artifact_text: str = "",
                        chunk_size: int = 500) -> dict:
        """
        Ingest a BMAD artifact (PRD, architecture doc, etc.) into memory.
        
        Splits the document into chunks and memorizes each one.
        The surprise metric naturally filters out redundant content.
        """
        if artifact_path:
            text = Path(artifact_path).read_text()
        elif artifact_text:
            text = artifact_text
        else:
            raise ValueError("Provide either artifact_path or artifact_text")

        # Split into semantic chunks (by headers or paragraphs)
        chunks = self._chunk_text(text, chunk_size)
        config = self.AGENT_CONFIGS.get(agent_role, {})

        items = []
        for chunk in chunks:
            items.append({
                "content": chunk,
                "content_type": config.get("default_content_type", "context"),
                "tags": config.get("default_tags", []) + ["artifact"],
                "source_agent": agent_role,
                "force": False,  # let surprise metric filter
            })

        # Use bulk memorize endpoint
        resp = requests.post(
            f"{self.api_url}/projects/{self.project_id}/memorize/bulk",
            json={"items": items},
        )
        resp.raise_for_status()
        return resp.json()

    def _chunk_text(self, text: str, max_words: int = 500) -> list[str]:
        """Split text into chunks, preferring header boundaries"""
        lines = text.split("\n")
        chunks = []
        current = []
        word_count = 0

        for line in lines:
            is_header = line.strip().startswith("#")
            line_words = len(line.split())

            if is_header and current and word_count > 50:
                chunks.append("\n".join(current))
                current = []
                word_count = 0

            current.append(line)
            word_count += line_words

            if word_count >= max_words:
                chunks.append("\n".join(current))
                current = []
                word_count = 0

        if current:
            chunks.append("\n".join(current))

        return [c.strip() for c in chunks if c.strip()]

    # ── Recall ───────────────────────────────────────────────────────

    def recall(self, query: str, agent_role: str = "",
               content_type: str = "", top_k: int = 10) -> list[dict]:
        """Search project memory"""
        params = {"query": query, "top_k": top_k}
        if content_type:
            params["content_type"] = content_type
        if agent_role:
            params["source_agent"] = agent_role

        result = self._post("/recall", params)
        return result.get("memories", [])

    # ── Chat ─────────────────────────────────────────────────────────

    def agent_chat(self, agent_role: str, message: str,
                   system_prompt: str = "") -> str:
        """
        Memory-augmented chat as a specific BMAD agent.
        The API automatically injects relevant project memory.
        """
        config = self.AGENT_CONFIGS.get(agent_role, {})

        if not system_prompt:
            system_prompt = (
                f"You are the {agent_role} agent in a BMAD development team. "
                f"Your focus areas: {config.get('context_focus', 'general development')}. "
                f"Use the project memory provided to maintain continuity across sessions."
            )

        result = self._post("/chat", {
            "message": message,
            "system_prompt": system_prompt,
            "source_agent": agent_role,
            "memorize_response": True,
        })

        return result.get("response", "")

    # ── Lifecycle ────────────────────────────────────────────────────

    def consolidate(self) -> dict:
        """Run memory consolidation (between sprints, phases, etc.)"""
        return self._post("/consolidate", {})

    def stats(self) -> dict:
        """Get project memory stats"""
        return self._get("")

    # ── Phase Transitions ────────────────────────────────────────────

    def transition_phase(self, from_phase: str, to_phase: str):
        """
        Handle BMAD phase transitions (e.g., Analysis → Architecture).
        Consolidates memory and prepares context for the next phase.
        """
        # Store the transition itself
        self.store_decision(
            "scrum_master",
            f"Phase transition: {from_phase} -> {to_phase}",
            tags=["phase", from_phase, to_phase],
            force=True,
        )

        # Consolidate to clean up low-value memories from previous phase
        self.consolidate()

        return {
            "transition": f"{from_phase} -> {to_phase}",
            "stats": self.stats(),
        }


# ── CLI Helper ───────────────────────────────────────────────────────────

def main():
    """Quick CLI for testing the bridge"""
    import argparse

    parser = argparse.ArgumentParser(description="BMAD <-> MIRAS Bridge CLI")
    parser.add_argument("--api", default="http://localhost:8100", help="MIRAS API URL")
    parser.add_argument("--project", required=True, help="Project ID")
    sub = parser.add_subparsers(dest="command")

    # Context
    ctx = sub.add_parser("context", help="Load agent context")
    ctx.add_argument("--agent", required=True)
    ctx.add_argument("--task", default="")

    # Store
    store = sub.add_parser("store", help="Store a memory")
    store.add_argument("--agent", required=True)
    store.add_argument("--type", choices=["decision", "fact", "code", "constraint"], required=True)
    store.add_argument("--content", required=True)
    store.add_argument("--tags", nargs="*", default=[])

    # Recall
    rcl = sub.add_parser("recall", help="Search memory")
    rcl.add_argument("--query", required=True)
    rcl.add_argument("--agent", default="")

    # Ingest
    ing = sub.add_parser("ingest", help="Ingest an artifact")
    ing.add_argument("--agent", required=True)
    ing.add_argument("--file", required=True)

    # Stats
    sub.add_parser("stats", help="Show memory stats")

    # Consolidate
    sub.add_parser("consolidate", help="Run memory consolidation")

    args = parser.parse_args()
    bridge = BMADBridge(api_url=args.api, project_id=args.project)

    if args.command == "context":
        print(bridge.load_agent_context(args.agent, args.task))
    elif args.command == "store":
        fn = {"decision": bridge.store_decision, "fact": bridge.store_fact,
              "code": bridge.store_code, "constraint": bridge.store_constraint}
        result = fn[args.type](args.agent, args.content, tags=args.tags)
        print(json.dumps(result, indent=2))
    elif args.command == "recall":
        results = bridge.recall(args.query, agent_role=args.agent)
        for r in results:
            print(f"[{r['content_type']}|s={r['surprise_score']:.2f}] {r['content'][:100]}")
    elif args.command == "ingest":
        result = bridge.ingest_artifact(args.agent, artifact_path=args.file)
        print(json.dumps(result, indent=2))
    elif args.command == "stats":
        print(json.dumps(bridge.stats(), indent=2))
    elif args.command == "consolidate":
        print(json.dumps(bridge.consolidate(), indent=2))


if __name__ == "__main__":
    main()
