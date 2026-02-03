"""
MIRAS Memory API — Per-Project Long-Term Memory Service

This API provides MIRAS-based memory for each project.
BMAD agents connect through this API to get persistent, 
surprise-weighted, long-term memory across sessions.

Endpoints:
  /projects           — CRUD for projects
  /projects/{id}/mem  — memorize, recall, forget, consolidate
  /projects/{id}/ctx  — get context window for LLM injection
  /projects/{id}/chat — memory-augmented LLM chat (uses Anthropic API)
"""

import os
import time
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from miras_memory import ProjectRegistry, PRESETS


# ── App Setup ────────────────────────────────────────────────────────────

registry: Optional[ProjectRegistry] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global registry
    storage = os.environ.get("MIRAS_STORAGE_DIR", "./memory_store")
    registry = ProjectRegistry(base_dir=storage)
    yield

app = FastAPI(
    title="MIRAS Memory API",
    description="Per-project long-term memory system based on Google's MIRAS framework. "
                "Each project gets isolated memory with configurable architecture, "
                "attentional bias, retention gate, and memory algorithm.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request/Response Models ──────────────────────────────────────────────

class CreateProjectRequest(BaseModel):
    project_id: str = Field(..., description="Unique project identifier")
    name: str = Field("", description="Human-readable project name")
    description: str = Field("", description="What this project is about")
    preset: str = Field("titans_default",
        description="MIRAS preset: titans_default, yaad_robust, moneta_strict, memora_stable, lightweight")
    config_overrides: Optional[dict] = Field(None,
        description="Override specific MIRAS settings")

class MemorizeRequest(BaseModel):
    content: str = Field(..., description="What to remember")
    content_type: str = Field("context",
        description="Type: decision, fact, constraint, code, context")
    tags: list[str] = Field(default_factory=list, description="Searchable tags")
    source_agent: str = Field("", description="Which BMAD agent: analyst, pm, architect, dev, qa")
    force: bool = Field(False, description="Bypass surprise threshold")

class RecallRequest(BaseModel):
    query: str = Field(..., description="What to search for")
    top_k: int = Field(10, description="Max results")
    content_type: Optional[str] = Field(None, description="Filter by type")
    source_agent: Optional[str] = Field(None, description="Filter by BMAD agent")
    tags: Optional[list[str]] = Field(None, description="Filter by tags")

class ContextRequest(BaseModel):
    query: str = Field("", description="Focus the context on this topic")
    max_tokens: int = Field(4000, description="Approximate token budget")

class ChatRequest(BaseModel):
    message: str = Field(..., description="User message")
    system_prompt: str = Field("You are a helpful assistant with long-term memory.",
        description="System prompt for the LLM")
    model: str = Field("claude-sonnet-4-5-20250929", description="Model to use")
    source_agent: str = Field("", description="Which BMAD agent is chatting")
    memorize_response: bool = Field(True, description="Store the exchange in memory")

class ConfigUpdateRequest(BaseModel):
    updates: dict = Field(..., description="MIRAS config fields to update")


# ── Project Endpoints ────────────────────────────────────────────────────

@app.post("/projects", tags=["Projects"])
async def create_project(req: CreateProjectRequest):
    """Create a new project with its own MIRAS memory system"""
    try:
        mem = registry.create_project(
            project_id=req.project_id,
            name=req.name,
            description=req.description,
            preset=req.preset,
            config=req.config_overrides,
        )
        return {
            "status": "created",
            "project_id": req.project_id,
            "preset": req.preset,
            "config": mem.config.to_dict(),
        }
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.get("/projects", tags=["Projects"])
async def list_projects():
    """List all projects"""
    return {"projects": registry.list_projects()}


@app.get("/projects/{project_id}", tags=["Projects"])
async def get_project(project_id: str):
    """Get project details and memory stats"""
    try:
        mem = registry.get_project(project_id)
        return mem.stats()
    except KeyError:
        raise HTTPException(404, f"Project '{project_id}' not found")


@app.delete("/projects/{project_id}", tags=["Projects"])
async def delete_project(project_id: str):
    """Delete a project and all its memories"""
    if registry.delete_project(project_id):
        return {"status": "deleted", "project_id": project_id}
    raise HTTPException(404, f"Project '{project_id}' not found")


@app.put("/projects/{project_id}/config", tags=["Projects"])
async def update_project_config(project_id: str, req: ConfigUpdateRequest):
    """Update a project's MIRAS configuration"""
    try:
        new_config = registry.update_config(project_id, req.updates)
        return {"status": "updated", "config": new_config.to_dict()}
    except KeyError:
        raise HTTPException(404, f"Project '{project_id}' not found")


@app.get("/presets", tags=["Configuration"])
async def list_presets():
    """List available MIRAS presets with their configurations"""
    return {
        name: config.to_dict()
        for name, config in PRESETS.items()
    }


# ── Memory Endpoints ─────────────────────────────────────────────────────

@app.post("/projects/{project_id}/memorize", tags=["Memory"])
async def memorize(project_id: str, req: MemorizeRequest):
    """
    Store information in project memory (Titans-style: only if surprising enough).
    
    The surprise metric determines if the content is worth remembering.
    Use force=true to bypass the surprise threshold for critical items.
    """
    try:
        mem = registry.get_project(project_id)
    except KeyError:
        raise HTTPException(404, f"Project '{project_id}' not found")

    entry = mem.memorize(
        content=req.content,
        content_type=req.content_type,
        tags=req.tags,
        source_agent=req.source_agent,
        force=req.force,
    )

    if entry:
        return {
            "status": "memorized",
            "memory_id": entry.id,
            "surprise_score": entry.surprise_score,
            "content_type": entry.content_type,
        }
    else:
        return {
            "status": "skipped",
            "reason": "surprise below threshold",
            "threshold": mem.config.surprise_threshold,
        }


@app.post("/projects/{project_id}/recall", tags=["Memory"])
async def recall(project_id: str, req: RecallRequest):
    """
    Retrieve relevant memories from project memory.
    
    Results are scored by: relevance to query × memory weight × recency.
    Accessing a memory also updates its weight (memory algorithm step).
    """
    try:
        mem = registry.get_project(project_id)
    except KeyError:
        raise HTTPException(404, f"Project '{project_id}' not found")

    results = mem.recall(
        query=req.query,
        top_k=req.top_k,
        content_type=req.content_type,
        source_agent=req.source_agent,
        tags=req.tags,
    )

    return {
        "count": len(results),
        "memories": [r.to_dict() for r in results],
    }


@app.delete("/projects/{project_id}/memory/{memory_id}", tags=["Memory"])
async def forget(project_id: str, memory_id: str):
    """Explicitly forget a specific memory"""
    try:
        mem = registry.get_project(project_id)
    except KeyError:
        raise HTTPException(404, f"Project '{project_id}' not found")

    if mem.forget(memory_id):
        return {"status": "forgotten", "memory_id": memory_id}
    raise HTTPException(404, f"Memory '{memory_id}' not found")


@app.post("/projects/{project_id}/consolidate", tags=["Memory"])
async def consolidate(project_id: str):
    """
    Run memory consolidation (like sleep).
    Applies full decay cycle and removes dead memories.
    Call this periodically or between development phases.
    """
    try:
        mem = registry.get_project(project_id)
    except KeyError:
        raise HTTPException(404, f"Project '{project_id}' not found")

    result = mem.consolidate()
    return {"status": "consolidated", **result}


# ── Context Window Endpoint ──────────────────────────────────────────────

@app.post("/projects/{project_id}/context", tags=["Context"])
async def get_context(project_id: str, req: ContextRequest):
    """
    Get a formatted context window from project memory.
    
    This is what you inject into LLM system prompts to give
    BMAD agents access to long-term project memory.
    """
    try:
        mem = registry.get_project(project_id)
    except KeyError:
        raise HTTPException(404, f"Project '{project_id}' not found")

    context = mem.get_context_window(query=req.query, max_tokens=req.max_tokens)

    return {
        "project_id": project_id,
        "context": context,
        "approx_tokens": len(context.split()) * 1.3,
    }


# ── Memory-Augmented Chat ───────────────────────────────────────────────

@app.post("/projects/{project_id}/chat", tags=["Chat"])
async def chat(project_id: str, req: ChatRequest):
    """
    Memory-augmented chat: recalls relevant memories, injects them
    as context, sends to LLM, then memorizes the exchange.
    
    This is the main endpoint BMAD agents use for intelligent,
    memory-aware conversations about a project.
    """
    try:
        mem = registry.get_project(project_id)
    except KeyError:
        raise HTTPException(404, f"Project '{project_id}' not found")

    # 1. Recall relevant memories
    context = mem.get_context_window(query=req.message, max_tokens=3000)

    # 2. Build augmented system prompt
    augmented_system = f"""{req.system_prompt}

## Project Long-Term Memory (MIRAS)
The following is your persistent memory for this project.
Memories are tagged with [type|surprise_score|weight].
Higher surprise = more novel information. Higher weight = more relevant.

{context if context else "(No memories yet for this project.)"}

## Memory Instructions
- Reference relevant memories naturally in your response
- Flag if you notice contradictions between memories
- Important new information will be automatically memorized after this response
"""

    # 3. Call LLM
    try:
        import anthropic
        client = anthropic.Anthropic()
        response = client.messages.create(
            model=req.model,
            max_tokens=4096,
            system=augmented_system,
            messages=[{"role": "user", "content": req.message}],
        )
        assistant_reply = response.content[0].text
    except ImportError:
        # Fallback if anthropic not installed
        assistant_reply = (
            f"[MIRAS Memory Active — {len(mem.memories)} memories loaded]\n\n"
            f"Anthropic SDK not installed. Install with: pip install anthropic\n"
            f"Set ANTHROPIC_API_KEY env var.\n\n"
            f"Context that would be injected:\n{context}"
        )
    except Exception as e:
        raise HTTPException(500, f"LLM call failed: {str(e)}")

    # 4. Memorize the exchange
    if req.memorize_response:
        mem.memorize(
            content=f"User ({req.source_agent}): {req.message}",
            content_type="context",
            tags=["chat", req.source_agent] if req.source_agent else ["chat"],
            source_agent=req.source_agent,
        )
        mem.memorize(
            content=f"Assistant response: {assistant_reply[:500]}",
            content_type="context",
            tags=["chat", "response"],
            source_agent="system",
        )

    return {
        "response": assistant_reply,
        "memories_used": len(context.split("\n")) if context else 0,
        "total_memories": len(mem.memories),
    }


# ── Bulk Operations ──────────────────────────────────────────────────────

class BulkMemorizeRequest(BaseModel):
    items: list[MemorizeRequest]

@app.post("/projects/{project_id}/memorize/bulk", tags=["Memory"])
async def bulk_memorize(project_id: str, req: BulkMemorizeRequest):
    """Memorize multiple items at once (for BMAD artifact ingestion)"""
    try:
        mem = registry.get_project(project_id)
    except KeyError:
        raise HTTPException(404, f"Project '{project_id}' not found")

    results = []
    for item in req.items:
        entry = mem.memorize(
            content=item.content,
            content_type=item.content_type,
            tags=item.tags,
            source_agent=item.source_agent,
            force=item.force,
        )
        results.append({
            "content": item.content[:80] + "..." if len(item.content) > 80 else item.content,
            "stored": entry is not None,
            "surprise": entry.surprise_score if entry else None,
        })

    stored = sum(1 for r in results if r["stored"])
    return {
        "total": len(results),
        "stored": stored,
        "skipped": len(results) - stored,
        "details": results,
    }


# ── Health ───────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
async def health():
    return {
        "status": "ok",
        "projects": len(registry._registry["projects"]),
        "presets": list(PRESETS.keys()),
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8100))
    uvicorn.run(app, host="0.0.0.0", port=port)
