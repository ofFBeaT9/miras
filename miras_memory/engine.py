"""
MIRAS Memory Engine — Per-Project Long-Term Memory

Implements the four MIRAS design choices:
1. Memory Architecture  — how information is stored (vector, matrix, MLP)
2. Attentional Bias     — what the memory prioritizes (loss function)
3. Retention Gate       — how it forgets (regularization)
4. Memory Algorithm     — how it updates (optimization)

Each project gets its own isolated memory instance.
"""

import time
import json
import hashlib
import math
from dataclasses import dataclass, field, asdict
from typing import Optional
from enum import Enum
from pathlib import Path


# ── MIRAS Design Choice 1: Memory Architecture ──────────────────────────

class MemoryArchitecture(str, Enum):
    """Structure that stores information"""
    VECTOR = "vector"       # simple key-value store (like linear RNNs)
    MATRIX = "matrix"       # associative matrix (like DeltaNet)
    MLP = "mlp"             # deep neural memory (Titans-style, most expressive)


# ── MIRAS Design Choice 2: Attentional Bias ─────────────────────────────

class AttentionalBias(str, Enum):
    """Internal loss function — what the memory cares about"""
    L2 = "l2"               # standard MSE (default, like most transformers)
    HUBER = "huber"         # YAAD variant — robust to outliers
    LP_NORM = "lp_norm"     # MONETA variant — stricter regularization
    KL_DIV = "kl_div"       # MEMORA variant — probability-map stability


# ── MIRAS Design Choice 3: Retention Gate ────────────────────────────────

class RetentionGate(str, Enum):
    """How the memory forgets — regularization strategy"""
    EXPONENTIAL = "exponential"   # standard exponential decay
    ADAPTIVE = "adaptive"         # Titans-style: decay based on surprise
    SLIDING = "sliding"           # fixed window, oldest drops off
    NONE = "none"                 # never forget (bounded by capacity)


# ── MIRAS Design Choice 4: Memory Algorithm ─────────────────────────────

class MemoryAlgorithm(str, Enum):
    """How the memory updates itself"""
    SGD = "sgd"                   # basic gradient descent
    SGD_MOMENTUM = "sgd_momentum" # with momentum (Titans default)
    ADAM = "adam"                  # adaptive learning rate


# ── Memory Entry ─────────────────────────────────────────────────────────

@dataclass
class MemoryEntry:
    """A single memory unit with MIRAS metadata"""
    id: str
    content: str
    content_type: str           # "decision", "fact", "constraint", "code", "context"
    surprise_score: float       # how unexpected this was (0.0 - 1.0)
    relevance_weight: float     # current weight after decay
    momentum: float             # accumulated gradient momentum
    created_at: float
    last_accessed: float
    access_count: int = 0
    tags: list[str] = field(default_factory=list)
    source_agent: str = ""      # which BMAD agent created this
    parent_id: Optional[str] = None  # for chained memories (momentum tracking)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(**d)


# ── MIRAS Configuration Per Project ──────────────────────────────────────

@dataclass
class MIRASConfig:
    """Full MIRAS configuration for a project's memory"""
    memory_architecture: MemoryArchitecture = MemoryArchitecture.MLP
    attentional_bias: AttentionalBias = AttentionalBias.L2
    retention_gate: RetentionGate = RetentionGate.ADAPTIVE
    memory_algorithm: MemoryAlgorithm = MemoryAlgorithm.SGD_MOMENTUM

    # Hyperparameters
    capacity: int = 500                 # max memories per project
    surprise_threshold: float = 0.3     # minimum surprise to store
    decay_rate: float = 0.98            # base decay per step
    momentum_factor: float = 0.9        # momentum coefficient
    learning_rate: float = 0.01         # memory update rate
    huber_delta: float = 1.0            # delta for Huber loss (YAAD)
    lp_norm_p: float = 1.5             # p for Lp-norm (MONETA)

    def to_dict(self):
        d = asdict(self)
        d["memory_architecture"] = self.memory_architecture.value
        d["attentional_bias"] = self.attentional_bias.value
        d["retention_gate"] = self.retention_gate.value
        d["memory_algorithm"] = self.memory_algorithm.value
        return d

    @classmethod
    def from_dict(cls, d):
        d["memory_architecture"] = MemoryArchitecture(d["memory_architecture"])
        d["attentional_bias"] = AttentionalBias(d["attentional_bias"])
        d["retention_gate"] = RetentionGate(d["retention_gate"])
        d["memory_algorithm"] = MemoryAlgorithm(d["memory_algorithm"])
        return cls(**d)


# ── MIRAS Presets (Named Configurations) ─────────────────────────────────

PRESETS = {
    "titans_default": MIRASConfig(
        memory_architecture=MemoryArchitecture.MLP,
        attentional_bias=AttentionalBias.L2,
        retention_gate=RetentionGate.ADAPTIVE,
        memory_algorithm=MemoryAlgorithm.SGD_MOMENTUM,
    ),
    "yaad_robust": MIRASConfig(
        memory_architecture=MemoryArchitecture.MLP,
        attentional_bias=AttentionalBias.HUBER,
        retention_gate=RetentionGate.ADAPTIVE,
        memory_algorithm=MemoryAlgorithm.SGD_MOMENTUM,
        huber_delta=1.0,
    ),
    "moneta_strict": MIRASConfig(
        memory_architecture=MemoryArchitecture.MATRIX,
        attentional_bias=AttentionalBias.LP_NORM,
        retention_gate=RetentionGate.EXPONENTIAL,
        memory_algorithm=MemoryAlgorithm.ADAM,
        lp_norm_p=1.5,
    ),
    "memora_stable": MIRASConfig(
        memory_architecture=MemoryArchitecture.MLP,
        attentional_bias=AttentionalBias.KL_DIV,
        retention_gate=RetentionGate.ADAPTIVE,
        memory_algorithm=MemoryAlgorithm.SGD_MOMENTUM,
    ),
    "lightweight": MIRASConfig(
        memory_architecture=MemoryArchitecture.VECTOR,
        attentional_bias=AttentionalBias.L2,
        retention_gate=RetentionGate.SLIDING,
        memory_algorithm=MemoryAlgorithm.SGD,
        capacity=100,
    ),
}


# ── Surprise Calculator ─────────────────────────────────────────────────

class SurpriseCalculator:
    """
    Computes surprise score based on the attentional bias.
    In full Titans: surprise = gradient magnitude of associative loss.
    Here: we approximate surprise using content divergence from memory state.
    """

    def __init__(self, config: MIRASConfig):
        self.config = config

    def compute(self, new_content: str, existing_memories: list[MemoryEntry]) -> float:
        """
        Surprise = how much the new content diverges from what memory expects.
        Higher surprise → more important to store.
        """
        if not existing_memories:
            return 1.0  # everything is surprising when memory is empty

        # Compute content overlap with existing memories
        new_tokens = set(new_content.lower().split())
        if not new_tokens:
            return 0.0

        # Weighted overlap against all memories
        total_overlap = 0.0
        total_weight = 0.0

        for mem in existing_memories:
            mem_tokens = set(mem.content.lower().split())
            if not mem_tokens:
                continue
            overlap = len(new_tokens & mem_tokens) / max(len(new_tokens | mem_tokens), 1)

            weight = mem.relevance_weight * mem.surprise_score
            total_overlap += overlap * weight
            total_weight += weight

        if total_weight == 0:
            return 1.0

        familiarity = total_overlap / total_weight

        # Apply attentional bias to shape the surprise curve
        raw_surprise = 1.0 - familiarity

        if self.config.attentional_bias == AttentionalBias.L2:
            # Standard MSE-like: quadratic surprise
            return raw_surprise ** 2

        elif self.config.attentional_bias == AttentionalBias.HUBER:
            # YAAD: gentler on small surprises, linear on large
            delta = self.config.huber_delta
            if raw_surprise <= delta:
                return 0.5 * raw_surprise ** 2
            else:
                return delta * (raw_surprise - 0.5 * delta)

        elif self.config.attentional_bias == AttentionalBias.LP_NORM:
            # MONETA: Lp-norm, stricter
            p = self.config.lp_norm_p
            return raw_surprise ** p

        elif self.config.attentional_bias == AttentionalBias.KL_DIV:
            # MEMORA: KL-divergence inspired, log-scale surprise
            epsilon = 1e-8
            return -math.log(max(familiarity, epsilon)) / 10.0  # normalized

        return raw_surprise


# ── Retention Manager ────────────────────────────────────────────────────

class RetentionManager:
    """
    Manages forgetting based on the retention gate strategy.
    MIRAS insight: forgetting = regularization, not data loss.
    """

    def __init__(self, config: MIRASConfig):
        self.config = config

    def apply_decay(self, memories: list[MemoryEntry], current_surprise: float = 0.0):
        """Apply retention gate to all memories"""
        for mem in memories:
            if self.config.retention_gate == RetentionGate.EXPONENTIAL:
                mem.relevance_weight *= self.config.decay_rate

            elif self.config.retention_gate == RetentionGate.ADAPTIVE:
                # Titans-style: decay is inversely proportional to surprise
                # High-surprise memories decay slower
                adaptive_rate = self.config.decay_rate + (1 - self.config.decay_rate) * mem.surprise_score * 0.5
                mem.relevance_weight *= adaptive_rate

            elif self.config.retention_gate == RetentionGate.SLIDING:
                # Age-based: older memories decay faster
                age = time.time() - mem.created_at
                age_factor = max(0.0, 1.0 - (age / (86400 * 30)))  # 30-day window
                mem.relevance_weight = mem.surprise_score * age_factor

            elif self.config.retention_gate == RetentionGate.NONE:
                pass  # no decay

    def evict(self, memories: list[MemoryEntry], capacity: int) -> list[MemoryEntry]:
        """Remove lowest-value memories when over capacity"""
        if len(memories) <= capacity:
            return memories

        # Score = relevance_weight * surprise * recency_bonus
        now = time.time()
        def score(m):
            recency = 1.0 / (1.0 + (now - m.last_accessed) / 3600)
            return m.relevance_weight * m.surprise_score * (1 + recency) * (1 + math.log1p(m.access_count))

        memories.sort(key=score, reverse=True)
        return memories[:capacity]


# ── Memory Update Engine ─────────────────────────────────────────────────

class MemoryUpdater:
    """
    Implements the memory algorithm — how memory weights are updated.
    Maps to MIRAS's "optimization algorithm" design choice.
    """

    def __init__(self, config: MIRASConfig):
        self.config = config
        self._velocity: dict[str, float] = {}  # for momentum
        self._m: dict[str, float] = {}          # Adam first moment
        self._v: dict[str, float] = {}          # Adam second moment
        self._step = 0

    def update(self, memory: MemoryEntry, surprise: float):
        """Update a memory's weight based on new interaction"""
        self._step += 1
        gradient = surprise - memory.relevance_weight  # "prediction error"

        if self.config.memory_algorithm == MemoryAlgorithm.SGD:
            memory.relevance_weight += self.config.learning_rate * gradient

        elif self.config.memory_algorithm == MemoryAlgorithm.SGD_MOMENTUM:
            vel = self._velocity.get(memory.id, 0.0)
            vel = self.config.momentum_factor * vel + gradient
            self._velocity[memory.id] = vel
            memory.relevance_weight += self.config.learning_rate * vel
            memory.momentum = vel

        elif self.config.memory_algorithm == MemoryAlgorithm.ADAM:
            m = self._m.get(memory.id, 0.0)
            v = self._v.get(memory.id, 0.0)
            m = 0.9 * m + 0.1 * gradient
            v = 0.999 * v + 0.001 * gradient ** 2
            m_hat = m / (1 - 0.9 ** self._step)
            v_hat = v / (1 - 0.999 ** self._step)
            memory.relevance_weight += self.config.learning_rate * m_hat / (math.sqrt(v_hat) + 1e-8)
            self._m[memory.id] = m
            self._v[memory.id] = v

        # Clamp to valid range
        memory.relevance_weight = max(0.0, min(1.0, memory.relevance_weight))
        memory.last_accessed = time.time()
        memory.access_count += 1


# ── Project Memory (Main Interface) ─────────────────────────────────────

class ProjectMemory:
    """
    Complete MIRAS memory system for a single project.
    This is the main class the API interacts with.
    """

    def __init__(self, project_id: str, config: Optional[MIRASConfig] = None,
                 storage_dir: Optional[str] = None):
        self.project_id = project_id
        self.config = config or PRESETS["titans_default"]
        self.memories: list[MemoryEntry] = []
        self.storage_dir = Path(storage_dir) if storage_dir else Path(f"./memory_store/{project_id}")

        # Initialize MIRAS components
        self.surprise_calc = SurpriseCalculator(self.config)
        self.retention = RetentionManager(self.config)
        self.updater = MemoryUpdater(self.config)

        # Load existing memories if available
        self._load()

    def _generate_id(self, content: str) -> str:
        return hashlib.sha256(f"{content}{time.time()}".encode()).hexdigest()[:16]

    # ── Core Operations ──────────────────────────────────────────────

    def memorize(self, content: str, content_type: str = "context",
                 tags: Optional[list[str]] = None, source_agent: str = "",
                 force: bool = False) -> Optional[MemoryEntry]:
        """
        Titans-style memorization: only store if surprise exceeds threshold.

        Args:
            content: what to remember
            content_type: "decision", "fact", "constraint", "code", "context"
            tags: searchable tags
            source_agent: which BMAD agent created this (analyst, architect, etc.)
            force: bypass surprise threshold (for critical items)

        Returns:
            MemoryEntry if stored, None if surprise too low
        """
        surprise = self.surprise_calc.compute(content, self.memories)

        if not force and surprise < self.config.surprise_threshold:
            return None  # not surprising enough to remember

        now = time.time()
        entry = MemoryEntry(
            id=self._generate_id(content),
            content=content,
            content_type=content_type,
            surprise_score=surprise,
            relevance_weight=surprise,  # initialize weight = surprise
            momentum=0.0,
            created_at=now,
            last_accessed=now,
            tags=tags or [],
            source_agent=source_agent,
        )

        self.memories.append(entry)

        # Apply retention gate (decay existing memories)
        self.retention.apply_decay(self.memories, current_surprise=surprise)

        # Momentum: if this is high-surprise, boost recent related memories
        if surprise > 0.7:
            self._apply_momentum(entry)

        # Evict if over capacity
        self.memories = self.retention.evict(self.memories, self.config.capacity)

        self._save()
        return entry

    def recall(self, query: str, top_k: int = 10,
               content_type: Optional[str] = None,
               source_agent: Optional[str] = None,
               tags: Optional[list[str]] = None) -> list[MemoryEntry]:
        """
        Retrieve relevant memories. Updates access patterns.

        Args:
            query: what to search for
            top_k: max results
            content_type: filter by type
            source_agent: filter by agent
            tags: filter by tags
        """
        candidates = self.memories

        # Apply filters
        if content_type:
            candidates = [m for m in candidates if m.content_type == content_type]
        if source_agent:
            candidates = [m for m in candidates if m.source_agent == source_agent]
        if tags:
            tag_set = set(tags)
            candidates = [m for m in candidates if tag_set & set(m.tags)]

        # Score by relevance to query
        query_tokens = set(query.lower().split())
        scored = []
        for mem in candidates:
            # Search content
            mem_tokens = set(mem.content.lower().split())
            # Also search tags
            tag_tokens = set(t.lower() for t in mem.tags)
            all_mem_tokens = mem_tokens | tag_tokens

            if not all_mem_tokens or not query_tokens:
                relevance = 0.0
            else:
                # Exact token overlap
                exact_overlap = len(query_tokens & all_mem_tokens)
                # Substring matching (e.g. "database" matches "PostgreSQL" context via tags,
                # "auth" matches "authentication")
                substring_hits = 0
                for qt in query_tokens:
                    for mt in all_mem_tokens:
                        if qt in mt or mt in qt:
                            substring_hits += 0.5
                            break

                total_hits = exact_overlap + substring_hits
                relevance = total_hits / max(len(query_tokens), 1)

            # Combined score: relevance * weight * recency
            recency = 1.0 / (1.0 + (time.time() - mem.last_accessed) / 3600)
            score = relevance * mem.relevance_weight * (1 + recency)
            scored.append((score, mem))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = [mem for _, mem in scored[:top_k] if _ > 0]

        # Update access patterns (memory algorithm step)
        for mem in results:
            self.updater.update(mem, mem.surprise_score)

        self._save()
        return results

    def forget(self, memory_id: str) -> bool:
        """Explicitly forget a memory"""
        before = len(self.memories)
        self.memories = [m for m in self.memories if m.id != memory_id]
        if len(self.memories) < before:
            self._save()
            return True
        return False

    def consolidate(self) -> dict:
        """
        Periodic consolidation — like sleep for the memory system.
        Applies full decay cycle and removes dead memories.
        """
        before_count = len(self.memories)

        # Full decay pass
        self.retention.apply_decay(self.memories)

        # Remove memories with near-zero weight
        self.memories = [m for m in self.memories if m.relevance_weight > 0.01]

        # Evict to capacity
        self.memories = self.retention.evict(self.memories, self.config.capacity)

        after_count = len(self.memories)
        self._save()

        return {
            "before": before_count,
            "after": after_count,
            "forgotten": before_count - after_count,
        }

    def get_context_window(self, query: str = "", max_tokens: int = 4000) -> str:
        """
        Build a context string from memories for injection into LLM prompts.
        This is what BMAD agents will receive as their "long-term memory".
        """
        if query:
            relevant = self.recall(query, top_k=20)
        else:
            relevant = []

        # If query didn't match anything, fall back to top-weight memories
        if not relevant:
            relevant = sorted(self.memories,
                            key=lambda m: m.relevance_weight * m.surprise_score,
                            reverse=True)[:20]

        lines = []
        approx_tokens = 0
        for mem in relevant:
            line = f"[{mem.content_type}|s={mem.surprise_score:.2f}|w={mem.relevance_weight:.2f}] {mem.content}"
            line_tokens = len(line.split()) * 1.3  # rough estimate
            if approx_tokens + line_tokens > max_tokens:
                break
            lines.append(line)
            approx_tokens += line_tokens

        return "\n".join(lines)

    def stats(self) -> dict:
        """Memory health and stats"""
        if not self.memories:
            return {"count": 0, "config": self.config.to_dict()}

        weights = [m.relevance_weight for m in self.memories]
        surprises = [m.surprise_score for m in self.memories]
        types = {}
        agents = {}
        for m in self.memories:
            types[m.content_type] = types.get(m.content_type, 0) + 1
            if m.source_agent:
                agents[m.source_agent] = agents.get(m.source_agent, 0) + 1

        return {
            "project_id": self.project_id,
            "count": len(self.memories),
            "capacity": self.config.capacity,
            "utilization": len(self.memories) / self.config.capacity,
            "avg_weight": sum(weights) / len(weights),
            "avg_surprise": sum(surprises) / len(surprises),
            "by_type": types,
            "by_agent": agents,
            "config": self.config.to_dict(),
        }

    # ── Internal ─────────────────────────────────────────────────────

    def _apply_momentum(self, trigger: MemoryEntry):
        """
        Titans momentum: when a high-surprise event occurs,
        boost the relevance of recent related memories.
        """
        recent = sorted(self.memories, key=lambda m: m.created_at, reverse=True)[:5]
        trigger_tokens = set(trigger.content.lower().split())

        for mem in recent:
            if mem.id == trigger.id:
                continue
            mem_tokens = set(mem.content.lower().split())
            overlap = len(trigger_tokens & mem_tokens) / max(len(trigger_tokens | mem_tokens), 1)
            if overlap > 0.1:
                boost = self.config.momentum_factor * trigger.surprise_score * overlap
                mem.relevance_weight = min(1.0, mem.relevance_weight + boost)
                mem.parent_id = trigger.id

    def _save(self):
        """Persist memories to disk"""
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        data = {
            "project_id": self.project_id,
            "config": self.config.to_dict(),
            "memories": [m.to_dict() for m in self.memories],
        }
        with open(self.storage_dir / "memory.json", "w") as f:
            json.dump(data, f, indent=2)

    def _load(self):
        """Load memories from disk"""
        path = self.storage_dir / "memory.json"
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            self.memories = [MemoryEntry.from_dict(m) for m in data.get("memories", [])]
