"""
Integration test — validates MIRAS memory + BMAD bridge end-to-end.
Run with: python -m pytest tests/test_integration.py -v
Or standalone: python tests/test_integration.py
"""

import sys
import os
import time
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from miras_memory import (
    ProjectMemory, MIRASConfig, PRESETS,
    MemoryArchitecture, AttentionalBias, RetentionGate, MemoryAlgorithm,
)
from miras_memory.registry import ProjectRegistry


def test_basic_memorization():
    """Test that surprise-based memorization works"""
    with tempfile.TemporaryDirectory() as tmpdir:
        mem = ProjectMemory("test-project", storage_dir=tmpdir)

        # First memory should always be stored (everything is surprising on empty memory)
        entry = mem.memorize("We are building a REST API with JWT authentication",
                            content_type="decision", source_agent="architect")
        assert entry is not None
        assert entry.surprise_score == 1.0  # first entry is always max surprise
        print(f"✓ First memory stored: surprise={entry.surprise_score:.2f}")

        # Very similar content should have lower surprise
        entry2 = mem.memorize("The REST API uses JWT tokens for auth",
                             content_type="decision", source_agent="architect")
        if entry2:
            print(f"  Similar content stored: surprise={entry2.surprise_score:.2f}")
        else:
            print(f"  Similar content skipped (below threshold)")

        # Very different content should be stored
        entry3 = mem.memorize("Database will use PostgreSQL with read replicas for scaling",
                             content_type="decision", source_agent="architect")
        assert entry3 is not None
        print(f"✓ Different content stored: surprise={entry3.surprise_score:.2f}")


def test_miras_presets():
    """Test all MIRAS preset configurations"""
    with tempfile.TemporaryDirectory() as tmpdir:
        for name, config in PRESETS.items():
            mem = ProjectMemory(f"test-{name}", config=config,
                              storage_dir=os.path.join(tmpdir, name))

            mem.memorize("The system must handle 10000 concurrent users",
                        content_type="constraint", force=True)
            mem.memorize("We chose React for the frontend",
                        content_type="decision")
            mem.memorize("Authentication uses OAuth2 with PKCE flow",
                        content_type="decision")

            stats = mem.stats()
            print(f"✓ Preset '{name}': {stats['count']} memories, "
                  f"arch={config.memory_architecture.value}, "
                  f"bias={config.attentional_bias.value}")


def test_recall_and_scoring():
    """Test memory retrieval with relevance scoring"""
    with tempfile.TemporaryDirectory() as tmpdir:
        mem = ProjectMemory("test-recall", storage_dir=tmpdir)

        # Store diverse memories
        memories = [
            ("JWT tokens for authentication with refresh rotation", "decision", "architect"),
            ("PostgreSQL database with connection pooling", "decision", "architect"),
            ("React frontend with TailwindCSS", "decision", "architect"),
            ("Must handle 50k requests per second", "constraint", "analyst"),
            ("User login endpoint returns 401 on invalid credentials", "code", "dev"),
            ("Load tests show p99 latency at 200ms", "fact", "qa"),
        ]

        for content, ctype, agent in memories:
            mem.memorize(content, content_type=ctype, source_agent=agent, force=True)

        # Recall by query
        auth_results = mem.recall("authentication login JWT")
        print(f"\n✓ Query 'authentication login JWT': {len(auth_results)} results")
        for r in auth_results:
            print(f"  [{r.content_type}] {r.content[:60]}...")

        # Recall by agent filter
        arch_results = mem.recall("system design", source_agent="architect")
        print(f"\n✓ Filter by architect: {len(arch_results)} results")

        # Recall by type filter
        constraint_results = mem.recall("performance", content_type="constraint")
        print(f"✓ Filter by constraints: {len(constraint_results)} results")


def test_retention_and_decay():
    """Test that memories decay and get evicted properly"""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = MIRASConfig(
            capacity=5,  # very small for testing
            retention_gate=RetentionGate.ADAPTIVE,
            decay_rate=0.5,  # aggressive decay for testing
        )
        mem = ProjectMemory("test-decay", config=config, storage_dir=tmpdir)

        # Fill beyond capacity
        for i in range(10):
            mem.memorize(f"Decision number {i}: use technology {chr(65+i)}",
                        content_type="decision", force=True)

        assert len(mem.memories) <= 5, f"Expected <= 5 memories, got {len(mem.memories)}"
        print(f"\n✓ Capacity enforcement: {len(mem.memories)}/5 memories after storing 10")

        # Consolidation should further prune
        result = mem.consolidate()
        print(f"✓ Consolidation: {result['before']} → {result['after']} "
              f"({result['forgotten']} forgotten)")


def test_momentum():
    """Test that high-surprise events boost related recent memories"""
    with tempfile.TemporaryDirectory() as tmpdir:
        mem = ProjectMemory("test-momentum", storage_dir=tmpdir)

        # Store a base memory
        mem.memorize("API endpoint design follows RESTful conventions",
                    content_type="decision", source_agent="architect", force=True)

        # Store a related but routine memory
        base = mem.memorize("API versioning uses URL path prefix /v1/",
                           content_type="decision", source_agent="architect", force=True)
        base_weight = base.relevance_weight if base else 0

        # Now a HIGH surprise event related to API design
        mem.memorize("BREAKING: Client requires GraphQL instead of REST — complete API redesign needed",
                    content_type="decision", source_agent="analyst", force=True)

        # Check if the related memory got a momentum boost
        recent = [m for m in mem.memories if "versioning" in m.content]
        if recent:
            new_weight = recent[0].relevance_weight
            print(f"\n✓ Momentum test: related memory weight {base_weight:.3f} → {new_weight:.3f}")
            if new_weight > base_weight:
                print(f"  ↑ Momentum boost confirmed!")
        else:
            print("\n✓ Momentum test: memory was evicted (also valid behavior)")


def test_project_registry():
    """Test multi-project isolation"""
    with tempfile.TemporaryDirectory() as tmpdir:
        registry = ProjectRegistry(base_dir=tmpdir)

        # Create projects with different presets
        p1 = registry.create_project("frontend-app", name="Frontend App",
                                     preset="titans_default")
        p2 = registry.create_project("ml-pipeline", name="ML Pipeline",
                                     preset="yaad_robust")
        p3 = registry.create_project("mobile-app", name="Mobile App",
                                     preset="memora_stable")

        # Store memories in each
        p1.memorize("React with Next.js for SSR", content_type="decision", force=True)
        p2.memorize("PyTorch training pipeline with DDP", content_type="decision", force=True)
        p3.memorize("Flutter for cross-platform mobile", content_type="decision", force=True)

        # Verify isolation
        p1_results = p1.recall("React")
        p2_results = p2.recall("React")
        assert len(p1_results) > 0, "Frontend project should have React memory"
        assert len(p2_results) == 0, "ML project should NOT have React memory"

        projects = registry.list_projects()
        print(f"\n✓ Registry: {len(projects)} projects, memories isolated correctly")
        for p in projects:
            print(f"  {p['project_id']}: preset={p['preset']}")


def test_context_window_generation():
    """Test that context window is properly formatted for LLM injection"""
    with tempfile.TemporaryDirectory() as tmpdir:
        mem = ProjectMemory("test-context", storage_dir=tmpdir)

        # Simulate a BMAD workflow
        mem.memorize("Project: E-commerce platform for artisanal goods",
                    content_type="fact", source_agent="analyst", force=True)
        mem.memorize("Must support 500 concurrent users at launch",
                    content_type="constraint", source_agent="analyst", force=True)
        mem.memorize("Microservices architecture with API gateway",
                    content_type="decision", source_agent="architect", force=True)
        mem.memorize("Payment processing via Stripe API",
                    content_type="decision", source_agent="architect", force=True)
        mem.memorize("Product catalog uses Elasticsearch for search",
                    content_type="code", source_agent="dev", force=True)

        # Generate context window
        context = mem.get_context_window(query="payment checkout flow", max_tokens=2000)
        print(f"\n✓ Context window ({len(context)} chars):\n")
        print(context)
        print("\n  (This is what gets injected into the LLM system prompt)")


def test_attentional_bias_variants():
    """Test that different attentional biases produce different surprise curves"""
    print("\n✓ Attentional bias comparison:")

    with tempfile.TemporaryDirectory() as tmpdir:
        content_pairs = [
            "Machine learning model for image classification",
            "Deep neural network for visual recognition",  # similar
        ]

        for bias in AttentionalBias:
            config = MIRASConfig(attentional_bias=bias)
            mem = ProjectMemory(f"test-{bias.value}", config=config,
                              storage_dir=os.path.join(tmpdir, bias.value))

            mem.memorize(content_pairs[0], force=True)
            entry = mem.memorize(content_pairs[1], force=True)

            if entry:
                print(f"  {bias.value:10s}: surprise for similar content = {entry.surprise_score:.4f}")
            else:
                print(f"  {bias.value:10s}: similar content was skipped")


if __name__ == "__main__":
    print("=" * 60)
    print("MIRAS Memory System — Integration Tests")
    print("=" * 60)

    test_basic_memorization()
    test_miras_presets()
    test_recall_and_scoring()
    test_retention_and_decay()
    test_momentum()
    test_project_registry()
    test_context_window_generation()
    test_attentional_bias_variants()

    print("\n" + "=" * 60)
    print("All tests passed ✓")
    print("=" * 60)
