"""Tests for context builder."""

import tempfile
from pathlib import Path

from nanobot.agent.context import ContextBuilder
from nanobot.agent.memory import MemoryStore
from nanobot.agent.memory_types import FractalNode, SuperpositionalState


def test_cognitive_directive_in_system_prompt():
    """Test that cognitive directive is included in system prompt."""
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        
        # Create a context builder
        context = ContextBuilder(workspace)
        
        # Build system prompt
        system_prompt = context.build_system_prompt()
        
        # Verify cognitive directive is present
        assert "# COGNITIVE DIRECTIVE" in system_prompt
        assert "Memory retrieved in the RESOURCES & MEMORY section is authoritative internal knowledge" in system_prompt
        assert "You must use retrieved memory as primary reasoning substrate" in system_prompt
        assert "First consult retrieved memory" in system_prompt
        assert "Prefer memory over tools" in system_prompt
        assert "Prefer memory over assumptions" in system_prompt
        assert "Use tools only if memory does not contain the answer" in system_prompt
        assert "Do not ignore relevant memory" in system_prompt


def test_cognitive_directive_placement():
    """Test that cognitive directive appears after bootstrap and before resources."""
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        
        # Create a context builder
        context = ContextBuilder(workspace)
        
        # Build system prompt
        system_prompt = context.build_system_prompt(user_query="test query")
        
        # Find positions
        cognitive_pos = system_prompt.find("# COGNITIVE DIRECTIVE")
        resources_pos = system_prompt.find("# RESOURCES & MEMORY")
        
        # Cognitive directive should be present
        assert cognitive_pos != -1, "Cognitive directive not found in system prompt"
        
        # If resources section exists, cognitive directive should come before it
        if resources_pos != -1:
            assert cognitive_pos < resources_pos, "Cognitive directive should appear before resources section"


def test_fractal_node_serialization():
    node = FractalNode(
        content="Test content",
        context_summary="Summary",
        entangled_ids={"node_abc": 0.8},
    )
    json_str = node.model_dump_json()
    loaded = FractalNode.model_validate_json(json_str)
    assert loaded.entangled_ids["node_abc"] == 0.8


def test_latent_state_parsing():
    raw_json = """
    {
        "hypotheses": [],
        "entropy": 0.5,
        "strategic_direction": "go"
    }
    """
    state = SuperpositionalState.model_validate_json(raw_json)
    assert state.entropy == 0.5
    assert state.hypotheses == []


def test_memory_normalization_logic():
    with tempfile.TemporaryDirectory() as tmpdir:
        memory = MemoryStore(Path(tmpdir))

        node_a = memory.save_fractal_node(
            content="alpha vector score candidate",
            tags=["alpha"],
            summary="alpha summary",
        )
        node_b = memory.save_fractal_node(
            content="beta",
            tags=["beta"],
            summary="beta summary",
        )
        node_c = memory.save_fractal_node(
            content="gamma",
            tags=["gamma"],
            summary="gamma summary",
        )

        node_b.entangled_ids[node_a.id] = 1.0
        node_c.entangled_ids[node_a.id] = 1.0
        memory._update_node(node_b)
        memory._update_node(node_c)

        ranked = memory.get_entangled_context("alpha", top_k=2)
        assert ranked
        assert ranked[0].id == node_a.id
