"""Tests for context builder."""

import tempfile
from pathlib import Path

from nanobot.agent.context import ContextBuilder


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
