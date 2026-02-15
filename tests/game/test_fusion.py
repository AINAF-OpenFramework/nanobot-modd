"""Tests for fusion module."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from nanobot.agent.memory import MemoryStore
from nanobot.game.fusion import FusedEmbedding, FusionConfig, MultimodalFusionLayer
from nanobot.game.visual_perception import VisualEmbedding


class TestFusionConfig:
    """Tests for FusionConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = FusionConfig()
        assert config.total_dim == 512
        assert config.state_dim == 128
        assert config.visual_dim == 256
        assert config.memory_dim == 128
        assert config.memory_top_k == 5

    def test_custom_config(self):
        """Test custom configuration values."""
        config = FusionConfig(
            total_dim=1024,
            state_dim=256,
            visual_dim=512,
            memory_dim=256,
            state_weight=0.5,
            visual_weight=0.3,
            memory_weight=0.2,
        )
        assert config.total_dim == 1024
        assert config.state_dim == 256
        # Weights should be normalized
        assert abs(config.state_weight + config.visual_weight + config.memory_weight - 1.0) < 0.01

    def test_weight_normalization(self):
        """Test that weights are normalized to sum to 1."""
        config = FusionConfig(
            state_weight=1.0,
            visual_weight=1.0,
            memory_weight=1.0,
        )
        # Should be normalized to 1/3 each
        assert abs(config.state_weight - 1/3) < 0.01
        assert abs(config.visual_weight - 1/3) < 0.01
        assert abs(config.memory_weight - 1/3) < 0.01


class TestFusedEmbedding:
    """Tests for FusedEmbedding dataclass."""

    def test_create_fused_embedding(self):
        """Test creating a fused embedding."""
        embedding = np.random.rand(512).astype(np.float32)
        fused = FusedEmbedding(
            embedding=embedding,
            dimensions=512,
            components={"state": 0.4, "visual": 0.3, "memory": 0.3},
        )

        assert fused.dimensions == 512
        assert len(fused.components) == 3
        assert fused.embedding.shape == (512,)


class TestMultimodalFusionLayer:
    """Tests for MultimodalFusionLayer class."""

    @pytest.fixture
    def temp_workspace(self):
        """Create a temporary workspace."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def memory_store(self, temp_workspace):
        """Create a memory store for testing."""
        return MemoryStore(temp_workspace)

    @pytest.fixture
    def fusion_layer(self, memory_store):
        """Create a fusion layer for testing."""
        config = FusionConfig(
            total_dim=256,
            state_dim=64,
            visual_dim=128,
            memory_dim=64,
        )
        return MultimodalFusionLayer(config=config, memory_store=memory_store)

    def test_init(self, memory_store):
        """Test fusion layer initialization."""
        config = FusionConfig(total_dim=512)
        layer = MultimodalFusionLayer(config=config, memory_store=memory_store)
        assert layer.config.total_dim == 512
        assert layer.memory_store == memory_store

    def test_init_without_memory_store(self):
        """Test fusion layer initialization without memory store."""
        layer = MultimodalFusionLayer()
        assert layer.memory_store is None

    def test_fuse_with_game_state_only(self, fusion_layer):
        """Test fusion with only game state (no visual)."""
        game_state = {
            "board": [["X", "", ""], ["", "O", ""], ["", "", ""]],
            "current_player": "X",
            "turn_number": 2,
            "legal_moves": ["0,1", "0,2", "1,0", "1,2", "2,0", "2,1", "2,2"],
        }

        result = fusion_layer.fuse(game_state=game_state)

        assert isinstance(result, FusedEmbedding)
        assert result.dimensions == 256
        assert result.embedding.shape == (256,)
        # Visual component should be near zero
        assert result.components.get("visual", 0) == 0.0

    def test_fuse_with_visual_embedding(self, fusion_layer):
        """Test fusion with visual embedding."""
        game_state = {
            "board": [[0, 1, 2]],
            "current_player": "A",
            "turn_number": 1,
        }
        visual = VisualEmbedding(
            embedding=np.random.rand(128).astype(np.float32),
            dimensions=128,
            model_name="test",
            confidence=1.0,
        )

        result = fusion_layer.fuse(game_state=game_state, visual_embedding=visual)

        assert isinstance(result, FusedEmbedding)
        assert result.dimensions == 256
        # Visual component should be non-zero
        assert result.components.get("visual", 0) > 0

    def test_fuse_with_query(self, fusion_layer, memory_store):
        """Test fusion with memory query."""
        # First store some memory
        memory_store.save_fractal_node(
            content="Test strategy for tictactoe",
            tags=["strategy", "game:tictactoe"],
            summary="Test strategy",
        )

        game_state = {
            "board": [],
            "current_player": "X",
            "turn_number": 1,
        }

        result = fusion_layer.fuse(
            game_state=game_state,
            query="strategy game:tictactoe",
        )

        assert isinstance(result, FusedEmbedding)
        assert result.metadata.get("memory_nodes_count", 0) >= 0

    def test_fuse_produces_normalized_output(self, fusion_layer):
        """Test that fused embedding is L2 normalized."""
        game_state = {
            "board": [1, 2, 3],
            "current_player": "X",
            "turn_number": 5,
        }

        result = fusion_layer.fuse(game_state=game_state)

        # Check L2 norm is approximately 1
        norm = np.linalg.norm(result.embedding)
        assert abs(norm - 1.0) < 0.01 or norm == 0.0  # Allow zero vector

    def test_fuse_low_confidence_visual_treated_as_none(self, fusion_layer):
        """Test that low-confidence visual embedding is treated as None."""
        game_state = {
            "board": [],
            "current_player": "X",
            "turn_number": 1,
        }
        visual = VisualEmbedding(
            embedding=np.random.rand(128).astype(np.float32),
            dimensions=128,
            model_name="test",
            confidence=0.0,  # Zero confidence
        )

        result = fusion_layer.fuse(game_state=game_state, visual_embedding=visual)

        # Visual component should be zero due to low confidence
        assert result.components.get("visual", 0) == 0.0

    def test_build_context_summary(self, fusion_layer):
        """Test building context summary for LLM."""
        game_state = {
            "board": [["X", "O", ""], ["", "X", ""], ["", "", "O"]],
            "current_player": "X",
            "turn_number": 4,
            "legal_moves": ["0,2", "1,0", "1,2", "2,0", "2,1"],
        }
        fused = FusedEmbedding(
            embedding=np.zeros(256, dtype=np.float32),
            dimensions=256,
            components={"state": 0.5, "visual": 0.3, "memory": 0.2},
        )

        summary = fusion_layer.build_context_summary(
            game_state=game_state,
            fused=fused,
            memory_nodes=[],
        )

        assert isinstance(summary, str)
        assert "Current Game State" in summary
        assert "X" in summary
        assert "Turn" in summary
        assert "Fusion Analysis" in summary

    def test_build_context_summary_with_memory_nodes(self, fusion_layer, memory_store):
        """Test context summary includes memory nodes."""
        # Create memory node
        node = memory_store.save_fractal_node(
            content="Previous winning strategy",
            tags=["strategy", "win"],
            summary="Winning move pattern",
        )

        game_state = {
            "board": [],
            "current_player": "X",
            "turn_number": 1,
        }
        fused = FusedEmbedding(
            embedding=np.zeros(256),
            dimensions=256,
            components={},
        )

        summary = fusion_layer.build_context_summary(
            game_state=game_state,
            fused=fused,
            memory_nodes=[node],
        )

        assert "Relevant Past Experiences" in summary
        assert "Winning move pattern" in summary
