"""Multimodal fusion layer for combining game state, visual, and memory embeddings."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from loguru import logger

from nanobot.agent.memory import MemoryStore
from nanobot.agent.memory_types import FractalNode
from nanobot.game.visual_perception import VisualEmbedding


@dataclass
class FusedEmbedding:
    """
    Represents a fused multimodal embedding.

    Combines state, visual, and memory embeddings into a single representation.

    Args:
        embedding: The fused embedding vector
        dimensions: Number of dimensions in the embedding
        components: Dictionary describing the contribution of each component
        metadata: Additional metadata about the fusion
    """

    embedding: np.ndarray
    dimensions: int
    components: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class FusionConfig:
    """
    Configuration for the multimodal fusion layer.

    Args:
        total_dim: Total output embedding dimension
        state_dim: Dimension for state encoding
        visual_dim: Dimension for visual embedding
        memory_dim: Dimension for memory encoding
        state_weight: Weight for state component (default: 0.4)
        visual_weight: Weight for visual component (default: 0.3)
        memory_weight: Weight for memory component (default: 0.3)
        memory_top_k: Number of memory nodes to retrieve (default: 5)
    """

    total_dim: int = 512
    state_dim: int = 128
    visual_dim: int = 256
    memory_dim: int = 128
    state_weight: float = 0.4
    visual_weight: float = 0.3
    memory_weight: float = 0.3
    memory_top_k: int = 5

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        # Normalize weights
        total_weight = self.state_weight + self.visual_weight + self.memory_weight
        if total_weight > 0:
            self.state_weight /= total_weight
            self.visual_weight /= total_weight
            self.memory_weight /= total_weight


class MultimodalFusionLayer:
    """
    Fuses game state, visual embeddings, and memory context into a unified representation.

    The fusion layer combines three modalities:
    1. Game state: Encoded using hash-based features
    2. Visual: From screenshot embeddings via VisualEncoder
    3. Memory: Retrieved from MemoryStore using get_entangled_context()

    The components are weighted, concatenated, and L2 normalized.

    Args:
        config: FusionConfig specifying dimensions and weights
        memory_store: MemoryStore instance for context retrieval
    """

    def __init__(
        self,
        config: FusionConfig | None = None,
        memory_store: MemoryStore | None = None,
    ):
        self.config = config or FusionConfig()
        self.memory_store = memory_store
        logger.info(
            f"game.fusion.multimodal_fusion_layer.init "
            f"total_dim={self.config.total_dim} "
            f"weights=({self.config.state_weight:.2f}, "
            f"{self.config.visual_weight:.2f}, {self.config.memory_weight:.2f})"
        )

    def fuse(
        self,
        game_state: dict[str, Any],
        visual_embedding: VisualEmbedding | None = None,
        query: str = "",
    ) -> FusedEmbedding:
        """
        Fuse game state, visual embedding, and memory context.

        Steps:
        1. Encode state using hash-based features
        2. Use visual embedding or zeros if not provided
        3. Encode memory via get_entangled_context()
        4. Apply weighted concatenation and L2 normalize

        Args:
            game_state: Dictionary representing the game state
            visual_embedding: Optional VisualEmbedding from screenshot
            query: Query string for memory retrieval

        Returns:
            FusedEmbedding containing the combined representation
        """
        # Step 1: Encode game state
        state_encoded = self._encode_state(game_state)

        # Step 2: Get visual embedding or use zeros
        if visual_embedding is not None and visual_embedding.confidence > 0:
            visual_encoded = self._adapt_visual(visual_embedding)
        else:
            visual_encoded = np.zeros(self.config.visual_dim, dtype=np.float32)

        # Step 3: Encode memory context
        memory_nodes: list[FractalNode] = []
        if self.memory_store and query:
            memory_nodes = self.memory_store.get_entangled_context(
                query=query, top_k=self.config.memory_top_k
            )
        memory_encoded = self._encode_memory(memory_nodes)

        # Step 4: Weighted concatenation
        state_weighted = state_encoded * self.config.state_weight
        visual_weighted = visual_encoded * self.config.visual_weight
        memory_weighted = memory_encoded * self.config.memory_weight

        # Concatenate
        fused = np.concatenate([state_weighted, visual_weighted, memory_weighted])

        # Pad or truncate to total_dim
        if len(fused) < self.config.total_dim:
            fused = np.pad(
                fused, (0, self.config.total_dim - len(fused)), mode="constant"
            )
        else:
            fused = fused[: self.config.total_dim]

        # L2 normalize
        norm = np.linalg.norm(fused)
        if norm > 0:
            fused = fused / norm

        result = FusedEmbedding(
            embedding=fused.astype(np.float32),
            dimensions=self.config.total_dim,
            components={
                "state": float(np.linalg.norm(state_weighted)),
                "visual": float(np.linalg.norm(visual_weighted)),
                "memory": float(np.linalg.norm(memory_weighted)),
            },
            metadata={
                "memory_nodes_count": len(memory_nodes),
                "visual_available": visual_embedding is not None,
            },
        )

        logger.debug(
            f"game.fusion.multimodal_fusion_layer.fuse "
            f"dimensions={self.config.total_dim} "
            f"memory_nodes={len(memory_nodes)}"
        )

        return result

    def build_context_summary(
        self,
        game_state: dict[str, Any],
        fused: FusedEmbedding,
        memory_nodes: list[FractalNode],
    ) -> str:
        """
        Build a text context summary for LLM reasoning.

        Creates a formatted text string combining game state information
        and relevant memory context for use in prompts.

        Args:
            game_state: Current game state dictionary
            fused: The fused embedding (for metadata)
            memory_nodes: Retrieved memory nodes

        Returns:
            Formatted text string for LLM context
        """
        lines = []

        # Game state section
        lines.append("## Current Game State")
        lines.append(f"- Board: {game_state.get('board', 'N/A')}")
        lines.append(f"- Current Player: {game_state.get('current_player', 'N/A')}")
        lines.append(f"- Turn: {game_state.get('turn_number', 'N/A')}")
        lines.append(f"- Legal Moves: {game_state.get('legal_moves', [])}")

        if game_state.get("game_over"):
            lines.append("- Game Over: Yes")
            lines.append(f"- Winner: {game_state.get('winner', 'None')}")

        # Fusion info
        lines.append("")
        lines.append("## Fusion Analysis")
        lines.append(
            f"- State contribution: {fused.components.get('state', 0):.3f}"
        )
        lines.append(
            f"- Visual contribution: {fused.components.get('visual', 0):.3f}"
        )
        lines.append(
            f"- Memory contribution: {fused.components.get('memory', 0):.3f}"
        )

        # Memory context
        if memory_nodes:
            lines.append("")
            lines.append("## Relevant Past Experiences")
            for i, node in enumerate(memory_nodes[:3]):  # Limit to top 3
                lines.append(f"### Memory {i + 1}")
                lines.append(f"- Summary: {node.context_summary}")
                lines.append(f"- Tags: {', '.join(node.tags)}")
                # Truncate content for context
                content_preview = (
                    node.content[:200] + "..."
                    if len(node.content) > 200
                    else node.content
                )
                lines.append(f"- Content: {content_preview}")

        context = "\n".join(lines)
        logger.debug(
            f"game.fusion.multimodal_fusion_layer.build_context_summary "
            f"length={len(context)}"
        )
        return context

    def _encode_state(self, game_state: dict[str, Any]) -> np.ndarray:
        """
        Encode game state using hash-based features.

        Args:
            game_state: Game state dictionary

        Returns:
            Numpy array of state features
        """
        features = []

        # Board encoding
        board = game_state.get("board")
        if board is not None:
            if isinstance(board, (list, tuple)):
                board_features = self._hash_encode_sequence(board, 32)
                features.extend(board_features)
            elif isinstance(board, dict):
                board_str = str(sorted(board.items()))
                board_features = self._hash_encode_string(board_str, 32)
                features.extend(board_features)

        # Current player encoding
        player = game_state.get("current_player", "")
        player_features = self._hash_encode_string(str(player), 8)
        features.extend(player_features)

        # Turn number (normalized)
        turn = game_state.get("turn_number", 0)
        features.append(min(turn / 100.0, 1.0))  # Normalize to [0, 1]

        # Legal moves count (normalized)
        legal_moves = game_state.get("legal_moves", [])
        features.append(min(len(legal_moves) / 50.0, 1.0))

        # Game over flag
        features.append(1.0 if game_state.get("game_over") else 0.0)

        # Score (normalized)
        score = game_state.get("score", 0.0)
        features.append(np.tanh(score / 10.0))  # Squash to [-1, 1]

        # Convert to numpy and resize to state_dim
        features_array = np.array(features, dtype=np.float32)

        if len(features_array) < self.config.state_dim:
            features_array = np.pad(
                features_array,
                (0, self.config.state_dim - len(features_array)),
                mode="constant",
            )
        else:
            features_array = features_array[: self.config.state_dim]

        return features_array

    def _adapt_visual(self, visual_embedding: VisualEmbedding) -> np.ndarray:
        """
        Adapt visual embedding to the configured visual dimension.

        Args:
            visual_embedding: VisualEmbedding from encoder

        Returns:
            Numpy array of visual_dim dimensions
        """
        embedding = visual_embedding.embedding

        if len(embedding) < self.config.visual_dim:
            return np.pad(
                embedding,
                (0, self.config.visual_dim - len(embedding)),
                mode="constant",
            ).astype(np.float32)
        else:
            return embedding[: self.config.visual_dim].astype(np.float32)

    def _encode_memory(self, nodes: list[FractalNode]) -> np.ndarray:
        """
        Encode memory nodes into a fixed-dimension vector.

        Args:
            nodes: List of FractalNode from memory retrieval

        Returns:
            Numpy array of memory features
        """
        if not nodes:
            return np.zeros(self.config.memory_dim, dtype=np.float32)

        features = []

        # Aggregate features from nodes
        for node in nodes[: self.config.memory_top_k]:
            # Use embedding if available
            if node.embedding:
                node_features = np.array(node.embedding[: 32], dtype=np.float32)
                if len(node_features) < 32:
                    node_features = np.pad(
                        node_features, (0, 32 - len(node_features)), mode="constant"
                    )
                features.extend(node_features.tolist())
            else:
                # Hash encode content
                content_features = self._hash_encode_string(node.content, 16)
                features.extend(content_features)

                # Hash encode tags
                tags_str = " ".join(node.tags)
                tags_features = self._hash_encode_string(tags_str, 8)
                features.extend(tags_features)

                # Importance
                features.append(node.importance)

                # Pad to 32
                while len(features) % 32 != 0:
                    features.append(0.0)

        features_array = np.array(features, dtype=np.float32)

        if len(features_array) < self.config.memory_dim:
            features_array = np.pad(
                features_array,
                (0, self.config.memory_dim - len(features_array)),
                mode="constant",
            )
        else:
            features_array = features_array[: self.config.memory_dim]

        return features_array

    def _hash_encode_string(self, s: str, dim: int) -> list[float]:
        """
        Hash-encode a string into a fixed-dimension vector.

        Args:
            s: String to encode
            dim: Target dimension

        Returns:
            List of float features
        """
        features = []
        h = hash(s)
        for i in range(dim):
            # Use different bits of the hash
            val = ((h >> (i * 4)) & 0xF) / 15.0  # Normalize to [0, 1]
            features.append(val)
        return features

    def _hash_encode_sequence(self, seq: list | tuple, dim: int) -> list[float]:
        """
        Hash-encode a sequence into a fixed-dimension vector.

        Args:
            seq: Sequence to encode
            dim: Target dimension

        Returns:
            List of float features
        """
        # Flatten and convert to string representation
        flat_str = str(seq)
        return self._hash_encode_string(flat_str, dim)
