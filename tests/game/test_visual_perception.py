"""Tests for visual perception module."""

from __future__ import annotations

import numpy as np
import pytest

from nanobot.game.visual_perception import (
    SimpleGridEncoder,
    VisualEmbedding,
    create_encoder,
)


class TestVisualEmbedding:
    """Tests for VisualEmbedding dataclass."""

    def test_create_embedding(self):
        """Test creating a visual embedding."""
        embedding = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        ve = VisualEmbedding(
            embedding=embedding,
            dimensions=4,
            model_name="test_model",
            image_size=(100, 100),
            confidence=0.95,
        )

        assert ve.dimensions == 4
        assert ve.model_name == "test_model"
        assert ve.image_size == (100, 100)
        assert ve.confidence == 0.95

    def test_normalize(self):
        """Test L2 normalization of embedding."""
        embedding = np.array([3.0, 4.0], dtype=np.float32)
        ve = VisualEmbedding(
            embedding=embedding,
            dimensions=2,
            model_name="test",
        )

        normalized = ve.normalize()
        expected_norm = np.sqrt(3**2 + 4**2)  # 5.0
        expected = embedding / expected_norm

        np.testing.assert_array_almost_equal(normalized, expected)

    def test_normalize_zero_vector(self):
        """Test normalization of zero vector."""
        embedding = np.zeros(4, dtype=np.float32)
        ve = VisualEmbedding(
            embedding=embedding,
            dimensions=4,
            model_name="test",
        )

        normalized = ve.normalize()
        np.testing.assert_array_equal(normalized, embedding)

    def test_to_list(self):
        """Test converting embedding to Python list."""
        embedding = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        ve = VisualEmbedding(
            embedding=embedding,
            dimensions=3,
            model_name="test",
        )

        result = ve.to_list()
        assert isinstance(result, list)
        assert len(result) == 3
        assert result == [1.0, 2.0, 3.0]


class TestSimpleGridEncoder:
    """Tests for SimpleGridEncoder class."""

    @pytest.fixture
    def encoder(self):
        """Create a SimpleGridEncoder for testing."""
        return SimpleGridEncoder(embedding_dim=64, grid_size=(4, 4))

    def test_init(self):
        """Test encoder initialization."""
        encoder = SimpleGridEncoder(embedding_dim=128, grid_size=(8, 8))
        assert encoder.embedding_dim == 128
        assert encoder.grid_size == (8, 8)

    def test_is_available(self, encoder):
        """Test is_available always returns True."""
        assert encoder.is_available() is True

    def test_encode_numpy_array(self, encoder):
        """Test encoding a numpy array image."""
        # Create a simple test image (32x32 RGB)
        image = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)

        result = encoder.encode(image)

        assert isinstance(result, VisualEmbedding)
        assert result.dimensions == 64
        assert result.embedding.shape == (64,)
        assert result.model_name == "SimpleGridEncoder"
        assert result.confidence == 1.0
        assert result.image_size == (32, 32)

    def test_encode_grayscale_image(self, encoder):
        """Test encoding a grayscale image."""
        # Create a grayscale image
        image = np.random.randint(0, 255, (32, 32), dtype=np.uint8)

        result = encoder.encode(image)

        assert isinstance(result, VisualEmbedding)
        assert result.dimensions == 64

    def test_encode_returns_float32(self, encoder):
        """Test that embedding is float32 dtype."""
        image = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        result = encoder.encode(image)
        assert result.embedding.dtype == np.float32

    def test_encode_pads_to_embedding_dim(self):
        """Test that output is padded to embedding_dim."""
        encoder = SimpleGridEncoder(embedding_dim=256, grid_size=(2, 2))
        image = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)

        result = encoder.encode(image)

        # 2x2 grid = 4 cells, 6 features per cell = 24 features
        # Should be padded to 256
        assert result.embedding.shape == (256,)

    def test_encode_truncates_if_too_large(self):
        """Test that output is truncated if features exceed embedding_dim."""
        encoder = SimpleGridEncoder(embedding_dim=10, grid_size=(4, 4))
        image = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)

        result = encoder.encode(image)

        # Should be truncated to 10
        assert result.embedding.shape == (10,)


class TestCreateEncoder:
    """Tests for create_encoder factory function."""

    def test_create_auto_encoder(self):
        """Test creating encoder with 'auto' type."""
        encoder = create_encoder(encoder_type="auto", embedding_dim=128)
        assert encoder is not None
        assert encoder.is_available() is True

    def test_create_grid_encoder(self):
        """Test creating grid encoder explicitly."""
        encoder = create_encoder(encoder_type="grid", embedding_dim=64)
        assert isinstance(encoder, SimpleGridEncoder)
        assert encoder.embedding_dim == 64

    def test_create_unknown_type_falls_back(self):
        """Test that unknown type falls back to auto."""
        encoder = create_encoder(encoder_type="unknown_type", embedding_dim=128)
        assert encoder is not None
        assert encoder.is_available() is True

    def test_encoder_respects_embedding_dim(self):
        """Test that encoder respects embedding_dim parameter."""
        encoder = create_encoder(encoder_type="grid", embedding_dim=256)
        image = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)

        result = encoder.encode(image)
        assert result.dimensions == 256
        assert result.embedding.shape == (256,)
