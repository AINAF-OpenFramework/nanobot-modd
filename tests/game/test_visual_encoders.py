"""Tests for visual encoder validation on game boards."""

from __future__ import annotations

import numpy as np
import pytest

from nanobot.game.rules.chess import ChessRules
from nanobot.game.rules.tictactoe import TicTacToeRules
from nanobot.game.visual_perception import (
    SimpleGridEncoder,
    VisualEmbedding,
    create_encoder,
)

# Try to import PIL for image generation
try:
    from PIL import Image, ImageDraw, ImageFont

    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


def _get_board_cell(board: list, row: int, col: int) -> str:
    """Helper to safely get a board cell value."""
    if row < len(board) and col < len(board[row]):
        return board[row][col]
    return ""


def create_tictactoe_board_image(state: dict) -> np.ndarray:
    """
    Create a synthetic image of a TicTacToe board.

    Args:
        state: TicTacToe game state

    Returns:
        numpy array representing the board image
    """
    if not PIL_AVAILABLE:
        # Fallback: return a simple synthetic board as numpy array
        size = 300
        img = np.ones((size, size, 3), dtype=np.uint8) * 255

        board = state.get("board", [])
        cell_size = size // 3

        # Draw grid lines
        for i in range(1, 3):
            pos = i * cell_size
            img[pos - 2 : pos + 2, :] = 0  # Horizontal lines
            img[:, pos - 2 : pos + 2] = 0  # Vertical lines

        # Draw pieces (simple colored blocks)
        for row in range(3):
            for col in range(3):
                piece = _get_board_cell(board, row, col)
                if piece == "X":
                    # Red X
                    y1, y2 = row * cell_size + 20, (row + 1) * cell_size - 20
                    x1, x2 = col * cell_size + 20, (col + 1) * cell_size - 20
                    img[y1:y2, x1:x2] = [255, 0, 0]
                elif piece == "O":
                    # Blue O
                    y1, y2 = row * cell_size + 20, (row + 1) * cell_size - 20
                    x1, x2 = col * cell_size + 20, (col + 1) * cell_size - 20
                    img[y1:y2, x1:x2] = [0, 0, 255]

        return img

    # Use PIL to create a better image
    size = 300
    img = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(img)

    board = state.get("board", [])
    cell_size = size // 3

    # Draw grid
    for i in range(1, 3):
        pos = i * cell_size
        draw.line([(0, pos), (size, pos)], fill="black", width=3)
        draw.line([(pos, 0), (pos, size)], fill="black", width=3)

    # Draw pieces
    for row in range(3):
        for col in range(3):
            piece = _get_board_cell(board, row, col)
            if piece:
                y = row * cell_size + cell_size // 2
                x = col * cell_size + cell_size // 2
                if piece == "X":
                    # Draw X
                    offset = cell_size // 3
                    draw.line(
                        [(x - offset, y - offset), (x + offset, y + offset)],
                        fill="red",
                        width=5,
                    )
                    draw.line(
                        [(x - offset, y + offset), (x + offset, y - offset)],
                        fill="red",
                        width=5,
                    )
                elif piece == "O":
                    # Draw O
                    radius = cell_size // 3
                    draw.ellipse(
                        [(x - radius, y - radius), (x + radius, y + radius)],
                        outline="blue",
                        width=5,
                    )

    return np.array(img)


def create_chess_board_image(state: dict) -> np.ndarray:
    """
    Create a synthetic image of a chess board.

    Args:
        state: Chess game state

    Returns:
        numpy array representing the board image
    """
    size = 400
    cell_size = size // 8

    # Create checkerboard pattern
    img = np.zeros((size, size, 3), dtype=np.uint8)

    # Alternate light and dark squares
    for row in range(8):
        for col in range(8):
            y1, y2 = row * cell_size, (row + 1) * cell_size
            x1, x2 = col * cell_size, (col + 1) * cell_size

            # Checkerboard pattern
            if (row + col) % 2 == 0:
                img[y1:y2, x1:x2] = [240, 217, 181]  # Light square
            else:
                img[y1:y2, x1:x2] = [181, 136, 99]  # Dark square

            # Add piece indicator (colored circle)
            board = state.get("board", [])
            if row < len(board) and col < len(board[row]):
                piece = board[row][col]
                if piece:
                    color = piece[0]  # 'w' or 'b'
                    # Draw a colored square for the piece
                    y_center = (y1 + y2) // 2
                    x_center = (x1 + x2) // 2
                    piece_size = cell_size // 3

                    if color == "w":
                        img[
                            y_center - piece_size : y_center + piece_size,
                            x_center - piece_size : x_center + piece_size,
                        ] = [255, 255, 255]
                    else:
                        img[
                            y_center - piece_size : y_center + piece_size,
                            x_center - piece_size : x_center + piece_size,
                        ] = [50, 50, 50]

    return img


class TestVisualEncoderValidation:
    """Tests for visual encoder validation on game boards."""

    @pytest.fixture
    def tictactoe_rules(self):
        """Create TicTacToe rules."""
        return TicTacToeRules()

    @pytest.fixture
    def chess_rules(self):
        """Create Chess rules."""
        return ChessRules()

    def test_simple_grid_encoder_on_tictactoe(self, tictactoe_rules):
        """Test SimpleGridEncoder on TicTacToe board."""
        encoder = SimpleGridEncoder(embedding_dim=128, grid_size=(3, 3))

        # Create various game states
        states = [
            tictactoe_rules.create_initial_state(),
            {
                "board": [
                    ["X", "", ""],
                    ["", "O", ""],
                    ["", "", ""],
                ],
                "current_player": "X",
                "move_count": 2,
            },
            {
                "board": [
                    ["X", "O", "X"],
                    ["O", "X", "O"],
                    ["O", "X", "X"],
                ],
                "current_player": "X",
                "move_count": 9,
            },
        ]

        embeddings = []
        for state in states:
            img = create_tictactoe_board_image(state)
            embedding = encoder.encode(img)

            # Validate embedding
            assert isinstance(embedding, VisualEmbedding)
            assert embedding.dimensions == 128
            assert embedding.embedding.shape == (128,)
            assert embedding.confidence > 0
            assert embedding.model_name == "SimpleGridEncoder"

            embeddings.append(embedding)

        # Different states should produce different embeddings
        assert not np.allclose(embeddings[0].embedding, embeddings[1].embedding)
        assert not np.allclose(embeddings[1].embedding, embeddings[2].embedding)

    def test_simple_grid_encoder_on_chess(self, chess_rules):
        """Test SimpleGridEncoder on chess board."""
        encoder = SimpleGridEncoder(embedding_dim=256, grid_size=(8, 8))

        # Create test positions
        positions = chess_rules.generate_test_positions()

        embeddings = []
        for state in positions:
            img = create_chess_board_image(state)
            embedding = encoder.encode(img)

            # Validate embedding
            assert isinstance(embedding, VisualEmbedding)
            assert embedding.dimensions == 256
            assert embedding.embedding.shape == (256,)
            assert embedding.confidence > 0

            embeddings.append(embedding)

        # Different positions should produce different embeddings
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                assert not np.allclose(embeddings[i].embedding, embeddings[j].embedding)

    def test_encoder_factory_creates_valid_encoder(self):
        """Test that create_encoder factory produces valid encoders."""
        encoder = create_encoder(encoder_type="grid", embedding_dim=128)
        assert encoder is not None
        assert encoder.is_available()
        assert isinstance(encoder, SimpleGridEncoder)

    def test_encoder_handles_grayscale_images(self):
        """Test encoder handles grayscale images."""
        encoder = SimpleGridEncoder(embedding_dim=64)

        # Create grayscale image
        gray_img = np.random.randint(0, 256, (100, 100), dtype=np.uint8)

        embedding = encoder.encode(gray_img)
        assert isinstance(embedding, VisualEmbedding)
        assert embedding.dimensions == 64
        assert embedding.confidence > 0

    def test_encoder_handles_different_sizes(self):
        """Test encoder handles images of different sizes."""
        encoder = SimpleGridEncoder(embedding_dim=128)

        sizes = [(50, 50), (100, 100), (200, 300), (400, 400)]

        for width, height in sizes:
            img = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
            embedding = encoder.encode(img)

            assert isinstance(embedding, VisualEmbedding)
            assert embedding.dimensions == 128
            assert embedding.image_size == (width, height)

    def test_encoder_consistency(self):
        """Test encoder produces consistent embeddings for same input."""
        encoder = SimpleGridEncoder(embedding_dim=128)

        # Create a test image
        img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

        # Encode multiple times
        embedding1 = encoder.encode(img)
        embedding2 = encoder.encode(img)

        # Should produce identical embeddings
        assert np.allclose(embedding1.embedding, embedding2.embedding)

    def test_encoder_latency(self):
        """Test encoder latency is reasonable."""
        import time

        encoder = SimpleGridEncoder(embedding_dim=256)

        # Create a test image
        img = np.random.randint(0, 256, (300, 300, 3), dtype=np.uint8)

        # Measure encoding time
        start = time.time()
        for _ in range(10):
            encoder.encode(img)
        elapsed = time.time() - start

        # Should encode at least 10 images per second (100ms per image)
        # With 10 images, should take less than 1 second
        assert elapsed < 1.0, f"Encoding too slow: {elapsed:.3f}s for 10 images"

    def test_embedding_accuracy_metric(self, tictactoe_rules):
        """Test that embeddings can distinguish different board states."""
        encoder = SimpleGridEncoder(embedding_dim=128, grid_size=(3, 3))

        # Create pairs of similar and dissimilar states
        empty_state = tictactoe_rules.create_initial_state()
        one_move_state = {
            "board": [["X", "", ""], ["", "", ""], ["", "", ""]],
            "current_player": "O",
            "move_count": 1,
        }
        full_state = {
            "board": [
                ["X", "O", "X"],
                ["O", "X", "O"],
                ["O", "X", "X"],
            ],
            "current_player": "X",
            "move_count": 9,
        }

        # Encode all states
        empty_emb = encoder.encode(create_tictactoe_board_image(empty_state))
        one_move_emb = encoder.encode(create_tictactoe_board_image(one_move_state))
        full_emb = encoder.encode(create_tictactoe_board_image(full_state))

        # Calculate similarities (cosine similarity via dot product of normalized vectors)
        empty_vs_one = np.dot(empty_emb.embedding, one_move_emb.embedding)
        empty_vs_full = np.dot(empty_emb.embedding, full_emb.embedding)
        one_vs_full = np.dot(one_move_emb.embedding, full_emb.embedding)

        # Empty and one_move should be more similar than empty and full
        assert empty_vs_one > empty_vs_full
        # One_move and full should be less similar than empty and one_move
        assert one_vs_full < empty_vs_one

    def test_encoder_error_handling(self):
        """Test encoder handles errors gracefully."""
        encoder = SimpleGridEncoder(embedding_dim=64)

        # Test with invalid input
        invalid_inputs = [
            None,
            [],
            {},
            "not an image",
        ]

        for invalid_input in invalid_inputs:
            try:
                embedding = encoder.encode(invalid_input)
                # Should return zero embedding with 0 confidence on error
                assert embedding.confidence == 0.0
                assert np.allclose(embedding.embedding, np.zeros(64))
            except Exception:
                # It's also acceptable to raise an exception
                pass
