"""Visual perception module for encoding game screenshots."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from loguru import logger

# Optional PyTorch dependency
try:
    import torch
    import torch.nn as nn
    from torch import Tensor

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None  # type: ignore
    nn = None  # type: ignore
    Tensor = None  # type: ignore

# Optional PIL dependency
try:
    from PIL import Image

    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    Image = None  # type: ignore


@dataclass
class VisualEmbedding:
    """
    Represents a visual embedding from a game screenshot.

    Args:
        embedding: The embedding vector as numpy array
        dimensions: Number of dimensions in the embedding
        model_name: Name of the model used to create the embedding
        image_size: Original image size (width, height)
        confidence: Confidence score for the embedding (0.0 to 1.0)
        metadata: Additional metadata about the embedding
    """

    embedding: np.ndarray
    dimensions: int
    model_name: str
    image_size: tuple[int, int] = (0, 0)
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def normalize(self) -> np.ndarray:
        """
        Return L2-normalized embedding vector.

        Returns:
            Normalized numpy array
        """
        norm = np.linalg.norm(self.embedding)
        if norm > 0:
            return self.embedding / norm
        return self.embedding

    def to_list(self) -> list[float]:
        """
        Convert embedding to a Python list.

        Returns:
            List of float values
        """
        return self.embedding.tolist()


class VisualEncoder(ABC):
    """
    Abstract base class for visual encoders.

    Visual encoders transform game screenshots into fixed-dimension
    embedding vectors for use in the fusion layer.
    """

    @abstractmethod
    def encode(self, image: Any) -> VisualEmbedding:
        """
        Encode an image into a visual embedding.

        Args:
            image: PIL Image or numpy array

        Returns:
            VisualEmbedding containing the encoded representation
        """
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if this encoder is available for use.

        Returns:
            True if encoder dependencies are satisfied
        """
        ...


class SimpleGridEncoder(VisualEncoder):
    """
    Simple grid-based visual encoder without PyTorch dependency.

    Divides the image into a grid and computes color features
    for each cell. Suitable for simple games without complex visuals.

    Args:
        embedding_dim: Target embedding dimension
        grid_size: Grid dimensions (rows, cols)
    """

    def __init__(
        self,
        embedding_dim: int = 256,
        grid_size: tuple[int, int] = (8, 8),
    ):
        self.embedding_dim = embedding_dim
        self.grid_size = grid_size
        logger.debug(
            f"game.visual_perception.simple_grid_encoder.init "
            f"embedding_dim={embedding_dim} grid_size={grid_size}"
        )

    def encode(self, image: Any) -> VisualEmbedding:
        """
        Encode an image using grid-based color features.

        Args:
            image: PIL Image or numpy array

        Returns:
            VisualEmbedding with grid-based features
        """
        try:
            # Convert to numpy array if needed
            if PIL_AVAILABLE and isinstance(image, Image.Image):
                img_array = np.array(image)
            elif isinstance(image, np.ndarray):
                img_array = image
            else:
                # Try to convert
                img_array = np.array(image)

            # Get image dimensions
            if len(img_array.shape) == 2:
                # Grayscale - expand to 3 channels
                img_array = np.stack([img_array] * 3, axis=-1)

            height, width = img_array.shape[:2]
            image_size = (width, height)

            # Compute grid cell features
            features = self._compute_grid_features(img_array)

            # Pad or truncate to embedding_dim
            if len(features) < self.embedding_dim:
                features = np.pad(
                    features, (0, self.embedding_dim - len(features)), mode="constant"
                )
            else:
                features = features[: self.embedding_dim]

            embedding = VisualEmbedding(
                embedding=features.astype(np.float32),
                dimensions=self.embedding_dim,
                model_name="SimpleGridEncoder",
                image_size=image_size,
                confidence=1.0,
                metadata={"grid_size": self.grid_size},
            )

            logger.debug(
                f"game.visual_perception.simple_grid_encoder.encode "
                f"image_size={image_size} embedding_dim={self.embedding_dim}"
            )
            return embedding

        except Exception as e:
            logger.error(f"game.visual_perception.simple_grid_encoder.encode error={e}")
            # Return zero embedding on error
            return VisualEmbedding(
                embedding=np.zeros(self.embedding_dim, dtype=np.float32),
                dimensions=self.embedding_dim,
                model_name="SimpleGridEncoder",
                confidence=0.0,
                metadata={"error": str(e)},
            )

    def is_available(self) -> bool:
        """SimpleGridEncoder is always available."""
        return True

    def _compute_grid_features(self, img_array: np.ndarray) -> np.ndarray:
        """
        Compute features for each grid cell.

        Args:
            img_array: Image as numpy array (H, W, C)

        Returns:
            Feature vector as numpy array
        """
        height, width = img_array.shape[:2]
        rows, cols = self.grid_size

        cell_height = height // rows
        cell_width = width // cols

        features = []

        for i in range(rows):
            for j in range(cols):
                # Extract cell
                y1, y2 = i * cell_height, (i + 1) * cell_height
                x1, x2 = j * cell_width, (j + 1) * cell_width
                cell = img_array[y1:y2, x1:x2]

                # Compute features: mean and std for each channel
                cell_mean = np.mean(cell, axis=(0, 1)) / 255.0
                cell_std = np.std(cell, axis=(0, 1)) / 255.0

                features.extend(cell_mean)
                features.extend(cell_std)

        return np.array(features, dtype=np.float32)


class LightweightCNNEncoder(VisualEncoder):
    """
    Lightweight CNN encoder using MobileNetV3-Small.

    Uses a pretrained MobileNetV3-Small backbone with a projection layer
    to generate compact embeddings. Efficient for real-time applications.

    Args:
        embedding_dim: Target embedding dimension
        device: PyTorch device ('cpu', 'cuda', etc.)
    """

    def __init__(
        self,
        embedding_dim: int = 256,
        device: str = "cpu",
    ):
        self.embedding_dim = embedding_dim
        self.device = device
        self._model = None
        self._projection = None
        self._transform = None

        if TORCH_AVAILABLE:
            self._initialize_model()

        logger.debug(
            f"game.visual_perception.lightweight_cnn_encoder.init "
            f"embedding_dim={embedding_dim} device={device}"
        )

    def _initialize_model(self) -> None:
        """Initialize the MobileNetV3-Small model."""
        try:
            from torchvision import models, transforms

            # Load pretrained MobileNetV3-Small
            weights = models.MobileNet_V3_Small_Weights.DEFAULT
            self._model = models.mobilenet_v3_small(weights=weights)
            # Remove classifier
            self._model.classifier = nn.Identity()
            self._model.eval()
            self._model.to(self.device)

            # Projection layer: 576 -> embedding_dim
            self._projection = nn.Linear(576, self.embedding_dim)
            self._projection.to(self.device)

            # Image transforms
            self._transform = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )

            logger.info(
                "game.visual_perception.lightweight_cnn_encoder "
                "model=MobileNetV3-Small initialized"
            )

        except Exception as e:
            logger.error(
                f"game.visual_perception.lightweight_cnn_encoder.init_model error={e}"
            )
            self._model = None

    @torch.no_grad() if TORCH_AVAILABLE else lambda f: f
    def encode(self, image: Any) -> VisualEmbedding:
        """
        Encode an image using MobileNetV3-Small.

        Args:
            image: PIL Image or numpy array

        Returns:
            VisualEmbedding with CNN features
        """
        if not self.is_available():
            logger.warning(
                "game.visual_perception.lightweight_cnn_encoder.encode "
                "model_not_available"
            )
            return VisualEmbedding(
                embedding=np.zeros(self.embedding_dim, dtype=np.float32),
                dimensions=self.embedding_dim,
                model_name="LightweightCNNEncoder",
                confidence=0.0,
                metadata={"error": "Model not available"},
            )

        try:
            # Convert to PIL Image if needed
            if isinstance(image, np.ndarray):
                pil_image = Image.fromarray(image)
            elif PIL_AVAILABLE and isinstance(image, Image.Image):
                pil_image = image
            else:
                pil_image = Image.fromarray(np.array(image))

            # Convert to RGB if needed
            if pil_image.mode != "RGB":
                pil_image = pil_image.convert("RGB")

            image_size = pil_image.size

            # Apply transforms
            input_tensor = self._transform(pil_image).unsqueeze(0)
            input_tensor = input_tensor.to(self.device)

            # Forward pass
            features = self._model(input_tensor)
            projected = self._projection(features)

            # Convert to numpy
            embedding = projected.squeeze(0).cpu().numpy()

            result = VisualEmbedding(
                embedding=embedding,
                dimensions=self.embedding_dim,
                model_name="LightweightCNNEncoder-MobileNetV3-Small",
                image_size=image_size,
                confidence=1.0,
            )

            logger.debug(
                f"game.visual_perception.lightweight_cnn_encoder.encode "
                f"image_size={image_size}"
            )
            return result

        except Exception as e:
            logger.error(
                f"game.visual_perception.lightweight_cnn_encoder.encode error={e}"
            )
            return VisualEmbedding(
                embedding=np.zeros(self.embedding_dim, dtype=np.float32),
                dimensions=self.embedding_dim,
                model_name="LightweightCNNEncoder",
                confidence=0.0,
                metadata={"error": str(e)},
            )

    def is_available(self) -> bool:
        """Check if PyTorch and model are available."""
        return TORCH_AVAILABLE and self._model is not None


class EfficientNetEncoder(VisualEncoder):
    """
    EfficientNet-B0 based visual encoder.

    Uses a pretrained EfficientNet-B0 backbone with a projection layer.
    Provides better accuracy than MobileNet at slightly higher cost.

    Args:
        embedding_dim: Target embedding dimension
        device: PyTorch device ('cpu', 'cuda', etc.)
    """

    def __init__(
        self,
        embedding_dim: int = 256,
        device: str = "cpu",
    ):
        self.embedding_dim = embedding_dim
        self.device = device
        self._model = None
        self._projection = None
        self._transform = None

        if TORCH_AVAILABLE:
            self._initialize_model()

        logger.debug(
            f"game.visual_perception.efficientnet_encoder.init "
            f"embedding_dim={embedding_dim} device={device}"
        )

    def _initialize_model(self) -> None:
        """Initialize the EfficientNet-B0 model."""
        try:
            from torchvision import models, transforms

            # Load pretrained EfficientNet-B0
            weights = models.EfficientNet_B0_Weights.DEFAULT
            self._model = models.efficientnet_b0(weights=weights)
            # Remove classifier
            self._model.classifier = nn.Identity()
            self._model.eval()
            self._model.to(self.device)

            # Projection layer: 1280 -> embedding_dim
            self._projection = nn.Linear(1280, self.embedding_dim)
            self._projection.to(self.device)

            # Image transforms
            self._transform = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )

            logger.info(
                "game.visual_perception.efficientnet_encoder "
                "model=EfficientNet-B0 initialized"
            )

        except Exception as e:
            logger.error(
                f"game.visual_perception.efficientnet_encoder.init_model error={e}"
            )
            self._model = None

    @torch.no_grad() if TORCH_AVAILABLE else lambda f: f
    def encode(self, image: Any) -> VisualEmbedding:
        """
        Encode an image using EfficientNet-B0.

        Args:
            image: PIL Image or numpy array

        Returns:
            VisualEmbedding with CNN features
        """
        if not self.is_available():
            logger.warning(
                "game.visual_perception.efficientnet_encoder.encode "
                "model_not_available"
            )
            return VisualEmbedding(
                embedding=np.zeros(self.embedding_dim, dtype=np.float32),
                dimensions=self.embedding_dim,
                model_name="EfficientNetEncoder",
                confidence=0.0,
                metadata={"error": "Model not available"},
            )

        try:
            # Convert to PIL Image if needed
            if isinstance(image, np.ndarray):
                pil_image = Image.fromarray(image)
            elif PIL_AVAILABLE and isinstance(image, Image.Image):
                pil_image = image
            else:
                pil_image = Image.fromarray(np.array(image))

            # Convert to RGB if needed
            if pil_image.mode != "RGB":
                pil_image = pil_image.convert("RGB")

            image_size = pil_image.size

            # Apply transforms
            input_tensor = self._transform(pil_image).unsqueeze(0)
            input_tensor = input_tensor.to(self.device)

            # Forward pass
            features = self._model(input_tensor)
            projected = self._projection(features)

            # Convert to numpy
            embedding = projected.squeeze(0).cpu().numpy()

            result = VisualEmbedding(
                embedding=embedding,
                dimensions=self.embedding_dim,
                model_name="EfficientNetEncoder-B0",
                image_size=image_size,
                confidence=1.0,
            )

            logger.debug(
                f"game.visual_perception.efficientnet_encoder.encode "
                f"image_size={image_size}"
            )
            return result

        except Exception as e:
            logger.error(f"game.visual_perception.efficientnet_encoder.encode error={e}")
            return VisualEmbedding(
                embedding=np.zeros(self.embedding_dim, dtype=np.float32),
                dimensions=self.embedding_dim,
                model_name="EfficientNetEncoder",
                confidence=0.0,
                metadata={"error": str(e)},
            )

    def is_available(self) -> bool:
        """Check if PyTorch and model are available."""
        return TORCH_AVAILABLE and self._model is not None


def create_encoder(
    encoder_type: str = "auto",
    embedding_dim: int = 256,
    device: str = "cpu",
) -> VisualEncoder:
    """
    Factory function to create a visual encoder.

    Args:
        encoder_type: Type of encoder ('auto', 'mobilenet', 'efficientnet', 'grid')
        embedding_dim: Target embedding dimension
        device: PyTorch device for neural network encoders

    Returns:
        VisualEncoder instance

    The 'auto' type selects MobileNet if PyTorch is available,
    otherwise falls back to the simple grid encoder.
    """
    logger.debug(
        f"game.visual_perception.create_encoder type={encoder_type} "
        f"embedding_dim={embedding_dim} device={device}"
    )

    if encoder_type == "auto":
        if TORCH_AVAILABLE:
            encoder = LightweightCNNEncoder(embedding_dim=embedding_dim, device=device)
            if encoder.is_available():
                return encoder
        # Fall back to grid encoder
        return SimpleGridEncoder(embedding_dim=embedding_dim)

    elif encoder_type == "mobilenet":
        if not TORCH_AVAILABLE:
            logger.warning(
                "game.visual_perception.create_encoder "
                "mobilenet_requires_torch falling_back=grid"
            )
            return SimpleGridEncoder(embedding_dim=embedding_dim)
        return LightweightCNNEncoder(embedding_dim=embedding_dim, device=device)

    elif encoder_type == "efficientnet":
        if not TORCH_AVAILABLE:
            logger.warning(
                "game.visual_perception.create_encoder "
                "efficientnet_requires_torch falling_back=grid"
            )
            return SimpleGridEncoder(embedding_dim=embedding_dim)
        return EfficientNetEncoder(embedding_dim=embedding_dim, device=device)

    elif encoder_type == "grid":
        return SimpleGridEncoder(embedding_dim=embedding_dim)

    else:
        logger.warning(
            f"game.visual_perception.create_encoder "
            f"unknown_type={encoder_type} falling_back=auto"
        )
        return create_encoder(encoder_type="auto", embedding_dim=embedding_dim, device=device)
