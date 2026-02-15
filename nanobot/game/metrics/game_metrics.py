"""Metrics tracker for chess game performance using IAS and CER metrics."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger


class MetricsTracker:
    """
    Tracks game metrics using IAS and CER measurements.
    
    - IAS (Intrinsic Alignment Score): Measures how well the chosen move aligns
      with the AI's strategic patterns (0.0-1.0)
    - CER (Contextual Evaluation Rate): Measures evaluation confidence/consistency (0.0-1.0)
    
    Logs metrics with timestamps for performance tracking and analysis.
    """

    # Normalization constants for score variance and consistency
    SEPARATION_NORMALIZER = 10.0  # Typical score separation range
    STD_DEV_NORMALIZER = 5.0  # Typical standard deviation range
    CONSISTENCY_NORMALIZER = 5.0  # Typical consistency deviation range

    def __init__(self, log_file: str | None = None):
        """
        Initialize metrics tracker.
        
        Args:
            log_file: Optional file path for persistent metric logging
        """
        self.metrics_history: list[dict[str, Any]] = []
        self.log_file = Path(log_file) if log_file else None
        
        if self.log_file:
            # Create parent directory if needed
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        logger.debug(f"game_metrics.MetricsTracker initialized log_file={log_file}")

    def compute_IAS_CER(
        self,
        move: str,
        scores: list[float],
        selected_idx: int | None = None,
        historical_scores: list[list[float]] | None = None,
    ) -> tuple[float, float]:
        """
        Compute IAS and CER metrics for a move.
        
        IAS (Intrinsic Alignment Score) measures strategic alignment:
        - Calculated as the normalized rank of the selected move
        - Higher values indicate better alignment with evaluation function
        - Range: 0.0 (worst) to 1.0 (best)
        
        CER (Contextual Evaluation Rate) measures confidence:
        - Calculated based on score variance and consistency
        - Higher values indicate more confident/consistent evaluation
        - Range: 0.0 (low confidence) to 1.0 (high confidence)
        
        Args:
            move: Move in algebraic notation
            scores: List of scores for all candidate moves
            selected_idx: Index of the selected move (if None, uses best)
            historical_scores: Optional historical scores for consistency check
            
        Returns:
            Tuple of (IAS, CER) values
            
        Example:
            >>> tracker = MetricsTracker()
            >>> scores = [0.5, 0.8, 0.3, 0.9]
            >>> ias, cer = tracker.compute_IAS_CER("d2d4", scores, selected_idx=3)
            >>> 0.0 <= ias <= 1.0 and 0.0 <= cer <= 1.0
            True
        """
        if not scores:
            logger.warning("game_metrics.compute_IAS_CER: no scores provided")
            return 0.0, 0.0
        
        # Determine selected index
        if selected_idx is None:
            selected_idx = scores.index(max(scores))
        
        # Compute IAS: normalized rank of selected move
        ias = self._compute_ias(scores, selected_idx)
        
        # Compute CER: evaluation confidence
        cer = self._compute_cer(scores, selected_idx, historical_scores)
        
        logger.debug(
            f"game_metrics.compute_IAS_CER move={move} IAS={ias:.3f} CER={cer:.3f}"
        )
        
        return ias, cer

    def _compute_ias(self, scores: list[float], selected_idx: int) -> float:
        """
        Compute Intrinsic Alignment Score.
        
        Measures how well the selected move aligns with the evaluation.
        
        Args:
            scores: List of move scores
            selected_idx: Index of selected move
            
        Returns:
            IAS value (0.0-1.0)
        """
        # Sort scores to find rank
        sorted_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )
        
        # Find rank of selected move (0 = best, n-1 = worst)
        rank = sorted_indices.index(selected_idx)
        
        # Normalize to 0-1 range (1 = best, 0 = worst)
        if len(scores) > 1:
            ias = 1.0 - (rank / (len(scores) - 1))
        else:
            ias = 1.0
        
        return ias

    def _compute_cer(
        self,
        scores: list[float],
        selected_idx: int,
        historical_scores: list[list[float]] | None,
    ) -> float:
        """
        Compute Contextual Evaluation Rate.
        
        Measures evaluation confidence based on:
        1. Score variance (lower variance = higher confidence)
        2. Score separation (clear winner = higher confidence)
        3. Historical consistency (if available)
        
        Args:
            scores: Current move scores
            selected_idx: Index of selected move
            historical_scores: Optional historical scores
            
        Returns:
            CER value (0.0-1.0)
        """
        selected_score = scores[selected_idx]
        
        # Component 1: Score separation
        # Higher separation means more confident choice
        if len(scores) > 1:
            sorted_scores = sorted(scores, reverse=True)
            best_score = sorted_scores[0]
            second_best = sorted_scores[1] if len(sorted_scores) > 1 else sorted_scores[0]
            separation = abs(best_score - second_best)
            
            # Normalize separation
            separation_component = min(separation / self.SEPARATION_NORMALIZER, 1.0)
        else:
            separation_component = 1.0
        
        # Component 2: Score variance
        # Lower variance means more consistent evaluation
        mean_score = sum(scores) / len(scores)
        variance = sum((s - mean_score) ** 2 for s in scores) / len(scores)
        std_dev = variance ** 0.5
        
        # Normalize variance
        variance_component = 1.0 - min(std_dev / self.STD_DEV_NORMALIZER, 1.0)
        
        # Component 3: Historical consistency (if available)
        if historical_scores:
            # Compare current score distribution with historical
            consistency_component = self._compute_consistency(scores, historical_scores)
        else:
            consistency_component = 0.5  # Neutral if no history
        
        # Weighted combination
        cer = (
            0.4 * separation_component +
            0.3 * variance_component +
            0.3 * consistency_component
        )
        
        return cer

    def _compute_consistency(
        self,
        current_scores: list[float],
        historical_scores: list[list[float]],
    ) -> float:
        """
        Compute consistency between current and historical scores.
        
        Args:
            current_scores: Current move scores
            historical_scores: List of historical score lists
            
        Returns:
            Consistency value (0.0-1.0)
        """
        if not historical_scores:
            return 0.5
        
        # Compare distributions (simplified)
        current_mean = sum(current_scores) / len(current_scores)
        historical_means = [
            sum(scores) / len(scores) for scores in historical_scores if scores
        ]
        
        if not historical_means:
            return 0.5
        
        # Average deviation from historical means
        avg_historical = sum(historical_means) / len(historical_means)
        deviation = abs(current_mean - avg_historical)
        
        # Normalize
        consistency = 1.0 - min(deviation / self.CONSISTENCY_NORMALIZER, 1.0)
        
        return consistency

    def log(
        self,
        IAS: float,
        CER: float,
        move: str | None = None,
        game_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Log metrics with timestamp.
        
        Args:
            IAS: Intrinsic Alignment Score
            CER: Contextual Evaluation Rate
            move: Optional move notation
            game_id: Optional game identifier
            metadata: Optional additional metadata
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "IAS": round(IAS, 3),
            "CER": round(CER, 3),
        }
        
        if move:
            entry["move"] = move
        if game_id:
            entry["game_id"] = game_id
        if metadata:
            entry["metadata"] = metadata
        
        # Add to history
        self.metrics_history.append(entry)
        
        # Write to log file if configured
        if self.log_file:
            self._write_to_file(entry)
        
        logger.info(
            f"game_metrics.log IAS={IAS:.3f} CER={CER:.3f} move={move}"
        )

    def _write_to_file(self, entry: dict[str, Any]) -> None:
        """Write metric entry to log file."""
        try:
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(entry) + '\n')
        except Exception as e:
            logger.error(f"game_metrics._write_to_file error: {e}")

    def get_statistics(self) -> dict[str, Any]:
        """
        Get summary statistics of tracked metrics.
        
        Returns:
            Dictionary with mean, min, max, and std dev of IAS and CER
        """
        if not self.metrics_history:
            return {
                "count": 0,
                "IAS": {"mean": 0.0, "min": 0.0, "max": 0.0, "std": 0.0},
                "CER": {"mean": 0.0, "min": 0.0, "max": 0.0, "std": 0.0},
            }
        
        ias_values = [m["IAS"] for m in self.metrics_history]
        cer_values = [m["CER"] for m in self.metrics_history]
        
        def compute_stats(values: list[float]) -> dict[str, float]:
            mean = sum(values) / len(values)
            variance = sum((v - mean) ** 2 for v in values) / len(values)
            return {
                "mean": round(mean, 3),
                "min": round(min(values), 3),
                "max": round(max(values), 3),
                "std": round(variance ** 0.5, 3),
            }
        
        return {
            "count": len(self.metrics_history),
            "IAS": compute_stats(ias_values),
            "CER": compute_stats(cer_values),
        }

    def clear(self) -> None:
        """Clear metrics history."""
        self.metrics_history = []
        logger.debug("game_metrics.MetricsTracker.clear")
