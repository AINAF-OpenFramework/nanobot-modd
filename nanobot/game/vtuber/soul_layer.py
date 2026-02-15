"""Soul Layer Integration for TanyalahD VTuber personality."""

from __future__ import annotations

import random
from typing import Any

import chess
from loguru import logger


class SoulLayerIntegration:
    """
    Generates personality-driven commentary for chess moves.
    
    Integrates TanyalahD VTuber personality traits:
    - Playful and enthusiastic
    - Analytical and thoughtful
    - Encouraging and supportive
    
    Varies responses based on move quality and game situation.
    """

    def __init__(self, personality: str = "TanyalahD"):
        """
        Initialize soul layer integration.
        
        Args:
            personality: VTuber personality preset (default: "TanyalahD")
        """
        self.personality = personality
        self._init_response_templates()
        logger.debug(f"soul_layer.SoulLayerIntegration initialized with personality={personality}")

    def _init_response_templates(self):
        """Initialize response templates for different move types."""
        self.responses = {
            "aggressive": [
                "Ooh, aggressive! I like it~ Let's put on the pressure! 🔥",
                "Going for the attack! That's the spirit! ⚔️",
                "Time to show them who's boss! 💪",
                "Bold move! Let's see how this plays out~",
            ],
            "defensive": [
                "Hmm, playing it safe... smart thinking! 🛡️",
                "Defense is an art too, you know~ 🎨",
                "Protecting our position, I see! Good call~",
                "Safety first! Let's hold the line! 💙",
            ],
            "brilliant": [
                "Wow! That's brilliant! Did you see that coming?! ✨",
                "Incredible! That's a masterpiece of a move! 🌟",
                "Genius! I'm impressed! 🧠",
                "That's... that's amazing! How did you think of that?! 😍",
            ],
            "blunder": [
                "Oh no... um... are you sure about that? 😅",
                "Hmm... that might not be the best idea... 😬",
                "Oopsie! Maybe we should think about that one? 💭",
                "Ah... well... let's see where this goes! 😓",
            ],
            "check": [
                "Check! Putting pressure on the king! 👑",
                "Check! They have to respond to this! ⚡",
                "Aha! Check! Let's see them get out of this~",
            ],
            "checkmate": [
                "CHECKMATE! WE DID IT!! 🎉🎊",
                "Victory! That was amazing! 🏆",
                "Checkmate! Game over! We won!! 🌟",
            ],
            "capture": [
                "Got 'em! One less piece to worry about~ 🎯",
                "Captured! Nice trade! ♟️",
                "Taking that piece! Good riddance~ ✨",
            ],
            "neutral": [
                "Hmm, let me think about this carefully... 🤔",
                "Interesting position... what's the plan? 💭",
                "Okay, developing our pieces nicely~ 📈",
                "Steady progress! Keep it up! 💙",
            ],
            "opening": [
                "Good opening! Let's establish control~ ♟️",
                "Starting strong! I like the setup! 🎯",
                "Classic opening! Solid choice~ 📚",
            ],
            "endgame": [
                "Endgame time! Every move counts now! ⏰",
                "Careful calculation needed here... 🧮",
                "The final stretch! Let's bring it home! 🏁",
            ],
        }

    def generate_comment(
        self,
        move: str,
        board_vector: list[float] | None = None,
        board: chess.Board | None = None,
        score: float | None = None,
        previous_score: float | None = None,
    ) -> str:
        """
        Generate personality-driven commentary for a chess move.
        
        Args:
            move: Move in UCI notation (e.g., "e2e4")
            board_vector: Optional board state vector
            board: Optional chess.Board object for context
            score: Optional move evaluation score
            previous_score: Optional previous position score for comparison
            
        Returns:
            Commentary string with TanyalahD personality
            
        Example:
            >>> soul = SoulLayerIntegration()
            >>> comment = soul.generate_comment("e2e4")
            >>> len(comment) > 0
            True
        """
        # Determine move type and context
        move_type = self._classify_move(move, board, score, previous_score)
        
        # Select appropriate response
        if move_type in self.responses:
            comment = random.choice(self.responses[move_type])
        else:
            comment = random.choice(self.responses["neutral"])
        
        logger.debug(
            f"soul_layer.generate_comment move={move} type={move_type} "
            f"comment_len={len(comment)}"
        )
        
        return comment

    def _classify_move(
        self,
        move: str,
        board: chess.Board | None,
        score: float | None,
        previous_score: float | None,
    ) -> str:
        """
        Classify the move type for appropriate commentary.
        
        Args:
            move: Move in UCI notation
            board: Optional chess.Board
            score: Optional evaluation score
            previous_score: Optional previous position score
            
        Returns:
            Move classification string
        """
        if not board:
            return "neutral"
        
        try:
            chess_move = chess.Move.from_uci(move)
            
            # Check for checkmate
            board_copy = board.copy()
            board_copy.push(chess_move)
            if board_copy.is_checkmate():
                return "checkmate"
            
            # Check for check
            if board_copy.is_check():
                return "check"
            
            # Check for capture
            if board.is_capture(chess_move):
                return "capture"
            
            # Evaluate move quality based on score change
            if score is not None and previous_score is not None:
                score_delta = score - previous_score
                
                if score_delta > 3.0:
                    return "brilliant"
                elif score_delta < -2.0:
                    return "blunder"
                elif score_delta > 1.0:
                    return "aggressive"
                elif score_delta < -0.5:
                    return "defensive"
            
            # Check game phase
            piece_count = len(board.piece_map())
            if piece_count >= 28:  # Opening phase
                return "opening"
            elif piece_count <= 10:  # Endgame
                return "endgame"
            
            # Default to neutral
            return "neutral"
            
        except (ValueError, chess.InvalidMoveError):
            return "neutral"

    def get_game_summary(
        self,
        result: str,
        move_count: int,
        highlights: list[str] | None = None,
    ) -> str:
        """
        Generate a game summary with personality.
        
        Args:
            result: Game result ("1-0", "0-1", "1/2-1/2")
            move_count: Total moves played
            highlights: Optional list of notable moments
            
        Returns:
            Summary commentary
        """
        if result == "1-0":
            summary = f"Victory! That was a great game! {move_count} moves of pure strategy~ 🏆"
        elif result == "0-1":
            summary = f"Tough loss, but we learned a lot! {move_count} moves fought hard! 💪"
        elif result == "1/2-1/2":
            summary = f"A draw! Well-played on both sides~ {move_count} moves of careful play! 🤝"
        else:
            summary = f"Game concluded after {move_count} moves! Thanks for playing~ ✨"
        
        if highlights:
            summary += "\n\nHighlights:\n" + "\n".join(f"- {h}" for h in highlights)
        
        return summary
