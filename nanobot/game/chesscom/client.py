"""Chess.com client orchestrator for VTuber integration."""

from __future__ import annotations

import asyncio
from typing import Any

from loguru import logger

from nanobot.game.chesscom.board_recognition import BoardRecognizer
from nanobot.game.chesscom.gui_automation import ChessComAutomation
from nanobot.game.chesscom.screen_capture import ChessComScreenCapture
from nanobot.game.chesscom.tts_integration import TTSIntegration
from nanobot.game.engines.chess_board import BoardStateManager
from nanobot.game.engines.chess_evaluator import MoveEvaluator
from nanobot.game.engines.chess_executor import MoveExecutor
from nanobot.game.engines.chess_moves import MoveGenerator
from nanobot.game.metrics.game_metrics import MetricsTracker
from nanobot.game.vtuber.soul_layer import SoulLayerIntegration
from nanobot.game.vtuber.vtuber_bridge import VTuberOutput


class ChessComClient:
    """
    Main Chess.com client with full VTuber integration.
    
    Orchestrates screen capture, board recognition, move generation,
    VTuber commentary, TTS, and GUI automation for autonomous chess play.
    """

    def __init__(
        self,
        personality: str = "TanyalahD",
        enable_tts: bool = True,
        human_like_play: bool = True,
        auto_play: bool = False,
    ):
        """
        Initialize Chess.com client with full VTuber integration.
        
        Args:
            personality: VTuber personality preset
            enable_tts: Enable text-to-speech commentary
            human_like_play: Enable human-like mouse movements and timing
            auto_play: If True, automatically make moves; if False, suggest only
        """
        self.personality = personality
        self.enable_tts = enable_tts
        self.human_like_play = human_like_play
        self.auto_play = auto_play
        self._paused = False
        
        # Initialize components
        self.screen_capture: ChessComScreenCapture | None = None
        self.board_recognizer = BoardRecognizer()
        self.gui_automation: ChessComAutomation | None = None
        self.tts: TTSIntegration | None = None
        
        # Chess engine components
        self.board_manager = BoardStateManager()
        self.move_generator = MoveGenerator()
        self.move_evaluator = MoveEvaluator()
        self.move_executor = MoveExecutor()
        
        # VTuber components
        self.soul_layer = SoulLayerIntegration(personality=personality)
        self.vtuber_output = VTuberOutput(enable_tts=enable_tts)
        
        # Metrics tracking
        self.metrics_tracker = MetricsTracker()
        
        # Game statistics
        self.stats = {
            "games_played": 0,
            "wins": 0,
            "losses": 0,
            "draws": 0,
            "moves_made": 0,
        }
        
        logger.info(
            f"ChessComClient initialized: personality={personality}, "
            f"tts={enable_tts}, auto_play={auto_play}"
        )

    async def start_game_loop(self) -> None:
        """
        Start the main game loop for autonomous play.
        
        Continuously monitors the board, generates moves, and executes them
        when it's the player's turn.
        """
        logger.info("Starting Chess.com game loop...")
        
        # Initialize screen capture
        self.screen_capture = ChessComScreenCapture()
        
        # Detect board region
        board_region = self.screen_capture.detect_board_region()
        
        # Initialize GUI automation
        if self.auto_play:
            self.gui_automation = ChessComAutomation(
                board_region=board_region,
                human_like=self.human_like_play,
            )
        
        # Initialize TTS if enabled
        if self.enable_tts:
            self.tts = TTSIntegration(provider="local")
        
        try:
            while True:
                # Check if paused
                if self._paused:
                    await asyncio.sleep(1.0)
                    continue
                
                # Check game status
                status = self.screen_capture.get_game_status()
                if status != "playing":
                    logger.debug(f"Game status: {status}, waiting...")
                    await asyncio.sleep(2.0)
                    continue
                
                # Check if it's our turn
                if not self.screen_capture.is_my_turn():
                    await asyncio.sleep(1.0)
                    continue
                
                # Process turn
                result = await self.process_turn()
                
                if result.get("success"):
                    logger.info(
                        f"Move executed: {result['move']} | "
                        f"IAS={result.get('ias', 0):.3f} "
                        f"CER={result.get('cer', 0):.3f}"
                    )
                else:
                    logger.warning(f"Turn processing failed: {result.get('error')}")
                
                # Wait before next check
                await asyncio.sleep(1.0)
                
        except KeyboardInterrupt:
            logger.info("Game loop interrupted by user")
        finally:
            self._cleanup()

    async def process_turn(self) -> dict[str, Any]:
        """
        Process a single turn.
        
        Steps:
        1. Capture board state
        2. Recognize pieces and convert to FEN
        3. Generate and evaluate moves
        4. Select best move using Q·K scoring
        5. Generate VTuber commentary
        6. Speak commentary via TTS
        7. Execute move (if auto_play enabled)
        8. Update metrics
        
        Returns:
            Dict with move, commentary, IAS, CER, and execution status
        """
        try:
            # 1. Capture board
            if self.screen_capture is None:
                return {"success": False, "error": "Screen capture not initialized"}
            
            board_image = self.screen_capture.capture_board()
            
            # 2. Recognize pieces
            board_state = self.board_recognizer.recognize_pieces(board_image)
            fen = self.board_recognizer.to_fen(board_state)
            orientation = self.board_recognizer.detect_orientation(board_image)
            
            # Load board state
            self.board_manager.load_fen(fen)
            
            # 3. Generate legal moves
            legal_moves = self.move_generator.generate_legal_moves_from_board(
                self.board_manager.board
            )
            
            if not legal_moves:
                return {"success": False, "error": "No legal moves available"}
            
            # 4. Evaluate moves
            # Use the score_moves API with board and moves
            # We pass empty move_vectors but provide board and moves for detailed analysis
            move_scores = []
            if legal_moves:
                # Call score_moves with board and moves for detailed evaluation
                # Note: move_vectors is required but we'll use dummy vectors
                dummy_vectors = [[0.0] * 133 for _ in legal_moves]
                move_scores = self.move_evaluator.score_moves(
                    board_vector=self.board_manager.get_board_vector(),
                    move_vectors=dummy_vectors,
                    board=self.board_manager.board,
                    moves=legal_moves,
                )
            
            # 5. Select best move
            best_move_idx = self.move_executor.select_best_move(move_scores)
            best_move = legal_moves[best_move_idx]
            best_score = move_scores[best_move_idx]
            
            # 6. Compute metrics
            ias, cer = self.metrics_tracker.compute_IAS_CER(
                move=best_move,
                scores=move_scores,
                selected_idx=best_move_idx,
            )
            
            # 7. Generate VTuber commentary
            commentary = self.soul_layer.generate_comment(
                move=best_move,
                board=self.board_manager.board,
                score=best_score,
            )
            
            # 8. Output with VTuber formatting
            output = self.vtuber_output.send_move_and_comment(
                move=best_move,
                comment=commentary,
                score=best_score,
            )
            
            # 9. Speak commentary if TTS enabled
            if self.tts:
                await self.tts.speak(commentary, block=False)
            
            # 10. Execute move if auto_play enabled
            executed = False
            if self.auto_play and self.gui_automation:
                executed = self.gui_automation.execute_move(
                    best_move, board_orientation=orientation
                )
            
            # 11. Update statistics
            self.stats["moves_made"] += 1
            
            return {
                "success": True,
                "move": best_move,
                "commentary": commentary,
                "score": best_score,
                "ias": ias,
                "cer": cer,
                "executed": executed,
                "output": output,
            }
            
        except Exception as e:
            logger.error(f"Error processing turn: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    def pause(self) -> None:
        """Pause auto-play mode."""
        self._paused = True
        logger.info("Game loop paused")

    def resume(self) -> None:
        """Resume auto-play mode."""
        self._paused = False
        logger.info("Game loop resumed")

    def get_statistics(self) -> dict[str, Any]:
        """
        Get game statistics.
        
        Returns:
            Dict with win rate, IAS/CER averages, and other stats
        """
        total_games = self.stats["games_played"]
        win_rate = (
            self.stats["wins"] / total_games if total_games > 0 else 0.0
        )
        
        # Get IAS/CER averages from metrics tracker
        if self.metrics_tracker.metrics_history:
            avg_ias = sum(
                m.get("IAS", 0.0) for m in self.metrics_tracker.metrics_history
            ) / len(self.metrics_tracker.metrics_history)
            avg_cer = sum(
                m.get("CER", 0.0) for m in self.metrics_tracker.metrics_history
            ) / len(self.metrics_tracker.metrics_history)
        else:
            avg_ias = 0.0
            avg_cer = 0.0
        
        return {
            **self.stats,
            "win_rate": win_rate,
            "avg_ias": avg_ias,
            "avg_cer": avg_cer,
        }

    def _cleanup(self) -> None:
        """Clean up resources."""
        if self.screen_capture:
            self.screen_capture.close()
        if self.tts:
            self.tts.stop()
        logger.info("ChessComClient cleanup complete")
