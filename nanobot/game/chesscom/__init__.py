"""Chess.com integration package for TanyalahD VTuber AI."""

from nanobot.game.chesscom.board_recognition import BoardRecognizer
from nanobot.game.chesscom.client import ChessComClient
from nanobot.game.chesscom.gui_automation import ChessComAutomation
from nanobot.game.chesscom.screen_capture import ChessComScreenCapture
from nanobot.game.chesscom.tts_integration import TTSIntegration

__all__ = [
    "ChessComScreenCapture",
    "BoardRecognizer",
    "ChessComAutomation",
    "TTSIntegration",
    "ChessComClient",
]
